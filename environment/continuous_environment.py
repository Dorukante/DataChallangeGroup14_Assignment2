import numpy as np
import pymunk
from typing import Tuple, List, Dict, Optional
from environment.continuous_gui import ContinuousGUI
from environment.environment_config import *  # contains only constants/type-aliases
from environment.environment_entities.goal import Goal
from environment.agent_state import AgentState
import json
from agents.dqn import DQNAgent
from tqdm import trange


class ContinuousEnvironment:
    """Simple continuous environment for a 2D agent with goals and obstacles.
    Obstacles and collision detection uses pymunk to avoid having to do expensive/complex collision detection
    inside python.

    Args:
        agent_initial_state (AgentState): Initial state of the agent.
        goals (list[Vector2]): Initial positions of the goals.
        start (Vector2): Starting position of the agent.
        extents (Tuple[int, int]): Size of the environment, default is (512, 512), note that this only affects where
            additional obstacles are placed.
        additional_obstacles (list[dict]): List of additional obstacles to add to the environment.
    """

    GUI_RENDER_INTERVAL: float = 2.5  # physics seconds, how often to render the GUI

    def __init__(
            self,
            agent_initial_state: AgentState,
            goals: list[Vector2],
            start: Vector2,
            extents: Tuple[int, int] = (512, 512),
            additional_obstacles: list[dict] = None,
            train_gui: bool = False,
            test_gui: bool = False,
    ):
        if additional_obstacles is None:  # avoiding mutable default arguments
            additional_obstacles = []
        self.agent_state: AgentState = agent_initial_state
        self.initial_goal_positions: list[Vector2] = goals
        self.start: Vector2 = start

        self.info: dict = self._reset_info()
        self.world_stats = self._reset_world_stats()

        # number of discrete actions, e.g. 8 for 8 directions (N, NE, E, SE, S, SW, W, NW)
        # note for some other RL algorithms we might want a continuous action space instead
        # but for deep Q-learning we can use discrete actions
        self.action_space_size = 4

        # environment size, note that without boundaries the agent can move outside this
        self.extents = extents

        self.train_gui = train_gui
        self.test_gui = test_gui
        self.gui = None
        if train_gui:
            self.gui = ContinuousGUI(extents=self.extents, window_size=(1024, 768))

        # physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # no gravity

        # define callbacks, these are called by the physics engine when collisions happen
        self.space.on_collision(AGENT_COLLISION_TYPE, GOAL_COLLISION_TYPE,
                                self._on_agent_goal_collision)
        self.space.on_collision(AGENT_COLLISION_TYPE, OBSTACLE_COLLISION_TYPE,
                                self._on_agent_obstacle_collision)

        # create obstacles
        self.other_actors = []  # perhaps it should avoid moving other actors in the future
        self.obstacles = []
        # boundary walls
        self.add_obstacle((0, 0), (self.extents[0], 10))  # bottom wall
        self.add_obstacle((0, 0), (10, self.extents[1]))  # left wall
        self.add_obstacle((self.extents[0] - 10, 0), (10, self.extents[1]))  # right wall
        self.add_obstacle((0, self.extents[1] - 10), (self.extents[0], 10))  # top wall

        for obs_data in additional_obstacles:
            pos = tuple(obs_data["position"])
            size = tuple(obs_data["size"])
            self.add_obstacle(pos, size)

        # things needed for the reward
        self.agent_collided_with_obstacle_count_before = 0  # initialize counter
        self.agent_collided_with_obstacle_count_after = 0

        self.goal_shapes_to_goals: Dict[pymunk.Shape, Goal] = {}  # maps pymunk shapes to Goal objects
        self.current_goals: Dict[Vector2, Goal] = {}  # current goals in the environment, maps position physical object

        self.agent_body: Optional[pymunk.Body] = None  # body of the agent
        self.agent_shape: Optional[pymunk.Circle] = None  # shape of the agent

        self.render_interval_timer = 0.0  # timer for rendering interval
        obstacles: List[Tuple[pymunk.Body, pymunk.Shape]]  # list of obstacles in the environment

        # things needed for reward calculations, but that the agent
        # should not have direct access to (think knowledge of goal positions)
        self.progress_to_goal = 0.0  # progress towards the current goal, used for reward calculation
        # while avoiding giving the position in our state, we can use this to calculate the reward

        self.dist_closest_goal_after = 0.0  # distance to the closest goal after the latest action
        self.dist_closest_goal_before = 0.0  # distance to the closest goal before the latest action

    def reset_goals(self) -> None:
        """Reset the goals in the environment to their initial positions."""

        for goal in self.current_goals:
            self.space.remove(self.current_goals[goal].body, self.current_goals[goal].shape)
        self.current_goals.clear()
        self.goal_shapes_to_goals.clear()
        for goal_start_position in self.initial_goal_positions:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = goal_start_position
            shape = pymunk.Circle(body, GOAL_RADIUS)
            shape.collision_type = GOAL_COLLISION_TYPE
            shape.filter = pymunk.ShapeFilter(
                categories=CATEGORY_GOAL,
            )
            shape.sensor = True  # agent can move through it instead of treating it as solid object
            self.space.add(body, shape)

            goalobject = Goal(goal_start_position, body, shape)

            self.current_goals[goal_start_position] = goalobject
            self.goal_shapes_to_goals[shape] = Goal(goal_start_position, body, shape)

    def reset_agent(self) -> None:
        """Reset the agent to its start position and clear its velocity.
        Works by removing the old agent body and shape, and creating a new one.
        """

        if self.agent_body:
            self.space.remove(self.agent_body, self.agent_shape)

        # create the agent body with a static mass and infinite moment of inertia
        agent_mass = 1.0  # in current implementation does not affect anything but is required for pymunk
        moment = np.inf  # infinite moment of inertia, so the agent does not rotate outside our direct control
        self.agent_body = pymunk.Body(agent_mass, moment)
        self.agent_body.position = self.start
        # define its shape as a circle that collides with everything
        self.agent_shape = pymunk.Circle(self.agent_body, AGENT_RADIUS)
        self.agent_shape.collision_type = AGENT_COLLISION_TYPE
        self.agent_shape.filter = pymunk.ShapeFilter(
            group=1,
            categories=CATEGORY_AGENT,
            mask=CATEGORY_AGENT | CATEGORY_GOAL | CATEGORY_OBSTACLE
        )
        # for tracking the number of collisions with obstacles before/after each action
        self.agent_collided_with_obstacle_count_after = 0
        self.agent_collided_with_obstacle_count_before = 0

        self.space.add(self.agent_body, self.agent_shape)

    @classmethod
    def load_from_file(cls, file_path: str, agent_state: AgentState,
                       train_gui: bool = True, test_gui: bool = True) -> 'ContinuousEnvironment':
        """Loads environment configuration from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing the environment configuration.
            agent_state (AgentState): Initial state of the agent.
            train_gui (bool): Whether to enable GUI for training.
            test_gui (bool): Whether to enable GUI for testing.

            """

        with open(file_path + '.json', 'r') as f:
            config = json.load(f)

        parsed_obstacles = []
        for obs_data in config.get("obstacles", []):
            parsed_obstacles.append({
                "position": tuple(obs_data["position"]),
                "size": tuple(obs_data["size"])
            })

        return cls(
            agent_initial_state=agent_state,
            goals=[tuple(g) for g in config["goals"]],
            start=tuple(config["start_position"]),
            extents=tuple(config["extents"]),
            additional_obstacles=parsed_obstacles,
            train_gui=train_gui,
            test_gui=test_gui,
        )

    def _on_agent_obstacle_collision(self, _arbiter, _space, _data) -> None:
        """Callback for when the agent collides with an obstacle.
        (The physics engine will call this when the agent collides with an obstacle)

        Args:
            _arbiter: The collision arbiter containing information about the collision.
            _space: The pymunk space where the collision occurred.
            _data: Additional data associated with the collision.
        """
        self.agent_collided_with_obstacle_count_after += 1

    def _on_agent_goal_collision(self, arbiter, space, _data) -> None:
        """Callback for when the agent collides with a goal.
        Increments the goals reached count, and removes the goal from the environment.
        (The physics engine will call this when the agent collides with a goal)

        Args:
            arbiter: The collision arbiter containing information about the collision.
            space: The pymunk space where the collision occurred.
            _data: Additional data associated with the collision.
        """

        agent_shape, goal_shape = arbiter.shapes
        goal_obj: Goal = self.goal_shapes_to_goals.get(goal_shape)

        body, shape = goal_obj.body, goal_obj.shape
        space.remove(body, shape)

        del self.current_goals[goal_obj.start_position]
        del self.goal_shapes_to_goals[goal_shape]

        self.agent_state.just_found_goal = True

    def add_obstacle(self, position: Vector2, size: Vector2):
        """Add a static rectangular obstacle to the environment at the given position and size.
        Args:
            position (Vector2): Position of the obstacle in the environment.
            size (Vector2): Size of the obstacle in the environment.
        """
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = tuple([position[0] + size[0] / 2, position[1] + size[1] / 2])
        shape = pymunk.Poly.create_box(body, size)
        shape.collision_type = OBSTACLE_COLLISION_TYPE
        self.space.add(body, shape)
        self.obstacles.append((body, shape))

    def step(self, agent_action: int, time_steps: int = 15,
             dt: float = 1.0 / 30.0,
             render: bool = False) -> Tuple[np.ndarray, bool]:
        """Advance the physics simulation by dt seconds.
        If using GUI, update the display. Returns agents new state, and a terminal flag
        (Unlike the given implementation the reward is not returned here)

        Args:
            agent_action: action to take, one of the ACTION_* constants defined in environment/environment_config.py
            time_steps: number of timesteps to advance the physics simulation, default is 15
            dt: time step in seconds, default is 1/30th of a second (30 FPS) this must be constant and small or the
                simulation will not be stable.
            render: whether to render the GUI, default is False, if True will render the GUI every GUI_RENDER_INTERVAL
                2.5 seconds of physics time. (useful for visual debugging with direct control)
        Returns:
            Tuple[np.ndarray, bool]: The new state of the agent as a numpy array and a boolean indicating if the episode
            is terminal (i.e. if the agent has reached all goals or collided with an obstacle).
        """
        # handle agent state updating before the action is applied
        self.agent_state.just_found_goal = False

        dx: float = 0
        dy: float = 0
        speed = AGENT_SPEED
        if agent_action == ACTION_ROTATE_LEFT:
            self.agent_body.angle -= dt * AGENT_TURN_SPEED * time_steps
            dx = 0
            dy = 0
        elif agent_action == ACTION_ROTATE_RIGHT:
            self.agent_body.angle += dt * AGENT_TURN_SPEED * time_steps
            dx = 0
            dy = 0
        elif agent_action == ACTION_MOVE_FORWARD:
            dx = np.cos(self.agent_body.angle)
            dy = np.sin(self.agent_body.angle)
        elif agent_action == ACTION_MOVE_BACKWARD:
            dx = -np.cos(self.agent_body.angle)
            dy = -np.sin(self.agent_body.angle)
            speed *= 0.2  # move backward is slower

        # --- PRE SIMULATION UPDATES ---
        self.agent_collided_with_obstacle_count_before = self.agent_collided_with_obstacle_count_after
        self.dist_closest_goal_before = self.dist_closest_goal_after

        self.set_agent_velocity(dx * speed, dy * speed)
        # do the simulation
        for _ in range(time_steps):
            # step the physics simulation
            self.space.step(dt)
            self.world_stats["total_time"] += dt

        # --- POST SIMULATION UPDATES ---

        if len(self.current_goals) == 0:
            self.progress_to_goal = 0.0
        else:
            self.dist_closest_goal_after = min(
                goal.body.position.get_distance(
                    self.agent_body.position) for goal in self.current_goals.values())
            self.progress_to_goal = self.dist_closest_goal_before - self.dist_closest_goal_after

        # render the GUI if enabled for training or testing
        if render:
            self.render_interval_timer += dt * time_steps
            if self.render_interval_timer >= self.GUI_RENDER_INTERVAL:
                self.render_interval_timer = 0.0
                self.gui.render(self)

        # update things that changed here
        self.agent_state.position = self.agent_body.position
        self.agent_state.rotation = self.agent_body.angle

        self.world_stats["collision_count"] = self.agent_collided_with_obstacle_count_after

        # do the sensor raycast for the front sensor and update the agent state
        self.agent_state.update_sensors(self.agent_body, self.space)

        is_terminal: bool = (len(self.current_goals) == 0)
        return self.get_agent_state(), is_terminal

    def set_agent_velocity(self, vx: float, vy: float) -> None:
        """Set the agent's linear velocity.
        Args:
            vx (float): Velocity in the x direction.
            vy (float): Velocity in the y direction.
        """
        self.agent_body.velocity = vx, vy

    def get_agent_state(self) -> np.ndarray:
        """Get the current state of the agent within the environment.
        Returns:
            np.ndarray: The agent's state as a numpy array, including position, rotation, and sensor values.
        """
        return self.agent_state.to_numpy()

    def reset(self) -> np.ndarray:
        """Reset the agent to its start position and clear velocity, reset the goals
        Returns:
            np.ndarray: The initial state of the agent after reset.
        """
        self.reset_goals()
        self.reset_agent()
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()
        return self.get_agent_state()

    def _reset_info(self) -> dict:
        """Resets the info dictionary.
        info is a dict with information of the most recent step
        consisting of whether the target was reached or the agent
        moved and the updated agent position.

        This function is kept the same for compatibility with the original environment code.

        Returns:
            dict: Dictionary containing the reset info.
        """
        return {"target_reached": False,
                "agent_moved": False,
                "actual_action": None}

    def _reset_world_stats(self):
        """Reset the world statistics.
        This is a new dictionary that tracks the total time and collision count in the environment.

        This function is kept the same for compatibility with the original environment code.

        Returns:
            dict: Dictionary containing the reset world statistics.
        """
        return {
            "total_time": 0,
            "collision_count": 0,
        }

    def evaluate_agent(self, agent: DQNAgent, max_steps: int = 1000,
                       random_seed: int | float | str | bytes | bytearray = 0):
        """
        Evaluates a trained agent in the environment.

        Args:
            agent (DQNAgent): Trained agent.
            max_steps (int): Max steps per episode.
            random_seed (int | float | str | bytes | bytearray): Random seed for reproducibility.
        """
        np.random.seed(random_seed)

        if self.test_gui:
            if self.gui is None:
                self.gui = ContinuousGUI(extents=self.extents, window_size=(1024, 768))
            self.gui.reset()
            self.render_interval_timer = 0.0

        state = self.reset()
        initial_position = self.agent_body.position
        agent_path = [initial_position]

        for step in trange(max_steps, desc="Evaluating agent"):
            # disable exploration
            action = agent.select_action(state, greedy=True)

            if action not in [0, 1, 2, 3]:
                print(f"Invalid action {action} at step {step}. Aborting evaluation.")
                break

            try:
                next_state, done = self.step(action, render=self.test_gui)
            except Exception as e:
                print(f"Error during env.step(): {e}")
                if self.test_gui:
                    self.gui.close()
                break

            if self.test_gui:
                self.render_interval_timer += self.GUI_RENDER_INTERVAL
                if self.render_interval_timer >= self.GUI_RENDER_INTERVAL:
                    self.render_interval_timer = 0.0
                    try:
                        self.gui.render(self, reward=0)
                    except Exception as e:
                        print(f"Error during GUI render: {e}")
                        if self.test_gui:
                            self.gui.close()
                        break

            agent_path.append(self.agent_body.position)

            state = next_state

            if done or len(self.current_goals) == 0:
                break

        self.world_stats["goals_reached"] = len(self.initial_goal_positions) - len(self.current_goals)
        self.world_stats["collision_count"] = self.agent_collided_with_obstacle_count_after

        return self.world_stats
