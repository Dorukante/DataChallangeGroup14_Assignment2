import numpy as np
import pygame
import copy
import pymunk
from typing import Tuple, TypeAlias, List, Dict
from continuous_gui import ContinuousGUI
from collections import deque
import json
from agents.dqn import DQNAgent
from tqdm import trange
from pathlib import Path
from warnings import warn
from datetime import datetime

Vector2: TypeAlias = Tuple[float, float]


ACTION_ROTATE_LEFT = 0
ACTION_ROTATE_RIGHT = 1
ACTION_MOVE_FORWARD = 2
ACTION_MOVE_BACKWARD = 3


class AgentSensor:
    NUMBER_OF_STATE_VALUES: int

    def __init__(self):
        """Initialize the sensor. This is called when the agent is created."""
        self.is_active = True

    def update(self, agentbody: pymunk.Body, space: pymunk.Space) -> np.array:
        """Update the sensor and return the state values it has read."""
        pass


class RaySensor(AgentSensor):
    NUMBER_OF_STATE_VALUES = 2

    name = "RaySensor"

    def __init__(self, ray_length: float = 250, ray_angle: float = 0.0, ray_offset: Vector2 = (0, 0)):
        super().__init__()
        print("Creating new RaySensor with parameters:")
        print(f"  ray_length: {ray_length}")
        print(f"  ray_angle: {ray_angle}")
        print(f"  ray_offset: {ray_offset}")

        self.is_active = True

        self.ray_length = ray_length
        self.ray_angle = ray_angle
        self.ray_offset: pymunk.Vec2d = ray_offset  # not implemented

        self.sensed_object_type: float = 0.0
        self.sensed_object_distance: float = 0.0

        # note this is only for drawing, the agent never receives actual coordinates
        self.sensed_object_position: pymunk.Vec2 = pymunk.Vec2d(0, 0)

        self.sensor_start: pymunk.Vec2 = pymunk.Vec2d(0, 0)
        self.sensor_end: pymunk.Vec2 = pymunk.Vec2d(0, 0)


    def update(self, agentbody: pymunk.Body, space: pymunk.Space) -> np.array:
        self.sensor_start = agentbody.position
        self.sensor_end = (
            self.sensor_start[0] + np.cos(
                agentbody.angle + self.ray_angle) * self.ray_length,
            self.sensor_start[1] + np.sin(
                agentbody.angle + self.ray_angle) * self.ray_length
        )
        sensed_object: pymunk.SegmentQueryInfo = space.segment_query_first(
            agentbody.position,
            self.sensor_end,
            1,
            pymunk.ShapeFilter(group=1)
        )
        if sensed_object is None:
            self.sensed_object_distance = self.ray_length
            self.sensed_object_type = 0.0
            self.sensed_object_position = self.sensor_end
        else:
            # print(f"Sensed object at position {sensed_object.point}, type: {sensed_object.shape.collision_type}")
            self.sensed_object_position = sensed_object.point # only for drawing

            self.sensed_object_distance = sensed_object.point.get_distance(
                agentbody.position)
            if sensed_object.shape.collision_type == ContinuousEnvironment.GOAL_COLLISION_TYPE:
                self.sensed_object_type = AgentState.SENSOR_TYPE_GOAL_VALUE
            elif sensed_object.shape.collision_type == ContinuousEnvironment.AGENT_COLLISION_TYPE:
                self.sensed_object_type = AgentState.SENSOR_TYPE_OTHER_AGENT_VALUE
            elif sensed_object.shape.collision_type == ContinuousEnvironment.OBSTACLE_COLLISION_TYPE:
                self.sensed_object_type = AgentState.SENSOR_TYPE_COLLISION_VALUE
            else:
                raise ValueError(
                    f"Unknown collision type: {sensed_object.shape.collision_type}")
        return np.array([self.sensed_object_distance, self.sensed_object_type])


class RaySensorNoType(RaySensor):
    NUMBER_OF_STATE_VALUES = 1
    name = "RaySensorNoType"
    def update(self, agentbody: pymunk.Body, space: pymunk.Space) -> np.array:
        values = super().update(agentbody, space)
        return np.array([values[0]])  # only return distance, not type

class AgentState:
    """State of the agent in the environment.
    You can modify it to include stuff like sensor readings, like it sensing distance to obstacles in front of it."""

    SENSOR_TYPE_NONE_VALUE: float = 0.0
    SENSOR_TYPE_GOAL_VALUE: float = 10.0
    SENSOR_TYPE_OTHER_AGENT_VALUE: float = -10.0
    SENSOR_TYPE_COLLISION_VALUE: float = -100.0

    ACTIONS_TO_REMEMBER: int = 3

    COLLISION_VALUE_ON_COLLISION: float = 100.0
    COLLISION_VALUE_DECAY: float = 0.5

    def __init__(self, sensors: List[AgentSensor] = []):
        # rotation in radians
        self.rotation = 0.0

        # whether the agent just found a goal
        self.just_found_goal = False

        # value that is set to a maximum on collision then quickly decays
        self.collision_value = 0

        # small deque to remember the last few actions taken by the agent
        # helps prevent loops to some degree but really needs a better solution
        # that allows for longer memory
        self.past_actions: deque[float] = deque([0.0] * AgentState.ACTIONS_TO_REMEMBER,
                                                maxlen=AgentState.ACTIONS_TO_REMEMBER)

        self.sensors = sensors
        self.total_sensor_values = sum(sensor.NUMBER_OF_STATE_VALUES for sensor in sensors)
        self.sensor_values = np.zeros(self.total_sensor_values, dtype=float)
        self.position: pymunk.Vec2 = pymunk.Vec2d(0, 0)

        self.past_sensor_values = np.zeros(self.total_sensor_values, dtype=float)

        self.sensor_time: int = 0

    def update_sensors(self, agent_body: pymunk.Body, space: pymunk.Space = None):
        self.past_sensor_values *= 0.5
        self.past_sensor_values += self.sensor_values
        index: int = 0
        for sensor in self.sensors:
            sensor_values = sensor.update(agent_body, space)
            if len(sensor_values) != sensor.NUMBER_OF_STATE_VALUES:
                raise ValueError(f"Sensor {sensor} returned {len(sensor_values)} values, expected {sensor.NUMBER_OF_STATE_VALUES}")
            self.sensor_values[index:index + sensor.NUMBER_OF_STATE_VALUES] = sensor_values
            index += sensor.NUMBER_OF_STATE_VALUES
        self.position = agent_body.position  # not given in state for now

    def size(self) -> int:
        """Returns the size of the state vector. Static method because the agents will
        initialize independently of the environment.
        modify this if you add more state variables"""
        return len(self.to_numpy())

    JUST_FOUND_GOAL = 0
    ROTATION_INDEX_COS = 1
    ROTATION_INDEX_SIN = 2
    COLLISION_VALUE_INDEX = 3

    def to_numpy(self) -> np.ndarray:
        """Convert the agent state to a numpy array. Most RL algorithms probably prefer this"""
        return np.array([
            self.just_found_goal,
            np.cos(self.rotation),
            np.sin(self.rotation),
            self.collision_value,
            *self.past_actions,
            *self.sensor_values,
            # *self.past_sensor_values,
            # self.position.x,
            # self.position.y,
        ], dtype=np.float32)


class Goal:
    """Small helper class to represent a goal in the environment."""
    body: pymunk.Body  # physical body of the goal
    shape: pymunk.Circle  # physical shape of the goal
    start_position: Vector2  # start position of the goal in the environment, prefer using the body position instead

    def __init__(self, start_position: Vector2, body: pymunk.Body, shape: pymunk.Circle):
        self.start_position = start_position
        self.body = body
        self.shape = shape


class ContinuousEnvironment:
    """Simple continuous environment for a 2D agent with goals and obstacles.
    Obstacles and collision detection uses pymunk to avoid having to do expensive/complex collision detection
    inside python."""

    GOAL_RADIUS = 16  # radius of the goal circles
    AGENT_RADIUS = 16  # radius of the agent circle
    AGENT_SPEED = 64  # units per second
    AGENT_TURN_SPEED = np.pi / 6
    AGENT_FRONT_SENSOR_RANGE = 500  # range of the front sensor in pixels

    GOAL_COLLISION_TYPE = 1  # collision type for goals
    AGENT_COLLISION_TYPE = 2  # collision type for agent
    OBSTACLE_COLLISION_TYPE = 3  # collision type for obstacles

    CATEGORY_AGENT = 0b1
    CATEGORY_GOAL = 0b10
    CATEGORY_OBSTACLE = 0b100

    GUI_RENDER_INTERVAL = 0.5  # physics seconds, how often to render the GUI

    def reset_goals(self):
        for goal in self.current_goals:
            self.space.remove(self.current_goals[goal].body, self.current_goals[goal].shape)
        self.current_goals.clear()
        self.goal_shapes_to_goals.clear()
        for goal_start_position in self.initial_goal_positions:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = goal_start_position
            shape = pymunk.Circle(body, self.GOAL_RADIUS)
            shape.collision_type = self.GOAL_COLLISION_TYPE
            shape.sensor = True  # agent can move through it instead of treating it as solid object
            self.space.add(body, shape)

            goalobject = Goal(goal_start_position, body, shape)

            self.current_goals[goal_start_position] = goalobject
            self.goal_shapes_to_goals[shape] = Goal(goal_start_position, body, shape)

    def reset_agent(self):
        if self.agent_body:
            self.space.remove(self.agent_body, self.agent_shape)
        agent_mass = 1.0  # probably unused since we only check for collisions with static obstacles
        moment = np.inf  # infinite moment of inertia, so the agent does not rotate outside of our direct control
        self.agent_body = pymunk.Body(agent_mass, moment)
        self.agent_body.position = self.start
        self.agent_shape = pymunk.Circle(self.agent_body, self.AGENT_RADIUS)
        self.agent_shape.collision_type = self.AGENT_COLLISION_TYPE
        self.agent_shape.filter = pymunk.ShapeFilter(
            group=1,
            categories=self.CATEGORY_AGENT,
            mask=self.CATEGORY_AGENT | self.CATEGORY_GOAL | self.CATEGORY_OBSTACLE
        )
        self.agent_collided_with_obstacle_count_after = 0
        self.agent_collided_with_obstacle_count_before = 0

        self.space.add(self.agent_body, self.agent_shape)

    def __init__(self,
            agent_initial_state: AgentState,
            goals: list[Vector2],
            start: Vector2,
            extents: Tuple[int, int] = (512, 512),
            additional_obstacles: list[tuple[Vector2, Vector2]] = [],
            use_gui: bool = True,
        ):
        self.agent_state = agent_initial_state
        self.initial_goal_positions = goals
        self.start = start

        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()

        # number of discrete actions, e.g. 8 for 8 directions (N, NE, E, SE, S, SW, W, NW)
        # note for some other RL algorithms we might want a continuous action space instead
        # but for deep Q-learning we can use discrete actions
        self.action_space_size = 4

        # environment size, note that without boundaries the agent can move outside this
        self.extents = extents

        self.use_gui = use_gui
        if use_gui:
            self.gui = ContinuousGUI(extents=self.extents, window_size=(1024, 768))

        # physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # no gravity
        self.space.on_collision(self.AGENT_COLLISION_TYPE, self.GOAL_COLLISION_TYPE, self._on_agent_goal_collision)
        self.space.on_collision(self.AGENT_COLLISION_TYPE, self.OBSTACLE_COLLISION_TYPE, self._on_agent_obstacle_collision)

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

        self.agent_body: pymunk.Body = None  # body of the agent
        self.agent_shape: pymunk.Circle = None  # shape of the agent

        self.render_interval_timer = 0.0  # timer for rendering interval
        obstacles: List[Tuple[pymunk.Body, pymunk.Shape]]  # list of obstacles in the environment

        # things needed for reward calculations, but that the agent
        # should not have direct access to (think knowledge of goal positions)
        self.progress_to_goal = 0.0  # progress towards the current goal, used for reward calculation
        # while avoiding giving the position in our state, we can use this to calculate the reward

        self.dist_closest_goal_after = 0.0  # distance to the closest goal after the latest action
        self.dist_closest_goal_before = 0.0  # distance to the closest goal before the latest action


    @classmethod
    def load_from_file(cls, file_path: str, agent_state:AgentState, use_gui: bool = True) -> 'ContinuousEnvironment':
        """Loads environment configuration from a JSON file."""
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
            use_gui=use_gui
        )

    def _on_agent_obstacle_collision(self, arbiter, space, data) -> None:
        self.agent_collided_with_obstacle_count_after += 1
        self.agent_state.collision_value = AgentState.COLLISION_VALUE_ON_COLLISION
        #print(f"Agent collided with an obstacle. Total collisions: {self.agent_collided_with_obstacle_count}")

    def _on_agent_goal_collision(self, arbiter, space, data) -> None:
        """Callback for when the agent collides with a goal.
        Increments the goals reached count."""

        agent_shape, goal_shape = arbiter.shapes
        goal_obj: Goal = self.goal_shapes_to_goals.get(goal_shape)

        body, shape = goal_obj.body, goal_obj.shape
        space.remove(body, shape)

        del self.current_goals[goal_obj.start_position]
        del self.goal_shapes_to_goals[goal_shape]

        self.agent_state.just_found_goal = True

    def add_obstacle(self, position: Vector2, size: Vector2):
        """Add a static rectangular obstacle."""
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = tuple([position[0] + size[0] / 2, position[1] + size[1] / 2])
        shape = pymunk.Poly.create_box(body, size)
        shape.collision_type = self.OBSTACLE_COLLISION_TYPE
        self.space.add(body, shape)
        self.obstacles.append((body, shape))

    def step(self, agent_action: int, time_steps: int = 5, dt: float = 1.0 / 30.0):
        """Advance the physics simulation by dt seconds.
        If using GUI, update the display. Returns agents new state, and a terminal flag
        (Unlike the given implementation the reward is not returned here)

        timesteps: number of timesteps to advance the simulation, default is 5 can consider this like the reaction time of the agent
        (eg it can change action only every 5 * dt seconds)
        dt: time step in seconds, default is 1/30th of a second (30 FPS) this must be constant and small or the
        simulation will not be stable.
        """
        # handle agent state updating before the action is applied
        self.agent_state.just_found_goal = False

        dx: float = 0
        dy: float = 0
        speed = self.AGENT_SPEED
        if agent_action == ACTION_ROTATE_LEFT:
            self.agent_body.angle -= dt * self.AGENT_TURN_SPEED * time_steps
            dx = 0
            dy = 0
        elif agent_action == ACTION_ROTATE_RIGHT:
            self.agent_body.angle += dt * self.AGENT_TURN_SPEED * time_steps
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
        self.agent_state.collision_value *= AgentState.COLLISION_VALUE_DECAY

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

        if self.use_gui:
            self.render_interval_timer += dt * time_steps
            if self.render_interval_timer >= self.GUI_RENDER_INTERVAL:
                self.render_interval_timer = 0.0
                self.gui.render(self)

        # update things that changed here
        self.agent_state.position = self.agent_body.position
        self.agent_state.rotation = self.agent_body.angle

        self.world_stats["collision_count"] = self.agent_collided_with_obstacle_count_after

        # update the memory of the agent state
        self.agent_state.past_actions.append(float(agent_action))

        # do the sensor raycast for the front sensor and update the agent state
        self.agent_state.update_sensors(self.agent_body, self.space)

        is_terminal: bool = (len(self.current_goals) == 0)
        return self.get_agent_state(), is_terminal

    def set_agent_velocity(self, vx: float, vy: float):
        """Set the agent's linear velocity."""
        self.agent_body.velocity = vx, vy

    def get_agent_state(self) -> np.ndarray:
        """Get the current state of the agent within the environment."""
        return self.agent_state.to_numpy()

    def reset(self):
        """Reset the agent to its start position and clear velocity, reset the goals"""
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
        """
        return {"target_reached": False,
                "agent_moved": False,
                "actual_action": None}

    def _reset_world_stats(self):
        """Reset the world statistics."""
        return {
            "total_time": 0,
            "collision_count":0,
        }
    def evaluate_agent(self,
                       agent: DQNAgent,
                       max_steps: int = 1000,
                       show_images: bool = False,
                       agent_start_pos: tuple[int, int] = None,
                       random_seed: int | float | str | bytes | bytearray = 0,
                       file_prefix: str = "evaluation"):
        """
        Evaluates a trained DQN agent in the environment.

        Args:
            agent (DQNAgent): Trained agent.
            max_steps (int): Max steps per episode.
            show_images (bool): Whether to display final path image.
            agent_start_pos (tuple[int, int]): Optional start position for agent.
            random_seed (int | float | str | bytes | bytearray): Random seed for reproducibility.
            file_prefix (str): Prefix for result filenames.
        """
        np.random.seed(random_seed)

        if agent_start_pos is not None:
            self.start = agent_start_pos

        goals_reached = 0

        state = self.reset()
        initial_position = self.agent_body.position
        agent_path = [initial_position]

        for step in trange(max_steps, desc="Evaluating agent"):
            #disable exploration
            action = agent.select_action(state, greedy=True)

            if action not in [0, 1, 2, 3]:
                print(f"Invalid action {action} at step {step}. Aborting evaluation.")
                break

            try:
                next_state, done = self.step(action)
            except Exception as e:
                print(f"Error during env.step(): {e}")
                break

            agent_path.append(self.agent_body.position)

            state = next_state

            if done or len(self.current_goals) == 0:
                break

        self.world_stats["goals_remaining"] = len(self.current_goals)
        self.world_stats["goals_reached"] = len(self.initial_goal_positions) - len(self.current_goals)
        self.world_stats["collision_count"] = self.agent_collided_with_obstacle_count_after

        file_name = f"{file_prefix}__" + datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        save_results(file_name, self.world_stats)


def save_results(file_name, world_stats):
    out_dir = Path("results/")
    if not out_dir.exists():
        warn("Evaluation output directory does not exist. Creating the "
             "directory.")
        out_dir.mkdir(parents=True, exist_ok=True)

    # Print evaluation results
    print("Evaluation complete. Results:")
    # Text file
    out_fp = out_dir / f"{file_name}.txt"
    with open(out_fp, "w") as f:
        for key, value in world_stats.items():
            f.write(f"{key}: {value}\n")
            print(f"{key}: {value}")
