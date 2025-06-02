import numpy as np
import pygame
import copy
import pymunk
from typing import Tuple, TypeAlias, List, Dict
from continuous_gui import ContinuousGUI
import json

Vector2: TypeAlias = Tuple[float, float]


class AgentState:
    """State of the agent in the environment.
    You can modify it to include stuff like sensor readings, like it sensing distance to obstacles in front of it"""
    def __init__(self, position: Vector2):
        self.position = position
        self.goals_reached = 0  # number of goals reached, can be used for reward shaping

    @staticmethod
    def size():
        """Returns the size of the state vector. Static method because likely the agents will
        initialize independently of the environment."""
        return 3  # modify this if you add more state variables

    def to_numpy(self) -> np.ndarray:
        """Convert the agent state to a numpy array. Most RL algorithms probably prefer this"""
        return np.array([
            self.goals_reached,
            self.position[0],
            self.position[1],
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

    GOAL_COLLISION_TYPE = 1  # collision type for goals
    AGENT_COLLISION_TYPE = 2  # collision type for agent

    goal_shapes_to_goals: Dict[pymunk.Shape, Goal] = {}  # maps pymunk shapes to Goal objects
    current_goals: Dict[Vector2, Goal] = {}  # current goals in the environment, maps position physical object

    agent_body: pymunk.Body = None  # body of the agent
    agent_shape: pymunk.Circle = None  # shape of the agent

    obstacles: List[Tuple[pymunk.Body, pymunk.Shape]]  # list of obstacles in the environment

    def reset_goals(self):
        for goal in self.current_goals:
            self.space.remove(self.current_goals[goal].body, self.current_goals[goal].shape)
        self.current_goals.clear()
        self.goal_shapes_to_goals.clear()
        for goal_start_position in self.goal_positions:
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
        moment = pymunk.moment_for_circle(agent_mass, 0, self.AGENT_RADIUS)
        self.agent_body = pymunk.Body(agent_mass, moment)
        self.agent_body.position = self.start
        self.agent_shape = pymunk.Circle(self.agent_body, self.AGENT_RADIUS)
        self.agent_shape.collision_type = self.AGENT_COLLISION_TYPE
        self.space.add(self.agent_body, self.agent_shape)

        self.agent_state = AgentState(self.start)

    def __init__(self, 
            goals: list[Vector2],
            start: Vector2,
            extents: Tuple[int, int] = (512, 512),
            additional_obstacles: list[tuple[Vector2, Vector2]] = [],
            use_gui: bool = True
        ):
        self.goal_positions = goals
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

    @staticmethod
    def action_to_direction(action: int) -> Tuple[int, int]:
        """Maps a discrete action index to a 2D direction vector (Down, Up, Left, Right)."""
        directions = {
            0: (0, 1),   # Down
            1: (0, -1),  # Up
            2: (-1, 0),  # Left
            3: (1, 0),   # Right
        }
        return directions[action]

    @classmethod
    def load_from_file(cls, file_path: str, use_gui: bool = True) -> 'ContinuousEnvironment':
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
            goals=[tuple(g) for g in config["goals"]],
            start=tuple(config["start_position"]),
            extents=tuple(config["extents"]),
            additional_obstacles=parsed_obstacles,
            use_gui=use_gui
        )

    def _on_agent_goal_collision(self, arbiter, space, data) -> None:
        """Callback for when the agent collides with a goal.
        Increments the goals reached count.
        Todo ensure we return a terminal state in step when all goals are reached."""
        agent_shape, goal_shape = arbiter.shapes
        goal_obj: Goal = self.goal_shapes_to_goals.get(goal_shape)

        body, shape = goal_obj.body, goal_obj.shape
        space.remove(body, shape)

        del self.current_goals[goal_obj.start_position]
        del self.goal_shapes_to_goals[goal_shape]

        self.agent_state.goals_reached += 1

        print(f"Goal reached at position {goal_obj.body.position}. Total goals reached: {self.agent_state.goals_reached}")


    def add_obstacle(self, position: Vector2, size: Vector2):
        """Add a static rectangular obstacle."""
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = tuple([position[0] + size[0] / 2, position[1] + size[1] / 2])
        shape = pymunk.Poly.create_box(body, size)
        self.space.add(body, shape)
        self.obstacles.append((body, shape))

    def step(self, agent_action: int, timesteps = 5, dt=1 / 30.0):
        """Advance the physics simulation by dt seconds.
        If using GUI, update the display. Returns agents new state, and a terminal flag
        (Unlike the given implementation the reward is not returned here)

        timesteps: number of timesteps to advance the simulation, default is 5 can consider this like the reaction time of the agent
        (eg it can change action only every 5 * dt seconds)
        dt: time step in seconds, default is 1/30th of a second (30 FPS) this must be constant and small or the
        simulation will not be stable.
        """

        dx, dy = self.action_to_direction(agent_action)
        speed = self.AGENT_SPEED
        self.set_agent_velocity(dx * speed, dy * speed)

        for _ in range(timesteps):
            # step the physics simulation
            self.space.step(dt)
        if self.use_gui:
            self.gui.render(self)

        # update things that changed here
        self.agent_state.position = self.agent_body.position

        self.world_stats["total_time"] += dt

        return self.get_agent_state(), False

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
        }