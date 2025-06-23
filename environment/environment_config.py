"""
Project wide environment constants/macros, that must be accessible by multiple modules while avoiding
circular imports/strong dependencies.
Prefer internal constants unless they need to be accessed by multiple modules.

These are internal constants; not meant to be changed by the user. Changing these may break the environment or
cause unexpected behavior, as the physics engine accuracy is highly dependent on these values.

See environment level .json files for user-configurable parameters.
"""

from typing import Tuple, TypeAlias
import numpy as np


Vector2: TypeAlias = Tuple[float, float]

ACTION_ROTATE_LEFT = 0
ACTION_ROTATE_RIGHT = 1
ACTION_MOVE_FORWARD = 2
ACTION_MOVE_BACKWARD = 3

GOAL_RADIUS = 16  # radius of the goal circles - this must
AGENT_RADIUS = 16  # radius of the agent circle
AGENT_SPEED = 64  # units per second
AGENT_TURN_SPEED = np.pi / 9  # rotation per action step in radians Note that this is multiplied by the time step Delta

GOAL_COLLISION_TYPE = 1  # collision type for goals
AGENT_COLLISION_TYPE = 2  # collision type for agent
OBSTACLE_COLLISION_TYPE = 4  # collision type for obstacles

CATEGORY_AGENT = 0b1  # bitmask for agents
CATEGORY_GOAL = 0b10  # bitmask for goals
CATEGORY_OBSTACLE = 0b100  # bitmask for obstacles

# note these are used only in RaySensor with type detection, which are NOT used in any experiments.
SENSOR_TYPE_NONE_VALUE: float = 0.0
SENSOR_TYPE_OTHER_AGENT_VALUE: float = -10.0
SENSOR_TYPE_COLLISION_VALUE: float = -100.0
