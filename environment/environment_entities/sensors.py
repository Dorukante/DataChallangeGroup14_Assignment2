import pymunk
from environment.environment_config import *  # only contains type aliases and constants


class AgentSensor:
    """Base class for agent sensors. All sensors should inherit from this class."""

    # number of state values returned by the sensor when updated.
    # essential for determining how big the state vector will be
    NUMBER_OF_STATE_VALUES: int

    # used for identifying how to visualize the sensor in the GUI without having to pass all the class types
    name = "UnknownSensor"

    def __init__(self):
        self.is_active = True

    def update(self, agentbody: pymunk.Body, space: pymunk.Space) -> np.array:
        """Update the sensor and return the state values it has read."""
        pass


class RaySensor(AgentSensor):
    """Ray sensor that detects obstacles and other agents in the environment.
    Args:
        ray_length (float): Length of the ray in pixels.
        ray_angle (float): Angle of the ray in radians, relative to the agent's orientation.
        ray_offset (Vector2): Offset of the ray from the agent's position (not implemented).
        verbose (bool): If True, prints debug information about the sensor creation.
    """

    NUMBER_OF_STATE_VALUES = 2

    name = "RaySensor"

    def __init__(self, ray_length: float = 250, ray_angle: float = 0.0, ray_offset: Vector2 = (0, 0), verbose=False):
        super().__init__()
        if verbose:
            print("Creating new RaySensor with parameters:")
            print(f"  ray_length: {ray_length}")
            print(f"  ray_angle: {ray_angle}")
            print(f"  ray_offset: {ray_offset}")

        self.is_active = True
        self.verbose = verbose

        self.ray_length = ray_length
        self.ray_angle = ray_angle

        self.sensed_object_type: float = 0.0
        self.sensed_object_distance: float = 0.0

        # note this is only for drawing, the agent never receives actual coordinates
        self.sensed_object_position: pymunk.Vec2d = pymunk.Vec2d(0, 0)

        self.sensor_start: pymunk.Vec2d = pymunk.Vec2d(0, 0)
        self.sensor_end: pymunk.Vec2d = pymunk.Vec2d(0, 0)

    def update(self, agentbody: pymunk.Body, space: pymunk.Space) -> np.array:
        """Cast a ray from the agent's position in the direction of the ray_angle
        and return the distance to the first sensed object and its type. 
        
        Note that this class should not be used in current experiments since it does not detect goals and there is
        only one agent: meaning it will pass a useless value.
        Prefer using RaySensorNoType which does not return the type of the sensed object.
        
        Args:
            agentbody (pymunk.Body): The body of the agent to which this sensor is attached.
            space (pymunk.Space): The pymunk space in which the ray is cast.
        Returns:
            np.array: An array containing the distance to the sensed object and its type.
        """

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
            pymunk.ShapeFilter(group=1,  # our agents group
                               mask=CATEGORY_OBSTACLE),
        )
        if self.verbose:
            print(f"RaySensor: Sensing from {self.sensor_start} to {self.sensor_end}")
            print(f"\tsensed: {sensed_object})")

        if sensed_object is None:
            self.sensed_object_distance = self.ray_length
            self.sensed_object_type = 0.0
            self.sensed_object_position = self.sensor_end
        else:
            # print(f"Sensed object at position {sensed_object.point}, type: {sensed_object.shape.collision_type}")
            self.sensed_object_position = sensed_object.point  # only for drawing

            self.sensed_object_distance = sensed_object.point.get_distance(
                agentbody.position)

            assert(sensed_object.shape.collision_type != GOAL_COLLISION_TYPE), \
                "RaySensor should not sense goals, only obstacles and other agents."
            if sensed_object.shape.collision_type == AGENT_COLLISION_TYPE:
                self.sensed_object_type = SENSOR_TYPE_OTHER_AGENT_VALUE
            elif sensed_object.shape.collision_type == OBSTACLE_COLLISION_TYPE:
                self.sensed_object_type = SENSOR_TYPE_COLLISION_VALUE
            else:
                raise ValueError(
                    f"Unknown collision type: {sensed_object.shape.collision_type}")
        return np.array([self.sensed_object_distance, self.sensed_object_type])


class RaySensorNoType(RaySensor):
    """Ray sensor that detects obstacles and only gives the distance to the sensed object."""

    NUMBER_OF_STATE_VALUES = 1
    name = "RaySensorNoType"

    def update(self, agentbody: pymunk.Body, space: pymunk.Space) -> np.array:
        """Update the sensor and return only the distance to the sensed object.
        Reuses the RaySensor's update method but only returns the distance value.
        (This is useful for keeping verbose output informative as we maintain access to the collision type even
        if the agent does not receive it)

        Args:
            agentbody (pymunk.Body): The body of the agent to which this sensor is attached.
            space (pymunk.Space): The pymunk space in which the ray is cast."""
        values = super().update(agentbody, space)
        return np.array([values[0]])  # only return distance, not type
