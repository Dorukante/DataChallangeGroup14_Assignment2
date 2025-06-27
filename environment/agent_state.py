import numpy as np
import pymunk
from environment.environment_config import * # only contains type aliases and constants


class AgentState:
    """State of the agent in the environment.
    You can modify it to include stuff like sensor readings, like it sensing distance to obstacles in front of it."""

    def __init__(self, sensors=None):
        if sensors is None:
            sensors = []
        self.rotation = 0.0
        self.sensors = sensors
        self.total_sensor_values = sum(sensor.NUMBER_OF_STATE_VALUES for sensor in sensors)
        self.sensor_values = np.zeros(self.total_sensor_values, dtype=float)
        self.position: pymunk.Vec2d = pymunk.Vec2d(0, 0)

    def update_sensors(self, agent_body: pymunk.Body, space: pymunk.Space = None):
        """Update each sensor and store the values in the sensor_values array."""
        index: int = 0
        for sensor in self.sensors:
            sensor_values = sensor.update(agent_body, space)
            if len(sensor_values) != sensor.NUMBER_OF_STATE_VALUES:
                raise ValueError(f"Sensor {sensor} returned {len(sensor_values)} values, expected {sensor.NUMBER_OF_STATE_VALUES}")
            self.sensor_values[index:index + sensor.NUMBER_OF_STATE_VALUES] = sensor_values
            index += sensor.NUMBER_OF_STATE_VALUES
        self.position = agent_body.position

    def size(self) -> int:
        """Returns the size of the state vector. Static method because the agents will
        initialize independently of the environment.
        modify this if you add more state variables"""
        return len(self.to_numpy())

    # indices for the state vector, avoid using these outside of debugging
    # (if you need specific values simply take them from the environment instead)
    POSITION_INDEX_X = 0
    POSITION_INDEX_Y = 1
    ROTATION_INDEX_COS = 2
    ROTATION_INDEX_SIN = 3

    def to_numpy(self) -> np.ndarray:
        """Convert the agent state to a numpy array. This is to be utilized directly by the agents."""
        return np.array([
            self.position.x,
            self.position.y,
            np.cos(self.rotation),
            np.sin(self.rotation),
            *self.sensor_values,
        ], dtype=np.float32)
