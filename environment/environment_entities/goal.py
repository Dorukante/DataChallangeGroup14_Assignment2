import pymunk
from environment.environment_config import *  # only contains type aliases and constants


class Goal:
    """Small helper class to represent a goal in the environment, holds reference to the pymunk body, shape
    and the integer tuple start_position which can be used to identify it.

    Args:
        start_position (Vector2): Initial position of the goal in the environment.
        body (pymunk.Body): The physical body of the goal.
        shape (pymunk.Circle): The physical shape of the goal, typically a circle.
    """
    body: pymunk.Body  # physical body of the goal
    shape: pymunk.Circle  # physical shape of the goal
    start_position: Vector2  # start position of the goal in the environment, prefer using the body position instead

    def __init__(self, start_position: Vector2, body: pymunk.Body, shape: pymunk.Circle):
        self.start_position = start_position
        self.body = body
        self.shape = shape
