"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np

from agents import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that performs a random action every time. """

    def __init__(self, action_space_size: int):
        super().__init__()
        self.action_space_size = action_space_size

    def update(self, state: tuple[int, int], reward: float, action):
        pass

    def take_action(self, state: np.ndarray) -> int:
        return randint(0, self.action_space_size - 1)