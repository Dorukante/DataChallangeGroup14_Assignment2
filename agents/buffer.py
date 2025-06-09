from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union, Any
import torch
import numpy as np
from collections import deque
import random

@dataclass
class Transition:
    """Base transition class that can be extended for different algorithms."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    
    def to_tuple(self) -> Tuple:
        """Convert transition to tuple for unpacking."""
        return (self.state, self.action, self.reward, self.next_state, self.done)
    
    @classmethod
    def to_tensors(cls, transitions: List['Transition'], device: torch.device) -> Tuple[torch.Tensor, ...]:
        s, a, r, s_next, done = zip(*[t.to_tuple() for t in transitions])
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.LongTensor(a).to(device),
            torch.FloatTensor(np.array(r)).to(device),
            torch.FloatTensor(np.array(s_next)).to(device),
            torch.FloatTensor(np.array(done)).to(device)
        )

@dataclass
class PPOTransition(Transition):
    """PPO-specific transition with additional fields."""
    log_prob: float
    value: float
    
    def to_tuple(self) -> Tuple:
        """Convert transition to tuple for unpacking."""
        return super().to_tuple() + (self.log_prob, self.value)
    
    @classmethod
    def to_tensors(cls, transitions: List['PPOTransition'], device: torch.device) -> Tuple[torch.Tensor, ...]:
        s, a, r, s_next, done, log_probs, values = zip(*[t.to_tuple() for t in transitions])
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.LongTensor(a).to(device),
            torch.FloatTensor(np.array(r)).to(device),
            torch.FloatTensor(np.array(s_next)).to(device),
            torch.FloatTensor(np.array(done)).to(device),
            torch.FloatTensor(log_probs).to(device),
            torch.FloatTensor(values).to(device)
        )

class Buffer:
    """Generic buffer for storing transitions.
    
    Args:
        device (torch.device): Device to store tensors on
        capacity (int): Maximum number of transitions to store
    """
    def __init__(self, device: torch.device, capacity: int = 10000):
        self.device = device
        self.buffer = deque(maxlen=capacity)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
    
    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(transition)
    
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """Get a batch of transitions.
        
        Args:
            batch_size (Optional[int]): If provided, randomly sample this many transitions.
                                      If None, return all transitions.
        
        Returns:
            Tuple[torch.Tensor, ...]: Batch of transitions as tensors
        """
        if self.is_empty():
            return None
            
        # Select transitions to process
        if batch_size is not None:
            transitions = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        else:
            transitions = self.buffer
            
        # Let the transition type handle the conversion
        return Transition.to_tensors(transitions, self.device)

    def __len__(self) -> int:
        return len(self.buffer)