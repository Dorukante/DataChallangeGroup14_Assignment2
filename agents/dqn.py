"""
Deep Q-Network (DQN) implementation.

This module provides the components for building and training a DQN agent,
including the Q-network model, a replay buffer, and the agent class itself.

1) TODO IMPROVE THE MODEL
2) TODO improve the selection method
"""

#for dqn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#for buffer
from collections import deque
import random
from typing import Tuple, Any, Optional


# ----- DQN Model -----
class DQN(nn.Module):
    """Deep Q-Network model.

    A simple feedforward neural network with two hidden layers.

    Args:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Number of possible actions.
        hidden_dim (int, optional): Number of units in each hidden layer. Defaults to 128.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)  # Outputs Q-values for each action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # Q-values for all actions

# ----- Replay Buffer -----
class Buffer:
    """Replay buffer to store and sample experiences.

    Uses a deque to store experiences up to a maximum capacity.

    Args:
        capacity (int): The maximum number of experiences to store in the buffer.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Adds an experience to the buffer.

        Args:
            state: The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state: The state reached after taking the action.
            done (bool): True if the episode terminated, False otherwise.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple[5 * torch.Tensor]: A tuple containing tensors for states, actions, rewards,
                next states, and done flags.
        """
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)

        return (
            torch.FloatTensor(s),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(s_next),
            torch.FloatTensor(done)
        )

    def __len__(self) -> int:
        return len(self.buffer)

# ----- DQN Agent -----
class DQNAgent:
    """DQN Agent that interacts with and learns from an environment.

    Args:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Number of possible actions.
        hidden_dim (int, optional): Number of units in hidden layers of the DQN. Defaults to 128.
        buffer_capacity (int, optional): Maximum capacity of the replay buffer. Defaults to 10000.
        batch_size (int, optional): Batch size for learning. Defaults to 64.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        epsilon_start (float, optional): Initial value for epsilon. Defaults to 1.0.
        epsilon_end (float, optional): Minimum value for epsilon. Defaults to 0.01.
        epsilon_decay (float, optional): Decay rate for epsilon. Defaults to 0.995.
    """
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            hidden_dim: int = 128, 
            buffer_capacity: int = 10000, 
            batch_size: int = 64, 
            gamma: float = 0.99, 
            lr: float = 1e-3, 
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01, 
            epsilon_decay: float = 0.995
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = Buffer(buffer_capacity)

    def select_action(self, state: Any) -> int:
        """Selects an action using an epsilon-greedy policy.

        With probability epsilon, a random action is chosen (exploration).
        Otherwise, the action with the highest Q-value is chosen (exploitation).

        Args:
            state: The current state from the environment.

        Returns:
            int: The selected action index.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Explore
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()  # Exploit

    def store_experience(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Stores an experience tuple in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update_epsilon(self) -> None:
        """Updates the exploration rate epsilon according to its decay schedule."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _compute_loss_for_batch(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, s_next: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Computes the DQN loss for a given batch of experiences.

        Args:
            s (torch.Tensor): Batch of current states.
            a (torch.Tensor): Batch of actions taken.
            r (torch.Tensor): Batch of rewards received.
            s_next (torch.Tensor): Batch of next states.
            done (torch.Tensor): Batch of done flags (1.0 if terminal, 0.0 otherwise).

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        # Q(s, a; θ)
        q_sa = self.q_network(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # max_{a'} Q(s', a'; θ⁻)
        with torch.no_grad():
            max_q_s_next = self.target_network(s_next).max(1)[0]
            target_q_sa = r + self.gamma * max_q_s_next * (1 - done)

        # Loss: (target - prediction)^2
        loss = F.mse_loss(q_sa, target_q_sa)
        return loss

    def learn(self) -> Optional[float]:
        """Performs a learning step.

        Samples a batch from the replay buffer, computes the loss, and updates
        the Q-network parameters.

        Returns:
            Optional[float]: The loss value for the current learning step, or None if
                           the buffer does not have enough samples yet.
        """
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s_next, done = self.buffer.sample(self.batch_size)

        # Move to device
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s_next = s_next.to(self.device)
        done = done.to(self.device)

        loss = self._compute_loss_for_batch(s, a, r, s_next, done)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """Updates the target Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())