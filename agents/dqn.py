import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.buffer import Buffer, Transition
import random
from typing import Any, Optional

"""
Deep Q-Network (DQN) implementation.

This module provides the components for building and training a DQN agent,
including the Q-network model, a replay buffer, and the agent class itself.

"""

# ----- DQN Model -----
class DQN(nn.Module):
    """
    Deep Q-Network model.

    A fully connected feedforward neural network with three hidden layers
    and layer normalization. Outputs Q-values for each possible action.

    Args:
        state_dim (int): Dimensionality of the input state space.
        action_dim (int): Number of possible discrete actions.
        hidden_dim (int, optional): Number of hidden units per layer. Defaults to 128.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.out(x)

# ----- DQN Agent -----
class DQNAgent:
    """
    DQN Agent that interacts with and learns from an environment.

    Implements experience replay, epsilon-greedy exploration, and soft updates
    to a target network for training stability.

    Args:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Number of possible actions.
        hidden_dim (int, optional): Number of units in the hidden layers. Defaults to 128.
        buffer_capacity (int, optional): Replay buffer size. Defaults to 10000.
        batch_size (int, optional): Size of each mini-batch used for training. Defaults to 128.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
        lr (float, optional): Learning rate for the optimizer. Defaults to 3e-4.
        epsilon_start (float, optional): Initial exploration rate for epsilon-greedy strategy. Defaults to 1.0.
        epsilon_end (float, optional): Minimum value epsilon can decay to. Defaults to 0.01.
        epsilon_decay (float, optional): Multiplicative factor for epsilon decay after each episode. Defaults to 0.98.
        tau (float, optional): Soft update coefficient for target network parameters. Defaults to 0.01.
    """
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            hidden_dim: int = 128, 
            buffer_capacity: int = 10000, 
            batch_size: int = 128, 
            gamma: float = 0.99, 
            lr: float = 3e-4, 
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01, 
            epsilon_decay: float = 0.98,
            tau: float=0.01
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
        self.buffer = Buffer(self.device, buffer_capacity)
        self.tau = tau

    def select_action(self, state: Any, greedy: bool = False) -> int:
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            state (Any): Current state of the environment.
            greedy (bool, optional): If True, ignores epsilon and always exploits. Defaults to False.

        Returns:
            int: Index of the selected action.
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Explore
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()  # Exploit

    def store_experience(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Stores a transition in the replay buffer.

        Args:
            state (Any): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (Any): Next state observed.
            done (bool): Whether the episode terminated.
        """
        self.buffer.add(Transition(state, action, reward, next_state, done))

    def update_epsilon(self) -> None:
        """
        Updates the epsilon value based on decay rate.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _compute_loss_for_batch(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, s_next: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """
        Computes the temporal difference loss for a batch of transitions.

        Args:
            s (torch.Tensor): Batch of current states.
            a (torch.Tensor): Batch of actions taken.
            r (torch.Tensor): Batch of rewards received.
            s_next (torch.Tensor): Batch of next states.
            done (torch.Tensor): Batch of terminal state indicators.

        Returns:
            torch.Tensor: Smooth L1 loss between predicted and target Q-values.
        """
        # Q(s, a; θ)
        q_sa = self.q_network(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # max_{a'} Q(s', a'; θ⁻)
        with torch.no_grad():
            max_q_s_next = self.target_network(s_next).max(1)[0]
            target_q_sa = r + self.gamma * max_q_s_next * (1 - done)

        loss = F.smooth_l1_loss(q_sa, target_q_sa)
        return loss

    def learn(self) -> Optional[float]:
        """Performs a learning step.

        Samples a batch from the replay buffer, computes the loss, and updates
        the Q-network parameters.

        Returns:
            Optional[float]: The loss value for the current learning step, or None if
                           the buffer does not have enough samples yet.
        """
        if self.buffer.size() < self.batch_size:
            return None

        s, a, r, s_next, done = self.buffer.get_batch(self.batch_size)
        loss = self._compute_loss_for_batch(s, a, r, s_next, done)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """
        Soft updates the target network parameters toward the online network.
        """
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)