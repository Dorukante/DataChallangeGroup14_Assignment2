import torch
import torch.nn as nn
import torch.optim as optim
from agents.dqn import DQNAgent
from agents.buffer import PPOTransition
import numpy as np
from typing import Tuple, Optional, Dict, Union

class ActorCritic(nn.Module):
    """
    Actor-Critic network used in PPO. It has a shared feature extractor (MLP),
    followed by separate actor (policy) and critic (value) heads.

    Attributes:
        actor (nn.Sequential): Outputs action probabilities (policy head).
        critic (nn.Sequential): Outputs state value estimate (value head).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initializes the Actor-Critic network.

        Args:
            state_dim (int): Dimension of input state.
            action_dim (int): Number of discrete actions.
            hidden_dim (int, optional): Size of hidden layers. Defaults to 128.
        """
        super(ActorCritic, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
)

    def forward(self, x):
        """
        Forward pass through shared layers, policy head, and value head.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, state_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (policy distribution, value estimate)
        """
         
        x = self.fc(x)
        policy_dist = self.actor(x)
        value = self.critic(x)
        return policy_dist, value


class PPOAgent(DQNAgent):
    """
    Proximal Policy Optimization (PPO) Agent extending DQNAgent interface.

    PPO uses actor-critic architecture and generalized advantage estimation (GAE).
    This class reuses parts of DQNAgent for interface consistency.
    """
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 128,
                 batch_size: int = 64, 
                 gamma: float = 0.99, 
                 lr: float = 1e-3, 
                 lam: float = 0.95, 
                 clip_eps: float = 0.2, 
                 entropy_coeff: float = 0.01, 
                 epochs: int = 4, 
                 kl: float = 0.01
    ):
        """
        Initializes the PPO agent.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Number of possible actions.
            hidden_dim (int): Size of the hidden layers.
            batch_size (int): Batch size used in training.
            gamma (float): Discount factor.
            lr (float): Learning rate.
            lam (float): GAE lambda.
            clip_eps (float): Clipping threshold for PPO.
            entropy_coeff (float): Entropy regularization coefficient.
            epochs (int): Number of PPO update epochs per batch.
            kl (float): KL early stopping threshold
        """
        super().__init__(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
            buffer_capacity=None,  # PPO is on-policy, so no capacity limit
            batch_size=batch_size, gamma=gamma, lr=lr,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0
        )

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.lam = lam
        self.clip_epsilon = clip_eps
        self.entropy_coeff = entropy_coeff
        self.epochs = epochs
        self.target_kl = kl


    def select_action(self, state: np.ndarray, greedy: bool = False) -> Union[int, Tuple[int, float, float]]:
        """
        Selects an action based on policy distribution.

        Args:
            state (np.ndarray): Current state.
            greedy (bool): If True, selects deterministic action (argmax).

        Returns:
            If greedy=False: Tuple (action, log_prob, value)
            If greedy=True: Integer action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, value = self.policy(state_tensor)

        dist = torch.distributions.Categorical(probs)

        if greedy:
            action = torch.argmax(probs).item()
            return action  # Evaluation mode: just return action
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), value.item()

    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool, log_prob: float, 
                        value: float) -> None:
        """
        Stores transition in PPO on-policy buffer.

        Args:
            state (np.ndarray): State at time t.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Terminal flag.
            log_prob (float): Log probability of action.
            value (float): Estimated value of state.
        """

        # Store reward
        self.buffer.add(PPOTransition(state, action, reward, next_state, done, log_prob, value))

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Computes Generalized Advantage Estimation (GAE) advantage and returns.

        Args:
            rewards (torch.Tensor): Collected rewards.
            values (torch.Tensor): Predicted state values.
            dones (torch.Tensor): Done flags.
            next_value (float): Bootstrap value for final state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (advantages, target returns)
        """

        # Compute deltas in one vectorized operation
        deltas = rewards + self.gamma * torch.cat([values[1:], torch.tensor([next_value], device=values.device)]) * (1 - dones) - values
        
        # Compute advantages using tensor operations
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # Compute returns in one vectorized operation
        returns = advantages + values
        
        return advantages, returns

    def learn(self) -> Optional[Dict[str, float]]:
        """
        Performs PPO update based on collected trajectory.

        - Uses clipped surrogate objective.
        - Uses normalized GAE advantages.
        - Includes entropy bonus.
        - Uses value clipping for stable critic updates.
        - KL divergence based early stopping inside epochs.

        Returns:
            Optional[Dict[str, float]]: Training metrics including:
                - policy_loss (float)
                - value_loss (float)
                - entropy (float)
                - total_loss (float)
                - early_stopping (Optional[str])
        """

        if self.buffer.is_empty():
            return None

        # Retrieve collected batch data from on-policy buffer
        states, actions, rewards, next_states, dones, log_probs_old, values = self.buffer.get_batch()

        # Estimate value of the final state for bootstrapping
        with torch.no_grad():
            _, next_value = self.policy(next_states[-1].unsqueeze(0))
            next_value = next_value.item()

        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        # Normalize advantages to stabilize policy updates
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0
        early_stopping_message = None

        for epoch in range(self.epochs):
            # Forward pass through policy network
            probs, values_pred = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO clipped objective (actor loss)
            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (exactly same as your code — keep as is)
            values_pred = values_pred.squeeze()
            values_old = values.detach()

            ret_mean = returns.mean()
            ret_std = returns.std() + 1e-8

            returns_norm = (returns - ret_mean) / ret_std
            values_pred_norm = (values_pred - ret_mean) / ret_std
            values_old_norm = (values_old - ret_mean) / ret_std

            value_clip_eps = 0.2
            values_clipped = values_old_norm + (values_pred_norm - values_old_norm).clamp(-value_clip_eps, value_clip_eps)
            value_loss_unclipped = (values_pred_norm - returns_norm).pow(2)
            value_loss_clipped = (values_clipped - returns_norm).pow(2)
            critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            # Total loss
            loss = actor_loss + critic_loss - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics:
            total_policy_loss += actor_loss.item()
            total_value_loss += critic_loss.item()
            total_entropy += entropy.item()

            # # KL-based early stopping:
            # with torch.no_grad():
            #     approx_kl = (log_probs_old - log_probs).mean().item()

            # if approx_kl > 1.5 * self.target_kl:
            #     early_stopping_message = f"Early stopping PPO update at epoch {epoch+1} due to KL={approx_kl:.5f}"
            #     break

        self.buffer.clear()  # Clear on-policy buffer

        return {
            "policy_loss": total_policy_loss / self.epochs,
            "value_loss": total_value_loss / self.epochs,
            "entropy": total_entropy / self.epochs,
            "total_loss": (total_policy_loss + total_value_loss - self.entropy_coeff * total_entropy) / self.epochs,
            "early_stopping": early_stopping_message
        }