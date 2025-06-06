import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.dqn import DQNAgent
import numpy as np

class ActorCritic(nn.Module):
    """
    Actor-Critic network used in PPO.

    Attributes:
        actor (nn.Sequential): Outputs action probabilities.
        critic (nn.Linear): Outputs state value estimate.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initializes the Actor-Critic network.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Number of possible actions.
            hidden_dim (int): Size of the hidden layer.
        """
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass through the actor-critic network.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy distribution and state value.
        """
        x = self.fc(x)
        policy_dist = self.actor(x)
        value = self.critic(x)
        return policy_dist, value


class PPOAgent(DQNAgent):
    """
    Proximal Policy Optimization (PPO) Agent extending DQNAgent for shared interface.

    PPO uses an actor-critic architecture and on-policy training.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, buffer_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3,
                 lam=0.95, clip_eps=0.2, entropy_coeff=0.01, epochs=4):
        """
        Initializes the PPO agent.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Number of possible actions.
            hidden_dim (int): Size of the hidden layers.
            buffer_capacity (int): Placeholder for compatibility.
            batch_size (int): Batch size used in training.
            gamma (float): Discount factor.
            lr (float): Learning rate.
            lam (float): GAE lambda.
            clip_eps (float): Clipping threshold for PPO.
            entropy_coeff (float): Entropy regularization coefficient.
            epochs (int): Number of PPO update epochs per batch.
        """
        super().__init__(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
            buffer_capacity=buffer_capacity, batch_size=batch_size, gamma=gamma, lr=lr,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0
        )

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.lam = lam
        self.clip_epsilon = clip_eps
        self.entropy_coeff = entropy_coeff
        self.epochs = epochs

        self.buffer = []  # On-policy buffer

    def select_action(self, state, greedy=False):
        """
        Selects an action given a state.

        Args:
            state (np.ndarray): Current environment state.
            greedy (bool): If True, use deterministic policy (for evaluation).

        Returns:
            Union[int, Tuple[int, float, float]]: Action index or tuple (action, log_prob, value).
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

    def store_experience(self, state, action, reward, next_state, done, log_prob, value):
        """
        Stores a single transition for PPO updates.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Episode done flag.
            log_prob (float): Log probability of the taken action.
            value (float): Estimated state value.
        """
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))

    def compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        """
        Computes Generalized Advantage Estimation (GAE) for trajectory.

        Args:
            rewards (List[float]): Rewards.
            values (List[float]): Value estimates.
            dones (List[bool]): Done flags.
            next_value (float): Value of final state.
            gamma (float): Discount factor.
            lam (float): GAE lambda.

        Returns:
            Tuple[List[float], List[float]]: Advantages and target returns.
        """
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def learn(self):
        """
        Performs PPO update using collected trajectories.

        Returns:
            dict: A dictionary of average policy loss, value loss, entropy, and total loss.
        """
        if not self.buffer:
            return None

        # Unpack and convert buffer
        states, actions, rewards, next_states, dones, log_probs_old, values = zip(*self.buffer)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        log_probs_old = torch.FloatTensor(np.array(log_probs_old)).to(self.device)
        values = torch.FloatTensor(np.array(values)).to(self.device)

        # Estimate value of the final state for bootstrapping
        with torch.no_grad():
            _, next_value = self.policy(torch.FloatTensor(next_states[-1]).unsqueeze(0).to(self.device))
            next_value = next_value.item()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            rewards.tolist(), values.tolist(), dones.tolist(), next_value, self.gamma, self.lam
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0

        for _ in range(self.epochs):
            probs, values_pred = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values_pred.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += actor_loss.item()
            total_value_loss += critic_loss.item()
            total_entropy += entropy.item()

        self.buffer = []  # Clear on-policy buffer

        return {
            "policy_loss": total_policy_loss / self.epochs,
            "value_loss": total_value_loss / self.epochs,
            "entropy": total_entropy / self.epochs,
            "total_loss": (total_policy_loss + 0.5 * total_value_loss - self.entropy_coeff * total_entropy) / self.epochs
        }
