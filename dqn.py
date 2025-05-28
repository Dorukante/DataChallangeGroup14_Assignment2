"""
DQN will take continious state space as input and will output Q values for each of the possible actions

1) TODO IMPROVE THE MODEL
2) TODO suitable Reward Function. For now I will make a very somple one
3) TODO improve the selection method
"""

#for dqn
import torch
import torch.nn as nn
import torch.nn.functional as F
#for buffer
from collections import deque
import random

#discrete action space (we already have this)
def action_to_direction(action):
    directions = {
        0: (0, 1),   # Down
        1: (0, -1),  # Up
        2: (-1, 0),  # Left
        3: (1, 0),   # Right
    }
    return directions[action]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- ACTION SELECTION METHOD (ε-greedy) -----
def select_action(state, q_network, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)  # Explore
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()  # Exploit


def reward_func(state, action, next_state, done):
    # Very basic reward shaping: encourage forward movement
    if done:
        return -10.0
    else:
        return 1.0  # Placeholder: tune based on your environment's goal

def loss_func(batch, q_network, target_network, gamma=0.99):
    """
    Computes the DQN loss using the Bellman equation:

        L(θ) = (r + γ * max_{a'} Q(s', a'; θ⁻) - Q(s, a; θ))²

    Parameters:
        batch: tuple (s, a, r, s_next, done)
        q_network: the current Q-network (θ)
        target_network: the target Q-network (θ⁻)
        gamma: discount factor

    Returns:
        MSE loss between target and predicted Q-values.
    """
    # Unpack batch
    s, a, r, s_next, done = batch

    # Move to device
    s       = s.to(device)
    a       = a.to(device)
    r       = r.to(device)
    s_next  = s_next.to(device)
    done    = done.to(device)

    # Q(s, a; θ)
    q_sa = q_network(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # max_{a'} Q(s', a'; θ⁻)
    with torch.no_grad():
        max_q_s_next = target_network(s_next).max(1)[0]
        target = r + gamma * max_q_s_next * (1 - done)

    # Loss: (target - prediction)^2
    loss = F.mse_loss(q_sa, target)

    return loss

#Create a buffer to store the transtions and sample mini-batches for policy
class Buffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)

        return (
            torch.FloatTensor(s),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(s_next),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)
"""
Very simple DQN model. It needs to be improved. 
Apply droput, batch norm, layer norm etc, once the envrironment is set
"""
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)  # Outputs Q-values for each action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # Q-values for all actions
    

