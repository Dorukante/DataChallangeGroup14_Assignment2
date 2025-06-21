# Data Challange - Assignment 2 [Group14]

A reinforcement learning framework for training and evaluating DQN and PPO agents in a continuous 2D environment with obstacles and goals. The project supports both headless and GUI-based training/evaluation, and provides detailed metrics and results output.

---

## Installation

#### 1) Clone the repository

Open Command Prompt and run:

```cmd
git clone https://github.com/Dorukante/DataChallangeGroup14_Assignment2
cd DataChallangeGroup14_Assignment2
```

#### 2) Set up a Python environment

```cmd
python -m venv venv
venv\Scripts\activate
```

#### 3) Install dependencies

```cmd
pip install -r requirements.txt
```

---

## Running the Project

The main entry point is `train.py`, which supports both DQN and PPO agents, multiple environment levels, and various training/evaluation options.

### Basic usage
Running the file without additional arguments will trigger all default values (if applicable).
```cmd
python train.py
```

### Key Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--level_file=<name>` | Environment JSON file (without `.json`) | `warehouse_level_1` |
| `--agent=<dqn\|ppo>` | Agent type | `dqn` |
| `--num_episodes=<int>` | Training episodes | `300` |
| `--max_steps=<int>` | Maximum steps per episode | `3000` |
| `--train-gui` | Show GUI during training | `False` |
| `--test-gui` | Show GUI during evaluation | `False` |
| `--verbose` | Print detailed logs | `False` |

| DQN Arguments | Description | Default |
|----------|-------------|---------|
| `--epsilon_start=<float>` | Initial epsilon value | `1.0` |
| `--epsilon_end=<float>` | Final epsilon value | `0.001` |
| `--epsilon_decay=<float>` | Epsilon decay rate | `0.98` |
| `--buffer=<int>` | Replay buffer size | `10000` |
| `--tau=<float>` | Target network soft update parameter | `0.01` |

| PPO Arguments | Description | Default |
|----------|-------------|---------|
| `--lamda=<float>` | GAE lambda parameter | `0.95` |
| `--clip_eps=<float>` | PPO clipping epsilon | `0.2` |
| `--ppo_epochs=<int>` | Number of PPO update epochs | `4` |
| `--entropy_coeff=<float>` | Entropy bonus coefficient | `0.4` |

#### Example: Train PPO agent with GUI on warehouse level 2

```cmd
python train.py --agent=ppo --level_file=warehouse_level_2 --num_episodes=500 --train-gui
```

#### Example: Train DQN agent headless on warehouse level 1

```cmd
python train.py --agent=dqn --level_file=warehouse_level_1 --num_episodes=200
```

---

## Results and Outputs

- Training metrics are saved as `.json` files in the `results/` directory, named according to agent type, environment, and hyperparameters.
- Evaluation summaries are saved as `.txt` files in `results/`, named according to agent type, environment, and hyperparameters.

### Metrics Explanation

**Training Metrics:**
- `episode`: Episode number
- `avg_reward_per_step`: Average reward received per step in the episode
- `episode_length`: Number of steps taken in the episode
- `epsilon`: Current exploration rate **(DQN agent)**
- `avg_td_loss`: Average temporal difference loss **(DQN agent)**
- `policy_loss`: Policy loss value **(PPO agent)**
- `value_loss`: Value function loss **(PPO agent)**
- `entropy`: Policy entropy **(PPO agent)**
- `total_loss`: Combined loss value **(PPO agent)**

**Evaluation Metrics:**
- `total_time`: Total simulation time in seconds
- `collision_count`: Number of collisions with obstacles
- `goals_reached`: Number of goals successfully reached

---
## Project Structure

```
root/
│
├── agents/
│   ├── dqn.py              # DQN agent and Q-network
│   ├── ppo.py              # PPO agent and Actor-Critic network
│   ├── buffer.py           # Buffers
│   ├── reward_function.py  # Reward calculation logic
│   └── train_agents.py     # Training loops for agents
│
├── environment/
│   ├── continuous_environment.py  # Main environment logic
│   ├── continuous_gui.py          # Pygame-based GUI for visualization
│   ├── warehouse_level_1.json     # Warehouse environment configs
│   ├── warehouse_level_2.json
│   └── warehouse_level_3.json
│
├── results/                # Output directory for metrics and evaluation logs
│
├── utility/
│   └── helper.py           # Helper functions for logging, saving, etc.
│
├── train.py                # Main script for training and evaluation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---
## Environment Overview

The project implements a **continuous 2D environment** where an agent navigates through a world with obstacles and goals. The environment uses physics simulation for realistic movement and collision detection.

### Agent Capabilities

**Sensors:**
- **Ray sensors**: 8 directional sensors that detect distance to obstacles
- **Position awareness**: Agent knows its current (x, y) coordinates
- **Orientation**: Agent tracks its rotation angle

---

### Environment Creation

Environments are defined using `.json` configuration files in the `environment/` directory. Each environment file specifies:

- **Agent starting position**: Initial coordinates (x, y) for the agent
- **Goals**: List of target positions (x, y) the agent must reach
- **Obstacles**: Rectangular barriers that the agent must avoid
- **Environment dimensions**: Size (x, y) of the 2D world (`"extents"`)

Example environment file structure:
```json
{
    "name": "warehouse_level_1",
    "extents": [600, 480],
    "start_position": [40, 240],
    "goals": [
        [520, 240]
    ],
    "obstacles": [{
        "position": [110, 40],
        "size": [75, 120],
        "name": "shelf_A1"
    }]
}
```
