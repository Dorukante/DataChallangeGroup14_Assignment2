from continuous_environment import ContinuousEnvironment, AgentState
import argparse
import sys
import numpy as np
from typing import Tuple, List, Any, Optional

try:
    from agents.dqn import DQNAgent
except ImportError:
    print("Warning: RandomAgent not found. Exiting...")
    sys.exit(1)

def reward_func(env, state, action, next_state, done):
    goal_positions = list(env.current_goals.keys())
    if not goal_positions:
        return 10.0  # all goals reached

    # Distance to the closest goal before and after
    prev_dist = min(np.linalg.norm(np.array(state[1:]) - np.array(g)) for g in goal_positions)
    next_dist = min(np.linalg.norm(np.array(next_state[1:]) - np.array(g)) for g in goal_positions)

    progress_reward = prev_dist - next_dist
    return progress_reward - 0.1

    
def main(args):
    max_steps_per_episode = args.max_steps

    try:
        env = ContinuousEnvironment.load_from_file(args.level_file, use_gui=args.use_gui)
        print(f"Environment loaded successfully from {args.level_file}.json")
    except FileNotFoundError:
        print(f"Error: {args.level_file}.json not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading environment: {e}")
        sys.exit(1)

    if args.agent == "dqn":
        agent = DQNAgent(
            state_dim=AgentState.size(),
            action_dim=4,
            hidden_dim=args.hidden_dim,
            buffer_capacity=args.buffer,
            batch_size=args.batch,
            gamma=args.gamma,
            lr=args.lr,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay
        )
    else:
        print(f"Error: Unknown agent type {args.agent}")
        sys.exit(1)

    for episode in range(args.num_episodes):
        print(f"\n--- Episode {episode + 1} / {args.num_episodes} ---")
        continuous_state = env.reset()
        if env.use_gui:
            env.gui.reset()
        done = False

        for env_step_idx in range(max_steps_per_episode):
            action = agent.select_action(continuous_state)
            try:
                next_state, done = env.step(action)
            except Exception as e:
                print(f"Error during env.step(): {e}")
                if env.use_gui:
                    env.gui.close()
                sys.exit(1)

            reward = reward_func(env, continuous_state, action, next_state, done)

            agent.store_experience(continuous_state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                print(f"Step {env_step_idx + 1}: Reward = {reward:.3f}, Loss = {loss:.4f}" if loss else "")
            agent.update_epsilon()
            if env_step_idx % 50 == 0:
                agent.update_target_network()

            continuous_state = next_state

            if done:
                break

        if done:
            print(f"Episode finished after {env_step_idx + 1} steps.")
        else:
            print(f"Episode reached max steps ({max_steps_per_episode}).")
        print("Final state:", continuous_state)
        print(f"Time simulated: {env.world_stats['total_time']:.2f} seconds")
        print(f"Goals remaining: {len(env.current_goals)}")

    if env.use_gui:
        env.gui.close()
    print("\nSimulation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a continuous environment simulation.")
    parser.add_argument("-l", "--level_file", type=str, default="level_1",
                        help="Name of the level JSON file (without .json extension) to load. Default: level_1")
    parser.add_argument("-e", "--num_episodes", type=int, default=5,
                        help="Number of episodes to run. Default: 5")
    parser.add_argument("-s", "--max_steps", type=int, default=1000,
                        help="Maximum steps per episode. Default: 1000")
    parser.add_argument("--use-gui", action="store_true",
                        help="Run the simulation with the GUI.")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn"],
                        help="Type of agent to use. Default: random")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--buffer", type=int, default=10000, help="Maximum capacity of the replay buffer")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for learning")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial value for epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Minimum value for epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help=" Decay rate for epsilon")
    parser.add_argument("--state_dim", type=int, default=0.995, help="  Dimensionality of the state space.")
    parser.add_argument("--hidden_dim", type=int, default=128, help=" Number of units in hidden layers of the DQN")
    

    args = parser.parse_args()
    main(args)
