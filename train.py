from continuous_environment import ContinuousEnvironment, AgentState, AgentSensor, RaySensor, RaySensorNoType
import argparse
import sys
import numpy as np
from typing import Tuple, List, Any, Optional
import ast

try:
    from agents.dqn import DQNAgent
except ImportError:
    print("Warning: RandomAgent not found. Exiting...")
    sys.exit(1)

def reward_func(env, state, action, next_state, done):
    goal_positions = list(env.current_goals.keys())
    if not goal_positions:
        return 100.0
    # all goals reached needs a big reward or its just going to
    # abuse distance rewards when theres multiple goals

    progress_reward = np.tanh(env.progress_to_goal * 0.01)
    collisions_this_step = env.agent_collided_with_obstacle_count_after - env.agent_collided_with_obstacle_count_before
    reached_goal = next_state[AgentState.JUST_FOUND_GOAL]

    return -0.5 + reached_goal * 30.0 \
           - collisions_this_step + progress_reward * 5 \

    
def main(args):
    max_steps_per_episode = args.max_steps
    start_position = ast.literal_eval(args.position)

    # define what kind of sensors the agent has
    agent_state: AgentState = AgentState(

        sensors = [
            # front sensor, these determine distance/type
            RaySensor(ray_angle=0, ray_length=1000),
            # semi-front sensors
            RaySensor(ray_angle=np.pi * (1.0 / 32), ray_length=600,
                      ray_offset=(0, 16)),
            RaySensor(ray_angle=np.pi * (-1.0 / 32), ray_length=600,
                      ray_offset=(0, -16)),
            RaySensor(ray_angle=np.pi * (1.0/16), ray_length=400,
                      ray_offset=(0, 16)),
            RaySensor(ray_angle=np.pi * (-1.0/16), ray_length=400,
                      ray_offset=(0, -16)),
            RaySensor(ray_angle=np.pi * (-2.0 / 16), ray_length=300,
                      ray_offset=(0, -32)),
            RaySensor(ray_angle=np.pi * (2.0 / 16), ray_length=300,
                      ray_offset=(0, 32)),

            RaySensor(ray_angle=np.pi / 4, ray_length=100),
            RaySensor(ray_angle=np.pi / 2, ray_length=100),
            RaySensor(ray_angle=3 * np.pi / 4, ray_length=100),
            RaySensor(ray_angle=5 * np.pi / 4, ray_length=100),
            RaySensor(ray_angle=3 * np.pi / 2, ray_length=100),
            RaySensor(ray_angle=7 * np.pi / 4, ray_length=100),

            RaySensorNoType(ray_angle=np.pi, ray_length=50), # back sensor
        ]
    )
    print("Agent State space is size: ", agent_state.size())

    try:
        env = ContinuousEnvironment.load_from_file(args.level_file, agent_state=agent_state, use_gui=args.use_gui)
        print(f"Environment loaded successfully from {args.level_file}.json")
    except FileNotFoundError:
        print(f"Error: {args.level_file}.json not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading environment: {e}")
        sys.exit(1)

    if args.agent == "dqn":
        agent = DQNAgent(
            state_dim=agent_state.size(),
            action_dim=env.action_space_size,
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

    step_idx = 0
    for episode in range(args.num_episodes):
        print(f"\n--- Episode {episode + 1} / {args.num_episodes} ---")
        print(f"Agent Epsilon: {agent.epsilon:.4f}")
        continuous_state = env.reset()
        if env.use_gui:
            env.gui.reset()
        done = False

        total_reward = 0.0
        for env_step_idx in range(max_steps_per_episode):
            step_idx += 1
            # better use this to track steps instead of env_step_idx for low stepcount episodes
            action = agent.select_action(continuous_state)
            if action not in [0,1,2,3]:
                break
            try:
                next_state, done = env.step(action)
            except Exception as e:
                print(f"Error during env.step(): {e}")
                if env.use_gui:
                    env.gui.close()
                sys.exit(1)

            reward = reward_func(env, continuous_state, action, next_state, done)
            total_reward += reward

            agent.store_experience(continuous_state, action, reward, next_state, done)
            loss = agent.learn()
            # if loss is not None:
            #     print(f"Step {env_step_idx + 1}: Reward = {reward:.3f}, Action = {action}, Loss = {loss:.4f}" if loss else "")
            agent.update_epsilon()
            if step_idx % 50 == 0:
                agent.update_target_network()

            continuous_state = next_state

            if done:
                break

        if done:
            print(f"Episode finished after {env_step_idx + 1} steps.")
        else:
            print(f"Episode reached max steps ({max_steps_per_episode}).")
        print(f"Total reward for episode {episode + 1}: {total_reward:.2f}")
        print(f"Average Reward per Step: {total_reward / (env_step_idx + 1):.2f}")
        print("Final state:", continuous_state)
        print(f"Time simulated: {env.world_stats['total_time']:.2f} seconds")
        print(f"Goals remaining: {len(env.current_goals)}")

    if env.use_gui:
        env.gui.close()
    print("\nTraining is finished")

    print("\nStarting evaluation...")
    env.evaluate_agent(agent=agent,
                       max_steps=max_steps_per_episode,
                       agent_start_pos=None,
                       random_seed=42,
                       file_prefix="post_training_eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a continuous environment simulation.")
    parser.add_argument("-l", "--level_file", type=str, default="level_1",
                        help="Name of the level JSON file (without .json extension) to load. Default: level_1")
    parser.add_argument("-e", "--num_episodes", type=int, default=50,
                        help="Number of episodes to run. Default: 50")
    parser.add_argument("-s", "--max_steps", type=int, default=1000,
                        help="Maximum steps per episode. Default: 1000")
    parser.add_argument("--use-gui", action="store_true",
                        help="Run the simulation with the GUI.")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn"],
                        help="Type of agent to use. Default: random")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor gamma.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--buffer", type=int, default=10000, help="Maximum capacity of the replay buffer")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for learning")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial value for epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Minimum value for epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.9999, help=" Decay rate for epsilon")
    parser.add_argument("--state_dim", type=int, default=0.995, help="  Dimensionality of the state space.")
    parser.add_argument("--hidden_dim", type=int, default=128, help=" Number of units in hidden layers of the DQN")
    parser.add_argument("--position", type=str, default="(3,11)", help="Start position of the agent")


    args = parser.parse_args()
    start_position = ast.literal_eval(args.position)
    main(args)
