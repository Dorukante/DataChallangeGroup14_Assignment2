from continuous_environment import ContinuousEnvironment, AgentState, RaySensorNoType
from helper import Helper
import argparse
import sys
import numpy as np
import ast
import os

try:
    from agents.dqn import DQNAgent
    from agents.ppo import PPOAgent
    from agents.train_agents import Train

except ImportError:
    print("Warning: Agent not found. Exiting...")
    sys.exit(1)


def main(args):
    """
    Main Function. Sets up environment, agents, training loop, 
    evaluation, and metrics saving for DQN and PPO agents.

    Args:
        args: Parsed argparse command-line arguments.
    """

    max_steps_per_episode = args.max_steps
    start_position = ast.literal_eval(args.position)

    episode_metrics = []

    results_path = os.path.join(os.path.curdir, "results")

    # define what kind of sensors the agent has
    agent_state: AgentState = AgentState(

        sensors = [
            # RaySensorNoType(ray_angle=0, ray_length=1000),
            RaySensorNoType(ray_angle=0, ray_length=600, verbose=args.verbose),
            RaySensorNoType(ray_angle=np.pi * 0.5, ray_length=600, verbose=args.verbose),
            RaySensorNoType(ray_angle=np.pi * -0.5, ray_length=600, verbose=args.verbose),
            RaySensorNoType(ray_angle=np.pi, ray_length=600, verbose=args.verbose),

            RaySensorNoType(ray_angle=np.pi * 1.25, ray_length=600, verbose=args.verbose),
            RaySensorNoType(ray_angle=np.pi * 0.75, ray_length=600, verbose=args.verbose),
            RaySensorNoType(ray_angle=np.pi * 0.25, ray_length=600, verbose=args.verbose),
            RaySensorNoType(ray_angle=np.pi * 1.75, ray_length=600, verbose=args.verbose),

        ]
    )
    Helper.v_print(f"Agent State space is size: {agent_state.size()}", args.verbose)

    try:
        env = ContinuousEnvironment.load_from_file(args.level_file, agent_state=agent_state, train_gui=args.train_gui, test_gui=args.test_gui)
        Helper.v_print(f"Environment loaded successfully from {args.level_file}.json", args.verbose)
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
            epsilon_decay=args.epsilon_decay,
            tau = args.tau
        )

        results_path = Helper.get_results_path(agent, args, results_path)
        Train.train_dqn_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics)

        #save the training metrices
        Helper.save_training_metrices(episode_metrics, results_path)

        if args.verbose:
            print("\nTraining is finished\nStarting evaluation...")

        #evaluate the agent
        env.evaluate_agent(agent=agent,
                       max_steps=max_steps_per_episode,
                       agent_start_pos=None,
                       random_seed=42,
                       file_prefix="post_training_eval_dqn")
        
        if env.test_gui:
            env.gui.close()

    elif args.agent == "ppo":
        agent = PPOAgent(
            state_dim=agent_state.size(),
            action_dim=env.action_space_size,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch,
            gamma=args.gamma,
            lr=args.lr,
            lam=args.lamda,
            clip_eps=args.clip_eps,
            entropy_coeff=args.entropy_coeff,
            epochs=args.ppo_epochs,
            kl = args.kl
            
        )
        results_path = Helper.get_results_path(agent, args, results_path)
        episode_metrics = Train.train_ppo_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics)

        #save the training metrices
        Helper.save_training_metrices(episode_metrics, results_path)

        if args.verbose:
            print("\nTraining is finished\nStarting evaluation...")

        #evaluate the agent
        env.evaluate_agent(agent=agent,
                       max_steps=max_steps_per_episode,
                       agent_start_pos=None,
                       random_seed=42,
                       file_prefix="post_training_eval_ppo")
        
        if env.test_gui:
            env.gui.close()

    else:
        print(f"Error: Unknown agent type {args.agent}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a continuous environment simulation.")
    # Common Arguments
    parser.add_argument("--level_file", type=str, default="level_1",
                        help="Name of the level JSON file (without extension). Default: level_1")
    parser.add_argument("--num_episodes", type=int, default=300,
                        help="Number of episodes to train. Default: 300")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Maximum steps per episode. Default: 3000")
    parser.add_argument("--train-gui", action="store_true",
                        help="Run environment GUI during training.")
    parser.add_argument("--test-gui", action="store_true",
                        help="Run GUI during evaluation phase only.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed training information.")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ppo"],
                        help="Agent type: dqn or ppo. Default: dqn")

    # Shared hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (gamma).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--buffer", type=int, default=10000, help="Replay buffer capacity (only used for DQN).")
    parser.add_argument("--batch", type=int, default=256, help="Batch size for learning.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer size.")
    parser.add_argument("--position", type=str, default="(3,11)", help="Agent start position as tuple string.")

    # DQN specific
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon (DQN only).")
    parser.add_argument("--epsilon_end", type=float, default=0.001, help="Final epsilon (DQN only).")
    parser.add_argument("--epsilon_decay", type=float, default=0.98, help="Epsilon decay rate (DQN only).")

    # PPO specific
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE lambda parameter (PPO only).")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon.")
    parser.add_argument("--ppo_epochs", type=int, default=6, help="Number of PPO update epochs.")
    parser.add_argument("--entropy_coeff", type=float, default=0.003, help="Entropy bonus coefficient.")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update tau")
    parser.add_argument("--kl", type=float, default=0.01, help="Target KL for PPO early stopping.")

    args = parser.parse_args()
    start_position = ast.literal_eval(args.position)
    main(args)

 