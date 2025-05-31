from continuous_environment import ContinuousEnvironment
import argparse
import sys

try:
    from agents.random_agent import RandomAgent
except ImportError:
    print("Warning: RandomAgent not found. Exiting...")
    sys.exit(1)

def main(args):
    """Main function to run the environment simulation."""
    max_steps_per_episode = args.max_steps

    # Load environment from file
    try:
        env = ContinuousEnvironment.load_from_file(args.level_file, use_gui=args.use_gui)
        print(f"Environment loaded successfully from {args.level_file}.json")
    except FileNotFoundError:
        print(f"Error: {args.level_file}.json not found. Please ensure the file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading environment from {args.level_file}.json: {e}")
        sys.exit(1)

    if args.agent == "random":
        agent = RandomAgent(action_space_size=env.action_space_size)
    elif args.agent == "dqn":
        raise NotImplementedError("DQN agent not implemented yet")
    else:
        print(f"Error: Unknown agent type {args.agent}")
        sys.exit(1)

    for episode in range(args.num_episodes):
        print(f"\n--- Episode {episode + 1} / {args.num_episodes} ---")
        continuous_state = env.reset()
        if env.use_gui:
            env.gui.reset()
        print("Initial state:", continuous_state)
        done = False
        for env_step_idx in range(max_steps_per_episode):
            action = agent.take_action(continuous_state)
            try:
                continuous_state, done = env.step(action)
            except Exception as e:
                print(f"Error during env.step(): {e}")
                if env.use_gui:
                    env.gui.close()
                sys.exit(1)
                
            if done:
                break
        
        if done:
            print(f"Episode finished after {env_step_idx + 1} steps.")
        else:
            print(f"Episode reached max steps ({max_steps_per_episode}) or ended without explicit termination signal.")
        print("Final state:", continuous_state)
        total_time_simulated = env.world_stats["total_time"]
        print(f"Time simulated: {total_time_simulated:.2f} of physics time in seconds.")
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
    parser.add_argument("--agent", type=str, default="random", choices=["random", "dqn"],
                        help="Type of agent to use. Default: random")

    args = parser.parse_args()
    main(args)
