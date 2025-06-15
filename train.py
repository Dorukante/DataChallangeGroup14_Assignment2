from continuous_environment import ContinuousEnvironment, AgentState, RaySensorNoType
from herper import Helper, Reward_Func
import argparse
import sys
import numpy as np
import ast
import json
import os
from tqdm import tqdm

try:
    from agents.dqn import DQNAgent
    from agents.ppo import PPOAgent
except ImportError:
    print("Warning: RandomAgent not found. Exiting...")
    sys.exit(1)

#early stopping for dqn
EARLY_STOPPING_WINDOW = 50
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_DELTA = 0.05

def main(args):
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
            epsilon_decay=args.epsilon_decay
        )
        results_path = Helper.get_results_path(agent, args, results_path)
        dqn_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics, results_path)

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
            epochs=args.ppo_epochs
        )
        results_path = Helper.get_results_path(agent, args, results_path)
        ppo_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics, results_path)

    else:
        print(f"Error: Unknown agent type {args.agent}")
        sys.exit(1)
   
def dqn_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics, results_path):
    step_idx = 0
    episode_range = range(args.num_episodes)

    best_avg_reward = -np.inf
    epochs_no_improve = 0
    reward_history = []
    
    # Setup tqdm progress bar if not verbose
    if not args.verbose:
        episode_range = tqdm(episode_range, desc="Training agent")
        episode_range.set_postfix({'eps': round(agent.epsilon, 3)})

    for episode in episode_range:
        if args.verbose:
            print(f"\n--- Episode {episode + 1} / {args.num_episodes} ---")
            print(f"Agent Epsilon: {agent.epsilon:.4f}")
        
        continuous_state = env.reset()
        if env.train_gui:
            env.gui.reset()
        done = False
        td_losses = []
        total_reward = 0.0

        learn_every = 4

        for env_step_idx in range(max_steps_per_episode):
            step_idx += 1
            action = agent.select_action(continuous_state)
            next_state, done = env.step(action, render=env.train_gui)
            reward = Reward_Func.reward_func(env, continuous_state, action, next_state, done)
            agent.store_experience(continuous_state, action, reward, next_state, done)

            total_reward += reward

            if step_idx % learn_every == 0:
                loss = agent.learn()
                if loss is not None:
                    td_losses.append(loss)

            if step_idx % 50 == 0:
                agent.update_target_network()

            continuous_state = next_state
            if done:
                break

        # Decay epsilon after each episode
        agent.update_epsilon()

        if args.verbose:
            if done:
                print(f"Episode finished after {env_step_idx + 1} steps.")
            else:
                print(f"Episode reached max steps ({max_steps_per_episode}).")
        else:
            episode_range.set_postfix({'eps': round(agent.epsilon, 3)})

        # Log episode metrics
        avg_td_loss = np.mean(td_losses) if td_losses else None
        episode_metrics.append({
            "episode": episode + 1,
            "avg_reward_per_step": total_reward / (env_step_idx + 1),
            "episode_length": env_step_idx + 1,
            "epsilon": agent.epsilon,
            "avg_td_loss": avg_td_loss,
        })
        
        reward_history.append(total_reward / (env_step_idx + 1))
        if len(reward_history) >= EARLY_STOPPING_WINDOW:
            current_avg = np.mean(reward_history[-EARLY_STOPPING_WINDOW:])
            
            if current_avg > best_avg_reward + EARLY_STOPPING_DELTA:
                best_avg_reward = current_avg
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                if args.verbose():
                
                    print(f"\nEarly stopping triggered after episode {episode+1}.")
                break


        # Write metrics to JSON every 10 episodes
        if (episode + 1) % 10 == 0:
            with open(results_path, "w") as f:
                json.dump(episode_metrics, f, indent=2)

    # Save final metrics once more after all episodes
    with open(results_path, "w") as f:
        json.dump(episode_metrics, f, indent=2)

    if env.train_gui and not env.test_gui:
        env.gui.close()

    if args.verbose:
        print("\nTraining is finished\nStarting evaluation...")

    env.evaluate_agent(agent=agent,
                       max_steps=max_steps_per_episode,
                       agent_start_pos=None,
                       random_seed=42,
                       file_prefix="post_training_eval_dqn")
    
    if env.test_gui:
        env.gui.close()


def ppo_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics, results_path):
    episode_range = range(args.num_episodes)
    if not args.verbose:
        episode_range = tqdm(episode_range, desc="Training agent")

    for episode in episode_range:
        if args.verbose and (episode+1) % 10 == 0:
            print(f"Episode {episode+1}/{args.num_episodes}")

        continuous_state = env.reset()
        if env.train_gui:
            env.gui.reset()

        done = False
        total_reward = 0.0
        episode_steps = 0

        while not done and episode_steps < max_steps_per_episode:
            episode_steps += 1
            action, log_prob, value = agent.select_action(continuous_state)

            try:
                next_state, done = env.step(action, render=env.train_gui)
            except Exception as e:
                print(f"Error during env.step(): {e}")
                if env.train_gui:
                    env.gui.close()
                sys.exit(1)

            reward = Reward_Func.reward_func(env, continuous_state, action, next_state, done)
            total_reward += reward

            agent.store_experience(
                state=continuous_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value
            )

            continuous_state = next_state

        # PPO Learning Step
        metrics = agent.learn()

        # Metrics extraction (with safe fallback)
        policy_loss = metrics.get("policy_loss") if metrics else None
        value_loss = metrics.get("value_loss") if metrics else None
        entropy = metrics.get("entropy") if metrics else None
        total_loss = metrics.get("total_loss") if metrics else None
        early_stop = metrics.get("early_stopping") if metrics else None

        episode_metrics.append({
            "episode": episode + 1,
            "avg_reward_per_step": total_reward / episode_steps,
            "episode_length": episode_steps,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
            "early_stopping": early_stop
        })

        # Save metrics every 10 episodes
        if (episode + 1) % 10 == 0:
            with open(results_path, "w") as f:
                json.dump(episode_metrics, f, indent=2)

    # Save final metrics
    with open(results_path, "w") as f:
        json.dump(episode_metrics, f, indent=2)

    if env.train_gui and not env.test_gui:
        env.gui.close()

    if args.verbose:
        print("\nTraining is finished\nStarting evaluation...")

    env.evaluate_agent(agent=agent,
                       max_steps=max_steps_per_episode,
                       agent_start_pos=None,
                       random_seed=42,
                       file_prefix="post_training_eval_ppo")

    if env.test_gui:
        env.gui.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a continuous environment simulation.")
    parser.add_argument("--level_file", type=str, default="level_1",
                        help="Name of the level JSON file (without .json extension) to load. Default: level_1")
    parser.add_argument("--num_episodes", type=int, default=300,
                        help="Number of episodes to run. Default: 50")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Maximum steps per episode. Default: 1000")
    parser.add_argument("--train-gui", action="store_true",
                        help="Run the simulation with the GUI.")
    parser.add_argument("--test-gui", action="store_true",
                        help="Only enable GUI during evaluation phase.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed training information.")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ppo"],
                        help="Type of agent to use. Default: random")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--buffer", type=int, default=10000, help="Maximum capacity of the replay buffer")
    parser.add_argument("--batch", type=int, default=256, help="Batch size for learning")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial value for epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.001, help="Minimum value for epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.98, help=" Decay rate for epsilon")
    parser.add_argument("--hidden_dim", type=int, default=256, help=" Number of units in hidden layers of the DQN")
    parser.add_argument("--position", type=str, default="(3,11)", help="Start position of the agent")
    parser.add_argument("--lamda", type=float, default=0.95, help=" λ for Generalized Advantage Estimation")
    parser.add_argument("--clip_eps", type=float, default=0.2, help=" PPO Clipping Parameter")
    parser.add_argument("--ppo_epochs", type=int, default=6, help="Number of PPO Update Epochs per Rollout")
    parser.add_argument("--entropy_coeff", type=float, default=0.003, help="Entropy Coefficient")

    args = parser.parse_args()
    start_position = ast.literal_eval(args.position)
    main(args)
