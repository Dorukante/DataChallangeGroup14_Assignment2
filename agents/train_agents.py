from tqdm import tqdm
from agents.reward_function import Reward_Func
import sys
import numpy as np

class Train():

    """
    Class that trains the PPO and DQN agents
    """

    @staticmethod
    def train_ppo_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics):
        """
        Train a PPO agent on the given environment.

        Args:
            agent: PPO agent instance.
            args: Namespace object containing arguments.
            env: The environment instance.
            max_steps_per_episode: Maximum allowed steps per episode.
            start_position: Starting position of the agent.
            episode_metrics: List to accumulate training metrics per episode.

        Returns:
            List of dictionaries containing episode metrics after PPO training.
        """
        episode_range = range(args.num_episodes)
        if not args.verbose:
            episode_range = tqdm(episode_range, desc="Training PPO")

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

        if env.train_gui and not env.test_gui:
            env.gui.close()

        return episode_metrics
    
    @staticmethod
    def train_dqn_agent(agent, args, env, max_steps_per_episode, 
                  start_position, episode_metrics):
        """
        Train a DQN agent with early stopping.

        Early stopping is based on average reward over a moving window.
        If no improvement is seen over `early_stop_patience` episodes,
        training stops early.

        Args:
            agent: DQN agent instance.
            args: Namespace object containing arguments.
            env: The environment instance.
            max_steps_per_episode: Maximum allowed steps per episode.
            start_position: Starting position of the agent.
            episode_metrics: List to accumulate training metrics per episode.

        Returns:
            List of dictionaries containing episode metrics after DQN training.
        """

        #early stopping for dqn
        early_stop_window = 50
        early_stop_patience = 25
        early_stop_delta = 0.05

        step_idx = 0
        episode_range = range(args.num_episodes)

        best_avg_reward = -np.inf
        epochs_no_improve = 0
        reward_history = []
        
        # Setup tqdm progress bar if not verbose
        if not args.verbose:
            episode_range = tqdm(episode_range, desc="Training DQN")
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
            if len(reward_history) >= early_stop_window:
                current_avg = np.mean(reward_history[-early_stop_window:])
                
                if current_avg > best_avg_reward + early_stop_delta:
                    best_avg_reward = current_avg
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stop_patience:
                    if args.verbose:
                    
                        print(f"\nEarly stopping triggered after episode {episode+1}.")
                    break

        if env.train_gui and not env.test_gui:
            env.gui.close()

        return episode_metrics