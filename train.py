from continuous_environment import ContinuousEnvironment, AgentState, AgentSensor, RaySensor, RaySensorNoType
import argparse
import sys
import numpy as np
import ast
import json
import os

try:
    from agents.dqn import DQNAgent
    from agents.ppo import PPOAgent
except ImportError:
    print("Warning: RandomAgent not found. Exiting...")
    sys.exit(1)

def reward_func(env, state, action, next_state, done):
    goal_positions = list(env.current_goals.keys())
    if not goal_positions:
        return 10000.0
    # all goals reached needs a big reward or its just going to
    # abuse distance rewards when theres multiple goals

    progress_reward = np.tanh(env.progress_to_goal * 0.01)
    collisions_this_step = env.agent_collided_with_obstacle_count_after - env.agent_collided_with_obstacle_count_before
    reached_goal = next_state[AgentState.JUST_FOUND_GOAL]

    return -0.5 + reached_goal * 30.0 \
           - collisions_this_step + progress_reward * 5 \

def format_args_summary(agent_type: str, args) -> str:
    if agent_type == "dqn":
        return (
            f"{args.level_file}"
            f"_episode{args.num_episodes}"
            f"_max_steps_per_episode{args.max_steps}"
            f"_gamma{args.gamma}"
            f"_lr{args.lr}"
            f"_buffer_capacity{args.buffer}"
            f"_batch_size{args.batch}"
            f"_hidden{args.hidden_dim}"
            f"_eps_start{args.epsilon_start}"
            f"_eps_end{args.epsilon_end}"
            f"_eps_decay{args.epsilon_decay}"
        )
    elif agent_type == "ppo":
        return (
            f"{args.level_file}"
            f"_episode{args.num_episodes}"
            f"_max_steps_per_episode{args.max_steps}"
            f"_gamma{args.gamma}"
            f"_lr{args.lr}"
            f"_buffer_cap{args.buffer}"
            f"_batch_size{args.batch}"
            f"_hidden_dim{args.hidden_dim}"
            f"_lamda{args.lamda}"
            f"_clip{args.clip_eps}"
            f"_entropy{args.entropy_coeff}"
            f"_ppo_epochs{args.ppo_epochs}"
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

def get_results_path(agent, args, results_dir: str) -> str:
    agent_name = agent.__class__.__name__
    args_summary = format_args_summary(args.agent, args)
    filename = f"{agent_name}_training_metrics_{args_summary}.json"
    return os.path.join(results_dir, filename)


def main(args):
    max_steps_per_episode = args.max_steps
    start_position = ast.literal_eval(args.position)

    episode_metrics = []

    results_path = os.path.join(os.path.curdir, "results")

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
        results_path = get_results_path(agent, args, results_path)
        dqn_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics, results_path)

    elif args.agent == "ppo":
        agent = PPOAgent(
            state_dim=agent_state.size(),
            action_dim=env.action_space_size,
            hidden_dim=args.hidden_dim,
            buffer_capacity=args.buffer,
            batch_size=args.batch,
            gamma=args.gamma,
            lr=args.lr,
            lam=args.lamda,
            clip_eps=args.clip_eps,
            entropy_coeff=args.entropy_coeff,
            epochs=args.ppo_epochs
        )
        results_path = get_results_path(agent, args, results_path)
        ppo_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics, results_path)

    else:
        print(f"Error: Unknown agent type {args.agent}")
        sys.exit(1)
   
def dqn_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics, results_path):
    
    step_idx = 0
    for episode in range(args.num_episodes):
        print(f"\n--- Episode {episode + 1} / {args.num_episodes} ---")
        print(f"Agent Epsilon: {agent.epsilon:.4f}")
        continuous_state = env.reset()
        if env.use_gui:
            env.gui.reset()
        done = False
        td_losses = []

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
            if loss is not None:
                td_losses.append(loss)
            if step_idx % 50 == 0:
                agent.update_target_network()

            continuous_state = next_state

            if done:
                break

        agent.update_epsilon()
        if done:
            print(f"Episode finished after {env_step_idx + 1} steps.")
        else:
            print(f"Episode reached max steps ({max_steps_per_episode}).")
        
        # ---- Log episode metrics ----
        avg_td_loss = np.mean(td_losses) if td_losses else None
        episode_metrics.append({
            "episode": episode + 1,
            "avg_reward_per_step": total_reward / (env_step_idx + 1),
            "episode_length": env_step_idx + 1,
            "epsilon": agent.epsilon,
            "avg_td_loss": avg_td_loss,
        })

        with open(results_path, "w") as f:
            json.dump(episode_metrics, f, indent=2)
        print("\nEpisode metrics saved to training_metrics.json")
    
    if env.use_gui:
        env.gui.close()
    print("\nTraining is finished")

    print("\nStarting evaluation...")
    env.evaluate_agent(agent=agent,
                       max_steps=max_steps_per_episode,
                       agent_start_pos=None,
                       random_seed=42,
                       file_prefix="post_training_eval")

def ppo_agent(agent, args, env, max_steps_per_episode, start_position, episode_metrics, results_path):

    step_idx = 0
    for episode in range(args.num_episodes):
        print(f"\n--- Episode {episode + 1} / {args.num_episodes} ---")
        continuous_state = env.reset()
        if env.use_gui:
            env.gui.reset()
        done = False

        total_reward = 0.0
        episode_steps = 0

        while not done and episode_steps < max_steps_per_episode:
            step_idx += 1
            episode_steps += 1

            action, log_prob, value = agent.select_action(continuous_state)
            try:
                next_state, done = env.step(action)
            except Exception as e:
                print(f"Error during env.step(): {e}")
                if env.use_gui:
                    env.gui.close()
                sys.exit(1)

            reward = reward_func(env, continuous_state, action, next_state, done)
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

        print(f"Episode ended after {episode_steps} steps.")

        # ---- PPO Learning Step ----
        metrics = agent.learn()  # returns a dictionary of losses

        # Fallback if learn() returns None
        policy_loss = metrics.get("policy_loss") if metrics else None
        value_loss = metrics.get("value_loss") if metrics else None
        entropy = metrics.get("entropy") if metrics else None
        total_loss = metrics.get("total_loss") if metrics else None

        # ---- Log episode metrics ----
        episode_metrics.append({
            "episode": episode + 1,
            "avg_reward_per_step": total_reward / episode_steps,
            "episode_length": episode_steps,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        })

        with open(results_path, "w") as f:
            json.dump(episode_metrics, f, indent=2)

        print("Episode metrics saved to training_metrics.json")

    if env.use_gui:
        env.gui.close()
    print("\nTraining is finished")

    print("\nStarting evaluation...")
    env.evaluate_agent(
        agent=agent,
        max_steps=max_steps_per_episode,
        agent_start_pos=None,
        random_seed=42,
        file_prefix="post_training_eval"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a continuous environment simulation.")
    parser.add_argument("--level_file", type=str, default="level_1",
                        help="Name of the level JSON file (without .json extension) to load. Default: level_1")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to run. Default: 50")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode. Default: 1000")
    parser.add_argument("--use-gui", action="store_true",
                        help="Run the simulation with the GUI.")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ppo"],
                        help="Type of agent to use. Default: random")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor gamma.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--buffer", type=int, default=10000, help="Maximum capacity of the replay buffer")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for learning")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial value for epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Minimum value for epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.95, help=" Decay rate for epsilon")
    parser.add_argument("--state_dim", type=int, default=0.995, help="  Dimensionality of the state space.")
    parser.add_argument("--hidden_dim", type=int, default=128, help=" Number of units in hidden layers of the DQN")
    parser.add_argument("--position", type=str, default="(3,11)", help="Start position of the agent")
    parser.add_argument("--lamda", type=float, default=0.95, help=" λ for Generalized Advantage Estimation")
    parser.add_argument("--clip_eps", type=float, default=0.2, help=" PPO Clipping Parameter")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO Update Epochs per Rollout")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy Coefficient")

    args = parser.parse_args()
    start_position = ast.literal_eval(args.position)
    main(args)
