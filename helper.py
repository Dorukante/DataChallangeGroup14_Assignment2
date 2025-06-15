import numpy as np
import os 
import json

class Helper():

    @staticmethod
    def v_print(text: str, verbose: bool):
        """
        Verbose print function.
        only if verbose is True, print the text.
        """
        if verbose:
            print(text)

    @staticmethod
    def format_args_summary(agent_type: str, args) -> str:
        if agent_type == "dqn":
            return (
                f"{args.level_file}"
                f"_e{args.num_episodes}"
                f"_max_steps{args.max_steps}"
                f"_g{args.gamma}"
                f"_lr{args.lr}"
                f"_bcap{args.buffer}"
                f"_bsize{args.batch}"
                f"_hdim{args.hidden_dim}"
                f"_eps_s{args.epsilon_start}"
                f"_eps_e{args.epsilon_end}"
                f"_eps_d{args.epsilon_decay}"
            )
        elif agent_type == "ppo":
            return (
                f"{args.level_file}"
                f"_e{args.num_episodes}"
                f"_max_steps{args.max_steps}"
                f"_g{args.gamma}"
                f"_lr{args.lr}"
                f"_bcap{args.buffer}"
                f"_bsize{args.batch}"
                f"_hdim{args.hidden_dim}"
                f"_lam{args.lamda}"
                f"_c{args.clip_eps}"
                f"_ent{args.entropy_coeff}"
                f"_ppo_ep{args.ppo_epochs}"
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    @staticmethod
    def get_results_path(agent, args, results_dir: str) -> str:
        agent_name = agent.__class__.__name__
        args_summary = Helper.format_args_summary(args.agent, args)
        filename = f"{agent_name}_training_metrics_{args_summary}.json"
        return os.path.join(results_dir, filename)
    
    @staticmethod
    def save_training_metrices(episode_metrics, results_path):
            
        # Save final metrics
        with open(results_path, "w") as f:
            json.dump(episode_metrics, f, indent=2)


class Reward_Func():

    @staticmethod
    def reward_func(env, state, action, next_state, done):
        goal_positions = list(env.current_goals.keys())
        if not goal_positions:
            return 150.0

        progress_reward = np.tanh(env.progress_to_goal * 0.05)
        collisions_this_step = env.agent_collided_with_obstacle_count_after - env.agent_collided_with_obstacle_count_before

        return -0.5 + collisions_this_step + progress_reward * 10 