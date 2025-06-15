import numpy as np
import os 
import json

class Helper:
    """
    Utility helper class containing static methods for logging, 
    argument formatting, file saving, and verbosity control.
    """

    @staticmethod
    def v_print(text: str, verbose: bool):
        """
        Print message only if verbose mode is enabled.

        Args:
            text (str): Message to print.
            verbose (bool): If True, print the message.
        """
        if verbose:
            print(text)

    @staticmethod
    def format_args_summary(agent_type: str, args) -> str:
        """
        Formats argument values into a string summary for filename generation.

        Args:
            agent_type (str): The agent type ('dqn' or 'ppo').
            args: Namespace object containing command-line arguments.

        Returns:
            str: Formatted string summarizing key arguments.
        """
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
        """
        Constructs full file path for saving training metrics based on agent type and hyperparameters.

        Args:
            agent: The agent instance.
            args: Namespace object with arguments.
            results_dir (str): Directory where results will be stored.

        Returns:
            str: Full file path for saving metrics.
        """
        agent_name = agent.__class__.__name__
        args_summary = Helper.format_args_summary(args.agent, args)
        filename = f"{agent_name}_training_metrics_{args_summary}.json"
        return os.path.join(results_dir, filename)
    
    @staticmethod
    def save_training_metrices(episode_metrics, results_path):
        """
        Saves training metrics to JSON file.

        Args:
            episode_metrics (list): List of episode dictionaries containing metrics.
            results_path (str): Full path to output file.
        """
        with open(results_path, "w") as f:
            json.dump(episode_metrics, f, indent=2)


class Reward_Func:
    """
    Reward function class providing static reward calculation method.
    """

    @staticmethod
    def reward_func(env, state, action, next_state, done) -> float:
        """
        Computes the reward signal based on current environment state and agent's behavior.

        Reward combines:
        - Base penalty (-0.5)
        - Collision penalty (number of collisions in this step)
        - Goal progress reward (scaled by progress and passed through tanh)

        Args:
            env: The environment instance.
            state: Current state before action.
            action: Action taken.
            next_state: Next state after action.
            done (bool): Whether episode terminated.

        Returns:
            float: Computed reward value.
        """
        goal_positions = list(env.current_goals.keys())
        if not goal_positions:
            return 150.0  # Successful goal completion reward

        progress_reward = np.tanh(env.progress_to_goal * 0.05)
        collisions_this_step = (
            env.agent_collided_with_obstacle_count_after - 
            env.agent_collided_with_obstacle_count_before
        )

        reward = -0.5 + collisions_this_step + progress_reward * 10 
        return reward
