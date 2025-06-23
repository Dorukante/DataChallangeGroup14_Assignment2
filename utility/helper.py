import os 
import json
from pathlib import Path
from warnings import warn

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
                f"_tau{args.tau}"
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

    @staticmethod
    def save_eval_results(agent_type, args, world_stats):
        """
        Saves evaluation results to a text file using formatted filename based on agent type and args.

        Args:
            agent_type (str): The agent type ('dqn' or 'ppo').
            args: Namespace object containing argument values.
            world_stats (dict): Dictionary of evaluation statistics.
        """
        out_dir = Path("results/")
        if not out_dir.exists():
            warn("Evaluation output directory does not exist. Creating the directory.")
            out_dir.mkdir(parents=True, exist_ok=True)

        # Construct file name using the argument summary
        args_summary = Helper.format_args_summary(agent_type, args)
        file_name = f"{agent_type}_evaluation_results_{args_summary}"
        out_fp = out_dir / f"{file_name}.txt"

        # Save results to text file
        print("Evaluation complete. Results:")
        with open(out_fp, "w") as f:
            for key, value in world_stats.items():
                f.write(f"{key}: {value}\n")
                print(f"{key}: {value}")


