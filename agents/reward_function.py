import numpy as np

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
            return 20000.0  # Successful goal completion reward

        progress_reward = np.tanh(env.progress_to_goal * 0.1)*10
        # print("progress",progress_reward)
        collisions_this_step = (
            env.agent_collided_with_obstacle_count_after - 
            env.agent_collided_with_obstacle_count_before
        )
        # print("collision",collisions_this_step)

        reward = -0.5 - collisions_this_step*10 + progress_reward 
        # print("reward", reward)
        return reward
