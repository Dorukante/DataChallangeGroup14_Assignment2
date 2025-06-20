import numpy as np

class Reward_Func:
    """
    A static reward function class for computing custom reward signals
    based on the agent's interaction with the environment.
    
    Designed for grid-based navigation or goal-seeking environments with:
    - Progress tracking toward goals
    - Obstacle collisions
    - Episode termination
    """

    @staticmethod
    def reward_func(env, state, action, next_state, done) -> float:
        """
        Computes the reward signal based on the agent's transition.

        The reward is composed of:
        - A constant base penalty (-0.5) to encourage shorter paths
        - A penalty for each collision with obstacles (-10 per collision)
        - A progress reward toward the goal, scaled and smoothed using tanh

        Additionally, if all goals are completed (i.e., no current goals remain),
        the agent receives a large terminal reward (+20000).

        Args:
            env: The environment instance, expected to expose:
                - current_goals (dict)
                - progress_to_goal (float)
                - agent_collided_with_obstacle_count_before (int)
                - agent_collided_with_obstacle_count_after (int)
            state: Current state before the action.
            action: Action taken by the agent.
            next_state: Resulting state after the action.
            done (bool): Whether the episode has terminated.

        Returns:
            float: Computed reward value.
        """
        goal_positions = list(env.current_goals.keys())
        if not goal_positions:
            return 20000.0  # Large reward for completing all goals

        progress_reward = np.tanh(env.progress_to_goal * 0.1) * 10

        collisions_this_step = (
            env.agent_collided_with_obstacle_count_after - 
            env.agent_collided_with_obstacle_count_before
        )

        reward = -0.5 - collisions_this_step * 10 + progress_reward
        return reward
