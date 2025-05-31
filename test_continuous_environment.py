from continuous_environment import ContinuousEnvironment

from agents.random_agent import RandomAgent


max_steps_per_episode = 1000
env = ContinuousEnvironment(
    goals=[(50, 50), (200, 50), (400, 50),
           (50, 200),             (400, 200),
           (50, 400), (200, 400), (400, 400)],
    start=(200, 200),
    additional_obstacles=[((130, 150), (300, 50)), ((250, 250), (50, 50))],
    use_gui=True
)


agent: RandomAgent = RandomAgent(action_space_size=env.action_space_size)

for episode in range(5):
    continuous_state = env.reset()
    print("Initial state:", continuous_state)
    done = False
    for env_step_idx in range(max_steps_per_episode):
        action = agent.take_action(continuous_state)
        continuous_state, done = env.step(action)
        if done:
            break
    if done:
        print("Episode finished after", env_step_idx + 1, "steps.")
    else:
        print("Episode ended without reaching a terminal state.")
    print("Final state:", continuous_state)
    print("Time simulated: ", env.world_stats["total_time"], " of phsyics time in seconds.")
