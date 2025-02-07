from gymnasium.envs.registration import register

register(
    id="InvertedPendulumEnv-v1",  # Unique identifier for the environment
    entry_point="my_envs.invertedPendulumEnv:InvertedPendulumEnv",  # Path to your environment
    max_episode_steps=500,
)

