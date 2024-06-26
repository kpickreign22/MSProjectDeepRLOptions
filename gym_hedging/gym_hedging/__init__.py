from gymnasium.envs.registration import register

# register(
#      id="test-env",
#      entry_point="gym_examples.envs:GridWorldEnv"
#     #  max_episode_steps=300,
# )
register(id='DeltaHedging-v0', entry_point="gym_hedging.envs:DeltaHedging")