from gym.envs.registration import register

register(
    id='multiple_gym-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='multiple_gym.envs:multipleEnv',              # Expalined in envs/__init__.py
)
# register(
#     id='multiple_gym_extend-v0',
#     entry_point='multiple_gym.envs:multipleEnvExtend',
# )