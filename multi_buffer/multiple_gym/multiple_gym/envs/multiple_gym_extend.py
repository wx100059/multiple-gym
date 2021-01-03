import gym

class multipleEnvExtend(gym.Env):
    def __init__(self):
        print('CustomEnvExtend Environment initialized')
    def step(self):
        print('CustomEnvExtend Step successful!')
    def reset(self):
        print('CustomEnvExtend Environment reset')