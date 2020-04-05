import gym

class EnvironmentFactory:
    def create(self):
        env = gym.make('PredatorPrey5x5-v0', n_agents=4)
        return env
