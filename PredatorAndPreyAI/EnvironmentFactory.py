import gym

class EnvironmentFactory:
    def create(self):
        env = gym.make('PredatorPrey5x5-v0', n_agents=2)
        # env = gym.make('PredatorPrey5x5-v0', n_agents=2, grid_shape=(10, 10))
        return env
