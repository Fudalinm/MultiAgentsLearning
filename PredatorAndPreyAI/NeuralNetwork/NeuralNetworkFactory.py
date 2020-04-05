from NeuralNetwork.FullyConnectedNeuralNetwork import FullyConnectedNeuralNetwork
from NeuralNetwork.SimpleConvNeuralNetwork import SimpleConvNeuralNetwork
import numpy as np

class NeuralNetworkFactory:
    def __init__(self, env):
        self.n_obs = env.observation_space[0].shape[0]
        self.n_actions = env.action_space[0].n
        print(self.n_obs)
        print(self.n_actions)

    def create(self, nn_id):
        if nn_id == 'fully-connected':
            return FullyConnectedNeuralNetwork(self.n_obs, self.n_actions)
        elif nn_id == 'simple-conv':
            return SimpleConvNeuralNetwork(self.n_obs, self.n_actions)
        else:
            raise Exception('Invalid id of neural network')
