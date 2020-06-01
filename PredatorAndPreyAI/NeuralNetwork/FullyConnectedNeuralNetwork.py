from torch import nn
import torch
import numpy as np

from NeuralNetwork.NeuralNetwork import NeuralNetwork

device = torch.device("cuda") if torch.cuda.is_available() and torch.cuda.current_device() > 0 else torch.device("cpu")


class FullyConnectedNeuralNetwork(NeuralNetwork):
    def __init__(self, n_obs, n_actions):
        super(FullyConnectedNeuralNetwork, self).__init__(n_obs, n_actions)
        self.fc_dims = [256]
        fc_layers = self.create_fully_connected_layers(n_obs, n_actions)
        self.fc_net = nn.Sequential(*fc_layers).to(device)

    def forward(self, state):
        if not isinstance(state, np.ndarray):
            state = np.asarray(state)
            state = state.reshape((1, -1))
        state = torch.tensor(state, dtype=torch.float32)
        x = self.fc_net.forward(state.to(device))
        return x

    def get_flatten(self):
        return True
