import torch
import numpy as np
import math

from NeuralNetwork.NeuralNetwork import NeuralNetwork

device = torch.device("cuda") if torch.cuda.is_available() and torch.cuda.current_device() > 0 else torch.device("cpu")


class AbstractConvNeuralNetwork(NeuralNetwork):
    @staticmethod
    def conv2d_out_size(size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def __init__(self, n_obs, n_actions):
        super(AbstractConvNeuralNetwork, self).__init__(n_obs, n_actions)

    def forward(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state).reshape(1, -1)
        state = state[:, 2:-1].copy()
        state = torch.tensor(state, dtype=torch.float32)
        size = int(math.sqrt(state.shape[-1]))
        state = state.view(-1, 1, size, size)
        state = state.to(device)
        x = self.conv_net.forward(state)
        x = x.view(x.shape[0], -1)
        x = self.fc_net.forward(x)
        return x

    def get_flatten(self):
        return False

