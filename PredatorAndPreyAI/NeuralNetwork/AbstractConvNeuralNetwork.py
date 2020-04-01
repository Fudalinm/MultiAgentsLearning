import torch
import numpy as np

from NeuralNetwork.NeuralNetwork import NeuralNetwork

device = torch.device("cuda") if torch.cuda.is_available() and torch.cuda.current_device() > 0 else torch.device("cpu")


class AbstractConvNeuralNetwork(NeuralNetwork):
    @staticmethod
    def conv2d_out_size(size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def __init__(self, n_obs, n_actions):
        super(AbstractConvNeuralNetwork, self).__init__(n_obs, n_actions)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = state.copy()
        if isinstance(state, np.ndarray) or isinstance(state, LazyFrames):
            state = torch.tensor(state, dtype=torch.float32)
        state = state / 255
        state = state.view(-1, *self.n_obs)
        state = torch.transpose(state, 1, 3).to(device)
        x = self.conv_net.forward(state)
        x = x.view(x.shape[0], -1)
        x = self.fc_net.forward(x)
        return x

    def get_flatten(self):
        return False

