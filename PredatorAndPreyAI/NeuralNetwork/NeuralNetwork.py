import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_obs, n_actions):
        super(NeuralNetwork, self).__init__()
        self.n_obs = n_obs
        self.n_actions = n_actions

    def forward(self, state):
        pass

    def create_fully_connected_layers(self, n_start, n_end):
        if not self.fc_dims:
            fc_layers = [torch.nn.Linear(n_start, n_end), torch.nn.ReLU()]
        else:
            fc_layers = [torch.nn.Linear(n_start, self.fc_dims[0]), torch.nn.ReLU()]
            for idx in range(len(self.fc_dims) - 1):
                fc_layers += [torch.nn.Linear(self.fc_dims[idx], self.fc_dims[idx + 1]), torch.nn.ReLU()]
            fc_layers += [torch.nn.Linear(self.fc_dims[-1], n_end)]
        return fc_layers
