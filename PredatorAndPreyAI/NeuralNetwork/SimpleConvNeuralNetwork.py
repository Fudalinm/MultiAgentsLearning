import torch
from NeuralNetwork.AbstractConvNeuralNetwork import AbstractConvNeuralNetwork

device = torch.device("cuda") if torch.cuda.is_available() and torch.cuda.current_device() > 0 else torch.device("cpu")


class SimpleConvNeuralNetwork(AbstractConvNeuralNetwork):
    def __init__(self, n_obs, n_actions):
        super(SimpleConvNeuralNetwork, self).__init__(n_obs, n_actions)
        self.fc_dims = [512]
        out_height = self.n_obs[0]
        out_width = self.n_obs[1]

        conv_layers = []

        print(out_height, out_width, 1, out_height * out_width)

        conv_layers.append(torch.nn.Conv2d(self.n_obs[2], 32, kernel_size=8, stride=4))
        out_height = self.conv2d_out_size(out_height, 8, 4)
        out_width = self.conv2d_out_size(out_width, 8, 4)
        conv_layers.append(torch.nn.ReLU())
        # conv_layers.append(torch.nn.MaxPool2d(2))
        # out_height = (int)(out_height / 2)
        # out_width = (int)(out_width / 2)
        print(out_height, out_width, 32, out_height * out_width * 32)

        conv_layers.append(torch.nn.Conv2d(32, 64, kernel_size=4, stride=2))
        out_height = self.conv2d_out_size(out_height, 4, 2)
        out_width = self.conv2d_out_size(out_width, 4, 2)
        conv_layers.append(torch.nn.ReLU())
        # conv_layers.append(torch.nn.MaxPool2d(2))
        # out_height = (int)(out_height / 2)
        # out_width = (int)(out_width / 2)
        print(out_height, out_width, 64, out_height * out_width * 64)

        conv_layers.append(torch.nn.Conv2d(64, 64, kernel_size=3, stride=1))
        out_height = self.conv2d_out_size(out_height, 3, 1)
        out_width = self.conv2d_out_size(out_width, 3, 1)
        conv_layers.append(torch.nn.ReLU())
        # conv_layers.append(torch.nn.MaxPool2d(2))
        # out_height = (int)(out_height / 2)
        # out_width = (int)(out_width / 2)
        print(out_height, out_width, 64, out_height * out_width * 64)
        pixels_amount = out_height * out_width * 64

        self.conv_net = torch.nn.Sequential(*conv_layers).to(device)

        fc_layers = self.create_fully_connected_layers(pixels_amount, n_actions)
        self.fc_net = torch.nn.Sequential(*fc_layers).to(device)
