import torch
import math
from NeuralNetwork.AbstractConvNeuralNetwork import AbstractConvNeuralNetwork

device = torch.device("cuda") if torch.cuda.is_available() and torch.cuda.current_device() > 0 else torch.device("cpu")


class SimpleConvNeuralNetwork(AbstractConvNeuralNetwork):
    def __init__(self, n_obs, n_actions):
        super(SimpleConvNeuralNetwork, self).__init__(n_obs, n_actions)
        self.fc_dims = [256, 128]
        print(n_obs)
        size = int(math.sqrt(n_obs - 3))
        out_height = size
        out_width = size

        conv_layers = []

        print(out_height, out_width, 1, out_height * out_width)

        conv_layers.append(torch.nn.Conv2d(1, 8, kernel_size=3, stride=1))
        out_height = self.conv2d_out_size(out_height, 3, 1)
        out_width = self.conv2d_out_size(out_width, 3, 1)
        conv_layers.append(torch.nn.ReLU())
        # conv_layers.append(torch.nn.MaxPool2d(2))
        # out_height = (int)(out_height / 2)
        # out_width = (int)(out_width / 2)
        print(out_height, out_width, 8, out_height * out_width * 8)
        pixels_amount = out_height * out_width * 8


        self.conv_net = torch.nn.Sequential(*conv_layers).to(device)

        fc_layers = self.create_fully_connected_layers(pixels_amount, n_actions)
        self.fc_net = torch.nn.Sequential(*fc_layers).to(device)
