import torch
import ma_gym
from Action.Action import Action
from EnvironmentFactory import EnvironmentFactory
import os

from NeuralNetwork.NeuralNetworkFactory import NeuralNetworkFactory


class Create(Action):
    def __init__(self, path, nn_id):
        if os.path.exists(path):
            raise Exception('Neural network has already existed on this path')
        super(Create, self).__init__(path)
        self.nn_id = nn_id
        self.env = EnvironmentFactory().create()
        self.nn_factory = NeuralNetworkFactory(self.env)

    def execute(self):
        net = self.nn_factory.create(self.nn_id)
        torch.save(net, self.path)
