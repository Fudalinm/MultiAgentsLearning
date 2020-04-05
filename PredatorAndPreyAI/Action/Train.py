import torch

from Action.Action import Action
from EnvironmentFactory import EnvironmentFactory
from Runner.MultiAgentMachineLearning import MultiAgentMachineLearning
from QNetwork import QNetwork
import os


class Train(Action):
    def __init__(self, paths):
        for path in paths:
            if not os.path.exists(path):
                raise Exception(f'Neural network doesnt exist on path: {path}')
        super(Train, self).__init__(paths)
        self.Qs = [self.create_q(path) for path in paths]
        self.env = EnvironmentFactory().create()

    @staticmethod
    def create_q(path):
        net = torch.load(path)
        net = net.train()
        return QNetwork(net, path, lr=1e-4)

    def execute(self):
        ml = MultiAgentMachineLearning(self.Qs, self.env)
        ml.train()
