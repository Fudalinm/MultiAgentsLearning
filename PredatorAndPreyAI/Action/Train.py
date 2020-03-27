import torch

from Action.Action import Action
from EnvironmentFactory import EnvironmentFactory
from Runner.MultiAgentMachineLearning import MultiAgentMachineLearning
from QNetwork import QNetwork
import os


class Train(Action):
    def __init__(self, path):
        if not os.path.exists(path):
            raise Exception('Neural network doesnt exist on this path')
        super(Train, self).__init__(path)
        self.net = torch.load(path)
        self.net = self.net.train()
        self.Q = QNetwork(self.net, lr=1e-4)
        self.env = EnvironmentFactory().create()

    def execute(self):
        ml = MultiAgentMachineLearning(self.Q, self.env, self.path)
        ml.train()
        torch.save(self.net, self.path)
