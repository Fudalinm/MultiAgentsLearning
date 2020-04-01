import torch

from Action.Action import Action
from EnvironmentFactory import EnvironmentFactory
from Runner.Evaluating import Evaluating
from QNetwork import QNetwork
import os


class Evaluate(Action):
    def __init__(self, path):
        if not os.path.exists(path):
            raise Exception('Neural network doesnt exist on this path')
        super(Evaluate, self).__init__(path)
        self.net = torch.load(path)
        self.net = self.net.eval()
        self.Q = QNetwork(self.net, lr=1e-4)
        self.env = EnvironmentFactory().create()

    def execute(self):
        evaluating = Evaluating(self.Q, self.env)
        evaluating.evaluate()
