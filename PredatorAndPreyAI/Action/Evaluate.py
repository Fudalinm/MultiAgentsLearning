import torch

from Action.Action import Action
from EnvironmentFactory import EnvironmentFactory
from QNetwork import QNetwork
import os

from Runner.MultiAgentEvaluating import MultiAgentEvaluating


class Evaluate(Action):
    def __init__(self, paths):
        for path in paths:
            if not os.path.exists(path):
                raise Exception(f'Neural network doesnt exist on path: {path}')
        super(Evaluate, self).__init__(paths)
        self.Qs = [self.create_q(path) for path in paths]
        self.env = EnvironmentFactory().create()

    @staticmethod
    def create_q(path):
        print(path)
        net = torch.load(path)
        net = net.train()
        return QNetwork(net, path, lr=1e-4)

    def execute(self):
        evaluating = MultiAgentEvaluating(self.Qs, self.env)
        evaluating.evaluate()
