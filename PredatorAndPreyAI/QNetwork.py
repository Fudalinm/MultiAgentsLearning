import torch
import numpy as np


class QNetwork():
    def __init__(self, net, lr=0.001):
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def update(self, deltas):
        """
        Aktualizuje wartość Q dla podanej delty.
        """
        # Poprawianie parametrów sieci na podstawie kosztu - nie trzeba rozumieć.
        loss = torch.mean(torch.stack(deltas).pow(2))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state):
        """
        Wybiera optymalną akcje na podstawie obecnych wartości Q dla podanej obserwacji
        """
        action_values = self.net.forward(state).cpu().detach().numpy()
        return np.argmax(action_values)

    def get_q_value(self, states, actions=None):
        """
        Wybiera optymalną akcje na podstawie obecnych wartości Q dla podanej
        obserwacji.
        """
        q_values = self.net.forward(states)
        if actions is not None:
            q_values = [q_value[action] for q_value, action
                        in zip(q_values, actions)]
        return q_values

    def get_max_q_value(self, states):
        """
        Daje nam wartość maksymalnej akcji w podanym stanie - przydatne do
        liczenia Q-target.
        """
        return self.get_q_value(states).cpu().detach().numpy().max(1)
