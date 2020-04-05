from random import randint

import numpy as np
import torch
import time

from ReplayBuffer import ReplayBuffer


class LearningAgent:
    def __init__(self, Q, num_actions):
        self.Q = Q
        self.num_actions = num_actions
        self.eps_decay = 0.999992
        self.final_eps = 0.01
        self.batch_size = 32
        self.update_freq = 4
        self.gamma = 0.9
        self.eps = 1.0
        self.replay_buffer = ReplayBuffer(int(1e5))
        self.step_idx = 0

    def evaluate(self, observation):
        return self.Q.act(observation)

    def act(self, observation):
        if np.random.rand() < self.eps:
            action = randint(0, self.num_actions - 1)
        else:
            action = self.Q.act(observation)

        self.step_idx += 1
        self.eps = max(self.final_eps, self.eps * self.eps_decay)
        return action

    def train(self, observation, action, reward, next_observation, done):
        self.replay_buffer.add((observation, action, reward, next_observation, done))

        # Jeżeli mamy wystarczająco krotek w historii, wziąć batcha:
        if len(self.replay_buffer) >= self.batch_size and self.step_idx % self.update_freq == 0:
            batch = self.replay_buffer.get_batch(self.batch_size)
            # Liczymy delty i poprawiamy naszą sieć
            deltas = self.calculate_deltas(batch)
            self.Q.update(deltas)

        if done:
            self.Q.save()

    def calculate_deltas(self, batch):
        obs, actions, rewards, next_obs, dones = batch

        next_obs_vals = self.Q.get_max_q_value(next_obs)
        targets = (rewards + (1 - dones) * next_obs_vals) * self.gamma

        obs_vals = self.Q.get_q_value(obs, actions)

        deltas = []
        for obs_val, target in zip(obs_vals, targets):
            deltas += [obs_val - target]
        return deltas
