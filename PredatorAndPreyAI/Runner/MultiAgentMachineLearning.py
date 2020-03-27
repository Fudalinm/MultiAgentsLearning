import numpy as np
import torch
import time

from ReplayBuffer import ReplayBuffer


class MultiAgentMachineLearning:
    def __init__(self, Q, env, path):
        self.Q = Q
        self.env = env
        self.path = path
        self.eps_decay = 0.999992
        self.final_eps = 0.01
        self.episodes = 100000
        self.batch_size = 32
        self.update_freq = 4
        self.gamma = 0.9
        self.eps = 1.0
        self.current_episode = 0
        self.replay_buffer = ReplayBuffer(int(1e5))

    def train(self):
        while True:
            self.train_single(self.env)
            self.current_episode += 1
            if self.current_episode > self.episodes:
                return
            torch.save(self.Q.net, self.path)

    def train_single(self, env):
        # zresetuj środowisko
        observations = env.reset()
        observations = observations.copy()

        # zapisuj nagrody z epizodu
        episode_reward = 0
        step_idx = 0
        dones = [False] * env.n_agents

        while not all(dones):
            if np.random.rand() < self.eps:
                actions = env.action_space.sample()
            else:
                actions = [self.Q.act(observation) for observation in observations]

            next_observations, rewards, dones, info = env.step(actions)
            # dodaj nagrode do obecnych z epizodu
            for observation, action, reward, next_observation, done in \
                    zip(observations, actions, rewards, next_observations, dones):
                self.replay_buffer.add((observation.copy(), action, reward, next_observation.copy(), done))
            # Jeżeli mamy wystarczająco krotek w historii, wziąć batcha:
            if len(self.replay_buffer) >= self.batch_size and step_idx % self.update_freq == 0:
                batch = self.replay_buffer.get_batch(self.batch_size)
                # Liczymy delty i poprawiamy naszą sieć
                deltas = self.calculate_deltas(batch)
                self.Q.update(deltas)
            episode_reward += sum(rewards)
            observations = next_observations.copy()
            step_idx += 1

            self.eps = max(self.final_eps, self.eps * self.eps_decay)
            env.render()

        print(f"\r episode: {self.current_episode+1}/{self.episodes} reward: {episode_reward:.2f} eps: {self.eps:.2f}")

    def calculate_deltas(self, batch):
        obs, actions, rewards, next_obs, dones = batch

        next_obs_vals = self.Q.get_max_q_value(next_obs)
        targets = (rewards + (1 - dones) * next_obs_vals) * self.gamma

        obs_vals = self.Q.get_q_value(obs, actions)

        deltas = []
        for obs_val, target in zip(obs_vals, targets):
            deltas += [obs_val - target]
        return deltas
