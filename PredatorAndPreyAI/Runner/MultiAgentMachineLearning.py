import time
import numpy as np
from LearningAgent import LearningAgent


class MultiAgentMachineLearning:
    def __init__(self, Qs, env):
        self.agents = [LearningAgent(Q, env.action_space[0].n) for Q in Qs]
        self.env = env
        self.episodes = 1000#000
        self.current_episode = 0

    def train(self):
        while True:
            self.train_single_episode()
            self.current_episode += 1
            if self.current_episode > self.episodes:
                return

    def train_single_episode(self):
        # zresetuj środowisko
        observations = self.env.reset()
        observations = observations.copy()

        # zapisuj nagrody z epizodu
        episode_reward = 0
        dones = [False] * self.env.n_agents

        while not all(dones):
            actions = [agent.act(observation) for agent, observation in zip(self.agents, observations)]
            next_observations, rewards, dones, info = self.env.step(actions)
            # dodaj nagrode do obecnych z epizodu
            for agent, observation, action, reward, next_observation, done in \
                    zip(self.agents, observations, actions, rewards, next_observations, dones):
                agent.train(observation.copy(), action, reward, next_observation.copy(), done)


            episode_reward += sum(rewards)
            observations = next_observations.copy()

            self.env.render()
            #print(np.reshape(next_observations[0][2:-1], (5, 5)))
            # print(rewards)
            # time.sleep(5)

        killed_preys = sum([0 if preyAlive else 1 for preyAlive in info['prey_alive']])
        print(f"\r episode: {self.current_episode+1}/{self.episodes} reward: {episode_reward:.2f}, killed preys: {killed_preys}")

