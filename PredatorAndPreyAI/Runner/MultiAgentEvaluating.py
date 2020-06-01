import time
import numpy as np
from LearningAgent import LearningAgent


class MultiAgentEvaluating:
    def __init__(self, Qs, env):
        self.agents = [LearningAgent(Q, env.action_space[0].n) for Q in Qs]
        self.env = env
        self.current_episode = 0
        self.episodes_amount = 100

    def evaluate(self):
        whole_reward_predators = 0
        whole_reward_preys = 0
        whole_killed_preys = 0
        for _ in range(self.episodes_amount):
            episode_reward_predators, episode_reward_preys, killed_preys = self.evaluate_single()
            print(f"\n Reward from this episode predators: {episode_reward_predators}, preys: {episode_reward_preys} killed preys {killed_preys}",)
            whole_reward_predators += episode_reward_predators
            whole_reward_preys += episode_reward_preys
            whole_killed_preys += killed_preys
            self.current_episode += 1

        print(f"\nWhole reward predators: {whole_reward_predators}, preys: {whole_reward_preys} whole killed preys: {whole_killed_preys}",)

    def evaluate_single(self):
        # zresetuj środowisko
        observations = self.env.reset()
        observations = observations.copy()

        # zapisuj nagrody z epizodu
        episode_reward_predators = 0
        episode_reward_preys = 0
        dones = [False] * self.env.n_agents

        while not all(dones):
            actions = [agent.evaluate(observation) for agent, observation in zip(self.agents, observations)]
            next_observations, rewards, dones, info = self.env.step(actions)
            # dodaj nagrode do obecnych z epizodu
            for agent, observation, action, reward, next_observation, done in \
                    zip(self.agents, observations, actions, rewards, next_observations, dones):
                agent.train(observation.copy(), action, reward, next_observation.copy(), done)

            episode_reward_predators += sum(rewards[:2])
            episode_reward_preys += sum(rewards[2:])
            observations = next_observations.copy()

            self.env.render()
            # print(np.reshape(next_observations[0][:25], (5, 5)))
            # print(np.reshape(next_observations[0][25:], (5, 5)))
            # print(rewards)
            # time.sleep(2)

        killed_preys = sum([0 if preyAlive else 1 for preyAlive in info['prey_alive']])
        return episode_reward_predators, episode_reward_preys, killed_preys
