import time

from LearningAgent import LearningAgent


class MultiAgentEvaluating:
    def __init__(self, Qs, env):
        self.agents = [LearningAgent(Q, env.action_space[0].n) for Q in Qs]
        self.env = env
        self.current_episode = 0
        self.episodes_amount = 100

    def evaluate(self):
        whole_reward = 0
        whole_killed_preys = 0
        for _ in range(self.episodes_amount):
            reward, killed_preys = self.evaluate_single()
            print(f"\n Reward from this episode: {reward}, killed preys {killed_preys}",)
            whole_reward += reward
            whole_killed_preys += killed_preys
            self.current_episode += 1

        print(f"\nWhole reward: {whole_reward}, whole killed preys: {whole_killed_preys}",)

    def evaluate_single(self):
        dones = [False] * self.env.n_agents
        observations = self.env.reset()
        self.env.render()

        episode_reward = 0
        while not all(dones):

            actions = [agent.evaluate(observation) for agent, observation in zip(self.agents, observations)]
            next_observations, rewards, dones, info = self.env.step(actions)
            episode_reward += sum(rewards)

            self.env.render()
        killed_preys = sum([0 if preyAlive else 1 for preyAlive in info['prey_alive']])
        print(killed_preys)
        return episode_reward, killed_preys
