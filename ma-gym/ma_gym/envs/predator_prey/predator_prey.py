import copy
import logging
import random

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

logger = logging.getLogger(__name__)


class PredatorPrey(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(5, 5), n_agents=2, n_preys=1, full_observable=False, penalty=-0.5,
                 step_cost=-0.01, prey_capture_reward=5, max_steps=100):
        self._grid_shape = grid_shape
        self.n_predators = n_agents
        self.n_preys = n_preys
        self.n_agents = n_agents + n_preys
        self._max_steps = max_steps
        self._step_count = None
        self._penalty = penalty
        self._step_cost = step_cost
        self._prey_capture_reward = prey_capture_reward
        self._agent_view_mask = (5, 5)

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_predators+self.n_preys)])

        self.predator_pos = {_: None for _ in range(self.n_predators)}
        self.prey_pos = {_: None for _ in range(self.n_preys)}
        self.agent_pos = {_: None for _ in range(self.n_agents)}

        self._prey_alive = None

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]

        self.viewer = None
        self.full_observable = full_observable

        # agent pos (2), prey (25), step (1)
        mask_size = np.prod(self._agent_view_mask)
        self._obs_high = np.array([1., 1.] + [1.] * mask_size + [1.0])
        self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0])
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents + self.n_preys)
            self._obs_low = np.tile(self._obs_low, self.n_agents + self.n_preys)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents + self.n_preys)])

        self._total_episode_reward = None

    """
     This one tels where agent on n_pos is going to move [ 0,1,2,1] -> [DOWN,LEFT,UP,LEFT] 
     0 agent is going down
     1 agent is going left
     ....
    """

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    """ This one return  list of action spaces whatever it is i cannot tell it is some higher class """

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_agent_space]

    """ Drawing map"""

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    """ create grid """

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    """ 
    This is initialization of places of initial location of predators and prays on map
    It is completely random but prey cannot be located as a neighbour of predator (i think so)
    """

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        # TODO: why predators and prays are distinguished in this case???
        # TODO: check if __update need to be changed
        for agent_i in range(self.n_predators):
            while True:
                pos = [random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos):
                    self.predator_pos[agent_i] = pos
                    self.agent_pos[agent_i] = pos
                    break
            self.__update_predator_view(agent_i)

        for prey_i in range(self.n_preys):
            while True:
                pos = [random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (self._neighbour_predators(pos)[0] == 0):
                    self.prey_pos[prey_i] = pos
                    self.agent_pos[self.n_predators + prey_i] = pos
                    break
            self.__update_prey_view(prey_i)

        self.__draw_base_img()

    """ it updates observable map for predators they don't know where are their colleagues  """

    def get_predator_obs(self):
        _obs = []
        for agent_i in range(self.n_predators):
            pos = self.predator_pos[agent_i]
            _agent_i_obs = [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            # check if prey is in the view area
            _prey_pos = np.zeros(self._agent_view_mask)  # prey location in neighbour
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['prey'] in self._full_obs[row][col]:
                        _prey_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the prey loc.
                    if PRE_IDS['predator'] in self._full_obs[row][col]: # coleagues has been seen as -1
                        _prey_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = -1

            _agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
            _agent_i_obs += [self._step_count / self._max_steps]  # adding time
            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    """ it updates observable map for preys they don't know where are their colleagues """

    def get_prey_obs(self):
        _obs = []
        for agent_i in range(self.n_preys):
            pos = self.agent_pos[agent_i + self.n_predators]
            _agent_i_obs = [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            # check if prey is in the view area
            _predator_pos = np.zeros(self._agent_view_mask)  # prey location in neighbour
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['predator'] in self._full_obs[row][col]:
                        _predator_pos[
                            row - (pos[0] - 2), col - (pos[1] - 2)] = -1  # get relative position for the predator loc.
                    if PRE_IDS['prey'] in self._full_obs[row][col]:
                        _predator_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1

            _agent_i_obs += _predator_pos.flatten().tolist()  # adding prey pos in observable area
            _agent_i_obs += [self._step_count / self._max_steps]  # adding time
            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def get_agent_obs(self):
        prey_obs = self.get_prey_obs()
        pred_obs = self.get_predator_obs()
        pred_obs.extend(prey_obs)
        return pred_obs

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}
        self.predator_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_alive = [True for _ in range(self.n_preys)]

        return self.get_agent_obs()

    def __wall_exists(self, pos):
        row, col = pos
        return PRE_IDS['wall'] in self._base_grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_predator_pos(self, agent_i, move):
        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self.predator_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_predator_view(agent_i)
        else:
            # poss not updated
            pass

    def __update_prey_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        if self._prey_alive[prey_i]:
            next_pos = None
            if move == 0:  # down
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif move == 1:  # left
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif move == 2:  # up
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif move == 3:  # right
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif move == 4:  # no-op
                pass
            else:
                raise Exception('Action Not found!')

            if next_pos is not None and self._is_cell_vacant(next_pos):
                self.agent_pos[self.n_predators + prey_i] = next_pos
                self.prey_pos[prey_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_prey_view(prey_i)
            else:
                # print('pos not updated')
                pass
        else:
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __update_predator_view(self, agent_i):
        # print(self.agent_pos[agent_i])
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['predator'] + str(agent_i + 1)

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def __update_prey_view(self, prey_i):
        self._full_obs[self.prey_pos[prey_i][0]][self.prey_pos[prey_i][1]] = PRE_IDS['prey'] + str(prey_i + 1)

    def _neighbour_predators(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['predator'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['predator'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['predator'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['predator'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        predator_id = []
        for x, y in neighbours_xy:
            predator_id.append(int(self._full_obs[x][y].split(PRE_IDS['predator'])[1]) - 1)
        return _count, predator_id

    def step(self, agents_action):
        self._step_count += 1

        pray_alive = 0
        for i in range(len(self._prey_alive)):
            if self._prey_alive[i]:
                pray_alive += 1

        # updating moves for both predators and preys
        for agent_i, action in enumerate(agents_action):
            if agent_i >= self.n_predators:
                # update preys
                if not (self._agent_dones[agent_i]):
                    self.__update_prey_pos(agent_i - self.n_predators, action)
            else:
                # update predators
                if not (self._agent_dones[agent_i]):
                    self.__update_predator_pos(agent_i, action)

# S: REWARDING FRAGMENT
        predator_additional_reward = 0
        prey_additional_reward = 0
        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                predator_neighbour_count, n_i = self._neighbour_predators(self.prey_pos[prey_i])

                if predator_neighbour_count >= 1:
                    _reward = self._penalty if predator_neighbour_count == 1 else self._prey_capture_reward
                    f_alive = (predator_neighbour_count == 1)
                    self._prey_alive[prey_i] = f_alive
                    self._agent_dones[self.n_predators+prey_i] = not f_alive

                    if not f_alive:
                        prey_additional_reward -= _reward
                        predator_additional_reward += _reward

        # All of them has the same reward this is team reward!
        rewards = [self._step_cost * pray_alive + predator_additional_reward for _ in range(self.n_predators)]
        rewards_preys = [-2 * self._step_cost * pray_alive + prey_additional_reward for _ in range(self.n_preys)]
        rewards.extend(rewards_preys)
# E: REWARDING FRAGMENT

        if (self._step_count >= self._max_steps) or (True not in self._prey_alive):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {'prey_alive': self._prey_alive}

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        for agent_i in range(self.n_predators):
            for neighbour in self.__get_neighbour_coordinates(self.agent_pos[agent_i]):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for agent_i in range(self.n_predators):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                draw_circle(img, self.prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                write_cell_text(img, text=str(prey_i + 1), pos=self.prey_pos[prey_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# CONSTS
AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
PREY_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'predator': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}
