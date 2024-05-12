import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import math
import copy
from multiagent.multi_discrete import MultiDiscrete
from Torch_maddpg import ActorDiscrete
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MAEnv():

    def __init__(self, psi, agent_num, action_dim, observation_dim, data_size, energy_consumption):
        # self.quality = quality
        # self.price = price
        self.observation_dim = observation_dim
        self.data_size = data_size
        self.energy_consumption = energy_consumption
        self.agent_num = agent_num
        self.action_dim = action_dim
        # Hyper-parameters
        self.psi = psi
    def reset(self):
        self.price = np.random.random(10)
        self.loss_pro = np.ones(10) * 3
        self.loss_local = np.ones(10) * 2
        self.users_set = [i for i in range(10)]
        return self.price,self.loss_pro,self.loss_local,self.users_set

    def get_state(self, price, quality):
        self.observation = []
        for agent in self.agent_num:
            observation_agent = np.concatenate(([self.quality[agent], self.data_size[agent], self.energy_consumption[agent], price[agent]], quality, self.data_size, self.energy_consumption, self.price))
            self.observation.append(observation_agent)
        state = copy.deepcopy(self.observation)
        return state

    def get_action(self, state):
        action_all = []
        for agent in self.agent_num:
            observation = state[agent]
            action_test = np.clip(ActorDiscrete(observation, self.action_dim))
            # action_test = np.clip(ActorDiscrete(observation, self.action_dim) + np.random.randn(1) * var, -1, 1)
            if action_test > 0:
                action = 1
            else:
                action = 0
            action_all.append(action)
        return action_all

    def get_reward(self, action, price, quality):
        reward = []
        for agent in self.agent_num:
            reward_agent = action[agent] * (price * (1-1/math.exp(self.psi*(self.data_size[agent] * quality[agent]))) - self.energy_consumption[agent])
            reward.append(reward_agent)
        return reward

