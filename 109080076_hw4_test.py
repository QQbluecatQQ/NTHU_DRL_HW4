import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import collections
import time
from osim.env import L2M2019Env

LOG_STD_MIN = -20
LOG_STD_MAX = 2

def flatten_dict(d):
    flattened = []
    for key, value in d.items():
        if isinstance(value, dict):
            flattened.extend(flatten_dict(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    flattened.extend(flatten_dict(item))
                else:
                    flattened.append(item)
        elif isinstance(value, np.ndarray):
            flattened.extend(value.flatten())
        else:
            flattened.append(value)
    return flattened

def flatten_obs(obs):
    flattened = flatten_dict(obs)
    flattened = np.array(flattened,dtype=np.float32)
    flattened = flattened.reshape(1,-1)
    return flattened


class police_net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(police_net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = torch.nn.Linear(hidden_dim, action_dim)
        self.action_scale = torch.tensor((1.0-0.0)/2.0,dtype=torch.float32)
        self.action_bias = torch.tensor((1.0+0.0)/2.0,dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std
    
    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t)
        
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
class SAC_agent:
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.police_net = police_net(state_dim, hidden_dim, action_dim).to(device)


        
    def get_action(self,state):
        state = torch.tensor(state).to(self.device)
        action, _, _ = self.police_net.get_action(state)
        action = action.detach().cpu().numpy()
        return action.flatten()

    def load_model(self, path):
        model_dict = torch.load(path, map_location=self.device)
        self.police_net.load_state_dict(model_dict['policy'])
        
class Agent():
    def __init__(self):
        self.sac_agent = SAC_agent(339, 22, 256, 'cpu')
        # self.sac_agent = SAC_agent(97, 22, 256, 'cpu')
        self.sac_agent.load_model('109080076_hw4_data')
        self.count_frame = 0
        self.last_action = None
    def act(self, observation):
        if self.count_frame % 4 == 0:
            self.count_frame = 0
            obs = flatten_obs(observation)
            # obs = obs[0][242::]
            # obs = obs.reshape(1,-1)
            self.last_action = self.sac_agent.get_action(obs)
            return self.last_action
        else:
            self.count_frame += 1
            return self.last_action
        
# test_time = 10
# agent = Agent()
# env = L2M2019Env(visualize=True,difficulty=2)
# total_reward = 0
# for i in range(test_time):
#     observation = env.reset()
#     done = False
#     episiode_reward = 0
#     while not done:
#         action = agent.act(observation)
#         observation, reward, done, info = env.step(action)
#         episiode_reward += reward
#     total_reward += episiode_reward
#     print('Episode:',i,'Reward:',episiode_reward)
# env.close()
# print('Final Score:',total_reward/test_time)
        