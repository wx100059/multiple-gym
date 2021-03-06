# -*- coding: utf-8 -*-
"""
DDPG implemented by pytorch
pytorch == 1.5.1
@author: li xiang
"""
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiple_gym   
import os
        
class Actor(nn.Module):
    '''
    3 layers full connected network. Remove bias
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size,bias=False)
        self.linear2 = nn.Linear(hidden_size, output_size,bias=False)
        
        # Initialize weights between -3e-3 and 3e-3
        self.linear1.weight.data.normal_(-3e-3,3e-3)
        self.linear2.weight.data.normal_(-3e-3,3e-3)
        
    def forward(self, s):
        x = self.linear1(s)
        x = self.linear2(x)

        return x


class Critic(nn.Module):
    '''
    3 layers full connected network. Remove bias
    '''
    def __init__(self, state_size, action_size, hidden_size, output_size):
        super().__init__()
        self.state_path1 = nn.Linear(state_size, 50)
        self.state_path2 = nn.Linear(50, 25)
        self.action_path1 = nn.Linear(action_size, 25)
        self.common_path1 = nn.Linear(25,1)
        # Initialize weights between -3e-3 and 3e-3
        self.state_path1.weight.data.normal_(-3e-3,3e-3)
        self.state_path2.weight.data.normal_(-3e-3,3e-3)
        self.action_path1.weight.data.normal_(-3e-3,3e-3)
        self.common_path1.weight.data.normal_(-3e-3,3e-3)
    def forward(self, s, a):
        s = torch.tanh(self.state_path1(s))
        s = self.state_path2(s)
        a = self.action_path1(a)
        x = s + a
        x = torch.tanh(x)
        x = self.common_path1(x)
        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        self.actor = Actor(s_dim, 256, a_dim)
        self.actor_target = Actor(s_dim, 256, a_dim)
        self.critic = Critic(s_dim, a_dim, 256, a_dim)
        self.critic_target = Critic(s_dim, a_dim, 256, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        #initial target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()
        return a0
    
    def put(self, *transition): 
        #FIFO buffer
        if len(self.buffer)== self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        samples = random.sample(self.buffer, self.batch_size)
        
        s0, a0, r1, s1 = zip(*samples)
        
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)
        
        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
            
            y_pred = self.critic(s0, a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            
        def actor_learn():
            loss = -torch.mean( self.critic(s0, self.actor(s0)) )
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
                                           
if __name__ == '__main__':          
    env = gym.make('multiple_gym-v0')
    env.reset()

    params = {
    'env': env,
    'gamma': 0.99, 
    'actor_lr': 0.0001, 
    'critic_lr': 0.001,
    'tau': 0.001,
    'capacity': 1000000, 
    'batch_size': 64,
    }

    agent = Agent(**params)

    for episode in range(10):
        s0 = env.reset()
        episode_reward = 0
    
        for step in range(500):
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            agent.put(s0, a0, r1, s1)

            episode_reward += r1 
            s0 = s1

            agent.learn()

        print(episode, ': ', episode_reward)
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # solve the multi registration bug
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'multiple_gym-v0' in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
        if 'multiple_gym_extend-v0' in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]