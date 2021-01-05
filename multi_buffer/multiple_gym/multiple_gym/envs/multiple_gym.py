import gym
import numpy as np
import control
import control.matlab
from gym import error, spaces, utils
from gym.utils import seeding
import math


class multipleEnv(gym.Env):
    def __init__(self,state_num=2,condition_num=2, observation_num=2,
                 control_num=1,FPS=50, Q=None,R=None,N=None,Tf=None,X0=None):
        self.state_num = state_num
        self.condition_num = condition_num
        self.observation_num = observation_num
        self.control_num = control_num
        self.FPS = FPS
        if Tf != None:
            self.Tf = Tf
        else:
            self.Tf = 200/self.FPS
        if Q != None:
            self.Q = Q
        else:
            self.Q = np.eye(self.state_num)
        if R != None:
            self.R = R
        else:
            self.R = np.eye(self.control_num) 
        if N != None:
            self.N = N
        else:
            self.N = np.zeros((self.state_num,self.control_num))
        self.generate_canonical()
        self.ss = control.StateSpace(self.A,self.B,self.C,self.D)
        if X0 != None:
            self.X0 = X0
        else:
            self.X0 = np.zeros((self.state_num,self.control_num))
        self.action_space = spaces.Box(np.array([-10000]), np.array([10000]))
        self.observation_space = spaces.Box(np.array([-10000,-10000]), np.array([10000,10000]))
        self.reward_history = []
        self.seed(10)
        np.random.seed(10)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self,action):
        action = np.array(action)
        yout, T, xout =control.matlab.lsim(self.ss,U = action.reshape(()),T = [0,1/self.FPS],X0=self.X0)
        y = yout[1,:]
        x = np.transpose(xout[1,:])
        action = action.reshape(self.R.shape)
        reward = -(np.transpose(x)@self.Q@x+np.transpose(action)@self.R@action)/self.FPS
        reward = reward[0][0]
        self.X0 = x
        self.reward_history.append(reward)
        done = False
        if len(self.reward_history) >= 200:
            done = True
        return y,reward,done,{}
    
    def reset(self,X0 = None):
        if np.all(X0 == None):
            self.X0 = np.random.randn(self.state_num,self.control_num)
            #self.X0 = np.zeros((self.state_num,self.control_num))
        else:
            self.X0 = X0
        self.reward_history = []
        yout, T, xout =control.matlab.lsim(self.ss,U = np.array(0),T = [0,1/self.FPS],X0=self.X0)
        y = yout[0,:]
        return y
        
    def generate_canonical(self):
        self.A = np.eye(self.state_num)
        self.A[self.state_num-1,self.state_num-1] = self.condition_num
        self.B = np.ones((self.state_num,self.control_num))
        if self.observation_num == self.state_num:
            self.C = np.eye(self.state_num)
        else:
            raise ValueError("state_num and observation_num do not match!")
        self.D = np.zeros((self.observation_num,self.control_num))