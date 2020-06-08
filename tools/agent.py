import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class TanhMod(nn.Module):

    def __init__(self,low,high):
        super().__init__()
        self.low = torch.from_numpy(low).float()
        self.high = torch.from_numpy(high).float()

    def forward(self,input):
        scale = (self.high - self.low)/2
        mid_point = (self.high + self.low)/2
        return scale*torch.tanh(input) + mid_point


class MLP(nn.Module):
    def __init__(self,**pikwargs):
        super().__init__()

        # set up MLP network configuration
        if not 'layers' in pikwargs:
            raise KeyError('number of layers is required')
        if not 'hidden_sizes' in pikwargs:
            raise KeyError('List of hidden sizes must be included')
        if not 'activation' in pikwargs:
            raise KeyError('activation function is required')
        if not 'output_act' in pikwargs:
            raise KeyError('output activation function is required')
        if not 'input_dim' in pikwargs:
            raise KeyError('Input Dimension is required')
        if not 'output_dim' in  pikwargs:
            raise KeyError('Output Dimension is required')

        hs = pikwargs['hidden_sizes']

        if not len(hs) == pikwargs['layers']:
            # update number of hidden layers
            pikwargs['layers'] = len(hs)
            #raise ValueError('hidden_sizes is not the correct length')

        # make hidden layers
        self.layers = pikwargs['layers']
        in_N = pikwargs['input_dim']
        out_N = pikwargs['output_dim']
        self.hidden = nn.ModuleList([nn.Linear(in_N,hs[0])])
        for i in range(1,self.layers):
            self.hidden.append(nn.Linear(hs[i-1],hs[i]))
        self.hidden.append(nn.Linear(hs[-1],out_N))

        # set up hidden activation function
        self.ha = pikwargs['activation']
        # get output activation function
        self.oa = pikwargs['output_act']

        # turn off gradient tracking
        try:
            rg = pikwargs['requires_grad']
            if not rg:
                self.turn_off_grad()

        except:
            print('requires_grad not set. Continuing with gradient tracking.')


    def forward(self,x):
        x = self.ha(self.hidden[0](x))
        for i in range(1,self.layers):
            x = self.ha(self.hidden[i](x))
        return self.oa(self.hidden[-1](x))

    def turn_off_grad(self):
        """
        turn off gradient tracking for parameters
        """
        for p in self.parameters():
            p.requires_grad = False

class Agent:
    """
    general agent class
    """
    def __init__(self,env):
        self.hid_width = 12
        self.pikwargs = {
                         'layers': 2,
                         'hidden_sizes':self.hid_width*2,
                         'requires_grad': False
                        }
        self.pi = None

    def act(self, obs):
        raise NotImplementedError

class MLPTextAgent(Agent):
    """
    agent for discrete action spaces
    """
    def __init__(self,env,hs=None,activation=None):
        super().__init__(env)
        
        # set up pikwargs
        self.policy_setup(env)

        # update hidden layer widths if need be
        if hs:
            self.pikwargs['hidden_sizes'] = hs

        # update hidden layer activation function
        if activation:
            self.pikwargs['activation'] = activation

        # create parameterized policy
        self.pi = MLP(**self.pikwargs)

    def act(self,obs):
        p = self.pi(torch.from_numpy(obs).view(1,-1).float())
        _,a = p.max(1)
        return int(a)

    def policy_setup(self,env):
        self.pikwargs.update(
                             {
                              'activation': nn.Tanh(),
                              'output_act': nn.Softmax(dim=1),
                              'input_dim': env.observation_space.n,
                              'output_dim': env.action_space.n,
                             }
                            )



class MLPDiscreteAgent(Agent):
    """
    agent for discrete action spaces
    """
    def __init__(self,env,hs=None,activation=None):
        super().__init__(env)
        
        # set up pikwargs
        self.policy_setup(env)

        # update hidden layer widths if need be
        if hs:
            self.pikwargs['hidden_sizes'] = hs

        # update hidden layer activation function
        if activation:
            self.pikwargs['activation'] = activation

        # create parameterized policy
        self.pi = MLP(**self.pikwargs)

    def act(self,obs):
        p = self.pi(torch.from_numpy(obs).view(1,-1).float())
        _,a = p.max(1)
        return int(a)

    def policy_setup(self,env):
        self.pikwargs.update(
                             {
                              'activation': nn.Tanh(),
                              'output_act': nn.Softmax(dim=1),
                              'input_dim': env.observation_space.shape[0],
                              'output_dim': env.action_space.n,
                             }
                            )


class MLPContinuousAgent(Agent):
    """
    agent for continuous actions spaces
    """
    def __init__(self,env,hs=None,activation=None):
        super().__init__(env)
        
        # setup pikwargs
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.policy_setup(env)

        # update hidden layer widths if need be
        if hs:
            self.pikwargs['hidden_sizes'] = hs

        # update activation function
        if activation:
            self.pikwargs['activation'] = activation
        
        # make policy network
        self.pi = MLP(**self.pikwargs)

    def act(self,obs):
        p = self.pi(torch.from_numpy(obs).view(1,-1).float())
        return p.numpy().flatten()

    def policy_setup(self,env):
        self.pikwargs.update(
                             {
                              'activation': nn.Tanh(),
                              #'output_act': TanhMod(self.low,self.high),
                              'output_act': nn.Identity(),
                              'input_dim': env.observation_space.shape[0],
                              'output_dim': env.action_space.shape[0],
                             }
                            )

class MLPGaussianAgent(Agent):
    """
    agent for continuous action space that uses a Gaussian
    """
    def __init__(self,env,hs=None,activation=None):
        super().__init__(env)
        
        # setup pikwargs
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.policy_setup(env)

        # update hidden layer widths if need be
        if hs:
            self.pikwargs['hidden_sizes'] = hs

        # update activation function
        if activation:
            self.pikwargs['activation'] = activation
        
        # make policy network
        self.pi = MLP(**self.pikwargs)

        # make parameters for standard devation)
        init_sigma = 0.05
        self.std = nn.Parameter(init_sigma*torch.ones(self.pikwargs['input_dim']))

    def act(self,obs):
        # sample network for mean
        mu = self.pi(torch.from_numpy(obs).view(1,-1).float())
        # update normal distribution
        dist = Normal(mu.view(-1),self.std)
        # sample from normal distribution
        p = dist.sample()
        return p.numpy()

    def policy_setup(self,env):
        self.pikwargs.update(
                             {
                              'activation': nn.Tanh(),
                              #'output_act': TanhMod(self.low,self.high),
                              'output_act': nn.Identity(),
                              'input_dim': env.observation_space.shape[0],
                              'output_dim': env.action_space.shape[0],
                             }
                            )

def get_agent(env,hs=[12]*2,activation=None,policy_mode='deterministic'):
    """
    get correct agent based on the kind of action space
    """
    torch.manual_seed(0)
   
    if policy_mode == 'prob':
        return MLPGaussianAgent(env,hs=hs,activation=activation)
    
    if policy_mode == 'deterministic':

        if (type(env.observation_space) == Discrete) and (type(env.action_space) == Discrete):
            return MLPTextAgent(env,hs=hs,activation=activation)

        elif env.action_space.__class__ == Box:
            # continuous agent
            #print('Making MLPContinuousAgent')
            return MLPContinuousAgent(env,hs=hs,activation=activation)

        elif env.action_space.__class__ == Discrete:
            # discrete agent
            #print('Making MLPDiscreteAgent')
            return MLPDiscreteAgent(env,hs=hs,activation=activation)

        else:
            raise ValueError('Cannot get Agent for invalid action space type.')


