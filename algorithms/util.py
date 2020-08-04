"""
Usefule functions for testing rl with gym environments
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gym
import pybullet_envs
from tools.agent import get_agent

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def simulate_reward(agent, env, itr, max_steps=200, scale=1, num_eps=1):
    """
    Inputs:
        agent     -- MLPAgent
        env       -- openai gym class for simulated environments
        seed      -- seed for environment initialization
        max_steps -- how long to simulate environment
        scale     -- scaling factor for reward
        num_eps   -- number of expisodes to average over
    """
    env_steps = {
                   'Pendulum-v0':200,
                   'CartPole-v0':200,
                   'MountainCarContinuous-v0':999,
                   'HopperBulletEnv-v0':1000,
                   'InvertedPendulumBulletEnv-v0':1000,
                   'ReacherBulletEnv-v0':150,
                   'MountainCar-v0':200,
                   'Acrobot-v1':500,
                }

    # get name of environment
    try:
        max_steps = env_steps[env.spec.id]
    except:
        max_steps = env._max_episode_steps

    # list for rewards
    returns_list = []

    seed_list = list(num_eps * (itr+1) + np.arange(num_eps))
    for eps in range(num_eps):
        # get first observation
        env.seed(int(seed_list[eps]))
        o = env.reset()
        r = 0

        # simulate episode
        for i in range(max_steps):
            # agent chooses action based on observation
            a = agent.act(scale*o)
            # update environment based on this action
            o,this_r,eps_done_flag,_ = env.step(a)
            # update current returns_list
            r += this_r
            # check if episode is complete
            if eps_done_flag:
                #print(f"exited after {i} iterations of {max_steps}")
                break

        # record obtained return
        returns_list.append(r)

    # return average return
    return np.mean(returns_list)

@torch.no_grad()
def get_net_param(net):
    """
    Pull out the network parameters and turn them into
    numpy array

    Input:
        net -- neural network whose parameters we want

    Output:
        w   -- parameters of net in vector form
    """

    # total parameters
    num_params = count_vars(net)

    # initialize vector
    w = np.empty(num_params)

    # initialize parameter counter
    param_counter = 0

    for p in net.parameters():
        s = p.shape

        try:
            num_p = s[0]*s[1]
            w[param_counter:param_counter+num_p] = p.view(num_p).float().detach().numpy()
            #print(f"start {param_counter} | finish {param_counter+num_p}")

        except:
            num_p = s[0]
            w[param_counter:param_counter+num_p] = p.view(num_p).float().detach().numpy()
            #print(f"start {param_counter} | finish {param_counter+num_p}")

        # update param_counter
        param_counter += num_p

    return w

@torch.no_grad()
def update_net_param(net,xi):
    """
    Change the state_dict of net to be xi
    the dimension of xi and the number of network
    parameters in net must be the same!

    Inputs:
        net -- pytorch neural network
        xi  -- numpy vector of length = number of parameters in net
    """

    # counter for portion of xi used
    param_counter = 0

    for p in net.parameters():
        s = p.shape

        # number of parameters in p
        try:
            num_p = s[0]*s[1]
            new_p = torch.from_numpy(xi[param_counter:param_counter+num_p]).view(s[0],s[1]).float()
            #print(f"start {param_counter} | finish {param_counter+num_p}")

        except:
            num_p = s[0]
            new_p = torch.from_numpy(xi[param_counter:param_counter+num_p]).float()
            #print(f"start {param_counter} | finish {param_counter+num_p}")

        # update parameters
        p.data = new_p

        # update parameter count
        param_counter += num_p


def rl_j_eval(agent,env,net,xi,eval_f):
    """
    Inputs:
        agent  -- MLPAgent assocaited with net
        env    -- openai gym environment
        net    -- network which evaluates policy
        xi     -- search direction
        eval_f -- function handle which evaluates J

    Notes:
        An example of eval_f is
        lambda actor,env: simulate_reward(agent,env)
    """
    # update network parameters to be xi
    update_net_param(net,xi)

    # evaluate the network
    r = eval_f(agent,env)

    return r

def setup_agent_env(env_name, max_steps=None, hs=[12]*2,policy_mode='deterministic'):
    """
    sets up environment given env_name string

    Inputs:
        env_name    -- string name of valid gym enviornment
        max_steps   -- maximum number of steps of gym environment
        hs          -- list of hidden layer sizes

    Outpts:
        agent       -- MLPAgent
        env         -- gym environment
        net         -- network associated with agent
    """

    # make environemnt
    env = gym.make(env_name)

    # create agent
    agent = get_agent(env,net_arch='MLP',hs=hs,policy_mode=policy_mode)

    # set network for actor
    net = agent.pi

    return agent,env,net

def make_rl_j_fn(env_name,max_steps=200, hs=[12]*2, scale=1,policy_mode='deterministic'):
    """
    Reinforcement Learning

    returns function which takes as input the new vector of parameters
    as well as number of network parameters

    Inputs:
        env_name  -- string representation of gym env e.g. 'CartPole-v0'
        max_steps -- maximum number of steps of episodes in environment
        hs        -- list of hidden network sizes

    Outputs:
        J        -- function whose input is xi a vector of size d and output is reward
        d        -- number of network parameters associated with policy
    """
    a,e,n = setup_agent_env(env_name, max_steps=max_steps,hs=hs,policy_mode=policy_mode)

    # count parametrs
    d = count_vars(n)

    return lambda xi,itr: rl_j_eval(a,e,n,xi,lambda a,e: simulate_reward(a,e,itr,scale=scale,max_steps=max_steps)), d

def turn_off_grad(net):
    """
    turns off gradient tracking for the network net
    """
    for p in net.parameters():
        p.requires_grad = False


