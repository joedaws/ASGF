import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn 
from torch.distributions.normal import Normal 
from torch.distributions.categorical import Categorical

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

def box_space_info(space,mode):
    """get information for a continuous openAI space

    Args:
        space (gym.spaces.Box): a openAI space
        mode (str): either an 'action' space or an 'observation' space

    Returns:
        dictionary contain info on the space
    """
    # low and high values
    low = space.low
    high = space.high

    # dimension
    dim = space.shape[0]
    
    # setup keys for dictionary
    mode_dim = mode + '_dim'
    mode_low = mode + '_low'
    mode_high = mode + '_high'
    mode_space_type = mode + '_space'

    # dictionary to return
    info = {mode_dim:dim,mode_space_type:'Box',mode_low:low,mode_high:high}

    return info

def discrete_space_info(space,mode):
    """get information for a discrete openAI space

    Args:
        space (gym.spaces.Discrete): a openAI space
        mode (str): either an 'action' space or 'observation' space

    Returns:
        dictionary contain info on the space
    """
    # dimension
    dim = space.n
        
    # setup keys for dictionary
    mode_dim = mode + '_dim'
    mode_space_type = mode + '_space'

    # dictionary to return
    info = {mode_dim:dim,mode_space_type:'Discrete'}

    return info

def get_env_info(env,**kwargs):
    """setup dictionary with info on environment"""

    def check_space(space,env,mode):
        if type(space) == Box:
            info = box_space_info(space,mode)
        elif type(space) == Discrete:
            info = discrete_space_info(space,mode)
        else:
            raise ValueError(f'the env {env} has unknown {mode} space {str(space)}.')

        return info

    # check observation space to set up input
    input_space = env.observation_space
    input_info = check_space(input_space,str(env),'observation')
    
    # check action space to set up output
    output_space = env.action_space
    output_info = check_space(output_space,str(env),'action')

    # collect additional info
    envkwargs = kwargs
    envkwargs['name'] = str(env)

    # collect all info into one dictionary
    envkwargs.update(input_info)
    envkwargs.update(output_info)

    return envkwargs

def get_mlp_info(input_dim,output_dim,**kwargs):
    # setup defaults
    mlpkwargs = {
        'layers':2,
        'hidden_sizes':2*[8],
        'activation':nn.Tanh(),
        'output_act':nn.Identity(),
        'input_dim':input_dim,
        'output_dim':output_dim
    }

    # update if necessary
    kw_set = set(kwargs)
    allowed_kw = kw_set.intersection(mlpkwargs)
    for kw in allowed_kw:
        mlpkwargs[kw] = kwargs[kw]
   
    if 'std' in kwargs:
        mlpkwargs['std'] = kwargs['std']

    return mlpkwargs

def make_policy(mode,env_info,net_arch='MLP',**kwargs):
    """make a policy network given the mode and env_info
       
    Args:
        mode (str): either 'deterministic' or 'stochastic'
        env_info (dict): dictionary of information about the environment
        **kwargs (dict): misc other info that might be necessary

    Returns:
        net (torch.nn.Module): neural network for paramterizing the policy
    """
    
    def make_net(net_arch):
        """makes a network based on environment and pi variables

        Args:
            net_arch (str): string indicating what kind of archtecture the network 
                should have

        Returns:
            a pytorch network with appropriate configuration
        """   

        #TODO add support for other architectures ONLY SUPPORTS MLP now
        if net_arch == 'MLP':
            # get input and output dimension
            input_dim = env_info['observation_dim']
            output_dim = env_info['action_dim']
            
            # check mode
            if mode == 'deterministic':
                output_act = nn.Identity()
            elif mode == 'stochastic':
                if env_info['action_space'] == 'Box':
                    output_act = nn.Identity()
                elif env_info['action_space'] == 'Discrete':
                    output_act = nn.Softmax(dim=1)
                else:
                    raise ValueError(f"Cannot make policy. Unknown action_space for environment {env_info['name']}")
            else:
                raise ValueError(f'Cannot make policy. Unknown mode {mode}')

            # setup info
            info = get_mlp_info(input_dim,output_dim,output_act=output_act,**kwargs)
            
            # instantiate network
            net = MLP(**info)

        return net, info
    
    # instantiate network and get dictionary of paramters
    net, pikwargs = make_net(net_arch) 

    return net, pikwargs

class StochasticAgent:
    """
    Stochastic Agent class
    """
    def __init__(self,env,net_arch,**kwargs):
        # network architecture
        self.net_arch = net_arch

        # get environment info
        self.env_info = get_env_info(env)
        
        # get network and policy options
        if self.env_info['action_space'] == 'Box':
            net, piopts = make_policy('stochastic',self.env_info,net_arch,std=0.05,**kwargs)
        else:
            net, piopts = make_policy('stochastic',self.env_info,net_arch,**kwargs)

        # set policy network 
        self.pi = net
        self.pikwargs = piopts

    def _policy_input(self,obs):
        """converts observation into policy input"""
        if self.env_info['observation_space'] == 'Box':
            return torch.from_numpy(obs).view(1,-1).float()
        elif self.env_info['observation_space'] == 'Discrete':
            return torch.eye(self.env_info['observation_dim'])[obs].view(1,-1).float()

    def _distribution(self,policy_input):
        if self.env_info['action_space'] == 'Box':
            mu = self.pi(policy_input)
            std = self.pikwargs['std']
            return Normal(mu, std)
        
        elif self.env_info['action_space'] == 'Discrete':
            p = self.pi(policy_input)
            return Categorical(probs=p)

    def act(self,obs):
        p_in = self._policy_input(obs)
        p_out = self._distribution(p_in).sample()
        a = p_out.detach().numpy()[0]
        # return a in numpy form
        return a

    def __repr__(self):
        s =  "StochasticAgent(\n"
        s += f"  net_arch: {self.net_arch} \n"
        s += f"  observation_space: {self.env_info['observation_space']} {self.env_info['observation_dim']}\n"
        s += f"  action_space: {self.env_info['action_space']} {self.env_info['action_dim']} \n"
        s += ")"
        return s

class DeterministicAgent:
    """
    Deterministic Agent class
    """
    def __init__(self,env,net_arch,**kwargs):
        # network architecture
        self.net_arch = net_arch

        # get environment info
        self.env_info = get_env_info(env)

        # get network and policy options
        net, piopts = make_policy('deterministic',self.env_info,net_arch,**kwargs)

        # set policy network
        self.pi = net
        self.pikwargs = piopts

    def _policy_input(self,obs):
        """converts observation into policy input"""
        if self.env_info['observation_space'] == 'Box':
            return torch.from_numpy(obs).view(1,-1).float()
        elif self.env_info['observation_space'] == 'Discrete':
            return torch.eye(self.env_info['observation_dim'])[obs].view(1,-1).float()

    def act(self,obs):
        if self.env_info['action_space'] == 'Box':
            # query to network for an action
            a = self.pi(self._policy_input(obs))
            # return a in numpy form
            return a.detach().numpy()[0]
        elif self.env_info['action_space'] == 'Discrete':
            p = self.pi(self._policy_input(obs))
            _,a = p.max(1)
            return int(a)

    def __repr__(self):
        s =  "DeterministicAgent(\n"
        s += f"  net_arch: {self.net_arch} \n"
        s += f"  observation_space: {self.env_info['observation_space']} {self.env_info['observation_dim']}\n"
        s += f"  action_space: {self.env_info['action_space']} {self.env_info['action_dim']} \n"
        s += ")"
        return s

def get_agent(env,net_arch='MLP',hs=[8]*2,activation=nn.Tanh(),policy_mode='deterministic'):
    """get correct agent based on the kind of action space"""
    # seeding of neural networks is controlled here
    # TODO should the seeding be included here?
    torch.manual_seed(0)

    if policy_mode == 'deterministic':
        # build a deterministic agent
        return DeterministicAgent(env,net_arch,hs=hs,activation=activation)
    elif policy_mode == 'stochastic':
        # build a stochastic agent
        return StochasticAgent(env,net_arch,hs=hs,activation=activation)

