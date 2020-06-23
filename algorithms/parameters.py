"""
    file: parameters.py
"""
from types import SimpleNamespace

import numpy as np

def init_asgf(**kwargs):
    """
    with no additional inputs returns default parameters for the asgf algorithm
    example: param = init_asgf()

    with addtional inputs reutrns custom paramters for the asgf algorithm
    example: param = init_asgf(s_rate=0.3,maxiter=300)
    """
    asgf_param = {
                    's_rate':0.9,
                    'm_min':5,
                    'm_max':21,
                    'qtol':.1,
                    'A_grad':.1, 
                    'B_grad':.9, 
                    'A_dec':.95, 
                    'A_inc':1.02, 
                    'B_dec':.98, 
                    'B_inc':1.01,		
                    'L_avg':1, 
                    'L_lmb':.9, 
                    's_min':1e-03, 
                    's_max':None, 
                    'lr_min':1e-03, 
                    'lr_max':1e+03,
                    'restart':True, 
                    'num_res':2, 
                    'res_mult':10, 
                    'res_div':10, 
                    'fun_req':-np.inf,
                    'maxiter':5000, 
                    'xtol':1e-06, 
                    'verbose':0,
                    'optimizer':'grad'
                 }

    # update if necessary
    for k in kwargs:
        asgf_param[k] = kwargs[k]

    return SimpleNamespace(**asgf_param)

def init_dgs(**kwargs):
    """
    without additional inputs returns a default paramter set for the dgs algorithm
    example: param = init_dgs()

    each parameter can also be set
    example: param = init_dgs(M=9)
    """
    # initial values of paramters
    dgs_param = {
                    'lr':0.1,
                    'M':7,
                    'r':1.5,
                    'alpha':1.0,
                    'beta':0.3,
                    'gamma':0.1,
                    'maxiter':5000,
                    'gtol':1e-06,
                    'verbose':0,
                    'optimizer':'grad'
                }

    # update if necessary
    for k in kwargs:
        dgs_param[k] = kwargs[k] 
    
    return SimpleNamespace(**dgs_param)

# TODO not sure if best way to do this
def get_asgf_param(problem_type='low_d_opt'):
    """
    modify default parameters for asgf 
    for several different problem cases
    """
    asgf_param = init_defaults()

    if problem_type == 'low_d_opt':
        ''' low-dimensional optimizaiton '''
        pass
    
    elif problem_type == 'high_d_opt':
        ''' high-dimensional optimization '''
        pass
    
    elif problem_type == 'stochastic_opt':
        ''' stochastic optimization'''
        not_yet_implemented(problem_type)

    elif problem_type == 'rl':
        ''' reinforcement learning '''
        not_yet_implemented(problem_type)

    elif problem_type == 'sl':
        ''' supervised learning '''
        not_yet_implemented(problem_type)

    else:
        print(f"No known paramters for {problem_type} using defaults for optimization")

    return asgf_param

def not_yet_implemented(s):
    print(f"Sorry but defaults for {s} are not yet implemented")


