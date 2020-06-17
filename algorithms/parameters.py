"""
    file: parameters.py
"""
from types import SimpleNamespace

import numpy as np

def init_asgf():
    """
    generic defaults for asgf
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
                    'verbose':0
                 }

    return SimpleNamespace(**asgf_param)

def not_yet_implemented(s):
    print(f"Sorry but defaults for {s} are not yet implemented")

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
