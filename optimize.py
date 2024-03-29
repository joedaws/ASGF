"""
    file: optimize.py

    module used to launch various algorithms to solve optimization problems
    supported algorithms:
        - asgf
        - dgs
        - cma
        - powell
        - nelder-mead
        - bfgs
"""
import argparse
import numpy as np
from tools.function import target_function, initial_guess
from algorithms.asgf import asgf
from algorithms.dgs import dgs
import cma
from scipy.optimize import minimize


if __name__ == "__main__":
    # get arguements
    parser = argparse.ArgumentParser(description='Get problem set up variables.')
    # function name
    parser.add_argument('--fun',\
                         default='ackley',\
                         help='name of the benchmark function')
    # function dimensionality
    parser.add_argument('--dim',\
                         default='2',\
                         help='dimensionality of the benchmark function')
    # algorihtm
    parser.add_argument('--algo',\
                         default='asgf',\
                         help='name of algorithm to use (asgf / dgs / cma / powell / nelder-mead / bfgs)')
    # number of simulations
    parser.add_argument('--sim',\
                         default='1',\
                         help='number of simulations (i.e. optimization tests)')

    # parse arguements
    args = parser.parse_args()

    ''' run optimization tests '''
    if args.algo == 'asgf':
        # display the problem setup
        dim = int(args.dim)
        sim_num = int(args.sim)
        print('Optimizing {:d}d-{:s} using {:s} ({:d} simulations)'.\
            format(dim, args.fun, args.algo, sim_num))
        # setup optimization problem
        conv_sim, itr_num, fev_num = 0, 0, 0
        fun, x_min, x_dom = target_function(args.fun, dim)
        s0 = np.linalg.norm(x_dom[1] - x_dom[0]) / 10
        # run optimization tests
        for k in range(sim_num):
            np.random.seed(k)
            x0 = initial_guess(x_dom)
            x, itr_k, fev_k = asgf(fun, x0, s0)
            print('{:d}/{:d}  {:d}d-{:s}:  '.format(k+1, sim_num, dim, args.fun), end='')
            print(f'f = {fun(x):1.5e},  {itr_k:d} iterations,  {fev_k:d} evaluations')
            # record stats on successful simulations
            f_delta = np.abs((fun(x) - fun(x_min)) / (fun(x0) - fun(x)))
            if f_delta < 1e-04:
                conv_sim += 1
                itr_num += itr_k
                fev_num += fev_k
            conv_num = np.nan if conv_sim == 0 else conv_sim
        # report statistics
        print('\naverage number of iterations / evaluations / convergence rate for {:s}:'.format(args.algo))
        print('{:d}d-{:s} --- {:.0f} / {:.0f} / {:6.2f}%'.\
            format(dim, args.fun, itr_num/conv_num, fev_num/conv_num, 100*conv_sim/sim_num))

    elif args.algo == 'dgs':
        # display the problem setup
        dim = int(args.dim)
        sim_num = int(args.sim)
        print('Optimizing {:d}d-{:s} using {:s} ({:d} simulations)'.\
            format(dim, args.fun, args.algo, sim_num))
        # setup optimization problem
        conv_sim, itr_num, fev_num = 0, 0, 0
        fun, x_min, x_dom = target_function(args.fun, dim)
        dgs_params = {'ackley': {'lr': .1, 'M': 5, 'r': 5, 'beta': 1, 'gamma': .1},\
                      'levy': {'lr': .03, 'M': 17, 'r': 4, 'beta': .8, 'gamma': .001},\
                      'rastrigin': {'lr': .003, 'M': 21, 'r': 5, 'beta': 1, 'gamma': .001},\
                      'branin': {'lr': .03, 'M': 5, 'r': 1, 'beta': .2, 'gamma': .001},\
                      'cross-in-tray': {'lr': .03, 'M': 13, 'r': 2, 'beta': .4, 'gamma': .1},\
                      'dropwave': {'lr': .1, 'M': 17, 'r': 2, 'beta': .4, 'gamma': .1},\
                      # #'sphere': {'lr': .1, 'M': 5, 'r': 1, 'beta': .2, 'gamma': .01}}
                      'sphere': {'lr': 1/16, 'M': 7, 'r': 2**.5, 'beta': 2**.5/5, 'gamma': .01}}
        # run optimization tests
        for k in range(sim_num):
            np.random.seed(k)
            x0 = initial_guess(x_dom)
            x, itr_k, fev_k = dgs(fun, x0, **dgs_params[args.fun])
            print('{:d}/{:d}  {:d}d-{:s}:  '.format(k+1, sim_num, dim, args.fun), end='')
            print('f = {:.2e},  {:d} iterations,  {:d} evaluations'.format(fun(x), itr_k, fev_k))
            # record stats on successful simulations
            f_delta = np.abs((fun(x) - fun(x_min)) / (fun(x0) - fun(x)))
            if f_delta < 1e-04:
                conv_sim += 1
                itr_num += itr_k
                fev_num += fev_k
            conv_num = np.nan if conv_sim == 0 else conv_sim
        # report statistics
        print('\naverage number of iterations / evaluations / convergence rate for {:s}:'.format(args.algo))
        print('{:d}d-{:s} --- {:.0f} / {:.0f} / {:6.2f}%'.\
            format(dim, args.fun, itr_num/conv_num, fev_num/conv_num, 100*conv_sim/sim_num))

    elif args.algo == 'cma':
        # display the problem setup
        dim = int(args.dim)
        sim_num = int(args.sim)
        print('Optimizing {:d}d-{:s} using {:s} ({:d} simulations)'.\
            format(dim, args.fun, args.algo, sim_num))
        # setup optimization problem
        conv_sim, itr_num, fev_num = 0, 0, 0
        fun, x_min, x_dom = target_function(args.fun, dim)
        cma_sigma = {'ackley': 5, 'levy': 4, 'rastrigin': 5, 'branin': 1,\
                     'cross-in-tray': 2, 'dropwave': 2, 'sphere': 1}
        # run optimization tests
        for k in range(sim_num):
            np.random.seed(k)
            x0 = initial_guess(x_dom)
            cma_result = cma.fmin2(fun, x0, cma_sigma[args.fun], \
                {'tolx': 1e-06, 'maxiter': 10000, 'verb_disp': 0})[1].result
            x = cma_result[0]
            itr_k = cma_result[4]
            fev_k = cma_result[3]
            print('{:d}/{:d}  {:d}d-{:s}:  '.format(k+1, sim_num, dim, args.fun), end='')
            print('f = {:.2e},  {:d} iterations,  {:d} evaluations'.format(fun(x), itr_k, fev_k))
            # record stats on successful simulations
            f_delta = np.abs((fun(x) - fun(x_min)) / (fun(x0) - fun(x)))
            if f_delta < 1e-04:
                conv_sim += 1
                itr_num += itr_k
                fev_num += fev_k
            conv_num = np.nan if conv_sim == 0 else conv_sim
        # report statistics
        print('\naverage number of iterations / evaluations / convergence rate for {:s}:'.format(args.algo))
        print('{:d}d-{:s} --- {:.0f} / {:.0f} / {:6.2f}%'.\
            format(dim, args.fun, itr_num/conv_num, fev_num/conv_num, 100*conv_sim/sim_num))

    elif args.algo == 'powell':
        # display the problem setup
        dim = int(args.dim)
        sim_num = int(args.sim)
        print('Optimizing {:d}d-{:s} using {:s} ({:d} simulations)'.\
            format(dim, args.fun, args.algo, sim_num))
        # setup optimization problem
        conv_sim, itr_num, fev_num = 0, 0, 0
        fun, x_min, x_dom = target_function(args.fun, dim)
        # run optimization tests
        for k in range(sim_num):
            np.random.seed(k)
            x0 = initial_guess(x_dom)
            opt_result = minimize(fun, x0, method='Powell',\
                tol=1e-06, options={'gtol': 1e-06, 'norm': 2, 'maxiter': 10000})
            x = opt_result.x
            itr_k = opt_result.nit
            fev_k = opt_result.nfev
            print('{:d}/{:d}  {:d}d-{:s}:  '.format(k+1, sim_num, dim, args.fun), end='')
            print('f = {:.2e},  {:d} iterations,  {:d} evaluations'.format(fun(x), itr_k, fev_k))
            # record stats on successful simulations
            f_delta = np.abs((fun(x) - fun(x_min)) / (fun(x0) - fun(x)))
            if f_delta < 1e-04:
                conv_sim += 1
                itr_num += itr_k
                fev_num += fev_k
            conv_num = np.nan if conv_sim == 0 else conv_sim
        # report statistics
        print('\naverage number of iterations / evaluations / convergence rate for {:s}:'.format(args.algo))
        print('{:d}d-{:s} --- {:.0f} / {:.0f} / {:6.2f}%'.\
            format(dim, args.fun, itr_num/conv_num, fev_num/conv_num, 100*conv_sim/sim_num))

    elif args.algo == 'nelder-mead':
        # display the problem setup
        dim = int(args.dim)
        sim_num = int(args.sim)
        print('Optimizing {:d}d-{:s} using {:s} ({:d} simulations)'.\
            format(dim, args.fun, args.algo, sim_num))
        # setup optimization problem
        conv_sim, itr_num, fev_num = 0, 0, 0
        fun, x_min, x_dom = target_function(args.fun, dim)
        # run optimization tests
        for k in range(sim_num):
            np.random.seed(k)
            x0 = initial_guess(x_dom)
            opt_result = minimize(fun, x0, method='Nelder-Mead',\
                tol=1e-06, options={'gtol': 1e-06, 'norm': 2, 'maxiter': 10000})
            x = opt_result.x
            itr_k = opt_result.nit
            fev_k = opt_result.nfev
            print('{:d}/{:d}  {:d}d-{:s}:  '.format(k+1, sim_num, dim, args.fun), end='')
            print('f = {:.2e},  {:d} iterations,  {:d} evaluations'.format(fun(x), itr_k, fev_k))
            # record stats on successful simulations
            f_delta = np.abs((fun(x) - fun(x_min)) / (fun(x0) - fun(x)))
            if f_delta < 1e-04:
                conv_sim += 1
                itr_num += itr_k
                fev_num += fev_k
            conv_num = np.nan if conv_sim == 0 else conv_sim
        # report statistics
        print('\naverage number of iterations / evaluations / convergence rate for {:s}:'.format(args.algo))
        print('{:d}d-{:s} --- {:.0f} / {:.0f} / {:6.2f}%'.\
            format(dim, args.fun, itr_num/conv_num, fev_num/conv_num, 100*conv_sim/sim_num))

    elif args.algo == 'bfgs':
        # display the problem setup
        dim = int(args.dim)
        sim_num = int(args.sim)
        print('Optimizing {:d}d-{:s} using {:s} ({:d} simulations)'.\
            format(dim, args.fun, args.algo, sim_num))
        # setup optimization problem
        conv_sim, itr_num, fev_num = 0, 0, 0
        fun, x_min, x_dom = target_function(args.fun, dim)
        # run optimization tests
        for k in range(sim_num):
            np.random.seed(k)
            x0 = initial_guess(x_dom)
            opt_result = minimize(fun, x0, method='BFGS',\
                tol=1e-06, options={'gtol': 1e-06, 'norm': 2, 'maxiter': 10000})
            x = opt_result.x
            itr_k = opt_result.nit
            fev_k = opt_result.nfev
            print('{:d}/{:d}  {:d}d-{:s}:  '.format(k+1, sim_num, dim, args.fun), end='')
            print('f = {:.2e},  {:d} iterations,  {:d} evaluations'.format(fun(x), itr_k, fev_k))
            # record stats on successful simulations
            f_delta = np.abs((fun(x) - fun(x_min)) / (fun(x0) - fun(x)))
            if f_delta < 1e-04:
                conv_sim += 1
                itr_num += itr_k
                fev_num += fev_k
            conv_num = np.nan if conv_sim == 0 else conv_sim
        # report statistics
        print('\naverage number of iterations / evaluations / convergence rate for {:s}:'.format(args.algo))
        print('{:d}d-{:s} --- {:.0f} / {:.0f} / {:6.2f}%'.\
            format(dim, args.fun, itr_num/conv_num, fev_num/conv_num, 100*conv_sim/sim_num))

    else:
        raise SystemExit('algorithm {:s} is not recognized, supported algorithms are:'\
            ' asgf, dgs, cma, powell, nelder-mead, bfgs'.format(args.algo))

