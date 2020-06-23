"""
file: run/dgs.py

    file used to launch dgs to solve optimization problems
"""

import sys
import argparse
import numpy as np
from mpi4py import MPI
from tools.function import get_function, initial_guess, get_function_list
from tools.mpi_util import mpi_fork
from tools.util import make_rl_j_fn
from tools.scribe import FScribe, RLScribe, hs_to_str
from algorithms.dgs import dgs, dgs_parallel_main, dgs_parallel_aux
from algorithms.parameters import init_dgs
from scipy.optimize import minimize
import mlflow

dgs_params = {'ackley': {'lr': .1, 'M': 5, 'r': 5, 'beta': 1, 'gamma': .1},\
              'levy': {'lr': .03, 'M': 17, 'r': 4, 'beta': .8, 'gamma': .001},\
              'rastrigin': {'lr': .003, 'M': 21, 'r': 5, 'beta': 1, 'gamma': .001},\
              'branin': {'lr': .03, 'M': 5, 'r': 1, 'beta': .2, 'gamma': .001},\
              'cross-in-tray': {'lr': .03, 'M': 13, 'r': 2, 'beta': .4, 'gamma': .1},\
              'dropwave': {'lr': .1, 'M': 17, 'r': 2, 'beta': .4, 'gamma': .1},\
              'sphere': {'lr': .1, 'M': 5, 'r': 1, 'beta': .2, 'gamma': .01}}

def run_dgs(algo,args):
    if algo == 'dgs_parallel':
        
        # MPI related variables
        comm = MPI.COMM_WORLD
        L = comm.Get_size()
        rank = comm.Get_rank()
        
        if args.fun in get_function_list():
            # display the problem setup
            # setup optimization problem
            conv_sim, itr_num, fev_num = 0, 0, 0
            fun, x_min, x_dom = get_function(args.fun, dim,args.process, args.mean, args.std)

            # run optimization tests
            np.random.seed(args.seed)
            x0 = initial_guess(x_dom)
            if rank == 0:
                # set up scribe
                scribe = FScribe('data/'+algo+'/'+str(k),args.algo)

                # launch main process
                x, itr_k, fev_k = dgs_parallel_main(comm,L,rank,fun,x0,scribe,param)
                
            else:
                # launch aux process
                dgs_parallel_aux(comm,L,rank,fun,x0,param)
        
            # report statistics
            if rank == 0:
                log_metrics({
                            "iterations_used": itr_k,
                            "function_evalutions": fev_k
                            })              
                mlflow.log_param("minimizer",x)
                print(f'Done optimizing {args.fun}') 
                
        # reinforcement learning case
        else:
            # set up optimization problem
            J,dim = make_rl_j_fn(args.fun,hs=args.hidden_sizes)
            fun = lambda x,i: -J(x,i)
            x0 = np.random.randn(dim)
            
            # run optimization tests
            np.random.seed(args.seed)
            if rank == 0:
                scribe = RLScribe('data/'+algo,args.fun,hs_to_str(args.hidden_sizes))
                x, itr_k, fev_k = dgs_parallel_main(comm,L,rank,fun,x0,scribe,param)
            else:
                dgs_parallel_aux(comm,L,rank,fun,x0,param)
        
            # report statistics
            if rank == 0:
                log_metrics({
                            "iterations_used": itr_k,
                            "function_evalutions": fev_k
                            })              
                mlflow.log_param("minimizer",x)
                print(f'Done training {args.fun}')

    elif algo == 'dgs':
        # setup optimization problem
        fun, x_min, x_dom = get_function(args.fun,args.dim,args.process,args.mean,args.std)
        
        # run optimization tests
        np.random.seed(args.seed)
        x0 = initial_guess(x_dom)
        x, itr_k, fev_k = dgs(fun, x0, param)

        # TODO record mlflow stats
        mlflow.log_metrics({
            "iterations_used": itr_k,
            "function_evalutions": fev_k
        })
        mlflow.log_param("minimizer",x)

        print(f'Done optimizing {args.fun}')




if __name__ == "__main__":
    # get arguements
    parser = argparse.ArgumentParser(description='Get problem set up variables.')
    # function name
    parser.add_argument('fun',
                         type=str,
                         default='ackley',
                         help='name of the objective function')
    
    # function dimensionality
    parser.add_argument('--dim',
                         type=int,
                         default='10',
                         help='dimensionality of the benchmark function')
    
    # deterministic or stochastic
    parser.add_argument('--process',
                         default='deterministic',
                         help='either deterministic or stochastic function evaluations.')

    # mean of additive noise
    parser.add_argument('--mean',
                         default=0,
                         help='mean of the additive noise')

    # standard devation of additive noise
    parser.add_argument('--std',
                         default=0.05,
                         help='standard devation of the additive noise')

    # number of processes
    parser.add_argument('--nprocs',
                         type=int,
                         default=1,
                         help='number of parallel processes to use')

    # parameter for dgs
    parser.add_argument('--M',
                        type=int,
                        default=7,
                        help='number of quadrature points used in GH')
    
    # parameter for dgs
    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help='learning rate')
    # parameter for dgs
    parser.add_argument('--r',
                        type=float,
                        default=1.5,
                        help='radius')
    # parameter for dgs
    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help='alpha')
    # parameter for dgs
    parser.add_argument('--beta',
                        type=float,
                        default=0.3,
                        help='beta')
   
    # parameter for dgs
    parser.add_argument('--gamma',
                        type=float,
                        default=0.1,
                        help='gamma')
    # parameter for dgs
    parser.add_argument('--maxiter',
                        type=int,
                        default=5000,
                        help='maximum number of iterations')
    # parameter for dgs
    parser.add_argument('--gtol',
                        type=float,
                        default=1e-06,
                        help='tolerance for algorithm')
    # parameter for dgs
    parser.add_argument('--optimizer',
                        type=str,
                        default='grad',
                        help='type of optimization step, either grad or adam')

    # hidden layer sizes
    parser.add_argument('--hidden_sizes',
                         #nargs='+',
                         type=lambda x: [int(item) for item in x.split(',')],
                         default=[8,8],
                         help='list of ints for hidden layer sizes')

    # rng seed
    parser.add_argument('--seed',
                        default=0,help='numpy rng seed.')

    # parse arguements
    args = parser.parse_args()

    # decide to run in parallel or not
    if args.nprocs > 1:
        algo = 'dgs_parallel'
    else:
        algo = 'dgs'

    # get parameters
    param = init_dgs(lr=args.lr,
                     M=args.M,
                     r=args.r,
                     alpha=args.alpha,
                     beta=args.beta,
                     gamma=args.gamma,
                     verbose=0,
                     optimizer=args.optimizer)

    # fork processes if nprocs > 1 and kill the original
    if "parent" == mpi_fork(args.nprocs): sys.exit()

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        print(f"experiment_id is {experiment_id}")
        run_dgs(algo,args)

