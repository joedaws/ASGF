"""
    file: train.py

    module used to launch adgs and dgs for training neural networks
    to solve reinforcement learning tasks.
"""
import argparse
from mpi4py import MPI
from algorithms.asgf import asgf_parallel_train
from algorithms.dgs import dgs_parallel_train
from algorithms.es import es_parallel_train

def get_maxiter(env_name):
    """get the maximum number of iterations to run the solver
    for a reinforceemnt learning environment problem
    """
    DEFAULT_ITS = 200
    max_its = {
                   'Pendulum-v0':200,
                   'InvertedPendulumBulletEnv-v0':100,
                   'Acrobot-v1':200,
                   'CartPole-v1':100,
                   'MountainCarContinuous-v0':100,
                   'HopperBulletEnv-v0':400,
                   'ReacherBulletEnv-v0':150,
                }

    try:
        its = max_its[env_name]

    except:
        print(f"No default number of iterations for {env_name}. Using 200.")
        its = DEFAULT_ITS

    return its

if __name__ == "__main__":
    # get arguements
    parser = argparse.ArgumentParser(description='Get problem set up variables.')
    # environment name
    parser.add_argument('--env_name',
                         help='name of openAI gym environment')
    # algorihtm
    parser.add_argument('--algo',
                         default='asgf',
                         help='name of algorithm to use. Either dgs or asgf.')
    # RNG seed
    parser.add_argument('--seed',
                         default=0,
                         help='value of random seed used for environment')
    # hidden layer sizes
    parser.add_argument('--hidden_sizes',
                         #nargs='+',
                         type=lambda x: [int(item) for item in x.split(',')],
                         default=[8,8],
                         help='list of ints for hidden layer sizes')

    # policy mode
    parser.add_argument('--policy_mode',
                         default='deterministic',
                         help='mode by which agent chooses actions. either prob or deterministic.')

    # parse arguements
    args = parser.parse_args()

    # get maximum number of iterations
    maxiter = get_maxiter(args.env_name)

    # set up comm world and get rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if args.algo == 'asgf':
        if rank == 0:
            # print some info
            print(f"Begin Training for {args.env_name} using {args.algo} with {size} workers")

        # train agent
        asgf_parallel_train(rank, int(args.seed), args.env_name, maxiter, hidden_layers=args.hidden_sizes, policy_mode=args.policy_mode)

    elif args.algo == 'dgs':
        if rank == 0:
            # print some info
            print(f"Begin Training for {args.env_name} using {args.algo} with {size} workers")

        # train agent
        dgs_parallel_train(rank, int(args.seed), args.env_name, maxiter, hidden_layers=args.hidden_sizes, policy_mode=args.policy_mode)

    elif args.algo == 'es':
        if rank == 0:
            # print some info
            print(f"Begin Training for {args.env_name} using {args.algo} with {size} workers")

        # train agent
        es_parallel_train(rank, int(args.seed), args.env_name, maxiter, hidden_layers=args.hidden_sizes, policy_mode=args.policy_mode)


