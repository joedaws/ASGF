from inspect import signature
from mpi4py import MPI
import numpy as np
from tools.scribe import RLScribe, FScribe, hs_to_str
from tools.optimizer import AdamUpdater
from tools.util import make_rl_j_fn, setup_agent_env, update_net_param, get_net_param
from tools.function import get_function_list
from algorithms.parameters import init_dgs

# set print options
np.set_printoptions(linewidth=100, suppress=True, formatter={'float':'{: 0.4f}'.format})
# RNG seed for numpy
np.random.seed(0)
def get_g(f,x,xi,n):
    if n == 1:
        return lambda t: f(x + t*xi)
    elif n == 2:
        return lambda t,itr: f(x + t*xi,itr)

def get_dg_val(g,n,M,s_d,i):
    p,w = np.polynomial.hermite.hermgauss(M)
    if n == 1:
        g_val = np.array([g(p_i*s_d) for p_i in p])
    elif n == 2:
        g_val = np.array([g(p_i*s_d,i) for p_i in p])
    return np.sum(w*p*g_val) / (s_d*np.sqrt(np.pi)/2)

def eval_f(f,x,n,i):
    if n == 1:
        return f(x)
    elif n == 2:
        return f(x,i)

'''
        the following is the implementation of the directional gaussian smoothing
        optimization algorithm (https://arxiv.org/abs/2002.03001)
        the default values of hyperparameters are taken from the paper
        on reinforcement learning (https://arxiv.org/abs/2002.09077)

        inputs:
                fun -- function handle
                x0  -- initial guess
                lr  -- learning rate
                M   -- number of quadrature points to use
                r, alpha, beta, gamma -- other hyperparameters
                maxiter -- maximal number of iterations
                xtol -- tolerance for the change of x

        outputs:
                x -- the minimizer
'''
# Directional Gaussian Smoothing
def dgs(fun, x0, param):
    """
    Inputs:
        fun -- function to be minimized
        x0  -- initial guess of minimizer
        param -- SimpleNamespace for all the reqequired algorithm parameters

    Outputs:
        x_min -- minimizer obtained by algorithm
        itr_min -- number of iterations required to obtain minimizer
        fun_eval_min -- number of fucntion evaluations used.
    """
    # check signiture of fun
    sig_fun = signature(fun)
    num_fun_args = len(sig_fun.parameters)

    # initialize variables
    x, dim = np.copy(x0), len(x0)

    #TODO should we initialize to random directions?
    u = np.eye(dim)
    s = param.r * np.ones(dim)
    # save the initial state
    x_min = np.copy(x)
    #f_min = fun_x = fun(x)
    f_min = fun_x = eval_f(fun,x,num_fun_args,0)
    itr_min, fun_eval, fun_eval_min = 0, 1, 1

    for itr in range(param.maxiter):
        # initialize gradient
        dg = np.zeros(dim)

        # estimate gradient along each direction
        for d in range(dim):
            # define directional function
            g = get_g(fun,x,u[d],num_fun_args)
            # estimate smoothed gradient
            #p, w = np.polynomial.hermite.hermgauss(param.M)
            #g_val = np.array([g(p_i*s[d]) for p_i in p])
            fun_eval += param.M-1
            #dg[d] = np.sum(w*p * g_val) / (s[d] * np.sqrt(np.pi)/2)
            dg[d] = get_dg_val(g,num_fun_args,param.M,s[d],itr)

        # assemble the gradient
        df = np.matmul(dg, u)
        
        # step of gradient descent
        x -= param.lr * df

        #fun_x = fun(x)
        fun_x = eval_f(fun,x,num_fun_args,itr)
        fun_eval += 1

        # save the best state so far
        if fun_x < f_min:
            x_min = np.copy(x)
            f_min = fun_x
            itr_min = itr
            fun_eval_min = fun_eval
        # report the current state
        if param.verbose > 0:
            print('dgs-iteration {:d}: f = {:.2e}'.format(itr+1, fun_x))

        # update parameters
        if np.linalg.norm(param.lr*df) < param.gtol:
            break
        elif np.linalg.norm(df) < param.gamma:
            Du = np.random.random((dim,dim))
            u = np.eye(dim) + param.alpha * (Du - Du.T)
            s = (param.r + param.beta * (2*np.random.random(dim) - 1))

    return x_min, itr_min+1, fun_eval_min

def get_split_sizes(data,size):
    """gets correct split sizes for size number of processes to split data"""
    split = np.array_split(data,size,axis=0)
    split_sizes = []
    for i in range(len(split)):
        split_sizes = np.append(split_sizes, len(split[i]))

    return split_sizes

def dgs_parallel_main(comm,L,rank,fun,x0,scribe,param):
    """Directional Gaussian Smoothing Parallel implementation for main process
        the following is the implementation of the directional gaussian smoothing
        optimization algorithm (https://arxiv.org/abs/2002.03001)
        the default values of hyperparameters are taken from the paper
        on reinforcement learning (https://arxiv.org/abs/2002.09077)

        Inputs:
            comm -- mpi comm world
            L    -- number of processes
            rank -- rank of process
            fun -- function to be minimized
            x0  -- initial guess
            scribe -- instance of scribe class for recording
            param -- paramters for the algorithm

        Outputs:
                x   -- the minimizer
                its -- number of iterations until minimizer was obtained
    """
    # everybody does this
    x, dim = np.copy(x0), len(x0)

    # main process initializes variables
    # splits and prepares data for scatter
    # initialize u and s
    u = np.eye(dim)
    #u = np.random.rand(dim,dim)

    s = param.r * np.ones(dim)

    # get split sizes
    split_sizes_u = get_split_sizes(u,L)
    split_sizes_s = get_split_sizes(s,L)

    # number of matrix elements to be recieved by each process
    count_u = split_sizes_u*dim
    displacements_u = np.insert(np.cumsum(count_u),0,0)[0:-1]

    # number of vector elements to be recieved by each process
    count_s = split_sizes_s
    displacements_s = np.insert(np.cumsum(count_s),0,0)[0:-1]

    # main initializes gradient
    dg = np.zeros(dim)

    # main initializes optimizer
    opt = AdamUpdater()

    # initialize the best_val to be infinity
    best_val = np.inf

    # get exp_num
    exp_num = 1000*scribe.exp_num

    # initialize flags
    break_flag = False
    update_flag = False

    # broadcast necessary info to all 
    aux_sizes_u = comm.bcast(split_sizes_u, root = 0)
    aux_sizes_s = comm.bcast(split_sizes_s, root = 0)
    count_u = comm.bcast(count_u, root = 0)
    count_s = comm.bcast(count_s, root = 0)
    displacements_u = comm.bcast(displacements_u, root = 0)
    displacements_s = comm.bcast(displacements_s, root = 0)
    exp_num = comm.bcast(exp_num,root=0)

    # create chunks
    chunk_u = np.zeros((int(aux_sizes_u[rank]),dim))
    chunk_s = np.zeros(int(aux_sizes_s[rank]))
    chunk_dg = np.zeros(int(aux_sizes_s[rank]))

    # scatter u and s
    comm.Scatterv([u,count_u,displacements_u,MPI.DOUBLE], chunk_u,root=0)
    comm.Scatterv([s,count_s,displacements_s,MPI.DOUBLE], chunk_s,root=0)
   
    # initialize function evaluation counter
    fun_eval = 1

    # check signiture of fun
    sig_fun = signature(fun)
    num_fun_args = len(sig_fun.parameters)

    # wait for everyone
    comm.Barrier()

    # begin iterations for everyone
    for itr in range(param.maxiter):

        # broadcast new x
        x = comm.bcast(x,root=0)

        # scatter u and s
        if update_flag == True:
            comm.Scatterv([u,count_u,displacements_u,MPI.DOUBLE], chunk_u,root=0)
            comm.Scatterv([s,count_s,displacements_s,MPI.DOUBLE], chunk_s,root=0)
            update_flag = False

        #for d in range(int(displacements_s[rank]),int(displacements_s[rank]+count_s[rank])):
        for d in range(chunk_u.shape[0]):
            # define directional function
            #g = lambda t : fun(x + t*chunk_u[d])
            g = get_g(fun,x,chunk_u[d],num_fun_args)
            # estimate smoothed gradient
            #p, w = np.polynomial.hermite.hermgauss(param.M)
            #g_val = np.array([g(p_i*chunk_s[d]) for p_i in p])
            #chunk_dg[d] = np.sum(w * p * g_val)/(chunk_s[d] * np.sqrt(np.pi)/2)
            chunk_dg[d] = get_dg_val(g,num_fun_args,param.M,chunk_s[d],itr)

        # increment number of function evaluations
        fun_eval += dim*(param.M-1) 
        
        # syncronize
        comm.Barrier()

        # main collects dg
        comm.Gatherv(chunk_dg,[dg,count_s,displacements_s,MPI.DOUBLE], root = 0)

        # main updates df i.e. assembling the gradient
        df = np.matmul(dg, u)

        # step of gradient descent
        if param.optimizer == 'grad':
            x -= param.lr * df
        elif param.optimizer == 'adam':
            opt.step(x,param.lr,df)

        #new_val = fun(x)
        new_val = eval_f(fun,x,num_fun_args,itr)
        fun_eval += 1

        # scribe record the progress
        scribe.record_iteration_data(iteration=itr+1,reward=-new_val,inf_norm_diff=np.amax(np.abs(np.abs(param.lr*df))))
        print(f"iteration {itr+1:3d} | reward = {-new_val:6.5f} | inf-norm-diff = {np.amax(np.abs(param.lr*df)):4.5e}")

        # test if we have found new best value
        if new_val < best_val:
            best_val = new_val
            # checkpoint
            scribe.checkpoint(x,opt,itr+1,best=True)

        # check point for 100 iterations
        if itr+1 % 100 == 0:
            scribe.checkpoint(x,opt,itr+1)

        # update parameters
        if np.linalg.norm(df) < param.gtol:
            break_flag = True
        elif np.linalg.norm(df) < param.gamma:
            #print(f"main updating on it {i}")
            Du = np.random.random((dim,dim))
            u = np.eye(dim) + param.alpha * (Du - Du.T)
            s = (param.r + param.beta * (2*np.random.random(dim) - 1))
            update_flag = True

        # broadcast the flags
        break_flag = comm.bcast(break_flag,root=0)
        update_flag = comm.bcast(update_flag,root=0)

        # break out of loop if break flag is set to true
        if break_flag:
            break

    scribe.record_metadata(total_iterations=itr+1,final_value=-new_val)
    # main process returns values
    return x, itr+1, fun_eval 

def dgs_parallel_aux(comm,L,rank,fun,x0,param):
    """Directional Gaussian Smoothing Parallel implementation for auxiliary processes
        the following is the implementation of the directional gaussian smoothing
        optimization algorithm (https://arxiv.org/abs/2002.03001)
        the default values of hyperparameters are taken from the paper
        on reinforcement learning (https://arxiv.org/abs/2002.09077)

        inputs:
            comm -- mpi comm world
            L    -- number of processes
            rank -- rank of process
            fun -- function handle
            x0  -- initial guess
            param -- dgs parameters
    """
    # everybody does this
    x, dim = np.copy(x0), len(x0)

    # aux initialize placeholders
    aux_sizes_u = None
    aux_sizes_s = None
    split_sizes_u = None
    split_sizes_s = None
    count_u = None
    count_s = None
    displacements_u = None
    displacements_s = None
    u = None
    s = None
    dg = None
    exp_num = None

    # initialize flags
    break_flag = False
    update_flag = False

    # broadcast necessary info to all 
    aux_sizes_u = comm.bcast(split_sizes_u, root = 0)
    aux_sizes_s = comm.bcast(split_sizes_s, root = 0)
    count_u = comm.bcast(count_u, root = 0)
    count_s = comm.bcast(count_s, root = 0)
    displacements_u = comm.bcast(displacements_u, root = 0)
    displacements_s = comm.bcast(displacements_s, root = 0)
    exp_num = comm.bcast(exp_num,root=0)

    # create chunks
    chunk_u = np.zeros((int(aux_sizes_u[rank]),dim))
    chunk_s = np.zeros(int(aux_sizes_s[rank]))
    chunk_dg = np.zeros(int(aux_sizes_s[rank]))
    # scatter u and s
    comm.Scatterv([u,count_u,displacements_u,MPI.DOUBLE], chunk_u,root=0)
    comm.Scatterv([s,count_s,displacements_s,MPI.DOUBLE], chunk_s,root=0)

    # check signiture of fun
    sig_fun = signature(fun)
    num_fun_args = len(sig_fun.parameters)

    # wait for everyone
    comm.Barrier()

    # begin iterations for everyone
    for itr in range(param.maxiter):

        # broadcast new x
        x = comm.bcast(x,root=0)

        # scatter u and s
        if update_flag == True:
            #print(f"process {rank} on it {i} is updating")
            comm.Scatterv([u,count_u,displacements_u,MPI.DOUBLE], chunk_u,root=0)
            comm.Scatterv([s,count_s,displacements_s,MPI.DOUBLE], chunk_s,root=0)
            update_flag = False

        #for d in range(int(displacements_s[rank]),int(displacements_s[rank]+count_s[rank])):
        for d in range(chunk_u.shape[0]):
            # define directional function
            #g = lambda t : fun(x + t*chunk_u[d])
            g = get_g(fun,x,chunk_u[d],num_fun_args)
            # estimate smoothed gradient
            #p, w = np.polynomial.hermite.hermgauss(param.M)
            #g_val = np.array([g(p_i*chunk_s[d]) for p_i in p])
            #chunk_dg[d] = np.sum(w * p * g_val)/(chunk_s[d] * np.sqrt(np.pi)/2)
            chunk_dg[d] = get_dg_val(g,num_fun_args,param.M,chunk_s[d],itr)

        # syncronize
        comm.Barrier()

        # main collects dg
        comm.Gatherv(chunk_dg,[dg,count_s,displacements_s,MPI.DOUBLE], root = 0)

        # broadcast the flags
        break_flag = comm.bcast(break_flag,root=0)
        update_flag = comm.bcast(update_flag,root=0)

        # break out of loop if break flag is set to true
        if break_flag:
            break

def dgs_parallel(args):
    """parallel implementation of dgs

    kicks off dgs optimiation depending on the arguements in the parser object args
    """
    # MPI related variables
    comm = MPI.COMM_WORLD
    L = comm.Get_size()
    rank = comm.Get_rank()
    
    # check number of processes
    if args.nprocs <= 1:
        raise ValueError('Number of processes must be more than 1')
    
    if args.fun in get_function_list():
        # display the problem setup
        dim = int(args.dim)
        sim_num = int(args.sim)
        if rank == 0:
            print('Optimizing {:d}d-{:s} using {:s} ({:d} simulations)'.\
                   format(dim, args.fun, args.algo, sim_num))
        
        # setup optimization problem
        conv_sim, itr_num, fev_num = 0, 0, 0
        fun, x_min, x_dom = get_function(args.fun, dim,args.process, args.mean, args.std)
        dgs_params = {'ackley': {'lr': .1, 'M': 5, 'r': 5, 'beta': 1, 'gamma': .1},\
                      'levy': {'lr': .03, 'M': 17, 'r': 4, 'beta': .8, 'gamma': .001},\
                      'rastrigin': {'lr': .003, 'M': 21, 'r': 5, 'beta': 1, 'gamma': .001},\
                      'branin': {'lr': .03, 'M': 5, 'r': 1, 'beta': .2, 'gamma': .001},\
                      'cross-in-tray': {'lr': .03, 'M': 13, 'r': 2, 'beta': .4, 'gamma': .1},\
                      'dropwave': {'lr': .1, 'M': 17, 'r': 2, 'beta': .4, 'gamma': .1},\
                      'sphere': {'lr': .1, 'M': 5, 'r': 1, 'beta': .2, 'gamma': .01}}
    
        # get correct parameters
        param = init_dgs(**dgs_params[args.fun])

        # run optimization tests
        for k in range(sim_num):
            np.random.seed(k)
            x0 = initial_guess(x_dom)
            if rank == 0:
                scribe = FScribe('data/'+args.algo+'/'+str(k),args.algo)
                x, itr_k, fev_k = dgs_parallel_main(comm,L,rank,fun,x0,scribe,param)
                print('{:d}/{:d}  {:d}d-{:s}:  '.format(k+1, sim_num, dim, args.fun), end='')
                print('f = {:.2e},  {:d} iterations,  {:d} evaluations'.format(fun(x), itr_k, fev_k))
                # record stats on successful simulations
                f_delta = np.abs((fun(x) - fun(x_min)) / (fun(x0) - fun(x)))
                if f_delta < 1e-04:
                    conv_sim += 1
                    itr_num += itr_k
                    fev_num += fev_k
                conv_num = np.nan if conv_sim == 0 else conv_sim
            else:
                dgs_parallel_aux(comm,L,rank,fun,x0,param)
        
        # report statistics
        if rank == 0:
            print('\naverage number of iterations / evaluations / convergence rate for {:s}:'.format(args.algo))
            print('{:d}d-{:s} --- {:.0f} / {:.0f} / {:6.2f}%'.\
                format(dim, args.fun, itr_num/conv_num, fev_num/conv_num, 100*conv_sim/sim_num))
    
    # reinforcement learning case
    else:
        # get correct parameters
        param = init_dgs(lr=0.1,M=7,r=np.sqrt(2),alpha=2.,beta=0.2*np.sqrt(2),gamma=0.001,verbose=1,optimizer='adam')

        # set up optimization problem
        J,dim = make_rl_j_fn(args.fun,hs=args.hidden_sizes)
        fun = lambda x,i: -J(x,i)
        x0 = np.random.randn(dim)
        
        # get number of simulations
        sim_num = int(args.sim)

        # run optimization tests
        for k in range(sim_num):
            np.random.seed(k)
            if rank == 0:
                scribe = RLScribe('data/'+args.algo,args.fun,hs_to_str(args.hidden_sizes))
                scribe.exp_num = k
                x, itr_k, fev_k = dgs_parallel_main(comm,L,rank,fun,x0,scribe,param)
                print('{:d}/{:d}  {:d}d-{:s}:  '.format(k+1, sim_num, dim, args.fun), end='')
                print('f = {:.2e},  {:d} iterations,  {:d} evaluations'.format(fun(x), itr_k, fev_k))
            else:
                dgs_parallel_aux(comm,L,rank,fun,x0,param)
        
        # report statistics
        if rank == 0:
            print(f'Done training {args.fun}')

def dgs_parallel_train(rank,exp_num,env_name,maxiter,hidden_layers=[8.8],policy_mode='deterministic'):
    """
    train an agent to solve the env_name task using
    dgs optimization
    """

    # number of layers of the neural network
    net_layers = hidden_layers

    # set up scribe
    root_save = 'data/dgs'
    env_name = env_name
    arch_type = hs_to_str(net_layers)
    scribe = RLScribe(root_save,env_name,arch_type)
    scribe.exp_num = exp_num

    # generate reward function
    J,d = make_rl_j_fn(env_name, hs=net_layers)

    # setup agent
    agent,env,net = setup_agent_env(env_name,hs=net_layers,policy_mode=policy_mode)

    # initial guess of parameter vector
    #np.random.seed(0)
    #w0 = np.random.randn(d)/10
    w0 = get_net_param(net)

    if rank == 0:
        print('problem dimensionality:', d)
        print('iteration   0: reward = {:6.2f}'.format(J(w0,1)))

    # run dgs parallel implementation
    w, itr = dgs_parallel(lambda w,i: -J(w,i), w0, scribe=scribe,maxiter=maxiter,gamma=0)


