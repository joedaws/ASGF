from mpi4py import MPI
import numpy as np
from tools.scribe import RLScribe, FScribe, hs_to_str
from tools.optimizer import AdamUpdater
from tools.util import make_rl_j_fn, setup_agent_env, update_net_param, get_net_param
# set print options
np.set_printoptions(linewidth=100, suppress=True, formatter={'float':'{: 0.4f}'.format})
# RNG seed for numpy
np.random.seed(0)


def get_split_sizes(data,size):
    """gets correct split sizes for size number of workers to split data"""
    split = np.array_split(data,size,axis=0)
    split_sizes = []
    for i in range(len(split)):
        split_sizes = np.append(split_sizes, len(split[i]))
    return split_sizes

def es_master(comm,L,rank,fun, x0,
        scribe=RLScribe('data_es','unknown','unkonwn'),
        lr=.1,
        N=1000,
        s0=np.sqrt(2),
        maxiter=500,
        gtol=1e-06):
    """Evolution Strategies Parallel implementation for master

        inputs:
                fun -- function handle
                x0  -- initial guess
                scribe -- instance of scribe class for recording
                lr  -- learning rate
                N   -- number of Monte-Carlo samples to use
                s0  -- initial smoothneess parameter
                maxiter -- maximal number of iterations
                gtol    -- tolerance for the magnitude of the gradient

        outputs:
                x   -- the minimizer
                its -- number of iterations until minimizer was obtained
    """
    # everybody does this
    x, dim = np.copy(x0), len(x0)

    # master process initializes variables
    # splits and prepares data for scatter
    # initialize u and s
    u = np.random.randn(N, dim)
    s = s0 * np.ones(N)

    # get split sizes
    split_sizes_u = get_split_sizes(u,L)
    split_sizes_s = get_split_sizes(s,L)

    # number of matrix elements to be recieved by each worker
    count_u = split_sizes_u*dim
    displacements_u = np.insert(np.cumsum(count_u),0,0)[0:-1]

    # number of vector elements to be recieved by each worker
    count_s = split_sizes_s
    displacements_s = np.insert(np.cumsum(count_s),0,0)[0:-1]

    # master initializes gradient
    dg = np.zeros(N)

    # master initializes optimizer
    opt = AdamUpdater()

    # initialize the best_val to be infinity
    best_val = np.inf

    # get exp_num
    exp_num = 1000*scribe.exp_num

    # initialize flags
    break_flag = False
    update_flag = False

    # broadcast necessary info to all workers
    worker_sizes_u = comm.bcast(split_sizes_u, root = 0)
    worker_sizes_s = comm.bcast(split_sizes_s, root = 0)
    count_u = comm.bcast(count_u, root = 0)
    count_s = comm.bcast(count_s, root = 0)
    displacements_u = comm.bcast(displacements_u, root = 0)
    displacements_s = comm.bcast(displacements_s, root = 0)
    exp_num = comm.bcast(exp_num, root=0)

    # create worker_chunks
    worker_chunk_u = np.zeros((int(worker_sizes_u[rank]), dim))
    worker_chunk_s = np.zeros(int(worker_sizes_s[rank]))
    worker_chunk_dg = np.zeros(int(worker_sizes_s[rank]))
    # scatter u and s
    comm.Scatterv([u,count_u,displacements_u,MPI.DOUBLE], worker_chunk_u,root=0)
    comm.Scatterv([s,count_s,displacements_s,MPI.DOUBLE], worker_chunk_s,root=0)

    # wait for everyone
    comm.Barrier()

    # begin iterations for everyone
    for i in range(maxiter):

        # broadcast new x
        x = comm.bcast(x, root=0)
        # scatter u and s
        if update_flag == True:
            comm.Scatterv([u,count_u,displacements_u,MPI.DOUBLE], worker_chunk_u,root=0)
            comm.Scatterv([s,count_s,displacements_s,MPI.DOUBLE], worker_chunk_s,root=0)
            update_flag = False

        for d in range(worker_chunk_u.shape[0]):
            # estimate smoothed gradient
            # #worker_chunk_dg[d] = (fun(x + worker_chunk_s[d]*worker_chunk_u[d], i+exp_num)\
                                # #- fun(x - worker_chunk_s[d]*worker_chunk_u[d], i+exp_num))\
                                # #/ (2*worker_chunk_s[d])
            worker_chunk_dg[d] = fun(x + worker_chunk_s[d]*worker_chunk_u[d], i+exp_num)\
                                 / worker_chunk_s[d]

        # syncronize
        comm.Barrier()

        # master collects dg
        comm.Gatherv(worker_chunk_dg,[dg,count_s,displacements_s,MPI.DOUBLE], root = 0)

        # master updates df
        # assemble the gradient
        df = np.matmul(dg, u) / N

        # step of gradient descent
        opt.step(x,lr,df)
        new_val = fun(x, i+exp_num)

        # scribe record the progress
        scribe.record_iteration_data(iteration=i+1,reward=-new_val,inf_norm_diff=np.amax(np.abs(np.abs(lr*df))))
        print(f"es-iteration {i+1:3d} | reward = {-fun(x,i):6.5f} | inf-norm-diff = {np.amax(np.abs(lr*df)):4.5e}")

        # test if we have found new best value
        if new_val < best_val:
            best_val = new_val
            # checkpoint
            scribe.checkpoint(x,opt,i+1,best=True)

        # check point for 100 iterations
        if i+1 % 100 == 0:
            scribe.checkpoint(x,opt,i+1)

        # update parameters
        if np.linalg.norm(df) < gtol:
            break_flag = True

        # broadcast the flags
        break_flag = comm.bcast(break_flag,root=0)
        update_flag = comm.bcast(update_flag,root=0)

        # break out of loop if break flag is set to true
        if break_flag:
            break

    scribe.record_metadata(total_iterations=i+1,final_value=-fun(x,i+exp_num))
    # master process returns values
    return x, i+1

def es_worker(comm,L,rank,fun, x0,
               lr=.1,
               N=1000,
               s0=np.sqrt(2),
               maxiter=500,
               gtol=1e-06):
    """Evolution Strategies Parallel implementation for workers

        inputs:
                fun -- function handle
                x0  -- initial guess
                lr  -- learning rate
                N   -- number of Monte-Carlo samples to use
                s0  -- initial smoothness parameter
                maxiter -- maximal number of iterations
                gtol    -- tolerance for the magnitude of the gradient

        outputs:
                x   -- the minimizer
                its -- number of iterations until minimizer was obtained
    """
    # everybody does this
    x, dim = np.copy(x0), len(x0)

    # workers initialize placeholders
    worker_sizes_u = None
    worker_sizes_s = None
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

    # broadcast necessary info to all workers
    worker_sizes_u = comm.bcast(split_sizes_u, root = 0)
    worker_sizes_s = comm.bcast(split_sizes_s, root = 0)
    count_u = comm.bcast(count_u, root = 0)
    count_s = comm.bcast(count_s, root = 0)
    displacements_u = comm.bcast(displacements_u, root = 0)
    displacements_s = comm.bcast(displacements_s, root = 0)
    exp_num = comm.bcast(exp_num,root=0)

    # create worker_chunks
    worker_chunk_u = np.zeros((int(worker_sizes_u[rank]),dim))
    worker_chunk_s = np.zeros(int(worker_sizes_s[rank]))
    worker_chunk_dg = np.zeros(int(worker_sizes_s[rank]))
    # scatter u and s
    comm.Scatterv([u,count_u,displacements_u,MPI.DOUBLE], worker_chunk_u,root=0)
    comm.Scatterv([s,count_s,displacements_s,MPI.DOUBLE], worker_chunk_s,root=0)

    # wait for everyone
    comm.Barrier()

    # begin iterations for everyone
    for i in range(maxiter):

        # broadcast new x
        x = comm.bcast(x, root=0)

        # scatter u and s
        if update_flag == True:
            comm.Scatterv([u,count_u,displacements_u,MPI.DOUBLE], worker_chunk_u,root=0)
            comm.Scatterv([s,count_s,displacements_s,MPI.DOUBLE], worker_chunk_s,root=0)
            update_flag = False

        for d in range(worker_chunk_u.shape[0]):
            # #worker_chunk_dg[d] = (fun(x + worker_chunk_s[d]*worker_chunk_u[d], i+exp_num)\
                                # #- fun(x - worker_chunk_s[d]*worker_chunk_u[d], i+exp_num))\
                                # #/ (2*worker_chunk_s[d])
            worker_chunk_dg[d] = fun(x + worker_chunk_s[d]*worker_chunk_u[d], i+exp_num)\
                                 / worker_chunk_s[d]

        # syncronize
        comm.Barrier()

        # master collects dg
        comm.Gatherv(worker_chunk_dg,[dg,count_s,displacements_s,MPI.DOUBLE], root = 0)

        # broadcast the flags
        break_flag = comm.bcast(break_flag,root=0)
        update_flag = comm.bcast(update_flag,root=0)

        # break out of loop if break flag is set to true
        if break_flag:
            break

def es_parallel(fun,x0,
        scribe=RLScribe('saves','unkonwn','unknown'),
        lr=.1,
        N=2000,
        s0=np.sqrt(2),
        maxiter=500,
        gtol=1e-06):
    """parallel implementation of es

    Inputs:
        fun -- function handle
        x0  -- initial guess

    Outputs:
        xfinal -- minimizer obtained by algo
        itr    -- number of iterations
    """
    comm = MPI.COMM_WORLD
    L = comm.Get_size()
    rank = comm.Get_rank()

    x_es = None
    itr_es = None

    # run the optimization
    if rank == 0:
        x_es, itr_es = es_master(comm,L,rank,fun,x0,
                         scribe=scribe,
                         lr=lr,
                         N=N,
                         s0=s0,
                         maxiter=maxiter,
                         gtol=gtol)
        # report the result
        print('\nes-optimization terminated after {:d} iterations:'.format(itr_es),\
              '  x_min = {}\n  f_min = {}'.format(x_es[:10], fun(x_es,itr_es)), sep='\n')
    else:
        es_worker(comm,L,rank,fun,x0,lr,N,s0,maxiter,gtol)

    x_es = comm.bcast(x_es, root=0)
    itr_es = comm.bcast(itr_es, root=0)
    #MPI.Finalize()
    #print('after Finalize')

    return x_es, itr_es



def es_parallel_train(rank,exp_num,env_name,maxiter,hidden_layers=[8,8],policy_mode='deterministic'):
    """
    train an agent to solve the env_name task using
    es optimization
    """

    # number of layers of the neural network
    net_layers = hidden_layers

    # set up scribe
    root_save = 'data/es'
    env_name = env_name
    arch_type = hs_to_str(net_layers)
    scribe = RLScribe(root_save,env_name,arch_type)
    scribe.exp_num = exp_num

    # generate reward function
    J,d = make_rl_j_fn(env_name, hs=net_layers, policy_mode=policy_mode)

    # setup agent
    agent,env,net = setup_agent_env(env_name, hs=net_layers, policy_mode=policy_mode)

    # initial guess of parameter vector
    w0 = get_net_param(net)

    if rank == 0:
        print('problem dimensionality:', d)
        print('net_layers =', net_layers)
        print('iteration   0: reward = {:6.2f}'.format(J(w0,1)))

    # run es parallel implementation
    w, itr = es_parallel(lambda w,i: -J(w,i), w0, N=1500, scribe=scribe, maxiter=maxiter)


