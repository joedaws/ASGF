from mpi4py import MPI
from scipy.stats import ortho_group
import numpy as np
from tools.scribe import RLScribe, FScribe, hs_to_str
from tools.optimizer import AdamUpdater
from tools.util import make_rl_j_fn, setup_agent_env, update_net_param, get_net_param
# set print options
np.set_printoptions(linewidth=100, suppress=True, formatter={'float':'{: 0.4f}'.format})
# RNG seed for numpy
np.random.seed(0)

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
                gtol -- tolerance for the magnitude of the gradient

        outputs:
                x -- the minimizer
'''
# Directional Gaussian Smoothing
def dgs(fun, x0, lr=.1, M=7, r=1.5, alpha=2.0, beta=.3, gamma=.01, maxiter=500, gtol=1e-06):

    # initialize variables
        x, dim = np.copy(x0), len(x0)
        u = np.eye(dim)
        s = r * np.ones(dim)

        for i in range(maxiter):
            # initialize gradient
                dg = np.zeros(dim)

                # estimate gradient along each direction
                for d in range(dim):
                    # define directional function
                        g = lambda t : fun(x + t*u[d])
                        # estimate smoothed gradient
                        p, w = np.polynomial.hermite.hermgauss(M)
                        g_val = np.array([g(p_i*s[d]) for p_i in p])
                        dg[d] = np.sum(w*p * g_val) / (s[d] * np.sqrt(np.pi)/2)

                # assemble the gradient
                df = np.matmul(dg, u)
                # report the current state
                # #print('dgs-iteration {:d}:\n  x = {}\n df = {}\n  s = {}'.format(i, x, df, s))
                # step of gradient descent
                x -= lr * df
                print('iteration {:3d}: reward = {:6.2f}'.format(i+1, -fun(x)))

                # update parameters
                if np.linalg.norm(df) < gtol:
                    break
                elif np.linalg.norm(df) < gamma:
                    #print('updating u and s')
                    Du = np.random.random((dim,dim))
                    u = np.eye(dim) + alpha * (Du - Du.T)
                    s = (r + beta * (2*np.random.random(dim) - 1))

        return x, i+1

def get_split_sizes(data,size):
    """gets correct split sizes for size number of workers to split data"""
    split = np.array_split(data,size,axis=0)
    split_sizes = []
    for i in range(len(split)):
        split_sizes = np.append(split_sizes, len(split[i]))

    return split_sizes

def dgs_master(comm,L,rank,fun, x0, 
        scribe=RLScribe('data_dgs','unknown','unkonwn'),
        lr=.1, 
        M=7, 
        r=np.sqrt(2), 
        alpha=2.0, 
        beta=np.sqrt(2)/5, 
        gamma=.01, 
        maxiter=500, 
        gtol=1e-06):
    """Directional Gaussian Smoothing Parallel implementation for master 
        the following is the implementation of the directional gaussian smoothing
        optimization algorithm (https://arxiv.org/abs/2002.03001)
        the default values of hyperparameters are taken from the paper
        on reinforcement learning (https://arxiv.org/abs/2002.09077)

        inputs:
                fun -- function handle
                x0  -- initial guess
                scribe -- instance of scribe class for recording
                lr  -- learning rate
                M   -- number of quadrature points to use
                r, alpha, beta, gamma -- other hyperparameters
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
    u = np.eye(dim)
    s = r * np.ones(dim)
    
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
    dg = np.zeros(dim)
   
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
        x = comm.bcast(x,root=0)
        
        #print(f"master before update on it {i} has first {x[0]} middle {x[int(len(x)/2)]} last {x[-1]}")

        # scatter u and s
        if update_flag == True:
            comm.Scatterv([u,count_u,displacements_u,MPI.DOUBLE], worker_chunk_u,root=0)
            comm.Scatterv([s,count_s,displacements_s,MPI.DOUBLE], worker_chunk_s,root=0)
            update_flag = False

        #for d in range(int(displacements_s[rank]),int(displacements_s[rank]+count_s[rank])):
        for d in range(worker_chunk_u.shape[0]):
            # define directional function
            g = lambda t : fun(x + t*worker_chunk_u[d],i+exp_num)
            # estimate smoothed gradient
            p, w = np.polynomial.hermite.hermgauss(M)
            g_val = np.array([g(p_i*worker_chunk_s[d]) for p_i in p])
            worker_chunk_dg[d] = np.sum(w * p * g_val)/(worker_chunk_s[d] * np.sqrt(np.pi)/2)

        # syncronize
        comm.Barrier()

        # master collects dg
        comm.Gatherv(worker_chunk_dg,[dg,count_s,displacements_s,MPI.DOUBLE], root = 0)

        # master updates df 
        # assemble the gradient
        df = np.matmul(dg, u)
        
        # step of gradient descent
        #x -= lr * df
        opt.step(x,lr,df)

        new_val = fun(x,i+exp_num)

        # scribe record the progress
        scribe.record_iteration_data(iteration=i+1,reward=-new_val,inf_norm_diff=np.amax(np.abs(np.abs(lr*df))))
        print(f"v4 iteration {i+1:3d} | reward = {-fun(x,i):6.5f} | inf-norm-diff = {np.amax(np.abs(lr*df)):4.5e}")

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
        elif np.linalg.norm(df) < gamma:
            #print(f"master updating on it {i}")
            Du = np.random.random((dim,dim))
            u = np.eye(dim) + alpha * (Du - Du.T)
            s = (r + beta * (2*np.random.random(dim) - 1))
            #u = ortho_group.rvs(dim)
            update_flag = True

        # broadcast the flags
        break_flag = comm.bcast(break_flag,root=0)
        update_flag = comm.bcast(update_flag,root=0)

        # break out of loop if break flag is set to true
        if break_flag:
            break

    scribe.record_metadata(total_iterations=i+1,final_value=-fun(x,i+exp_num))
    # master process returns values
    return x, i+1

def dgs_worker(comm,L,rank,fun, x0, 
               lr=.1, 
               M=7, 
               r=1.5, 
               alpha=2.0, 
               beta=.3, 
               gamma=.01, 
               maxiter=500, 
               gtol=1e-06):
    """Directional Gaussian Smoothing Parallel implementation for workers
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
        x = comm.bcast(x,root=0)
        #print(f"{rank} on it {i} has first {x[0]} middle {x[int(len(x)/2)]} last {x[-1]}")
        
        # scatter u and s
        if update_flag == True:
            #print(f"worker {rank} on it {i} is updating")
            comm.Scatterv([u,count_u,displacements_u,MPI.DOUBLE], worker_chunk_u,root=0)
            comm.Scatterv([s,count_s,displacements_s,MPI.DOUBLE], worker_chunk_s,root=0)
            update_flag = False

        #for d in range(int(displacements_s[rank]),int(displacements_s[rank]+count_s[rank])):
        for d in range(worker_chunk_u.shape[0]):
            # define directional function
            g = lambda t : fun(x + t*worker_chunk_u[d],i+exp_num)
            # estimate smoothed gradient
            p, w = np.polynomial.hermite.hermgauss(M)
            g_val = np.array([g(p_i*worker_chunk_s[d]) for p_i in p])
            worker_chunk_dg[d] = np.sum(w * p * g_val)/(worker_chunk_s[d] * np.sqrt(np.pi)/2)

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

def dgs_parallel(fun,x0,
        scribe=RLScribe('saves','unkonwn','unknown'),
        lr=.1,
        M=7,
        r=np.sqrt(2),
        alpha=2.,
        beta=.2*np.sqrt(2),
        gamma=.001,
        maxiter=500,
        gtol=1e-06):
    """parallel implementation of dgs

    r --     1.0 or sqrt(2)?
    alpha -- 2.0 or sqrt(2)/2?

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

    x_dgs = None
    itr_dgs = None

    # run the optimization
    if rank == 0:
        x_dgs, itr_dgs = dgs_master(comm,L,rank,fun,x0,
                         scribe=scribe,
                         lr=lr,
                         M=M,
                         r=r,
                         alpha=alpha,
                         beta=beta,
                         gamma=gamma,
                         maxiter=maxiter,
                         gtol=gtol)
        # report the result
        print('\ndgs-optimization terminated after {:d} iterations:'.format(itr_dgs),\
              '  x_min = {}\n  f_min = {}'.format(x_dgs[:10], fun(x_dgs,itr_dgs)), sep='\n')
    else:
        dgs_worker(comm,L,rank,fun,x0,lr,M,r,alpha,beta,gamma,maxiter,gtol)
    
    x_dgs = comm.bcast(x_dgs,root=0)
    itr_dgs = comm.bcast(itr_dgs,root=0)
    #MPI.Finalize()
    #print('after Finalize')

    return x_dgs, itr_dgs



def dgs_parallel_train(rank,exp_num,env_name,maxiter,hidden_layers=[8.8]):
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
    agent,env,net = setup_agent_env(env_name,hs=net_layers)
    
    # initial guess of parameter vector
    #np.random.seed(0)
    #w0 = np.random.randn(d)/10
    w0 = get_net_param(net)

    if rank == 0:
        print('problem dimensionality:', d)
        print('iteration   0: reward = {:6.2f}'.format(J(w0,1)))

    # run dgs parallel implementation
    w, itr = dgs_parallel(lambda w,i: -J(w,i), w0, scribe=scribe,maxiter=maxiter)


