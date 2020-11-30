import time
import numpy as np
from mpi4py import MPI
from tools.scribe import RLScribe, FScribe, hs_to_str
from tools.optimizer import AdamUpdater
from tools.util import make_rl_j_fn, setup_agent_env, update_net_param, get_net_param

# set print options
np.set_printoptions(linewidth=100, suppress=True, formatter={'float':'{: 0.4f}'.format})

""" auxiliary functions """ 
def generate_directions(dim, vec=None):
    if vec is None:
        u = np.random.randn(dim,dim)
    else:
        u = np.concatenate((vec.reshape((-1,1)), \
                            np.random.randn(dim,dim-1)), axis=1)
    u /= np.linalg.norm(u, axis=0)
    u = np.linalg.qr(u)[0].T
    return u.copy()

def reset_params(args):
    u = generate_directions(len(args['x0']))
    s = np.float(args['s0'])
    L = np.float(args['L_avg'])
    A = np.float(args['A_grad'])
    B = np.float(args['B_grad'])
    return u, s, L, A, B

def gh_quad_aux(g, s, args):
    # initialize variables
    m_min = np.int(args['m_min'])
    xtol = np.float(args['xtol'])
    # Gauss-Hermite quadrature
    p, w = np.polynomial.hermite.hermgauss(m_min)
    g_val = np.array([g(p_i*s) for p_i in p])
    fun_eval = m_min - 1
    dg = np.sum(w*p * g_val) / (s*np.sqrt(np.pi)/2)
    # compute local Lipschitz constant
    L = max(xtol, np.amax(np.abs(g_val[1:] - g_val[:-1]) / (p[1:] - p[:-1]) / s))
    return dg, L, fun_eval

def gh_quad_main(g, s, args):
    # initialize variables
    m_min = np.int(args['m_min'])
    m_max = np.int(args['m_max'])
    qtol = np.float(args['qtol'])
    xtol = np.float(args['xtol'])
    dg_quad = np.array([np.inf])
    g_vals, p_vals = np.array([]), np.array([])
    fun_eval = 0
    # estimate smoothed derivative
    for m in range(max(3, m_min-2), m_max+1, 2):
        # Gauss-Hermite quadrature
        p, w = np.polynomial.hermite.hermgauss(m)
        g_val = np.array([g(p_i*s) for p_i in p])
        fun_eval += m - 1
        # append sampled values
        g_vals = np.append(g_vals, g_val)
        p_vals = np.append(p_vals, p)
        dg_quad = np.append(dg_quad, np.sum(w*p * g_val) / (s*np.sqrt(np.pi)/2))
        # compute relative difference for gradient estimate
        quad_delta = np.amin(np.abs(dg_quad[:-1] - dg_quad[-1]) \
                    / max(np.abs(dg_quad[-1]), xtol))
        if quad_delta < qtol:
            break
    # keep the unique values
    p, p_ind = np.unique(p_vals, return_index=True)
    g_val = g_vals[p_ind]
    # compute local Lipschitz constant
    L = max(xtol, np.amax(np.abs(g_val[1:] - g_val[:-1]) / (p[1:] - p[:-1]) / s))
    return dg_quad[-1], L, fun_eval



"""
    adaptive directional gaussian smoothing v2
    this version is aimed to remove hyperparameter selection
    however it does not provide 100% convergence rate, especially in low dimensions
"""
def asgf(fun, x0, s0, s_rate=.9, m_min=5, m_max=21, qtol=.1,\
            A_grad=.1, B_grad=.9, A_dec=.95, A_inc=1.02, B_dec=.98, B_inc=1.01,\
            L_avg=1, L_lmb=.9, s_min=1e-03, s_max=None, lr_min=1e-03, lr_max=1e+03,\
            restart=True, num_res=2, res_mult=10, res_div=10, fun_req=-np.inf,\
            maxiter=5000, xtol=1e-06, verbose=0):
    """Serial implemenation of the ASGF algorithm

    Args:
        fun -- function to be optimized which is callable and returns float values.
        x0  -- Starting value used in the algorithm. Should a vector stored as a numpy array.
        s0  -- initial starting paramter. 

    Returns:
        Minimizer obtained by algorithm at terminiation.
        Number of iterations the algorithm ran to obtain the returned minimum.
        Number of function evaluations of fun used by the algorithm.
    """
    # save the input arguments
    asgf_args = locals()
    # initialize variables
    x, dim = np.copy(x0), len(x0)
    u = generate_directions(dim)
    s = np.float(s0)
    s_max = np.float(1000*s0) if s_max is None else s_max
    s_status = '='
    # save the initial state
    x_min = np.copy(x)
    f_min = fun_x = fun(x)
    fun_eval = 1
    s_res = s0

    # iteratively construct minimizer
    for itr in range(maxiter):
        # initialize gradient and Lipschitz constants
        dg, L_loc = np.zeros(dim), np.zeros(dim)
        # estimate derivative in every direction
        for d in range(dim):
            # define directional function
            g = lambda t : fun(x + t*u[d])
            if d == 0:
                # main direction
                dg[0], L_loc[0], fun_eval_d = gh_quad_main(g, s, asgf_args)
            else:
                # auxiliary directions
                dg[d], L_loc[d], fun_eval_d = gh_quad_aux(g, s, asgf_args)
            # update number of function evaluations
            fun_eval += fun_eval_d

        # assemble the gradient
        df = np.matmul(dg, u)
        # average Lipschitz constant along the main direction
        L_avg = (1 - L_lmb) * L_loc[0] + L_lmb * L_avg
        # select learning rate
        lr = np.clip(s/L_avg, lr_min, lr_max)
        # report parameters
        if verbose > 2:
            print('iteration {:d}:  f(x) = {:.2e} / {:.2e} = f_min,  lr = {:.2e}, '\
                ' s = {:.2e} ({:s})  dx = {:.2e}'.format(itr+1, fun_x, f_min, lr, \
                s, s_status, np.amax(np.abs(lr * df))))
        if verbose > 3:
            print('  x = {}\n df = {}'.format(x[:10], df[:10]))
        # perform step of gradient descent
        x -= lr * df
        fun_x = fun(x)
        fun_eval += 1

        # save the best state so far
        if fun_x < f_min:
            x_min = np.copy(x)
            f_min = fun_x
            s_res = s
            # check if reauired value was attained
            if f_min < fun_req:
                if verbose > 1:
                    print('iteration {:d}: the required value is attained: '\
                        'fun(x) = {:.2e} < {:.2e} = fun_req'.format(itr+1, f_min, fun_req))
                # disable restarts and increase decay rate
                restart = False
                s_rate *= s_rate
                fun_req = -np.inf

        # check termination condition
        dx_norm = np.linalg.norm(lr*df)
        if dx_norm < xtol:
            break
        # check convergence to a local minimum
        elif restart*(num_res > -1) and s < s_min*res_mult:
            # reset parameters
            u, s, L_avg, A_grad, B_grad = reset_params(asgf_args)
            if num_res > 0:
                if verbose > 1:
                    print('iteration {:d}: reset the parameters'.format(itr+1))
            else:
                # restart from the best state
                x = np.copy(x_min)
                fun_x = f_min
                s = s_res
                if verbose > 1:
                    print('iteration {:d}: restarting from the state '\
                        'fun(x) = {:.2e}, s = {:.2e}'.format(itr+1, fun_x, s))
            num_res -= 1
        # check divergence
        elif restart and s > s_max/res_div:
            # reset parameters
            u, _, L_avg, A_grad, B_grad = reset_params(asgf_args)
            # restart from the best state
            if verbose > 1:
                print('iteration {:d}: restarting from the state fun(x) = {:.2e}, '\
                    's = {:.2e}\nsince the current value of s = {:.2e} is too large'\
                    .format(itr+1, f_min, s_res, s))
            x = np.copy(x_min)
            fun_x = f_min
            s = s_res
        # update parameters for next iteration
        else:
            # update directions
            u = generate_directions(dim, df)
            # adjust smoothing parameter
            s_norm = np.amax(np.abs(dg) / L_loc)
            if s_norm < A_grad:
                s *= s_rate
                A_grad *= A_dec
                s_status = '-'
            elif s_norm > B_grad:
                s /= s_rate
                B_grad *= B_inc
                s_status = '+'
            else:
                A_grad *= A_inc
                B_grad *= B_dec
                s_status = '='
            s = np.clip(s, s_min, s_max)

    # report the result of optimization
    if verbose > 0:
        print('asgf-optimization terminated after {:d} iterations and {:d} '\
            'function evaluations: f_min = {:.2e}'.format(itr+1, fun_eval, f_min))

    return x_min, itr+1, fun_eval


def asgf_get_split_sizes(data,size):
    """gets correct split sizes for size number of workers to split data

    In this case the master receives the first row and everyone else
    just gets split up however.

    Args:
        data: an 2-d numpy array to be split row-wise
        size: number of pieces to split the data into.

    Returns:
        Array whose ith entry represents the number of rows the ith work will recieve.
    """
    # get master row
    split = [data[0:1,:]]

    # split size-1 rows
    split = np.array_split(data[1:],size-1,axis=0)
    split_sizes = [1]
    for i in range(size-1):
        split_sizes = np.append(split_sizes, len(split[i]))

    return split_sizes

def asgf_master(comm,L,rank,fun, x0, s0, scribe = RLScribe('data_adgs','unknown','unknown'),
                s_rate=.9, m_min=5, m_max=21, qtol=.1,\
                A_grad=.1, B_grad=.9, A_dec=.95, A_inc=1.02, B_dec=.98, B_inc=1.01,\
                L_avg=1, L_lmb=.9, s_min=1e-03, s_max=None, lr_min=1e-03, lr_max=1e+03,\
                restart=True, num_res=2, res_mult=10, res_div=10, fun_req=-np.inf,\
                maxiter=1000, xtol=1e-06, verbose=0, optimizer='adam'):
    """Master process in a parallel implementation of ASGF.

    Inputs:
        comm -- MPI comm world
        L    -- number of workers a.k.a. comm.Get_size
        rank -- rank of worker
        fun  -- function to be minimized
        x0   -- initialize value of minimizer
        s0   -- initial value of s

    Returns:
        Minimizer obtain by the algorithm.
        Number of iterations the algorithm ran to obtain the returned minimum.
        Number of function evaluations of fun used by the algorithm.
    """

    # Everyone: save the input arguments
    asgf_args = locals()

    # Everyone: initialize variables
    x, dim = np.copy(x0), len(x0)
    break_flag = False

    # Master: setup optimizer
    if optimizer == 'adam':
        opt = AdamUpdater()

    # Master: initialize important variables
    u = np.eye(dim)
    s = np.float(s0)

    # Master: set up variables for adaptivitiy
    s_max = np.float(1000*s0) if s_max is None else s_max
    s_status = '='

    # Master: save the initial state
    x_min = np.copy(x)

    # initialize gradient and Lipschitz constants
    dg, L_loc = np.zeros(dim), np.zeros(dim)
    f_min = fun_x = fun(x,0)
    s_res = s0

    # Master: initialize master function eval counter
    fun_eval = 1

    # Master: get exp_num from scribe
    exp_num = 1000*scribe.exp_num

    # Master: set up communication information
    split_sizes = asgf_get_split_sizes(u,L)
    count_mat = split_sizes*dim
    count_vec = split_sizes
    displacements_mat = np.insert(np.cumsum(count_mat),0,0)[0:-1]
    displacements_vec = np.insert(np.cumsum(count_vec),0,0)[0:-1]

    # bcast necessary communication info
    worker_rows = comm.bcast(split_sizes, root = 0)
    count_mat = comm.bcast(count_mat, root = 0)
    count_vec = comm.bcast(count_vec, root = 0)
    displacements_mat = comm.bcast(displacements_mat, root = 0)
    displacements_vec = comm.bcast(displacements_vec, root = 0)
    exp_num = comm.bcast(exp_num,root=0)

    # initialize worker chunks
    worker_chunk_u = np.zeros((int(worker_rows[rank]),dim))
    worker_chunk_dg = np.zeros(int(worker_rows[rank]))
    worker_chunk_L_loc = np.zeros(int(worker_rows[rank]))

    # synchronize before iterations begin
    comm.Barrier()

    # iteratively construct minimizer
    for itr in range(maxiter):
        # broadcast x and s
        x = comm.bcast(x,root=0)
        s = comm.bcast(s,root=0)

        comm.Barrier()

        # scatter u
        comm.Scatterv([u,count_mat,displacements_mat,MPI.DOUBLE],worker_chunk_u,root=0)

        # reset worker function eval counter
        worker_fun_eval = np.array(0, dtype='i')
        master_collect = np.array(0, dtype='i')

        # estimate derivative in every direction
        for d in range(worker_chunk_u.shape[0]):
            # define directional function
            g = lambda t : fun(x + t*worker_chunk_u[d],itr+exp_num)
            # Master takes care of main direction
            worker_chunk_dg[d], worker_chunk_L_loc[d], worker_fun_eval_d = gh_quad_aux(g, s, asgf_args)
            # update number of function evaluations
            worker_fun_eval += int(worker_fun_eval_d)

        # Gather the worker chunks of dg and L_loc
        comm.Gatherv(worker_chunk_dg,[dg,count_vec,displacements_vec,MPI.DOUBLE],root=0)
        comm.Gatherv(worker_chunk_L_loc,[L_loc,count_vec,displacements_vec,MPI.DOUBLE],root=0)

        # print dg
        #print(f"master on {itr} has dg:")
        #print(dg)

        # Reduce worker_fun_eval and update total counts so far
        comm.Reduce([worker_fun_eval,1,MPI.INT],[master_collect,1,MPI.INT],MPI.SUM,root=0)
        comm.Barrier()
        fun_eval += master_collect

        # Master: assemble the gradient
        df = np.matmul(dg, u)

        # average Lipschitz constant along the main direction
        #L_avg = L_loc[0] if itr == 0 else (1 - L_lmb) * L_loc[0] + L_lmb * L_avg
        L_avg = (1 - L_lmb) * L_loc[0] + L_lmb * L_avg

        # select learning rate
        #lr = np.clip(s/10, lr_min, lr_max)
        lr = np.clip(s/L_avg, lr_min, lr_max)

        # perform step
        if optimizer == 'grad':
            x -= lr * df
        elif optimizer == 'adam':
            opt.step(x,lr,df)

        # evaluate function
        fun_x = fun(x,itr+exp_num)
        fun_eval += 1

        # scribe records data
        scribe.record_iteration_data(iteration=itr+1,reward=-fun_x,inf_norm_diff=np.amax(np.abs(np.abs(lr*df))))

        # report parameters
        if verbose > 2:
            print('iteration {:d}:  f(x) = {:7.2f} / {:7.2f} = f_min,  lr = {:.2e}, '\
                ' s = {:.2e} ({:s})  dx = {:.2e}'.format(itr+1, -fun_x, -f_min, lr, \
                s, s_status, np.amax(np.abs(lr * df))))
        if verbose > 3:
            print('  x = {}\n df = {}'.format(x[:10], df[:10]))

        # Master: save the best state so far
        if fun_x < f_min:
            x_min = np.copy(x)
            f_min = fun_x
            s_res = s

            # scribe records state of network
            # TODO: change 'SDG' to opt if using Adam
            scribe.checkpoint(x,'SDG',itr+1,best=True)

            # check if reauired value was attained
            if f_min < fun_req:
                if verbose > 1:
                    print('iteration {:d}: the required value is attained: '\
                        'fun(x) = {:7.2f} < {:7.2f} = fun_req'.format(itr+1, -f_min, -fun_req))
                # disable restarts and increase decay rate
                restart = False
                s_rate *= s_rate
                fun_req = -np.inf

        # check convergence to a local minimum
        if restart*(num_res > -1) and s < s_min*res_mult:
            # reset parameters
            u, s, L_avg, A_grad, B_grad = reset_params(asgf_args)
            if num_res > 0:
                if verbose > 1:
                    print('iteration {:d}: reset the parameters'.format(itr+1))
            else:
                # restart from the best state
                x = np.copy(x_min)
                fun_x = f_min
                s = s_res
                if verbose > 1:
                    print('iteration {:d}: restarting from the state '\
                        'fun(x) = {:7.2f}, s = {:.2e}'.format(itr+1, -fun_x, s))
            num_res -= 1

        # check divergence
        elif restart and s > s_max/res_div:
            # reset parameters
            u, _, L_avg, A_grad, B_grad = reset_params(asgf_args)
            # restart from the best state
            if verbose > 1:
                print('iteration {:d}: restarting from the state fun(x) = {:7.2f}, '\
                    's = {:.2e}\nsince the current value of s = {:.2e} is too large'\
                    .format(itr+1, -f_min, s_res, s))
            x = np.copy(x_min)
            fun_x = f_min
            s = s_res

        # update parameters for next iteration
        # Master: updates u and s
        else:
            # adjust smoothing parameter
            s_norm = np.amax(np.abs(dg) / L_loc)
            if s_norm < A_grad:
                s *= s_rate
                A_grad *= A_dec
                s_status = '-'
            elif s_norm > B_grad:
                s /= s_rate
                B_grad *= B_inc
                s_status = '+'
            else:
                A_grad *= A_inc
                B_grad *= B_dec
                s_status = '='
            s = np.clip(s, s_min, s_max)

        comm.Barrier()

        # Everyone: break if break_flag is set
        break_flag = comm.bcast(break_flag,root = 0)
        if break_flag:
            break

    # scribe records metadata
    scribe.record_metadata(total_iterations=itr+1,total_fun_evals=fun_eval,minimum_val=f_min)

    # report the result of optimization
    if verbose > 0:
        print('asgf-optimization terminated after {:d} iterations and {:d} '\
            'function evaluations: f_min = {:7.2f}'.format(itr+1, fun_eval, -f_min))

    return x_min, itr+1, fun_eval

def asgf_worker(comm,L,rank,fun, x0, s0, scribe = RLScribe('data_adgs','unknown','unknown'),
                s_rate=.9, m_min=5, m_max=21, qtol=.1,\
                A_grad=.1, B_grad=.9, A_dec=.95, A_inc=1.02, B_dec=.98, B_inc=1.01,\
                L_avg=1, L_lmb=.9, s_min=1e-03, s_max=None, lr_min=1e-03, lr_max=1e+03,\
                restart=True, num_res=2, res_mult=10, res_div=10, fun_req=-np.inf,\
                maxiter=1000, xtol=1e-06, verbose=0,optimizer='grad'):
    """Worker process version of asgf in the parallel implementation.

    Inputs:
        comm -- MPI comm world
        L    -- number of workers a.k.a. comm.Get_size
        rank -- rank of worker
        fun  -- function to be minimized
        x0   -- initialize value of minimizer
        s0   -- initial value of s
    """

    # Everyone: save the input arguments
    asgf_args = locals()

    # Everyone: initialize variables
    x, dim = np.copy(x0), len(x0)
    break_flag = False

    # Worker: placeholders for important variables
    u = None
    s = None
    dg = None
    L_loc = None

    # initialize gradient and Lipschitz constants
    #dg, L_loc = np.zeros(dim), np.zeros(dim)
    #f_min = fun_x = fun(x,0)
    #s_res = s0

    # Master: initialize master function eval counter
    #fun_eval = 1

    # Worker: set up communication information
    split_sizes = None
    count_mat = None
    count_vec = None
    displacements_mat = None
    displacements_vec = None

    # Everyone: bcast necessary communication info
    worker_rows = comm.bcast(split_sizes, root = 0)
    count_mat = comm.bcast(count_mat, root = 0)
    count_vec = comm.bcast(count_vec, root = 0)
    displacements_mat = comm.bcast(displacements_mat, root = 0)
    displacements_vec = comm.bcast(displacements_vec, root = 0)
    exp_num = None
    exp_num = comm.bcast(exp_num,root=0)

    # Everyone: initialize worker chunks
    worker_chunk_u = np.zeros((int(worker_rows[rank]),dim))
    worker_chunk_dg = np.zeros(int(worker_rows[rank]))
    worker_chunk_L_loc = np.zeros(int(worker_rows[rank]))

    # synchronize before iterations begin
    comm.Barrier()

    # iteratively construct minimizer
    for itr in range(maxiter):
        # broadcast x and s
        x = comm.bcast(x,root=0)
        s = comm.bcast(s,root=0)

        comm.Barrier()

        # scatter u
        comm.Scatterv([u,count_mat,displacements_mat,MPI.DOUBLE],worker_chunk_u,root=0)

        # reset worker function eval counter
        worker_fun_eval = np.array(0, dtype='i')
        master_collect = np.array(0,dtype='i')

        # estimate derivative in every direction
        for d in range(worker_chunk_u.shape[0]):
            # define directional function
            g = lambda t : fun(x + t*worker_chunk_u[d],itr+exp_num)
            # Master takes care of main direction
            worker_chunk_dg[d], worker_chunk_L_loc[d], worker_fun_eval_d = gh_quad_aux(g, s, asgf_args)
            # update number of function evaluations
            worker_fun_eval += int(worker_fun_eval_d)

        # Gather the worker chunks of dg and L_loc
        comm.Gatherv(worker_chunk_dg,[dg,count_vec,displacements_vec,MPI.DOUBLE],root=0)
        comm.Gatherv(worker_chunk_L_loc,[L_loc,count_vec,displacements_vec,MPI.DOUBLE],root=0)

        # Reduce worker_fun_eval and update total counts so far
        comm.Reduce([worker_fun_eval,1,MPI.INT],[master_collect,1,MPI.INT],MPI.SUM,root=0)
        comm.Barrier()
        #fun_eval += master_collect

        comm.Barrier()

        # Everyone: break if break_flag is set
        break_flag = comm.bcast(break_flag,root = 0)
        if break_flag:
            break

def asgf_parallel(fun, x0, s0, scribe = RLScribe('data_adgs','unknown','unknown'),
                  s_rate=.9, m_min=3, m_max=21, qtol=.1,
                  A_grad=.1, B_grad=.9, A_dec=.95, A_inc=1.02, B_dec=.98, B_inc=1.01,
                  L_avg=1, L_lmb=.9, s_min=1e-03, s_max=5, lr_min=1e-03, lr_max=1e+03,
                  restart=False, num_res=2, res_mult=10, res_div=10, fun_req=-np.inf,
                  maxiter=1000, xtol=1e-06, verbose=3, optimizer='grad'):
    """Function for calling the various processes in the parallel implemnetation of ASGF.

    This function is called by all processes launched by the main process. 
    It gives each worker either master or worker version of asgf algorithm.

    Args:
        fun  -- function to be minimized.
        x0   -- initialize value of minimizer.
        s0   -- initial value of s.
        scribe -- Kind of scribe to use to record algorithm info.
    """
    comm = MPI.COMM_WORLD
    L = comm.Get_size()
    rank = comm.Get_rank()

    x = None
    itr = None
    fun_eval = None

    if rank == 0:
        x,itr,fun_eval = asgf_master(comm,L,rank,fun, x0, s0, scribe,
                                     s_rate, m_min, m_max, qtol,
                                     A_grad, B_grad, A_dec, A_inc, B_dec, B_inc,
                                     L_avg, L_lmb, s_min, s_max, lr_min, lr_max,
                                     restart, num_res, res_mult, res_div, fun_req,
                                     maxiter, xtol, verbose,optimizer)
    else:
        asgf_worker(comm,L,rank,fun, x0, s0, scribe,
                    s_rate, m_min, m_max, qtol,
                    A_grad, B_grad, A_dec, A_inc, B_dec, B_inc,
                    L_avg, L_lmb, s_min, s_max, lr_min, lr_max,
                    restart, num_res, res_mult, res_div, fun_req,
                    maxiter, xtol, verbose,optimizer)

    return x,itr,fun_eval

def asgf_parallel_train(rank,exp_num,env_name,maxiter,hidden_layers=[8,8],policy_mode='deterministic'):
    """Sets up scribe, agent, and environment and then runs a reinforcement learning problem.
    """

    # number of layers of the neural network
    net_layers = hidden_layers

    # set up scribe
    root_save = 'data/asgf'
    env_name = env_name
    arch_type = hs_to_str(net_layers)
    scribe = RLScribe(root_save, env_name, arch_type, alg_name='asgf')
    scribe.exp_num = exp_num

    # generate reward function
    J,d = make_rl_j_fn(env_name, hs=net_layers,policy_mode=policy_mode)

    # setup agent
    agent,env,net = setup_agent_env(env_name,hs=net_layers,policy_mode=policy_mode)

    # initial guess of parameter vector
    w0 = get_net_param(net)

    if rank == 0:
        print('problem dimensionality:', d)
        print('iteration   0: reward = {:6.2f}'.format(J(w0,1)))

    asgf_args = dict(s0=np.sqrt(2), s_rate=1., m_min=5, m_max=21,\
                     L_avg=10, L_lmb=1, A_grad=np.inf, B_grad=np.inf,\
                     s_min=.01, s_max=100, lr_min=.001, lr_max=10, restart=False,\
                     maxiter=maxiter, xtol=1e-06, verbose=3, optimizer='adam')
    _, itr, fun_val = asgf_parallel(lambda w,i: -J(w,i), w0, scribe=scribe, **asgf_args)

    print(f"Finished training in {itr} iterations with final fun_val as {fun_val}")

