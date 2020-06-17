import time
from types import SimpleNamespace
import numpy as np
from mpi4py import MPI
from tools.scribe import RLScribe, FScribe, hs_to_str
from tools.optimizer import AdamUpdater
from tools.util import make_rl_j_fn, setup_agent_env, update_net_param, get_net_param
from algorithms.parameters import init_asgf

# set print options
np.set_printoptions(linewidth=100, suppress=True, formatter={'float':'{: 0.4f}'.format})

''' auxiliary functions '''
def generate_directions(dim, vec=None):
    if vec is None:
        u = np.random.randn(dim,dim)
    else:
        u = np.concatenate((vec.reshape((-1,1)), \
                            np.random.randn(dim,dim-1)), axis=1)
    u /= np.linalg.norm(u, axis=0)
    u = np.linalg.qr(u)[0].T
    return u

def reset_params(initial_args,len_x0):
    u = generate_directions(len_x0)
    s = initial_args.s
    L_avg = initial_args.L_avg
    A = initial_args.A_grad
    B = initial_args.B_grad
    return u, s, L, A, B

def gh_quad_aux(g, s, args):
    # TODO need to adjust args --> SimpleNamespace
    # initialize variables
    m_min = args.m_min
    xtol = args.xtol
    # Gauss-Hermite quadrature
    p, w = np.polynomial.hermite.hermgauss(m_min)
    g_val = np.array([g(p_i*s) for p_i in p])
    fun_eval = m_min - 1
    dg = np.sum(w*p * g_val) / (s*np.sqrt(np.pi)/2)
    # compute local Lipschitz constant
    L = max(xtol, np.amax(np.abs(g_val[1:] - g_val[:-1]) / (p[1:] - p[:-1]) / s))
    return dg, L, fun_eval

def gh_quad_main(g, s, args):
    # TODO need to adjust args --> SimpleNamespace
    # initialize variables
    m_min = args.m_min
    m_max = args.m_max
    qtol = args.qtol
    xtol = args.xtol
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



'''
    adaptive directional gaussian smoothing v2
    this version is aimed to remove hyperparameter selection
    however it does not provide 100% convergence rate, especially in low dimensions
'''
"""
def asgf(fun, x0, s0, s_rate=.9, m_min=5, m_max=21, qtol=.1,\
            A_grad=.1, B_grad=.9, A_dec=.95, A_inc=1.02, B_dec=.98, B_inc=1.01,\
            L_avg=1, L_lmb=.9, s_min=1e-03, s_max=None, lr_min=1e-03, lr_max=1e+03,\
            restart=True, num_res=2, res_mult=10, res_div=10, fun_req=-np.inf,\
            maxiter=5000, xtol=1e-06, verbose=0):
"""
def asgf(fun,x0,s0,param=init_asgf()):
    """
    Inputs:
        fun -- function to be optimized
        x0  -- initial guess of minimizer
        s0  -- initial smoothing parameter
        param -- SimpleNamedspace for all of the required parameters

    Outputs:
        
    TODO:
        need to compartmentalize parameters which will neverage change and those that 
        are updated throughout the algorithm
        
        Those that change and may want to be recorded:
            s
            u
            L_avg
            A_grad
            B_grad
            fun_req
            s_rate
            num_res
            restart

    """
    # unpack some of the parameters
    L_avg = np.copy(param.L_avg)
    A_grad = np.copy(param.A_grad)
    B_grad = np.copy(param.B_grad)
    fun_req = np.copy(param.fun_req)
    s_rate = np.copy(param.s_rate)
    num_res = np.copy(param.num_res)
    restart = np.copy(param.restart)

    # initialize variables
    x, dim = np.copy(x0), len(x0)
    u = generate_directions(dim)
    s = np.float(s0)
    s_max = np.float(1000*s0) if param.s_max is None else param.s_max
    s_status = '='
    # save the initial state
    x_min = np.copy(x)
    f_min = fun_x = fun(x)
    fun_eval = 1
    s_res = s0

    # iteratively construct minimizer
    for itr in range(param.maxiter):
        # initialize gradient and Lipschitz constants
        dg, L_loc = np.zeros(dim), np.zeros(dim)
        # estimate derivative in every direction
        for d in range(dim):
            # define directional function
            g = lambda t : fun(x + t*u[d])
            if d == 0:
                # main direction
                dg[0], L_loc[0], fun_eval_d = gh_quad_main(g, s, param)
            else:
                # auxiliary directions
                dg[d], L_loc[d], fun_eval_d = gh_quad_aux(g, s, param)
            # update number of function evaluations
            fun_eval += fun_eval_d

        # assemble the gradient
        df = np.matmul(dg, u)
        # average Lipschitz constant along the main direction
        L_avg = (1 - param.L_lmb) * L_loc[0] + param.L_lmb * L_avg
        # select learning rate
        lr = np.clip(s/L_avg, param.lr_min, param.lr_max)
        # report parameters
        if param.verbose > 2:
            print('iteration {:d}:  f(x) = {:.2e} / {:.2e} = f_min,  lr = {:.2e}, '\
                    ' s = {:.2e} ({:s})  dx = {:.2e}'.format(itr+1, fun_x, f_min, lr, \
                    s, s_status, np.amax(np.abs(lr * df))))
        if param.verbose > 3:
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
            if f_min < param.fun_req:
                if param.verbose > 1:
                        print('iteration {:d}: the required value is attained: '\
                                'fun(x) = {:.2e} < {:.2e} = fun_req'.format(itr+1, f_min, fun_req))
                # disable restarts and increase decay rate
                restart = False
                s_rate *= s_rate
                fun_req = -np.inf

        # check termination condition
        dx_norm = np.linalg.norm(lr*df)
        if dx_norm < param.xtol:
                break
        # check convergence to a local minimum
        elif restart*(num_res > -1) and s < param.s_min*param.res_mult:
            # reset parameters
            u, s, L_avg, A_grad, B_grad = reset_params(param,dim)
            if num_res > 0:
                if verbose > 1:
                    print('iteration {:d}: reset the parameters'.format(itr+1))
                else:
                    # restart from the best state
                    x = np.copy(x_min)
                    fun_x = f_min
                    s = s_res
                    if verbose > 1:
                        print('iteration {:d}: restarting from the state '
                              'fun(x) = {:.2e}, s = {:.2e}'.format(itr+1, fun_x, s))
                num_res -= 1
        
        # check divergence
        elif restart and s > s_max/param.res_div:
            # reset parameters
            u, _, L_avg, A_grad, B_grad = reset_params(param,dim)
            # restart from the best state
            if param.verbose > 1:
                print('iteration {:d}: restarting from the state fun(x) = {:.2e}, '
                      's = {:.2e}\nsince the current value of s = {:.2e} is too large'
                                .format(itr+1, f_min, s_res, s) )
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
                A_grad *= param.A_dec
                s_status = '-'
            elif s_norm > B_grad:
                s /= s_rate
                B_grad *= param.B_inc
                s_status = '+'
            else:
                A_grad *= param.A_inc
                B_grad *= param.B_dec
                s_status = '='
            s = np.clip(s, param.s_min, s_max)

        # report the result of optimization
        if param.verbose > 0:
            print('asgf-optimization terminated after {:d} iterations and {:d} '\
                  'function evaluations: f_min = {:.2e}'.format(itr+1, fun_eval, f_min))

    return x_min, itr+1, fun_eval

def asgf_get_split_sizes(data,size):
    """gets correct split sizes for size number of workers to split data

    In this case the master receives the first row and everyone else
    just gets split up however.
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
    """
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
        L_avg = (1 - L_lmb) * np.amax(L_loc) + L_lmb * L_avg

        # select learning rate
        #lr = np.clip(10 * s/L_avg, lr_min, lr_max)
        lr = np.clip(10 * s/L_avg, lr_min, lr_max)

        # TODO: do we need ADAM updates or not?
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

        # TODO: may want to add these steps back in
        # check convergence to a local minimum
        if restart*(num_res > -1) and s < s_min*res_mult:
            # reset parameters
            u, s, L_avg, A_grad, B_grad = reset_params(asgf_args,dim)
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
        # TODO make sure this is done
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
    """worker version of asgf
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
                  s_rate=.9, m_min=3, m_max=11, qtol=.1,
                  A_grad=.1, B_grad=.9, A_dec=.95, A_inc=1.02, B_dec=.98, B_inc=1.01,
                  L_avg=1, L_lmb=.9, s_min=1e-03, s_max=5, lr_min=1e-03, lr_max=1,
                  restart=False, num_res=2, res_mult=10, res_div=10, fun_req=-np.inf,
                  maxiter=1000, xtol=1e-06, verbose=3, optimizer='grad'):
    """
    function which gives each worker eitehr master or worker
    version of asgf algorithm
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
    """
    train an agent to solve the env_name task using
    dgs optimization
    """

    # number of layers of the neural network
    net_layers = hidden_layers

    # set up scribe
    root_save = 'data/asgf'
    env_name = env_name
    arch_type = hs_to_str(net_layers)
    scribe = RLScribe(root_save,env_name,arch_type)
    scribe.exp_num = exp_num

    # generate reward function
    J,d = make_rl_j_fn(env_name, hs=net_layers)

    # setup agent
    agent,env,net = setup_agent_env(env_name,hs=net_layers,policy_mode=policy_mode)

    # initial guess of parameter vector
    w0 = get_net_param(net)

    if rank == 0:
        print('problem dimensionality:', d)
        print('iteration   0: reward = {:6.2f}'.format(J(w0,1)))

    # run dgs parallel implementation
    asgf_args = dict(s0=3, s_rate=1.0, m_min=5, m_max=5, L_avg=1000, L_lmb=.9, \
                     s_min=.01, s_max=100, lr_min=.01, lr_max=1, \
                     maxiter=maxiter, xtol=1e-06, verbose=3, optimizer='adam')
    _, itr, fun_val = asgf_parallel(lambda w,i: -J(w,i), w0, scribe=scribe, **asgf_args)

    print(f"Finished training in {itr} iterations with final fun_val as {fun_val}")

