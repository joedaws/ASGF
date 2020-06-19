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
    #u = np.linalg.qr(u)[0].T
    u = np.linalg.qr(u)[0].T.copy() # so that the new matrix is c-ordered
    return u

def reset_params(initial_args,len_x0):
    """
    Inputs:
        initial_args -- SimpleNamespace filled with initial parameter values
        len_x0       -- dimension of the initial guess

    Outputs:
        u -- random collection of orthonormal directions
        s -- initial smoothing value
        L -- initial L_avg value
        A -- initial A_grad
        B -- initial B_grad
    """
    u = generate_directions(len_x0)
    s = initial_args.s0
    L = initial_args.L_avg
    A = initial_args.A_grad
    B = initial_args.B_grad
    return u, s, L, A, B

def gh_quad_aux(g, s, args):
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
    param.s0 = s0

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
        #print(u)
        for d in range(dim):
            # define directional function
            #print(f"u[{d}] is {u[d]}")
            g = lambda t : fun(x + t*u[d])
            if d == 0:
                # main direction
                dg[0], L_loc[0], fun_eval_d = gh_quad_main(g, s, param)
            else:
                # auxiliary directions
                dg[d], L_loc[d], fun_eval_d = gh_quad_aux(g, s, param)
            # update number of function evaluations
            fun_eval += fun_eval_d

        #print('dg')
        #print(dg)
        #print('L_loc')
        #print(L_loc)
        
        #print(f"fun eval this itr {fun_eval}")
        
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
        
        #print(f"serial: itr {itr} \n x {x} \n df {df} \n lr {lr}")
        
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
    """gets correct split sizes for number of processes to split data

    In this case the main receives the first row and everyone else
    just gets split up however.
    """
    # get main row
    split = [data[0:1,:]]

    # split size-1 rows
    split = np.array_split(data[1:],size-1,axis=0)
    split_sizes = [1]
    for i in range(size-1):
        split_sizes = np.append(split_sizes, len(split[i]))

    return split_sizes

def asgf_parallel_main(comm,L,rank,fun,x0,s0,scribe,param=init_asgf()):
    """
    Inputs:
        comm -- MPI comm world
        L    -- number of processes a.k.a. comm.Get_size
        rank -- rank of process 
        fun  -- function to be minimized
        x0   -- initialize value of minimizer
        s0   -- initial value of s
        scribe -- isntance of scribe
        param -- simplename space filled with
    """
    # unpack some of the parameters
    L_avg = np.copy(param.L_avg)
    A_grad = np.copy(param.A_grad)
    B_grad = np.copy(param.B_grad)
    fun_req = np.copy(param.fun_req)
    s_rate = np.copy(param.s_rate)
    num_res = np.copy(param.num_res)
    restart = np.copy(param.restart)
    param.s0 = s0

    # Everyone: initialize variables
    x, dim = np.copy(x0), len(x0)
    break_flag = False

    # main: setup optimizer
    if param.optimizer == 'adam':
        opt = AdamUpdater()

    # main: initialize important variables
    #u = np.eye(dim)
    u = generate_directions(dim)
    #print(f"main has generated u \n")
    #print(u)
    s = np.float(s0)

    # main: set up variables for adaptivitiy
    s_max = np.float(1000*s0) if param.s_max is None else param.s_max
    s_status = '='

    # main: save the initial state
    x_min = np.copy(x)

    # initialize gradient and Lipschitz constants
    dg, L_loc = np.zeros(dim), np.zeros(dim)
    f_min = fun_x = fun(x)
    s_res = s0

    # main: initialize main function eval counter
    fun_eval = 1

    # main: get exp_num from scribe
    exp_num = 1000*scribe.exp_num

    # main: set up communication information
    split_sizes = asgf_get_split_sizes(u,L)
    count_mat = split_sizes*dim
    count_vec = split_sizes
    displacements_mat = np.insert(np.cumsum(count_mat),0,0)[0:-1]
    displacements_vec = np.insert(np.cumsum(count_vec),0,0)[0:-1]

    # bcast necessary communication info
    split_rows = comm.bcast(split_sizes, root = 0)
    count_mat = comm.bcast(count_mat, root = 0)
    count_vec = comm.bcast(count_vec, root = 0)
    displacements_mat = comm.bcast(displacements_mat, root = 0)
    displacements_vec = comm.bcast(displacements_vec, root = 0)
    exp_num = comm.bcast(exp_num,root=0)

    # initialize chunks
    chunk_u = np.zeros((int(split_rows[rank]),dim))
    chunk_dg = np.zeros(int(split_rows[rank]))
    chunk_L_loc = np.zeros(int(split_rows[rank]))

    # synchronize before iterations begin
    comm.Barrier()

    # iteratively construct minimizer
    for itr in range(param.maxiter):
        # broadcast x and s
        x = comm.bcast(x,root=0)
        s = comm.bcast(s,root=0)

        comm.Barrier()
        #print(u)

        # scatter u
        comm.Scatterv([u,count_mat,displacements_mat,MPI.DOUBLE],chunk_u,root=0)

        # reset function eval counter
        aux_fun_eval = np.array(0, dtype='i')
        main_collect = np.array(0, dtype='i')

        # estimate derivative in every direction
        for d in range(chunk_u.shape[0]):
            # define directional function
            #print(f"process {rank} dim {d} chunk_u {chunk_u[d]}")
            g = lambda t : fun(x + t*chunk_u[d])
            # main takes care of main direction
            chunk_dg[d], chunk_L_loc[d], fun_eval_d = gh_quad_main(g, s, param)
            #print(f"aux {rank} and dim {rank+d} dg {chunk_dg[d]}")
            # update number of function evaluations
            aux_fun_eval += int(fun_eval_d)

        # Gather the chunks of dg and L_loc
        comm.Gatherv(chunk_dg,[dg,count_vec,displacements_vec,MPI.DOUBLE],root=0)
        comm.Gatherv(chunk_L_loc,[L_loc,count_vec,displacements_vec,MPI.DOUBLE],root=0)


        # Reduce aux_fun_eval and update total counts so far
        comm.Reduce([aux_fun_eval,1,MPI.INT],[main_collect,1,MPI.INT],MPI.SUM,root=0)
 
        #print('dg')
        #print(dg)
        #print('L_loc')
        #print(L_loc)       
        
        fun_eval += main_collect
        #print(f"fun eval this itr {fun_eval}")

        # assemble the gradient
        df = np.matmul(dg, u)

        # average Lipschitz constant along the main direction
        L_avg = (1 - param.L_lmb) * np.amax(L_loc) + param.L_lmb * L_avg

        # select learning rate
        lr = np.clip(s/L_avg, param.lr_min, param.lr_max)
        #lr = np.clip(10 * s/L_avg, param.lr_min, param.lr_max)

        # perform step
        if param.optimizer == 'grad':
            x -= lr * df
        elif param.optimizer == 'adam':
            opt.step(x,lr,df)
    
        #print(f"parallel: itr {itr} \n x {x} \n df {df} \n lr {lr}")
        
        # evaluate function
        fun_x = fun(x)
        fun_eval += 1

        # scribe records data
        scribe.record_iteration_data(iteration=itr+1,reward=-fun_x,inf_norm_diff=np.amax(np.abs(np.abs(lr*df))))

        # report parameters
        if param.verbose > 2:
            print('iteration {:d}:  f(x) = {:7.2f} / {:7.2f} = f_min,  lr = {:.2e}, '\
                ' s = {:.2e} ({:s})  dx = {:.2e}'.format(itr+1, -fun_x, -f_min, lr, \
                s, s_status, np.amax(np.abs(lr * df))))
        if param.verbose > 3:
            print('  x = {}\n df = {}'.format(x[:10], df[:10]))

        # main: save the best state so far
        if fun_x < f_min:
            x_min = np.copy(x)
            f_min = fun_x
            s_res = s

            # scribe records state of network
            scribe.checkpoint(x,param.optimizer,itr+1,best=True)

            # check if reauired value was attained
            if f_min < fun_req:
                if param.verbose > 1:
                    print('iteration {:d}: the required value is attained: '\
                        'fun(x) = {:7.2f} < {:7.2f} = fun_req'.format(itr+1, -f_min, -fun_req))
                # disable restarts and increase decay rate
                restart = False
                s_rate *= s_rate
                fun_req = -np.inf
        
        # check convergence to a local minimum
        if restart*(num_res > -1) and s < param.s_min*param.res_mult:
            # reset parameters
            u, s, L_avg, A_grad, B_grad = reset_params(param,dim)
            u = np.copy(u.T)
            if num_res > 0:
                if param.verbose > 1:
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
        elif restart and s > s_max/param.res_div:
            # reset parameters
            u, _, L_avg, A_grad, B_grad = reset_params(param,dim)
            u = np.copy(u.T)
            # restart from the best state
            if param.verbose > 1:
                print('iteration {:d}: restarting from the state fun(x) = {:7.2f}, '\
                    's = {:.2e}\nsince the current value of s = {:.2e} is too large'\
                    .format(itr+1, -f_min, s_res, s))
            x = np.copy(x_min)
            fun_x = f_min
            s = s_res

        # update parameters for next iteration
        else:
            # update directions 
            u = generate_directions(dim, df).T

            # adjust smoothing parameter
            s_norm = np.amax(np.abs(dg) / L_loc)
            if s_norm < A_grad:
                s *= param.s_rate
                A_grad *= param.A_dec
                s_status = '-'
            elif s_norm > B_grad:
                s /= param.s_rate
                B_grad *= param.B_inc
                s_status = '+'
            else:
                A_grad *= param.A_inc
                B_grad *= param.B_dec
                s_status = '='
            s = np.clip(s, param.s_min, param.s_max)
        
        # check termination condition
        dx_norm = np.linalg.norm(lr*df)
        if dx_norm < param.xtol:
            break_flag = True

        comm.Barrier()

        # Everyone: break if break_flag is set
        break_flag = comm.bcast(break_flag,root = 0)
        if break_flag:
            break

    # scribe records metadata
    scribe.record_metadata(total_iterations=itr+1,total_fun_evals=fun_eval,minimum_val=f_min)

    # report the result of optimization
    if param.verbose > 0:
        print('asgf-optimization terminated after {:d} iterations and {:d} '\
            'function evaluations: f_min = {:7.2f}'.format(itr+1, fun_eval, -f_min))

    return x_min, itr+1, fun_eval

def asgf_parallel_aux(comm,L,rank,fun,x0,s0,param=init_asgf()):
    """auxiliary version of asgf
    Inputs:
        comm -- MPI comm world
        L    -- number of processes a.k.a. comm.Get_size
        rank -- rank of process 
        fun  -- function to be minimized
        x0   -- initialize value of minimizer
        s0   -- initial value of s
        param -- SimpleNamespace filled with algorithm parameters
    """

    # Everyone: initialize variables
    x, dim = np.copy(x0), len(x0)
    break_flag = False

    # aux: placeholders for important variables
    u = None
    s = None
    dg = None
    L_loc = None

    # aux: set up communication information
    split_sizes = None
    count_mat = None
    count_vec = None
    displacements_mat = None
    displacements_vec = None

    # Everyone: bcast necessary communication info
    split_rows = comm.bcast(split_sizes, root = 0)
    count_mat = comm.bcast(count_mat, root = 0)
    count_vec = comm.bcast(count_vec, root = 0)
    displacements_mat = comm.bcast(displacements_mat, root = 0)
    displacements_vec = comm.bcast(displacements_vec, root = 0)
    exp_num = None
    exp_num = comm.bcast(exp_num,root=0)

    # Everyone: initialize chunks
    chunk_u = np.zeros((int(split_rows[rank]),dim))
    chunk_dg = np.zeros(int(split_rows[rank]))
    chunk_L_loc = np.zeros(int(split_rows[rank]))

    # synchronize before iterations begin
    comm.Barrier()

    # iteratively construct minimizer
    for itr in range(param.maxiter):
        # broadcast x and s
        x = comm.bcast(x,root=0)
        s = comm.bcast(s,root=0)

        comm.Barrier()

        # scatter u
        comm.Scatterv([u,count_mat,displacements_mat,MPI.DOUBLE],chunk_u,root=0)
        #print(f"process {rank} chunk_u {chunk_u}")

        # reset function eval counter
        aux_fun_eval = np.array(0, dtype='i')
        main_collect = np.array(0,dtype='i')

        # estimate derivative in every direction
        for d in range(chunk_u.shape[0]):
            # define directional function
            g = lambda t : fun(x + t*chunk_u[d])
            # aux directions 
            chunk_dg[d], chunk_L_loc[d], aux_fun_eval_d = gh_quad_aux(g, s, param)
            #print(f"aux {rank} and dim {rank+d} dg {chunk_dg[d]}")
            # update number of function evaluations
            aux_fun_eval += int(aux_fun_eval_d)

        # Gather the chunks of dg and L_loc
        comm.Gatherv(chunk_dg,[dg,count_vec,displacements_vec,MPI.DOUBLE],root=0)
        comm.Gatherv(chunk_L_loc,[L_loc,count_vec,displacements_vec,MPI.DOUBLE],root=0)

        # Reduce fun_eval and update total counts so far
        comm.Reduce([aux_fun_eval,1,MPI.INT],[main_collect,1,MPI.INT],MPI.SUM,root=0)

        # second wait for main
        comm.Barrier()

        # Everyone: break if break_flag is set
        break_flag = comm.bcast(break_flag,root = 0)
        if break_flag:
            break

# TODO may get rid of this function
def asgf_parallel(fun, x0, s0, scirbe, param=init_asgf()):
    """
    split up processes into main and aux
    """
    # use mpi_fork here
    comm = MPI.COMM_WORLD
    L = comm.Get_size()
    rank = comm.Get_rank()

    x = None
    itr = None
    fun_eval = None

    if rank == 0:
        x,itr,fun_eval = asgf_parallel_main(comm,L,rank,fun, x0, s0, scribe,param)
    else:
        asgf_parallel_aux(comm,L,rank,fun, x0, s0, param)

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

