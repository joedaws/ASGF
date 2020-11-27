from mpi4py import MPI
import numpy as np
import cma
from tools.scribe import RLScribe, FScribe, hs_to_str
from tools.optimizer import AdamUpdater
from tools.util import make_rl_j_fn, setup_agent_env, update_net_param, get_net_param
# set print options
np.set_printoptions(linewidth=100, suppress=True, formatter={'float':'{: 0.4f}'.format})
# RNG seed for numpy
np.random.seed(0)

def cma_train(rank, exp_num, env_name, maxiter, hidden_layers=[8,8], policy_mode='deterministic'):
    """
    train an agent to solve the env_name task using
    cma-es optimization
    """

    # running cma with 1 worker due to inability to parallelize
    if rank == 0:
        # number of layers of the neural network
        net_layers = hidden_layers

        # set up scribe
        root_save = 'data/cma'
        env_name = env_name
        arch_type = hs_to_str(net_layers)
        scribe = RLScribe(root_save, env_name, arch_type, alg_name='cma')
        scribe.exp_num = exp_num

        # generate reward function
        J,d = make_rl_j_fn(env_name, hs=net_layers, policy_mode=policy_mode)

        # setup agent
        agent,env,net = setup_agent_env(env_name, hs=net_layers, policy_mode=policy_mode)
        w = get_net_param(net)
        print('problem dimensionality:', d)
        print('net_layers =', net_layers)
        print('iteration   0: reward = {:6.2f}'.format(J(w,1)))

        # run cma
        feval = 0
        for itr in range(maxiter):
            cma_result = cma.fmin2(lambda w: -J(w,itr+exp_num), w, np.sqrt(2),
                                   {'maxiter': 1, 'verb_disp': 0})[1].result
            w = cma_result[0]
            feval += cma_result[3]
            scribe.record_iteration_data(iteration=itr+1, reward=-cma_result[1], feval=feval)
            print(f"cma-iteration {itr+1:3d} | reward = {-cma_result[1]:6.5f}")

        scribe.record_metadata(total_iterations=itr+1, final_value=-cma_result[1])


