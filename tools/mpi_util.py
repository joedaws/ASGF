from mpi4py import MPI
import os, subprocess, sys

def mpi_fork(n):
    """
    Inputs: 
        n -- number of workers
    
    Outputs:
        
    Re-launches the current script with multiple processes
    Returns "parent" for original parent, "child" for MPI children
    """
    if n<=1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        cmd = ["mpiexec", "-n", str(n),   sys.executable] +['-u']+ sys.argv
        print(cmd)
        subprocess.check_call(cmd, env=env)
        return "parent"
    else:
        return "child"


