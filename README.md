# ASGF

This repository contains the source code reproducing the numerical examples presented in the paper [An adaptive stochastic gradient-free approach for high-dimensional blackbox optimization](https://arxiv.org/abs/2006.10887).

## Requirements
An appropriate environment for these pieces of code can be created
using the environment yml file included. To create a conda environment 
use: 
```
conda env create -f environment.yml
```

Then to activate the constructed environment use:
```
conda activate asgf
```

NOTE: The creation of the asgf conda environment may take up to 5-7 minutes.

## Example usage for functional optimization

```
python -m optimize --fun=ackley --dim=10 --algo=asgf --sim=100
```
* fun -- a string for the name of function to be minimizer. Implemented functions are in tools/function.py
* dim -- input dimension of function
* algo -- name of algorithm to use to minimizer the function
* sim -- number of simulations or trials. Each simulation begins at a different inital point with different random seeding throughout.

## Examlpe usage for reinforcement learning

```
mpiexec -n 8 python -m train --env_name=InvertedPendulumBulletEnv-v0 --algo=asgf --hidden_sizes=8,8
```
* env\_name -- name of gym environment. Must be already registered in gym or in pybullet\_envs
* algo      -- name of algorithm to use to train the agent
* hidden\_sizes -- comma seperated values which represent the number of nodes to use on the hidden layers.
