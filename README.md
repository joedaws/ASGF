# ASGF

## Requirements
An appropriate environment for these pieces of code can be created
using the environment yml file included. To create a conda environment 
use: 
```
conda env create -f environment.yml
```

## Example usage for functional optimization

```
python -m optimize --fun=ackley --dim=10 --algo=asgf --sim=1
```

## Examlpe usage for reinforcement learning

```
mpiexec -n 8 python -m train --env_name=InvertedPendulumBulletEnv-v0 --algo=asgf --hidden_sizes=12,12
```
