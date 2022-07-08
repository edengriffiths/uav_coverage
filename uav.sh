#!/bin/bash

env_id=uav-v0
n_uavs=2
exp_name=dems

cd rl-baselines3-zoo || exit
python train.py --algo ppo --env $env_id --gym-packages uav_gym --env-kwargs n_uavs:$n_uavs -f ./logs/${exp_name}/fair_cov_right -tb ./tb

# python test.py
#python train.py --algo ppo --env uav-v0 --gym-packages uav_gym -n 100000 -optimize --n-trials 1000 --n-jobs 2 --sampler tpe --pruner median
#python train.py --algo ppo --env uav-v0 --gym-packages uav_gym -optimize \
#  --study-name test --storage sqlite:///example.db
