#!/bin/bash

cd rl-baselines3-zoo || exit
python train.py --algo ppo --env uav-v0 --gym-packages uav_gym --vec-env subproc
#python train.py --algo ppo --env uav-v0 --gym-packages uav_gym -n 100000 -optimize --n-trials 1000 --n-jobs 2 --sampler tpe --pruner median
#python train.py --algo ppo --env uav-v0 --gym-packages uav_gym -optimize \
#  --study-name test --storage sqlite:///example.db
