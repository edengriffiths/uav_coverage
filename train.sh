#!/bin/bash

env_id='uav-v0'
alpha=1
beta=1
gamma=1
delta=1

cd rl-baselines3-zoo || exit
python train.py --algo ppo --env $env_id --gym-packages uav_gym --env-kwargs alpha:$alpha beta:$beta gamma:$gamma delta:$delta -f ./logs/${alpha}_${beta}_${gamma}_${delta} -n 4000000
