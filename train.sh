#!/bin/bash

cd rl-baselines3-zoo || exit
python train.py --algo ppo --env $env_id --gym-packages uav_gym --env-kwargs alpha:$alpha beta:$beta gamma:$gamma -f ./logs/${alpha}_${beta}_${gamma} -n 4*10**6
