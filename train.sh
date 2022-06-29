#!/bin/bash

cd rl-baselines3-zoo || exit
python train.py --algo ppo --env $env_id --gym-packages uav_gym
