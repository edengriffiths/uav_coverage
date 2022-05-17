#!/bin/bash

cd rl-baselines3-zoo || exit
python train.py --algo ppo --env uav-v0 --gym-packages uav_gym