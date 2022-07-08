#!/bin/bash

env_id=$1
n_uavs=$2
cov_r=$3
prop=$4
pref_fac=$5

cd rl-baselines3-zoo || exit
python train.py --algo ppo --env $env_id --gym-packages uav_gym --env-kwargs n_uavs:$n_uavs cov_range:$cov_r pref_prop:$prop pref_factor:$pref_fac -f ./logs/${n_uavs}_${cov_r}_${prop}_${pref_fac}
