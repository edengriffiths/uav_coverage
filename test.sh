#!/bin/bash


env_id=$1
n_uavs=$2
cov_r=$3
prop=$4
pref_fac=$5

python test.py $env_id $n_uavs $cov_r $prop $pref_fac
