#!/bin/bash

# train or test
file=sbatch_test.script

def_u=4
def_cov=200
def_npref=15
def_multi=4

# train / test for number of uavs
us=( 3 4 5 6 7 8 )

for u in "${us[@]}"; do
    bash $file uav-v0 $u $def_cov $def_npref $def_multi r_n_uavs
  done


# train / test for cov range
cs=( 175 200 225 250 275 300 )

for c in "${cs[@]}"; do
    bash $file uav-v0 $def_u $c $def_npref $def_multi r_cov_range
  done


# train / test for proportion prioritised
nps=( 0 5 10 15 20 25 )

for np in "${nps[@]}"; do
    bash $file uav-v0 $def_u $def_cov $np $def_multi r_pref_prop
  done


# train / test for mul
ms=( 1 2 4 8 16 32 )

for m in "${ms[@]}"; do
    bash $file uav-v0 $def_u $def_cov $def_npref $m r_pref_fac
  done
