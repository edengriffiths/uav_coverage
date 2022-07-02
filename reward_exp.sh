#!/bin/bash

ds=( 1 5 10 50 100 )

# test delta
for d in "${ds[@]}"; do
    bash sbatch_train.script uav-v0 1 1 1 $d
  done


#weights=( 1 3 9 )

#for a in "${weights[@]}"; do
#  for b in "${weights[@]}"; do
#    for g in "${weights[@]}"; do
#      if [ $((a % 3)) != 0 ] || [ $((b % 3)) != 0 ] ||  [ $((g % 3)) != 0 ]; then
#        bash sbatch_train.script uav-v0 $a $b $g
#        fi;
#      done
#    done
#  done


#for b in "${weights[@]}"; do
#  for g in "${weights[@]}"; do
#    if [ $((b % 3)) != 0 ] ||  [ $((g % 3)) != 0 ]; then
##      bash sbatch_train.script uav-v0 2 $b $g
#       echo uav-v0 2 $b $g
#      fi;
#    done
#  done