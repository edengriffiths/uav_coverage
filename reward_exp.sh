#!/bin/bash

weights=( 1 3 9 )


for a in "${weights[@]}"; do
  for b in "${weights[@]}"; do
    for g in "${weights[@]}"; do
      if [ $((a % 3)) != 0 ] || [ $((b % 3)) != 0 ] ||  [ $((g % 3)) != 0 ]; then
        # sbatch_train.script
        bash sbatch_train.script uav-v0 $a $b $g
        fi;
      done
    done
  done
