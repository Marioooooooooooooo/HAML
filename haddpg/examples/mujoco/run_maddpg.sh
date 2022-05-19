#!/bin/bash

LOGDIR="results"
TASK=$1
ALGO=$2
NUM_AGENTS=$3
SAVE_DIR=$4

echo "Experiments started."
for seed in {6,42,7654,9876,12321}
do
    python mujoco_maddpg.py --task $TASK --algo $ALGO --num_agents $NUM_AGENTS --seed $seed --logdir $LOGDIR --save_dir $SAVE_DIR > ${TASK}_`date '+%m-%d-%H-%M-%S'`_seed_$seed.txt 2>&1 &
done
echo "Experiments ended."
