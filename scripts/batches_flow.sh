#!/bin/bash

save_dir="results"
if [ ! -d "$save_dir" ]  # Check if directory doesn't exist
then
  mkdir "$save_dir"
  echo "Directory '$save_dir' created successfully."
else
  echo "Directory '$save_dir' already exists."
fi

# Runing forward optical flow computation in parallel
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/flow.py \
  --config.flow.mode=forward  \
  --config.flow.indices='(0, 25)' \
  > results/fwd_0-25.log 2>&1 </dev/null &

export CUDA_VISIBLE_DEVICES=1
nohup python scripts/flow.py \
  --config.flow.mode=forward \
  --config.flow.indices='(25, 50)' \
  > results/fwd_25-50.log 2>&1 </dev/null &

export CUDA_VISIBLE_DEVICES=2
nohup python scripts/flow.py \
  --config.flow.mode=forward \
  --config.flow.indices='(50, 75)' \
  > results/fwd_50-75.log 2>&1 </dev/null &

export CUDA_VISIBLE_DEVICES=3
nohup python scripts/flow.py \
  --config.flow.mode=forward \
  --config.flow.indices='(75, 100)' \
  > results/fwd_75-100.log 2>&1 </dev/null &

echo "Forward flow scripts started in the background."

# Runing backward optical flow computation in parallel
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/flow.py \
  --config.flow.mode=backward  \
  --config.flow.indices='(0, 25)' \
  > results/bwd_0-25.log 2>&1 </dev/null &

export CUDA_VISIBLE_DEVICES=1
nohup python scripts/flow.py \
  --config.flow.mode=backward \
  --config.flow.indices='(25, 50)' \
  > results/bwd_25-50.log 2>&1 </dev/null &

export CUDA_VISIBLE_DEVICES=2
nohup python scripts/flow.py \
  --config.flow.mode=backward \
  --config.flow.indices='(50, 75)' \
  > results/bwd_50-75.log 2>&1 </dev/null &

export CUDA_VISIBLE_DEVICES=3
nohup python scripts/flow.py \
  --config.flow.mode=backward \
  --config.flow.indices='(75, 100)' \
  > results/bwd_75-100.log 2>&1 </dev/null &

echo "Backward flow scripts started in the background."