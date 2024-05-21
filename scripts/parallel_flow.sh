#!/bin/bash

# Directory to save the results
save_dir="results"

# Check if the directory doesn't exist and create it if necessary
if [ ! -d "$save_dir" ]; then
  mkdir "$save_dir"
  echo "Directory '$save_dir' created successfully."
else
  echo "Directory '$save_dir' already exists."
fi

# Function to run optical flow computation
run_optical_flow() {
  local mode=$1
  local indices=$2
  local gpu=$3
  local log_file=$4

  export CUDA_VISIBLE_DEVICES=$gpu
  nohup python scripts/flow.py \
    --config.flow.mode=$mode \
    --config.flow.indices=$indices \
    > $log_file 2>&1 </dev/null &
}

# Run forward optical flow computation in parallel
run_optical_flow "forward" "(0, 25)" 0 "results/fwd_0-25.log"
run_optical_flow "forward" "(25, 50)" 1 "results/fwd_25-50.log"
run_optical_flow "forward" "(50, 75)" 2 "results/fwd_50-75.log"
run_optical_flow "forward" "(75, 100)" 3 "results/fwd_75-100.log"
echo "Forward flow scripts started in the background."

# Run backward optical flow computation in parallel
run_optical_flow "backward" "(0, 25)" 4 "results/bwd_0-25.log"
run_optical_flow "backward" "(25, 50)" 5 "results/bwd_25-50.log"
run_optical_flow "backward" "(50, 75)" 6 "results/bwd_50-75.log"
run_optical_flow "backward" "(75, 100)" 7 "results/bwd_75-100.log"
echo "Backward flow scripts started in the background."