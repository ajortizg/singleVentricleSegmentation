#!/bin/bash

# Get the list of process IDs for 'flow.py' started by user 'antonio'
pids=$(pgrep -u antonio -f flow.py)

# Kill each of these processes
if [ -n "$pids" ]; then
  kill $pids
  echo "Killed processes: $pids"
else
  echo "No flow.py processes found for user antonio."
fi
