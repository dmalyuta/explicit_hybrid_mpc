#!/bin/bash
#
# Runs the example.

N_PROC=$(head -n 1 n_proc.txt) # Number of processes

# Setup files
python setup_files.py

# Spawn slave processes
for i in $(eval echo "{0..$((N_PROC-1))}")
do
    python slave.py $i &
done

# Display status
while true
do
    echo "$(cat status.txt)"
    sleep 1
done
