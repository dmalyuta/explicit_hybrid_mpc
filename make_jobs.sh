#!/bin/bash
#
# Creates runtime directories for the explicit MPC partitions that you wish to
# create. Optionally submits the jobs to the cluster.

SUBMIT_JOBS=true # true submits jobs to the cluster

create_jobs()
{ # creates and (if specified) submits jobs
    for i in ${!ABS_FRACS[@]}; do
	ABS_FRAC=${ABS_FRACS[$i]}
	REL_ERR=${REL_ERRS[$i]}
	JOB_DURATION=${JOB_DURATIONS[$i]}
	NODES=${NODES_LIST[$i]}

	RUNTIME_DIR=$((python ./lib/prepare.py -e $EXAMPLE -N $NODES -n $TASKS_PER_NODE -p $MPC_N -a $ABS_FRAC -r $REL_ERR -d $JOB_DURATION) 2>&1)

	if $SUBMIT_JOBS; then
	    echo sbatch -p stf -A stf $RUNTIME_DIR/hyak.slurm
	fi
    done    
}

# CWH z
EXAMPLE=cwh_z
MPC_N=4
ABS_FRACS=(0.01 0.03 0.1 0.25 0.5)
REL_ERRS=(0.01 0.05 0.1 1.0 2.0)
JOB_DURATIONS=('10:00:00' '10:00:00' '10:00:00' '10:00:00' '10:00:00')
NODES_LIST=(5 5 5 2 2)
TASKS_PER_NODE=28

create_jobs

# CWH xy
EXAMPLE=cwh_xy
MPC_N=4
ABS_FRACS=(0.01 0.03 0.1 0.25 0.5)
REL_ERRS=(0.01 0.05 0.1 1.0 2.0)
JOB_DURATIONS=('10:00:00' '10:00:00' '10:00:00' '10:00:00' '10:00:00')
NODES_LIST=(5 5 5 2 2)
TASKS_PER_NODE=28

create_jobs

# CWH xyz
EXAMPLE=cwh_xyz
MPC_N=4
ABS_FRACS=(0.01 0.03 0.1 0.25 0.5)
REL_ERRS=(0.01 0.05 0.1 1.0 2.0)
JOB_DURATIONS=('10:00:00' '10:00:00' '10:00:00' '10:00:00' '10:00:00')
NODES_LIST=(5 5 5 2 2)
TASKS_PER_NODE=28

create_jobs
