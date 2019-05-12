#!/bin/bash
#
# Creates runtime directories for the explicit MPC partitions that you wish to
# create. Optionally submits the jobs to the cluster.

SUBMIT_JOBS=false # true submits jobs to the cluster

create_jobs()
{ # creates and (if specified) submits jobs
    for i in ${!ABS_FRACS[@]}; do
	ABS_FRAC=${ABS_FRACS[$i]}
	REL_ERR=${REL_ERRS[$i]}
	JOB_DURATION=${JOB_DURATIONS[$i]}
	NODES=${NODES_LIST[$i]}

	RUNTIME_DIR=$((python ./lib/prepare.py -e $EXAMPLE -N $NODES -p $MPC_N -a $ABS_FRAC -r $REL_ERR -t $JOB_DURATION) 2>&1)

	if $SUBMIT_JOBS; then
	    sbatch -p stf -A stf $RUNTIME_DIR/hyak.slurm
	fi
    done    
}

EXAMPLE=pendulum
MPC_N=5
ABS_FRACS=(0.5)
REL_ERRS=(2.0)
JOB_DURATIONS=('10:00:00')
NODES_LIST=(1)

create_jobs

# # CWH z
# EXAMPLE=cwh_z
# MPC_N=4
# ABS_FRACS=(0.01 0.03 0.1 0.25 0.5)
# REL_ERRS=(0.01 0.05 0.1 1.0 2.0)
# JOB_DURATIONS=('10:00:00' '02:00:00' '00:30:00' '00:10:00' '00:10:00')
# NODES_LIST=(15 3 1 1 1)

# create_jobs

# # CWH xy
# EXAMPLE=cwh_xy
# MPC_N=4
# ABS_FRACS=(0.15 0.25 0.5)
# REL_ERRS=(0.5 1.0 2.0)
# JOB_DURATIONS=('20:00:00' '20:00:00' '05:00:00')
# NODES_LIST=(15 15 5)

# create_jobs

# # CWH xyz
# EXAMPLE=cwh_xyz
# MPC_N=4
# ABS_FRACS=(0.5)
# REL_ERRS=(2.0)
# JOB_DURATIONS=('20:00:00')
# NODES_LIST=(40)

# create_jobs
