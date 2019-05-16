#!/bin/bash
#
# Creates runtime directories for the explicit MPC partitions that you wish to
# create. Optionally submits the jobs to the cluster.

SUBMIT_JOBS_LOCAL=false # true runs jobs locally
SUBMIT_JOBS_CLUSTER=false # true submits jobs to the cluster

while getopts ::cl option
do
    case $option in
	c) SUBMIT_JOBS_CLUSTER=true
	   ;;
	l) SUBMIT_JOBS_LOCAL=true
	   ;;
	?)
	echo "usage: bash make_jobs.sh [-l] [-c]"
	echo "    -l submit the jobs to run on local machine (blocks at each jobs)"
	echo "    -c submit the jobs to run on cluster"
	exit 0
	;;
    esac
done

if $SUBMIT_JOBS_LOCAL && $SUBMIT_JOBS_CLUSTER; then
    echo "ERROR: cannot do both local and cluster submissions"
    exit 1
fi

create_jobs()
{ # creates and (if specified) submits jobs
    for i in ${!ABS_FRACS[@]}; do
	ABS_FRAC=${ABS_FRACS[$i]}
	REL_ERR=${REL_ERRS[$i]}
	JOB_DURATION=${JOB_DURATIONS[$i]}
	NUM_PROC=${NUM_PROC_LIST[$i]}

	RUNTIME_DIR=$((python ./lib/prepare.py -e $EXAMPLE -n $NUM_PROC -N $MPC_N -a $ABS_FRAC -r $REL_ERR -t $JOB_DURATION) 2>&1)

	if $SUBMIT_JOBS_CLUSTER; then
	    echo sbatch $RUNTIME_DIR/hyak.slurm
	    sbatch $RUNTIME_DIR/hyak.slurm
	elif $SUBMIT_JOBS_LOCAL; then
	    TIMESTAMP=$(date '+%d%m%YT%H%M%S')
	    echo bash $RUNTIME_DIR/run.sh '&>' $RUNTIME_DIR/log-$TIMESTAMP.txt
	    bash $RUNTIME_DIR/run.sh &> $RUNTIME_DIR/log-$TIMESTAMP.txt
	fi
    done    
}

round_to_nodes()
{ # rounds the desired number of processors to a number that
  # utilizes all processors of a node
    PROCS=$1
    PPN=28 # processors per node
    # local PROCS_ROUNDED=$((python -c "'print($PPN*($PROCS//$PPN+1))'") 2>&1)
    local PROCS_ROUNDED=$((python -c "print($PPN*($PROCS//$PPN+1))") 2>&1)
    echo $PROCS_ROUNDED
}

EXAMPLE=cwh_z
MPC_N=4
ABS_FRACS=(0.5 0.25)
REL_ERRS=(2.0 1.0)
JOB_DURATIONS=('20:00:00' '20:00:00')
NUM_PROC_LIST=($(round_to_nodes 1000) $(round_to_nodes 1000))

create_jobs
