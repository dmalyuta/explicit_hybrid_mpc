#!/bin/bash
#SBATCH --job-name=cwh_xy_N_4_abs_0_25_rel_1_0
#SBATCH --account=stf
#SBATCH --partition=stf
#SBATCH --time=40:00:00
#SBATCH --mem=60G
#SBATCH --workdir=/gscratch/stf/danylo/explicit_hybrid_mpc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=danylo@uw.edu

## ikt or MOX (ikt: 16 CPUs/node, 60 GB RAM/node; mox: 24 CPU/node, 1020 GB/RAM)

## edit -- nodes: number of nodes to use, tasks: nodes * cores per node 
#SBATCH --nodes=4 --ntasks-per-node=28

## module load abaqus_6.14
# TODO python
source activate py36

cd /gscratch/stf/danylo/explicit_hybrid_mpc

ipython master.py


