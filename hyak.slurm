#!/bin/bash
#SBATCH --job-name=cwh_xy_N_4_abs_0_25_rel_1_0
#SBATCH --account=stf
#SBATCH --partition=stf
#SBATCH --time=30:00:00
#SBATCH --mem=120G
#SBATCH --workdir=/gscratch/stf/danylo/explicit_hybrid_mpc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=danylo@uw.edu

# ikt or mox (ikt: 16 CPUs/node, 60 GB RAM/node; mox: 28 CPU/node, 120 GB RAM/node)

## edit -- nodes: number of nodes to use, tasks: nodes * cores per node 
#SBATCH --nodes=11 --ntasks-per-node=28

source activate py36
cd /gscratch/stf/danylo/explicit_hybrid_mpc
# Run partitioning process with 11*28=308 processes (1 scheduler, 307 workers)
mpiexec -n 308 ipython master.py


