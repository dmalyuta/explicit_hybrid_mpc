#!/bin/bash
#SBATCH --job-name=lcss_explicit_mpc
#SBATCH --account=stf
#SBATCH --partition=stf
#SBATCH --time=10:00:00
#SBATCH --mem=20G
#SBATCH --workdir=/gscratch/stf/danylo/explicit_hybrid_mpc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=danylo@uw.edu

# ikt or mox (ikt: 16 CPUs/node, 60 GB RAM/node; mox: 28 CPU/node, 120 GB RAM/node)

## edit -- nodes: number of nodes to use, tasks: nodes * cores per node 
#SBATCH --nodes=15 --ntasks-per-node=28

# Module loading
module load icc_17-impi_2017 # Intel MPI

# Run script
source activate py36
cd /gscratch/stf/danylo/explicit_hybrid_mpc
bash run.sh
