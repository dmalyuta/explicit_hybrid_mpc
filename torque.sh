#!/bin/bash

#MSUB -V
#MSUB -l nodes=4:ppn=20
#MSUB -l mem=128gb
#MSUB -M malyuta@uw.edu
#MSUB -m ae
#MSUB -l walltime=240:00:00 -q long
#MSUB -N random_partition

cd $PBS_O-WORKDIR/

module load python/3.6.7

source .bashrc
cd docs/mm_control/examples
source activate mm_control

# 1 instance of the test.py script
ipython random.py &
sleep 3

# Last instance without & to not exit (not sure if this is a problem)
