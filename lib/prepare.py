"""
Prepares a runtime directory for running the partitioning algorithm.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os

import global_vars
import tools

def make_runtime_dir():
    """
    Setup the runtime directory based on passed command-line
    arguments. Creates the following runtime directory structure (more may
    be added into this directory by the scheduler and worker scripts):
        
        runtime_<example>_N_<N>_abs_<abs>_rel_<rel>_nodes_<nodes>_tpn_<tasks_per_node>_<timestamp>
        |
        +--- data
        |    |
        |    +--- # data files added by scheduler and worker scripts
        |    +--- # ...
        +--- hyak.slurm # ``sbatch -p <partition> -A <account> hyak.slurm`` if running on cluster
        +--- run.sh # ``bash run.sh`` if running locally
    """
    args = tools.parse_args()
    timestamp = tools.set_global_variables()
    # Make runtime directory
    try:
        os.makedirs(global_vars.RUNTIME_DIR)
        os.makedirs(global_vars.DATA_DIR)
    except:
        raise ValueError('Directory %s already exists... being conservative and not overwriting it'%(global_vars.RUNTIME_DIR))
    # run.sh: bash script that can be run in interactive mode via ``bash run.sh``
    with open(global_vars.RUNTIME_DIR+'/run.sh','w') as runsh_file:
        runsh_file.write('\n'.join([
                '#!/bin/bash',
                '#',
                '# Run the partitioning process',
                '# NB: Make sure that you have set the right Python virtualenv',
                '',
                'NUM_WORKERS=%d # Number of worker processes'%(args['nodes']*args['tasks_per_node']-1),
                '',
                'mpirun -n $((NUM_WORKERS+1)) python %s/lib/main.py %s -t %s'%(global_vars.PROJECT_DIR,args['raw_cmdline_string'],timestamp)
                ])+'\n\n')
    # hyak.slurm: script to submit a task to University of Washington's Hyak cluster (mox scheduler)
    with open(global_vars.RUNTIME_DIR+'/hyak.slurm','w') as runsh_file:
        jobname = '%s_N_%d_abs_%s_rel_%s'%(args['example'],args['N'],
                                           str(args['abs_frac']).replace('.','_'),
                                           str(args['rel_err']).replace('.','_'))
        runsh_file.write('\n'.join([
                '#!/bin/bash',
                '#SBATCH --job-name=%s'%(jobname),
                '#SBATCH --account=stf',
                '#SBATCH --partition=stf',
                '#SBATCH --time=%s'%(args['job_duration']),
                '#SBATCH --mem=20G',
                '#SBATCH --workdir=%s'%(global_vars.RUNTIME_DIR),
                '#SBATCH --mail-type=ALL',
                '#SBATCH --mail-user=danylo@uw.edu',
                '#SBATCH --nodes=%d --ntasks-per-node=%d'%(args['nodes'],args['tasks_per_node']),
                '',
                '# Module loading',
                'module load icc_17-impi_2017 # Intel MPI',
                '',
                '# Run script',
                'source activate py36 # Python 3.6.7 virtualenv with necessary packages (MOSEK + requirements.txt)',
                'cd %s'%(global_vars.RUNTIME_DIR),
                'bash run.sh'
                ])+'\n\n')

if __name__=='__main__':
    make_runtime_dir()
    print(global_vars.RUNTIME_DIR) # can be captured by a caller bash script