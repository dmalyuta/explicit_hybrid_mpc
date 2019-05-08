"""
Prepares a runtime directory for running the partitioning algorithm.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os
import sys
import time
import argparse

import global_vars

"""
Some notes on usage:

- When calling ``prepare.py``, you should call it with the -N and -t commandline
  arguments in order to create a runtime directory properly.
- When calling ``main.py``, you may omit the -N and -t commandline arguments and
  instead use --runtime-dir=<runtime directory name>, which should be the one
  that was created by ``prepare.py``. In fact, it is the directory of the
  ``run.sh`` file that you want to execute.
- You may use the --ecc-tree commandline argument to specify the directory of a
  pre-computed feasible partition by the ECC 2019 algorithm
  (worker.py:Worker.ecc)
"""

def parse_args():
    """
    Parse the command-line arguments.
    
    Returns
    -------
    args : dict
        Dictionary of passed command-line arguments.
    """
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--example',action='store',dest='example',type=str,
                        help='which MPC algorithm to run',required=True)
    parser.add_argument('-N','--nodes',action='store',dest='nodes',type=int,
                        help='number of nodes on which to run',required=False)
    parser.add_argument('-p','--mpc-horizon',action='store',dest='N',type=int,
                        help='MPC prediction horizon length',required=True)
    parser.add_argument('-a','--abs',action='store',dest='abs_frac',type=float,
                        help='fraction by which to shrink the invariant set for'
                        ' absolute error computation for epsilon-suboptimality',
                        required=True)
    parser.add_argument('-r','--rel',dest='rel_err',type=float,
                        help='relative error for epsilon-suboptimality',
                        required=True)
    parser.add_argument('-t','--job-duration',action='store',
                        dest='job_duration',type=str,help='job time duration in'
                        ' HH:MM:SS format',required=False)
    parser.add_argument('--runtime-dir',action='store',dest='runtime_dir',
                        type=str,help='force-specify the runtime directory'
                        ' name',required=False)
    parser.add_argument('-b','--branches',action='store_true',
                        dest='use_branches',default=False,
                        help='whether to use',required=False)
    args = vars(parser.parse_args())
    return args

def make_runtime_dir_name(args):
    """
    Creates the runtime directory name.
    
    Parameters
    ----------
    args : dict
        Dictionary of passed command-line arguments.
        
    Returns
    -------
    dirname : str
        Runtime directory name.
    """
    timestamp = time.strftime("%d%m%YT%H%M%S")
    if args['runtime_dir'] is None:
        dirname = 'runtime_%s_N_%d_abs_%s_rel_%s_%s'%(
            args['example'],args['N'],str(args['abs_frac']).replace('.','_'),
            str(args['rel_err']).replace('.','_'),timestamp)
    else:
        dirname = args['runtime_dir']
    return dirname

def set_global_variables():
    """
    Set the global variables to passed command-line arguments.
        
    Returns
    -------
    timestamp : str
        Timestamp when the runtime directory was created.
    """
    args = parse_args()
    global_vars.CMD_LINE_ARGS = args
    # Optimization oracle parameters
    global_vars.EXAMPLE = args['example']
    global_vars.MPC_HORIZON = args['N']
    global_vars.ABS_FRAC = args['abs_frac']
    global_vars.REL_ERR = args['rel_err']    
    # Filenames
    dirname = make_runtime_dir_name(args)
    global_vars.RUNTIME_DIR = global_vars.PROJECT_DIR+'/runtime/'+dirname
    global_vars.DATA_DIR = global_vars.RUNTIME_DIR+'/data'
    global_vars.STATUS_FILE = global_vars.DATA_DIR+'/status.txt' # Overall program status (text file)
    global_vars.STATISTICS_FILE = global_vars.DATA_DIR+'/statistics.pkl' # Overall statistics
    global_vars.TREE_FILE = global_vars.DATA_DIR+'/tree.pkl' # Overall tree
    global_vars.ETA_RLS_FILE = global_vars.DATA_DIR+'/rls.pkl' # Overall tree
    global_vars.BRANCHES_FILE = global_vars.DATA_DIR+'/branches.pkl' # Tree branches, used for tree building
    global_vars.IDLE_COUNT_FILE = global_vars.DATA_DIR+'/idle_count.pkl' # Idle process count

def make_runtime_dir():
    """
    Setup the runtime directory based on passed command-line
    arguments. Creates the following runtime directory structure (more may
    be added into this directory by the scheduler and worker scripts):
        
        runtime_<example>_N_<N>_abs_<abs>_rel_<rel>_<timestamp>
        |
        +--- data
        |    |
        |    +--- # data files added by scheduler and worker scripts
        |    +--- # ...
        +--- hyak.slurm # ``sbatch -p <partition> -A <account> hyak.slurm`` if running on cluster
        +--- run.sh # ``bash run.sh`` if running locally
    """
    args = parse_args()
    set_global_variables()
    # Make runtime directory
    try:
        os.makedirs(global_vars.DATA_DIR)
    except:
        raise ValueError('Directory %s already exists... being conservative and not overwriting it'%(global_vars.RUNTIME_DIR))
    # run.sh: bash script that can be run in interactive mode via ``bash run.sh``
    with open(global_vars.RUNTIME_DIR+'/run.sh','w') as runsh_file:
        runsh_file.write('\n'.join([
            '#!/bin/bash',
            '#',
            '# Run the partitioning process by calling:',
            '# ```',
            '# $ bash run.sh',
            '# ```',
            '# NB: Make sure that you have set the right Python virtualenv',
            '',
            'mpirun python %s/lib/main.py -e %s -p %d -a %s -r %s --runtime-dir=%s'%
            (global_vars.PROJECT_DIR,args['example'],args['N'],
             str(args['abs_frac']),str(args['rel_err']),
             global_vars.RUNTIME_DIR.split('/')[-1])])+'\n\n')
    # hyak.slurm: script to submit a task to University of Washington's Hyak cluster (mox scheduler)
    with open(global_vars.RUNTIME_DIR+'/hyak.slurm','w') as runsh_file:
        jobname = '%s_N_%d_abs_%s_rel_%s'%(
            args['example'],args['N'],str(args['abs_frac']).replace('.','_'),
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
            '#SBATCH --nodes=%d'%(args['nodes']),
            '',
            '# Module loading',
            'module load gcc_4.8.5-impi_2017 # Intel(R) MPI Library',
            '',
            '# Run script',
            'source activate py36 # Python 3.6.7 virtualenv with necessary packages (MOSEK + requirements.txt)',
            'cd %s'%(global_vars.RUNTIME_DIR),
            'bash run.sh'
        ])+'\n\n')

if __name__=='__main__':
    make_runtime_dir()
    print(global_vars.RUNTIME_DIR) # can be captured by a caller bash script
