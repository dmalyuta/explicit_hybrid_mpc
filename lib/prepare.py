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

def parse_args(require_timestamp=False):
    """
    Parse the command-line arguments.
    
    Parameters
    ----------
    require_timestamp : bool, optional
        ``True`` to require that a timestamp was passed in the command-line
        arguments.
    
    Returns
    -------
    args : dict
        Dictionary of passed command-line arguments.
    """
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--example',action='store',dest='example',type=str,
                        choices=['cwh_z','cwh_xy'],help='which MPC algorithm to run',required=True)
    parser.add_argument('-N','--nodes',action='store',dest='nodes',type=int,
                        help='number of nodes on which to run',required=True)
    parser.add_argument('-n','--ntasks-per-node',action='store',dest='tasks_per_node',type=int,
                        help='number of tasks to invoke on each node',required=True)
    parser.add_argument('-p','--mpc-horizon',action='store',dest='N',type=int,
                        help='MPC prediction horizon length',required=True)
    parser.add_argument('-a','--abs',action='store',dest='abs_frac',type=float,
                        help='fraction by which to shrink the invariant set for absolute'
                        ' error computation for epsilon-suboptimality',required=True)
    parser.add_argument('-r','--rel',dest='rel_err',type=float,
                        help='relative error for epsilon-suboptimality',required=True)
    parser.add_argument('-d','--job-duration',action='store',dest='job_duration',type=str,
                        help='job time duration in HH:MM:SS format',required=True)
    parser.add_argument('-t','--timestamp',action='store',dest='timestamp',type=str,
                        help='runtime directory timestamp',required=False)
    parser.add_argument('--runtime-dir',action='store',dest='runtime_dir',type=str,
                        help='force-specify the runtime directory name',required=False)
    parser.add_argument('--ecc-tree',action='store',dest='ecc_tree',type=str,
                        help='absolute path to tree.pkl output by ECC algorithm',required=False)
    args = vars(parser.parse_args())
    if args['timestamp'] is None and args['runtime_dir'] is None and require_timestamp==True:
        raise NameError('Need either --timestamp or --runtime-dir to be specified')
    args['raw_cmdline_string'] = ' '.join(sys.argv[1:])
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
    timestamp : str
        Directory timestamp.
    """
    timestamp = args['timestamp'] if args['timestamp'] is not None else time.strftime("%d%m%YT%H%M%S")
    if args['runtime_dir'] is None:
        dirname = 'runtime_%s_N_%d_abs_%s_rel_%s_nodes_%d_tpn_%d_%s'%(
                args['example'],args['N'],str(args['abs_frac']).replace('.','_'),
                str(args['rel_err']).replace('.','_'),args['nodes'],args['tasks_per_node'],timestamp)
    else:
        dirname = args['runtime_dir']
    return dirname, timestamp

def set_global_variables(require_timestamp=False):
    """
    Set the global variables to passed command-line arguments.
    
    Parameters
    ----------
    require_timestamp : bool, optional
        ``True`` to require that a timestamp was passed in the command-line
        arguments.
        
    Returns
    -------
    timestamp : str
        Timestamp when the runtime directory was created.
    """
    args = parse_args(require_timestamp)
    # Optimization oracle parameters
    global_vars.EXAMPLE = args['example']
    global_vars.MPC_HORIZON = args['N']
    global_vars.ABS_FRAC = args['abs_frac']
    global_vars.REL_ERR = args['rel_err']    
    # Filenames
    dirname, timestamp = make_runtime_dir_name(args)
    global_vars.RUNTIME_DIR = global_vars.PROJECT_DIR+'/runtime/'+dirname
    global_vars.DATA_DIR = global_vars.RUNTIME_DIR+'/data'
    global_vars.STATUS_FILE = global_vars.DATA_DIR+'/status.txt' # Overall program status (text file)
    global_vars.STATISTICS_FILE = global_vars.DATA_DIR+'/statistics.pkl' # Overall statistics
    global_vars.TREE_FILE = global_vars.DATA_DIR+'/tree.pkl' # Overall tree
    global_vars.ETA_RLS_FILE = global_vars.DATA_DIR+'/rls.pkl' # Overall tree
    global_vars.BRANCHES_FILE = global_vars.DATA_DIR+'/branches.pkl' # Tree branches, used for tree building
    global_vars.IDLE_COUNT_FILE = global_vars.DATA_DIR+'/idle_count.pkl' # Idle process count
    return timestamp

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
    args = parse_args()
    timestamp = set_global_variables()
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