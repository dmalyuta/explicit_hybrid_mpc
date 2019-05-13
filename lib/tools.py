"""
Various utility functions.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import sys
import math
import itertools
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.spatial as scs

import mpi4py
# (maybe) resolves UnpicklingError (https://tinyurl.com/mpi4py-unpickling-issue)
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI as mpi4py_MPI

import global_vars
from tree import Tree, NodeData

class MPICommunicator:
    def __init__(self):
        self.comm = mpi4py_MPI.COMM_WORLD # MPI inter-process communicator
        self.async_send_counter = 0 # Counter of how many messages were send asynchronously
        self.req = None # Latest non-blocking MPI send Request object
        self.block_counter = 0 # Counter for how many times blocking send had to occur

    def size(self):
        """Get total number of processes."""
        return self.comm.Get_size()
    
    def rank(self):
        """Get the process number (MPI rank)."""
        return self.comm.Get_rank()
    
    def global_sync(self):
        """MPI barrier global synchronization."""
        self.comm.Barrier()
        return None

    def broadcast(self,*args,**kwargs):
        """MPI broadcast (blocking)."""
        return self.comm.bcast(*args,**kwargs)
    
    def nonblocking_receive(self,*args,**kwargs):
        """MPI receive message, non-blocking."""
        return self.comm.irecv(*args,**kwargs)

    def nonblocking_send(self,*args,**kwargs):
        """MPI send message, non-blocking."""
        return self.comm.issend(*args,**kwargs)

    def blocking_receive(self,*args,**kwargs):
        """MPI receive message, blocking."""
        return self.comm.recv(*args,**kwargs)
    
    def blocking_send(self,*args,**kwargs):
        """MPI send message, blocking."""
        return self.comm.ssend(*args,**kwargs)

    def send(self,*args,**kwargs):
        """Wrapper for MPI blocking/non-blocking sending respecting
        MAX_ASYNC_SEND."""
        self.async_send_counter += 1
        receive_posted = False if self.req is None else self.req.test()[0]
        if receive_posted:
            # matching receive is posted
            # https://www.mcs.anl.gov/research/projects/mpi/sendmode.html
            # see also:
            # https://stackoverflow.com/questions/21512975/what-is-the-difference-between-isend-and-issend
            self.async_send_counter = 0 # reset
        overflow = self.async_send_counter>global_vars.MAX_ASYNC_SEND
        if self.async_send_counter>global_vars.MAX_ASYNC_SEND:
            # wait for matching receive to post
            self.block_counter += 1
            info_print('MPI message overflow - waiting '
                       '(occured %d times)...'%(self.block_counter))
            self.req.wait()
            self.async_send_counter = 0 # reset
        self.req = self.nonblocking_send(*args,**kwargs)
        return self.req
            

MPI = MPICommunicator() # Object that abstracts calls to MPI library routines

class NonblockingMPIMessageReceiver:
    """Wraps a call to MPI's Comm.irecv."""
    def __init__(self,source,tag):
        """
        Parameters
        ----------
        source : int
            Rank of message source process.
        tag : int
            Message tag.
        """
        def update_receiver():
            self.req = MPI.nonblocking_receive(source=source,tag=tag)
            
        self.update_receiver = update_receiver
        self.update_receiver()
        
    def receive(self):
        """
        Non-blocking receive message, if one is available.
        
        Returns
        -------
        The received data, ``None`` if no data received. The type depends on
        what is sent by the source process.
        """
        msg_available,data = self.req.test()
        if msg_available:
            self.update_receiver()
            return data
        return None

def info_print(msg):
    """
    Print info message based on verbosity setting.
    
    Parameters
    ----------
    msg : string
        The message to print.
    """
    rank = MPI.rank()
    if global_vars.VERBOSE==2:
        print('%s (%d): %s'%('scheduler' if rank==global_vars.SCHEDULER_PROC
                             else 'worker',rank,msg))
        sys.stdout.flush()

def error_print(msg):
    """
    Print error message based on verbosity setting.
    
    Parameters
    ----------
    msg : string
        The message to print.
    """
    rank = MPI.rank()
    if global_vars.VERBOSE>=1:
        print('%s (%d): %s'%('scheduler' if rank==global_vars.SCHEDULER_PROC
                             else 'worker',rank,msg))
        sys.stdout.flush()

def simplex_volume(R):
    """
    Get the simplex volume.
    
    Parameters
    ----------
    R : np.array
        Matrix whose rows are the simplex vertices.

    Returns
    -------
    vol : float
        Volume of the simplex.
    """
    M = np.column_stack([v-R[0] for v in R[1:]])
    vol = np.abs(1./math.factorial(R.shape[0]-1)*la.det(M))
    return vol

def delaunay(R):
    """
    Partition polytope R into simplices.
    
    Parameters
    ----------
    R : np.array
        Matrix whose rows are the polytope vertices.
        
    Returns
    -------
    root : Tree
        Binary tree whose leaves are the partition. Unless both children are
        leaves, the left child is a leaf and the right child has "None" data
        and is itself a parent.
    Nsx : float
        Number of simplices that R was partitioned into.
    vol : float
        Volume of R.
    """
    vol = 0.
    delaunay = scs.Delaunay(R)
    root = Tree(NodeData(vertices=R))
    cursor = root # This is a temporary "pointer" that goes down the tree
    Nsx = delaunay.simplices.shape[0] # Number of simplices
    if Nsx==1:
        vol += simplex_volume(R)
    for i in range(Nsx-1):
        left = NodeData(vertices=R[delaunay.simplices[i]])
        vol += simplex_volume(R[delaunay.simplices[i]])
        if i<Nsx-2:
            right = None
        else:
            right = NodeData(vertices=R[delaunay.simplices[i+1]])
            vol += simplex_volume(R[delaunay.simplices[i+1]])
        cursor.grow(left,right)
        cursor = cursor.right
    return root, Nsx, vol

def join_triangulation(cursor,new_tree):
        """
        Append a new tree to an existing one at the rightmost leaf.
        This effectively union-izes several triangulations.
        **Modifies the tree associated with cursor and sets
        new_tree.top=False**

        Parameters
        ----------
        cursor : Tree
            Root of the tree to which the new tree is to be appended.
        new_tree : Tree
            Root of the new tree to be appended.
        """
        if cursor.is_leaf():
            # Performs the transformation:
            #   ...            ...
            #     \              \
            #      \              \
            #    cursor          None
            #                    /  \
            #                   /    \
            #              cursor  new_tree
            #                       /   \
            #                      /     \
            #                    ...     ...
            cursor.grow(cursor.data,None)
            cursor.right = new_tree
            new_tree.top = False
            cursor.data = None
        else:
            join_triangulation(cursor.right,new_tree)

def split_along_longest_edge(R):
    """
    Split simplex R at the midpoint of its longest edge (into two halves).
    
    Parameters
    ----------
    R : np.array
        Matrix whose rows are the simplex vertices.
        
    Returns
    -------
    S_1 : np.array
        First half of R.
    S_2 : np.array
        Second half of R.
    longest_edge_combo : tuple
        Tuple of two elements corresponding to row index of the two vertices
        constituting the longest edge. The first vertex is removed from S_1 and
        the second vertex is removed from S_2, substituted for v_mid.
    """
    # Find midpoint along longest edge of simplex
    N_vertices = R.shape[0]
    vertex_combinations = list(itertools.combinations(range(N_vertices),2))
    longest_edge_idx = np.argmax([la.norm(R[vx_combo[0]]-R[vx_combo[1]])
                                  for vx_combo in vertex_combinations])
    longest_edge_combo = vertex_combinations[longest_edge_idx]
    bar_v = R[longest_edge_combo[0]]
    bar_v_prime = R[longest_edge_combo[1]]
    v_mid = (bar_v+bar_v_prime)/2.
    # Split simplex in half at the midpoint
    S_1,S_2 = R.copy(),R.copy()
    S_1[longest_edge_combo[0]] = v_mid
    S_2[longest_edge_combo[1]] = v_mid
    return S_1, S_2, longest_edge_combo

def discretize(Ac,Bc,dt):
    """Dynamics discretization"""
    M = sla.expm(np.block([[Ac,Bc],
                           [np.zeros([Bc.shape[1],
                                      Ac.shape[1]+Bc.shape[1]])]])*dt)
    A = M[:Ac.shape[0],:Ac.shape[0]]
    B = M[:Ac.shape[0],Ac.shape[0]:]
    return A,B

def fullrange(a,b=None):
    """
    Return an integer range iterator, which includes the final point
    and which starts at 1 by default.
    
    Parameters
    ----------
    a : int
        Interval start. If b is not specified, interval start is
        taken to be 1 and a is used as the interval end.
    b : int, optional
        Interval end.
        
    Returns
    -------
    : list
        Integer range {a,a+1,a+2,...,b-1,b}.
    """
    if b is None:
        return range(1,a+1)
    else:
        return range(a,b+1)

def cluster_job_duration(m,N,n,e,lt,lv,s=1.5):
    """
    Compute the required job duration to ask from cluster scheduler.
    
    Parameters
    ----------
    m : int
        Number of processes used in local benchmark.
    N : int
        Number of nodes on cluster.
    n : int
        Number of tasks per node on cluster.
    e : int
        ECC runtime in seconds in local benchmark.
    lt : int
        L-CSS runtime in seconds in local benchmark.
    lv : float
        Achieved L-CSS volume filled (in [0,100]).
    s : float
        Safety factor (ask for this fraction more time than computed from the
        above data).
        
    Returns
    -------
    T : str
        Job duration in HH:MM:SS format, rounded to nearest minute.
    """
    lv /= 100.
    duration = (m-1)/(n*N-1)*(e+lt/lv)*s
    hours = int(duration//3600)
    minutes = int(np.ceil((duration-hours*3600)/60))
    T = '%s:%s:00'%(str(hours).zfill(2),str(minutes).zfill(2))
    print(T)
    return T
