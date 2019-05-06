"""
Various utility functions.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import math
import itertools
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.spatial as scs
from mpi4py import MPI

import global_vars
from tree import Tree, NodeData

class NonblockingMPIMessageReceiver:
    """Wraps a call to MPI's Comm.irecv."""
    def __init__(self,source,tag,buffer=None):
        """
        Parameters
        ----------
        source : int
            Rank of message source process.
        tag : int
            Message tag.
        buffer : int, optional
            Upper bound to the number of bytes of the incoming pickled message.
            E.g. ``bytearray(1<<20)`` means that the message is at most 1 MB in
            size. See ``https://tinyurl.com/y5craot6`` and
            ``https://tinyurl.com/yy42wrpm``.
        """
        def update_receiver():
            if buffer is not None:
                self.req = MPI.COMM_WORLD.irecv(buffer,source=source,tag=tag)
            else:
                self.req = MPI.COMM_WORLD.irecv(source=source,tag=tag)
            
        self.update_receiver = update_receiver
        self.update_receiver()
        
    def receive(self,which='oldest'):
        """
        Non-blocking receive message, if one is available.
        
        Parameters
        ----------
        which : {'oldest','newest','all'}, optional
            Which message in the buffer to receive; 'oldest' gets the first
            message in the queue, 'newest' receives the last message in the
            queue (clearing the queue in the process), and 'all' gets all the
            messages in the queue.
            
        Returns
        -------
        The received data, ``None`` if no data received. The type depends on
        what is sent by the source process. If which=='all', the type is a list.
        """
        data_list = []
        while True:
            msg_available,data = self.req.test()
            if msg_available:
                self.update_receiver()
                data_list.append(data)
                if which=='oldest':
                    break
            else:
                break
        if len(data_list)==0:
            return None
        elif which=='oldest':
            return data_list[0]
        elif which=='newest':
            return data_list[-1]
        else:
            return data_list

def debug_print(msg):
    """
    Print message based on verbosity setting.
    
    Parameters
    ----------
    msg : string
        The message to print.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    if global_vars.VERBOSE:
        print('%s (%d): %s'%('scheduler' if rank==global_vars.SCHEDULER_PROC else 'worker',rank,msg))

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
