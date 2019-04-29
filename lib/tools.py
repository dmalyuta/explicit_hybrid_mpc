"""
Various utility functions.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import sys
import glob
import itertools
import primefac
import math
import fcntl
import numpy as np
import numpy.linalg as la
import scipy.spatial as scs
import cvxpy as cvx
import progressbar
import matplotlib
import matplotlib.pyplot as plt

import global_vars
from tree import Tree, NodeData
from polytope import Polytope

class Progress:
    """
    Provides a progressbar that allows to monitor how much volume has been
    "filled in" already.
    """
    def __init__(self,total_volume,open_count):
        """
        Parameters
        ----------
        total_volume : float
            The total volume that is to be filled in.
        open_count : int
            How many simplices in the original tree that is to be partitioned.
        """
        self.widgets=['0','% vol, ','0','/',str(open_count),' [', progressbar.Timer(), '] ',
                      progressbar.Bar(),' ', progressbar.ETA(), ')']
        self.total_volume = total_volume
        self.open_count = open_count
        self.volume_closed = 0.
        self.closed_count = 0
        self.bar = progressbar.ProgressBar(max_value=1.,widgets=self.widgets)
        self.bar.start()
        
    def update(self,volume_increment=None,count_increment=None):
        """
        Update the progressbar and print it out. At least one of the two
        parameters must be provided.
        
        Parameters
        ----------
        volume_increment : float (optional)
            How much extra volume has been closed.
        count_increment : int (optional)
            How many extra open leaves have been added.
        """
        if volume_increment is not None:
            self.volume_closed += volume_increment
            self.closed_count += 1
            self.open_count -= 1
        elif count_increment is not None:
            self.open_count += count_increment
        else:
            raise RuntimeError('At least one argument must be provided')
        
        fraction_closed = min(self.volume_closed/self.total_volume,1.)
        self.widgets[0] = '%.2e'%(fraction_closed*100)
        self.widgets[2] = '%d'%(self.closed_count)
        self.widgets[4] = '%d'%(self.open_count+self.closed_count)
        self.bar.update(fraction_closed)

class Animator:
    """
    Allows to animate the simplicial partitioning process.
    """
    def __init__(self,fignum,mpc):
        """
        Parameters
        ----------
        fignum : int
            Figure number.
        mpc : MPC
            Optimization problem under consideration.
        """
        # Make the commutation to color map
        commutations_at_stage = [[int(i!=j) for j in range(mpc.Nu)] for i in range(mpc.Nu)]
        all_commutations = [np.array(list(itertools.chain.from_iterable(combo))) for combo in itertools.product(commutations_at_stage,repeat=mpc.N)]
        commutation2color = {str(all_commutations[i]):i for i in range(len(all_commutations))}
        cmap = plt.cm.jet
        norm = matplotlib.colors.Normalize(vmin=0,vmax=len(commutation2color)-1)
        self.get_color = (lambda commutation: cmap(norm(commutation2color[str(commutation)])))
        
        # Setup the figure
        self.fig = plt.figure(fignum)
        plt.clf()
        ax_labels = ['$x_'+str(i+1)+'$' for i in range(mpc.n_x)]
        self.handles = dict()
        self.projections = list(itertools.combinations(range(mpc.n_x),2))
        num_plots = len(self.projections)
        sp_layout = self.numSubplots(num_plots)
        self.axes = []
        for i in range(num_plots):
            self.axes.append(self.fig.add_subplot(sp_layout[0],
                                                  sp_layout[1],i+1))
            self.axes[-1].set_xlabel(ax_labels[self.projections[i][0]])
            self.axes[-1].set_ylabel(ax_labels[self.projections[i][1]])
        plt.tight_layout(True,rect=[0.1, 0.03, 1, 0.95])
        plt.ion()
        plt.show(block=False)
        self.draw = lambda: mypause(1e-5)
        self.draw()
        
    def update(self,obj,mem=None,**kwargs):
        """
        Draw projections of a point or a polytope onto all 2-axis
        projection combinations. Plot style arguments can be
        provided via **kwargs.
        
        Parameters
        ----------
        obj : Polytope
            The polytope to be drawn.
        mem : str, optional
            Name with which to memorize these plot handles, so that e.g.
            they may be erased later.
        """
        plot_handles = []
        for ax,proj_coords in zip(self.axes,self.projections):
            plot_handles.append(obj.plot(ax,coords=proj_coords,**kwargs))
        self.draw()
        if mem is not None:
            self.handles.update({mem:plot_handles})
            
    def erase(self,handles):
        """
        Erase plots given by handles.
        
        Parameters
        ----------
        handles : {str,list}
            self.handles dictionary field(s) whose associated plots are to
            be erased.
        """
        handles = handles if type(handles) is list else [handles]
        for handle in handles:
            if self.animate and handle in self.handles:
                for plot_handle in self.handles[handle]:
                    plot_handle.remove()
                self.draw()
                self.handles.pop(handle,None) # Remove field
            
    def title(self,msg):
        """
        Set a figure title.
        
        Parameters
        ----------
        msg : str
            The title.
        """
        self.fig.suptitle(msg)
            
    def numSubplots(self,n):
        """
        Calculate how many rows and columns of sub-plots are needed to neatly
        display n subplots. Converted to Python from Rob Campbell's MATLAB code
        [1].
        
        [1] https://www.mathworks.com/matlabcentral/fileexchange/26310-numsubplots-
        neatly-arrange-subplots
        
        Parameters
        ----------
        n : int
            The desired number of subplots.
            
        Returns
        -------
        p : list
            A list of length 2 defining the number of rows and number of columns
            required to show n plots.
        n_current : int
            The current number of subplots. This output is used only by this
            function for a recursive call.
        """
        isprime = lambda x: x>1 and all(x % i for i in range(2, x))
        while isprime(n) and n>4:
            n += 1
    
        p=list(primefac.primefac(n) if n>1 else [1])
    
        if len(p)==1:
            p=[1,p[0]]
            return p
    
        while len(p)>2:
            if len(p)>=4:
                p[0]=p[0]*p[-2]
                p[1]=p[1]*p[-1]
                del p[-2:]
            else:
                p[0]=p[0]*p[1]
                del p[1]
            p=list(np.sort(p))
    
        # Reformat if the column/row ratio is too large: we want a roughly
        # square design 
        while p[1]/float(p[0])>2.5:
            N = n+1
            p,n = self.numSubplots(N) # Recursive!
        return p

def mypause(interval):
    """
    See https://stackoverflow.com/questions/45729092/make-
    interactive-matplotlib-window-not-pop-to-front-on-each-
    update-windows-7/45734500#45734500
    """
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def get_nodes_in_queue():
    """
    Get subtree nodes in the multi-process partitioning queue.
    
    Returns
    -------
    : list
        List of node files.
    """
    return glob.glob(global_vars.NODE_DIR+'node_*.pkl')

def simplex_condition_number(R):
    """
    Get the simplex condition number.
    
    Parameters
    ----------
    R : np.array
        Matrix whose rows are the simplex vertices.

    Returns
        -------
        : float
            Condition number.
    """
    M = np.column_stack([v-R[0] for v in R[1:]])
    sigma_min = la.svd(M)[1][-1] # Smallest singular value
    cond = la.cond(M) if sigma_min>0 else np.inf
    return cond

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

def getM(H,h):
    """
    Find the smallest inf-norm "M" for a big-M mixed-integer constraint
    formulation of containement of a variable u in one of the polytopes
    {u: H[i]*u<=h[i]}.
    
    Parameters
    ----------
    H : list
        Matrices of polytope facet normals.
    h : list
        Vectors of polytope facet distances.
        
    Returns
    -------
    M : float
        The smallest M.
    """
    Nu = len(H)
    n_u = h[0].size
    M = cvx.Variable(n_u)
    cost = cvx.Minimize(cvx.norm_inf(M))
    # The constraint is that polytope {u: H[i]*u<=h[i]} must be contained in
    # the intersection of polytopes {u: H[j]*u<=h[j]+M} for all j!=i. This can
    # be checked as an LP by checking if the vertices of the former set are
    # contained in the latter set, since both sets are convex polytopes.
    constraints = []
    for i in range(Nu):
        for v in Polytope(A=H[i],b=h[i]).V:
                constraints += [H[j].dot(v) <= h[j]+M for j in range(Nu) if j!=i]
    problem = cvx.Problem(cost,constraints)
    problem.solve(**global_vars.SOLVER_OPTIONS)
    return M.value

class Mutex:
    def __init__(self,proc_num,verbose=False):
        """
        Parameters
        ----------
        proc_num : int
            Process number that mutex belongs to.
        verbose : bool, optional
            ``True`` to print debug output.
        """
        self.proc_num = proc_num
        self.verbose = verbose
        self.mutex = open(global_vars.MUTEX_FILE,'w')
        
    def __print(self,string):
        if self.verbose:
            print('proc %d: '%(self.proc_num)+string)
            sys.stdout.flush()
    
    def lock(self,msg=''):
        self.__print('trying to acquire mutex%s'%(' (%s)'%(msg) if msg!='' else ''))
        fcntl.lockf(self.mutex,fcntl.LOCK_EX)
        self.__print('acquired mutex')
    
    def unlock(self,msg=''):
        fcntl.lockf(self.mutex,fcntl.LOCK_UN)
        self.__print('released mutex%s'%(' (%s)'%(msg) if msg!='' else ''))
