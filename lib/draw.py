"""
Draw the partition.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import pickle

import global_vars
from mpc_library import SatelliteZ,SatelliteXY
from tools import Animator
from polytope import Polytope

def draw(root,animator,location='',handles=[]):
    """
    Grow an initial binary tree given by root using algorithm() (either
    ``ecc()`` or ``lcss()``).
    
    Parameters
    ----------
    algorithm : callable
        Either ``ecc()`` or ``lcss()``.
    oracle : Oracle
        Container of optimization problems used by the partitioning process.
    node : Tree
        Tree root. Sufficient that it just holds node.data.vertices for the
        simplex vertices.
    **kwargs : dict
        Passed on to algorithm().
    """
    # Animation preferences
    anim_open_leaf = dict(edgecolor='black',facecolor='none',linewidth=0.1)
    # Use "safe" rounding to int (so that conversion to int does not floor 0.99999999 down to 0)
    anim_closed_leaf = lambda delta: dict(facecolor=animator.get_color((delta>0.5).astype(int)))
    
    if root.is_leaf():
        #print(location)
        animator.update(Polytope(V=root.data.vertices,A=False),**anim_open_leaf)#**anim_closed_leaf(root.data.commutation))
    else:
        #animator.update(Polytope(V=root.data.vertices,A=False),**anim_open_leaf)
        draw(root.left,animator,location+'0')
        draw(root.right,animator,location+'1')
        
sat = SatelliteXY()
animator = Animator(1,sat)

partition = pickle.load(open(global_vars.TREE_FILE,'rb'))
draw(partition,animator)