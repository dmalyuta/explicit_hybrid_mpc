"""
Tree tools, namely a binary tree class and node data container.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

class NodeData:
    def __init__(self,vertices,commutation=None,vertex_costs=None,vertex_inputs=None):
        """
        Create node data container.
        
        Parameters
        ----------
        vertices : np.array
            2D array where each row is a vertex.
        commutation : np.array (optional)
            1D array that is the commutation.
        vertex_costs : np.array (optional)
            1D array of the same length as vertices.shape[0] of optimal costs
            at the vertices using the commutation.
        vertex_inputs : np.array (optional)
            2D array of the same length as vertices.shape[0] of optimal inputs
            at the vertices using the commutation (row i is for the i-th vertex).
        """
        self.vertices = vertices
        if commutation is not None:
            self.commutation = commutation
        if vertex_costs is not None:
            self.vertex_costs = vertex_costs
        if vertex_inputs is not None:
            self.vertex_inputs = vertex_inputs

class Tree:
    def __init__(self,data,top=True):
        """
        Create root node.
        
        Parameters
        ----------
        data : NodeData
            Data associated with the tree node.
        top : bool, optional
            If ``True``, the object corresponds to the top of the tree.
        """
        self.data = data
        self.top = top
        
    def grow(self,left,right):
        """
        Create a left and a right child.
        
        Parameters
        ----------
        left : NodeData
            Left child's data.
        right : NodeData
            Right child's data.
        """
        self.left = Tree(left,top=False)
        self.right = Tree(right,top=False)
    
    def is_leaf(self):
        """
        Checks if this node is a leaf.
        
        Returns
        -------
        : bool
            ``True`` if is a leaf.
        """
        return not hasattr(self,'left')
        