"""
Partition tools, namely a binary tree class and node data container.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

class NodeData:
    def __init__(self,vertices,commutation=None,vertex_costs=None):
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
        """
        self.vertices = vertices
        if commutation is not None:
            self.commutation = commutation
        if vertex_costs is not None:
            self.vertex_costs = vertex_costs

class Tree:
    def __init__(self,data):
        """
        Create root node.
        
        Parameters
        ----------
        data : NodeData
            Data associated with the tree node.
        """
        self.data = data
        
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
        self.left = Tree(left)
        self.right = Tree(right)
        