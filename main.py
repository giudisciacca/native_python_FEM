import numpy as np
import scipy

class Mesh:
    def __init__(self,nodes: np.array, elements: np.array):
        self.nodes = nodes
        self.elements = elements
        return
    def visualise2D(self):
        plt.figure()
        plt.scatter(self.nodes)
        plt.xlabel('x')
        plt.xlabel('y')
        plt.title('Mesh visualisation')
        return
    def visualise3D(self):
        raise NotImplementedError('Method not implemented')


def mesh_rectangle(rectangle_params: list, num_elements_x: int, num_elements_y ):
    """"
    This function get as input the extremes of a rectangular domain, stored as a list [a,b,c,d]
    and returns a mesh
    """
    a,b,c,d = rectangle_params
    x = np.linspace(a,b, num_elements_x)
    y = np.linspace(c,d, num_elements_y)
    nodes = np.hstack([mg.reshape(-1,1) for mg in np.meshgrid(x,y)])
    elements = find_triangular_elements(nodes)
    return nodes, elements

def find_triangular_elements(nodes: np.array):
    """"Given the nodes of a matrix defines a set of triangular elements"""
    if nodes.shape[1] != 2: raise IndexError('Second dimension of node array should be 2 for 2D elements')
    elements = np.zeros((nodes.shape[0],3))
    

    return elements

def set_basis(mesh, mode='linear'):
    if mode == 'linear':
        return set_linear_basis()
    else:
        raise NotImplementedError('Basis with mode ' + mode + ' is not existing' )

def solve_system()
    return
    
def get_stiffness_matrix():
    return
    

if __name__=='__main__':

    import matplotlib.pyplot as plt


    a = 0
    b = 1
    c = 0
    d = 1
    nx = 10
    ny = 10
    mesh = Mesh(*mesh_rectangle([a,b,c,d], nx, ny))
    basis  = set_basis(mesh, 'linear')
