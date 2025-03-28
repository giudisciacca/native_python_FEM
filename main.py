import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class Mesh:
    def __init__(self,nodes: np.array, elements: np.array):
        self.nodes = nodes
        self.elements = elements
        return
    def visualise2D(self):
        fig = plt.figure()
        plt.scatter(self.nodes[:,0],self.nodes[:,1])
        for el in self.elements:
            for i in range(3):
                p1, p2 = self.nodes[el[i]], self.nodes[el[(i + 1) % 3]]  # Get edge endpoints
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', alpha=0.6)  # Draw edge
        plt.xlabel('x')
        plt.xlabel('y')
        plt.title('Mesh visualisation')
        return fig

    def visualise3D(self):
        raise NotImplementedError('Method not implemented')




def mesh_rectangle(rectangle_params: list, num_elements_x: int, num_elements_y ):
    """"
    This function get as input the extremes of a rectangular domain, stored as a list [a,b,c,d]
    and returns a mesh
    """
    a,b,c,d = rectangle_params
    x = np.linspace(a,b, num_elements_x, endpoint=True)
    y = np.linspace(c,d, num_elements_y, endpoint=True)
    nodes = np.hstack([mg.reshape(-1,1) for mg in np.meshgrid(x,y)])
    elements = find_triangular_elements_in_rect_domain( [num_elements_x, num_elements_y])
    return nodes, elements


class TriangularElement:
    def __init__(self, indices):
        """Triangle defined by node indices."""
        self.indices = tuple(sorted(indices))  # Sort indices to ensure consistency

    def is_point_in_circumcircle(self, p_idx, nodes):
        """Check if point p is inside the circumcircle of this triangle."""
        ax, ay = nodes[self.indices[0]]
        bx, by = nodes[self.indices[1]]
        cx, cy = nodes[self.indices[2]]
        dx, dy = nodes[p_idx]

        # Compute determinant of the circumcircle matrix
        circum_matrix = np.array([
            [ax - dx, ay - dy, (ax - dx) ** 2 + (ay - dy) ** 2],
            [bx - dx, by - dy, (bx - dx) ** 2 + (by - dy) ** 2],
            [cx - dx, cy - dy, (cx - dx) ** 2 + (cy - dy) ** 2]
        ])
        det = np.linalg.det(circum_matrix)
        return det > 0  # Inside if determinant is positive


def find_triangular_elements_in_rect_domain(size):
    """
    """
    square = divide_in_square(size)
    elements = np.hstack( [np.array([square[0], square[1], square[2]]),
                          np.array([ square[0], square[2], square[3]])]).T



    return elements

def divide_in_square(size ):
    idx = np.arange(0, size[0] * size[1]).reshape(size)
    square = [idx[:-1, :-1].reshape(-1), idx[1:, :-1].reshape(-1), idx[1:, 1:].reshape(-1), idx[:-1, 1:].reshape(-1)]
    return square





def set_basis(mesh, mode='linear'):
    if mode == 'linear':
        return set_linear_basis()
    else:
        raise NotImplementedError('Basis with mode ' + mode + ' is not existing' )

def solve_system():
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
    mesh.visualise2D()
    basis  = set_basis(mesh, 'linear')
    print('End of Code')
