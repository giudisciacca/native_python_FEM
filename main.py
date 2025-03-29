'''
This files gather a set of function for the finite element solution of the Poisson equation

Author: Giuseppe Di Sciacca
'''


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class Mesh:
    def __init__(self,nodes: np.array, elements: np.array):
        self.nodes = nodes
        self.elements = elements
        self.numel = len(elements)
        self.numnd = len(nodes)
        self.boundary_nodes = self.find_boundary_nodes()
        self.vertices = self.nodes[self.elements]
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

    def find_boundary_nodes(self):
        """Find boundary nodes of a triangular mesh.

        :param elements: (M, 3) array of element connectivity (node indices for each triangle).
        :return: Set of boundary node indices.
        """
        # Create a dictionary to store edges and their occurrences

        edge_count = {}

        # Iterate through all elements and extract edges
        for elem in self.elements:
            # Create the 3 edges for each triangle (elements are assumed to be 3-node triangles)
            edges = [(elem[i], elem[(i + 1) % 3]) for i in range(3)]

            for edge in edges:
                # Sort edges to avoid directional issues (e.g., (0, 1) and (1, 0) should be the same)
                edge = tuple(sorted(edge))

                if edge not in edge_count:
                    edge_count[edge] = 0
                edge_count[edge] += 1

        # Find boundary edges (those that appear exactly once)
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        # Extract nodes that appear in boundary edges
        boundary_nodes = set()
        for edge in boundary_edges:
            boundary_nodes.update(edge)

        return list(boundary_nodes)



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



class Basis:
    def __init__(self, mesh, mode):
        self.mesh = mesh
        if mode == 'linear':
            self.basis_function = self.linear()
        else:
            raise NotImplementedError('Basis with mode ' + mode + ' is not existing' )

    def linear(self):
        return Linear2DFunction()


class Linear2DFunction:
    def __init__(self):
        return

    def _gradients(self, vertices,areas):
        gradient_matrix=  vertices[:,[2,0,1] ,:] - vertices[:,[1,2,0] ,:]
        return gradient_matrix


    def _element_areas(self,vertices, indices):
        areas = 0.5 * np.abs(
            (vertices[indices, 0, 0] - vertices[indices, 2, 0]) *
            (vertices[indices, 1, 1] - vertices[indices, 0, 1]) -
            (vertices[indices, 0, 0] - vertices[indices, 1, 0]) *
            (vertices[indices, 2, 1] - vertices[indices, 0, 1]))
        return areas

    def _compute_local_stiffnesses(self, mesh, areas):

        gradients = self._gradients(mesh.vertices, areas)

        local_stiffnesses = np.einsum("nij,nkj->nik", gradients, gradients) * areas[:, None, None]
        return local_stiffnesses

    def apply_dirichlet_boundary_conditions(self, K, F, boundary_nodes, value=0.0):
        """
        Apply Dirichlet boundary conditions (u = value) at specified nodes.

        :param K: Stiffness matrix (CSR format).
        :param F: Load vector.
        :param boundary_nodes: List of node indices with Dirichlet boundary conditions.
        :param value: The prescribed value at the boundary (default: 0.0).
        """
        # Ensure boundary_nodes is a numpy array (convert from set if needed)
        boundary_nodes = np.asarray(boundary_nodes).flatten()  # Convert to array and flatten

        # Step 1: Convert K to CSR format if it's not already
        if not isinstance(K, sparse.csr_matrix):
            K = K.tocsr()  # Convert COO to CSR format for efficient operations

        # Step 2: Create a mask for rows and columns corresponding to boundary nodes
        boundary_mask_rows = np.isin(K.indices, boundary_nodes)  # Mask for rows
        boundary_mask_cols = np.isin(K.indices, boundary_nodes)  # Mask for columns

        # Step 3: Set corresponding rows and columns to zero (except diagonal)
        K.data[boundary_mask_rows | boundary_mask_cols] = 0  # Set non-diagonal elements to 0

        # Step 4: Set diagonal entries corresponding to boundary nodes to 1
        # First, create an array of the indices of the diagonal elements
        row_indices = np.repeat(np.arange(K.shape[0]),
                         np.diff(K.indptr))  # Repeat each row index based on the count of non-zeros in that row
        col_indices = K.indices
        diag_mask = (row_indices == col_indices) & np.isin(row_indices, boundary_nodes)

        # Apply the mask to set diagonal entries to 1
        K.data[diag_mask] = 1
        # Step 5: Modify the load vector entries corresponding to boundary nodes to the prescribed value
        F[boundary_nodes] = value  # Direct assignment of the prescribed boundary value
        return K, F

    def build_freespace_stiffness_poisson(self, mesh, areas):

        local_stiffnesses = self._compute_local_stiffnesses(mesh, areas).ravel()
        i_idx = np.repeat(mesh.elements, 3, axis=1).ravel()
        j_idx = np.tile(mesh.elements, (1, 3)).ravel() # (M, 1, 3)

        # Build sparse matrix in COO format
        stiffness = sparse.coo_matrix((local_stiffnesses, (i_idx, j_idx)), shape=(mesh.numnd, mesh.numnd)).tocsr()

        return stiffness

    def build_load_vector(self, mesh, area, f):
        """
        Compute the load vector for the Poisson equation -∇²u = f using FEM.

        :param mesh: Mesh object containing nodes and elements.
        :param f: Function representing the source term to project onto the mesh.
        :return: Global load vector (right-hand side) for the FEM problem.
        """
        # Get the mesh properties
        nodes = mesh.nodes  # Shape (num_nodes, 2)
        elements = mesh.elements  # Shape (num_elements, 3)
        num_nodes = mesh.numnd

        # Evaluate the source function f at the node positions
        f_values = f(nodes[:, 0].copy(), nodes[:, 1].copy())  # Shape (num_nodes,)


        # Compute the function values at the element nodes (vectorized)
        f_values_element = np.vstack([f_values[elements[:, 0]],
                                      f_values[elements[:, 1]],
                                      f_values[elements[:, 2]]]).T  # Shape (num_elements, 3)

        # Compute the contribution of each element to the global load vector
        load_vector = np.zeros(num_nodes)  # Shape (num_nodes,)
        np.add.at(load_vector, elements[:, 0], f_values_element[:, 0] * area)
        np.add.at(load_vector, elements[:, 1], f_values_element[:, 1] * area)
        np.add.at(load_vector, elements[:, 2], f_values_element[:, 2] * area)

        return load_vector

    def apply_boundary_conditions(self,free_stiffness, free_load, mode='dirichlet'):
        if mode=='dirichlet':
            stiffness, load = self.apply_dirichlet_boundary_conditions(free_stiffness, free_load, mesh.boundary_nodes)
        else:
            raise NotImplementedError
        return stiffness, load

    def build_poisson_system(self, mesh, load_function, boundary_conditions):
        indices = np.arange(0,mesh.numel)
        areas = self._element_areas(mesh.vertices, indices)
        free_stiffness =  self.build_freespace_stiffness_poisson(mesh,areas)
        free_load = self.build_load_vector(mesh,areas, load_function)
        A, f = self.apply_boundary_conditions(free_stiffness, free_load, mode = boundary_conditions)
        return A, f


def solve_system(A, f):

    return sparse.linalg.cg(A,f)[0]
    

if __name__=='__main__':

    import matplotlib.pyplot as plt


    a = 0
    b = 10
    c = 0
    d = 10
    nx = 100
    ny = 100
    mesh = Mesh(*mesh_rectangle([a,b,c,d], nx, ny))
    #mesh.visualise2D()
    basis = Basis(mesh,'linear')
    system = basis.basis_function.build_poisson_system(mesh, np.sin, 'dirichlet')
    result = solve_system(*system)

    system = basis.basis_function.build_poisson_system(mesh, lambda x,y: np.sin(y)+np.cos(x)**2, 'dirichlet')
    result = solve_system(*system)

    print('End of Code')
