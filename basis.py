'''
This file contains classes for the building of the FEM system

Author: Giuseppe Di Sciacca
'''
from meshing import Mesh
import numpy as np
from scipy import sparse

class Basis:
    def __init__(self, mesh: 'Mesh', mode: str):
        """
        Initialize a Basis function for finite element analysis.

        Args:
            mesh (Mesh): The mesh object containing the nodes and elements.
            mode (str): The type of basis function to use. Currently supports 'linear'.

        Raises:
            NotImplementedError: If an unsupported mode is specified.
        """
        self.mesh = mesh
        if mode == 'linear':
            self.basis_function = self.linear()
        else:
            raise NotImplementedError(f'Basis with mode {mode} is not existing')

    def linear(self) -> 'Linear2DFunction':
        """
        Return a linear 2D basis function.

        Returns:
            Linear2DFunction: An instance of the Linear2DFunction class.
        """
        return Linear2DFunction()


class Linear2DFunction:
    def __init__(self):
        """
        Initialize the linear 2D basis function.
        """
        return

    def _gradients(self, vertices: np.ndarray, areas: np.ndarray) -> np.ndarray:
        """
        Compute the gradients of the linear basis functions for each triangle.

        Args:
            vertices (np.ndarray): The vertex coordinates of the elements (shape: Mx3x2).
            areas (np.ndarray): The areas of the elements (shape: M,).

        Returns:
            np.ndarray: The gradient matrix (shape: Mx3x2).
        """
        gradient_matrix = vertices[:, [2, 0, 1], :] - vertices[:, [1, 2, 0], :]
        return gradient_matrix

    def _element_areas(self, vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Compute the areas of each triangular element.

        Args:
            vertices (np.ndarray): The vertex coordinates of the mesh (shape: N x 3 x 2).
            indices (np.ndarray): The indices of the vertices for each element (shape: M x 3).

        Returns:
            np.ndarray: The areas of the elements (shape: M,).
        """
        x1, y1 = vertices[:, 0, 0], vertices[:, 0, 1]
        x2, y2 = vertices[:, 1, 0], vertices[:, 1, 1]
        x3, y3 = vertices[:, 2, 0], vertices[:, 2, 1]

        areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        return areas

    def _compute_local_stiffnesses(self, mesh: 'Mesh', areas: np.ndarray) -> np.ndarray:
        """
        Compute the local stiffness matrix for each element.

        Args:
            mesh (Mesh): The mesh object containing the mesh data.
            areas (np.ndarray): The areas of the elements (shape: M,).

        Returns:
            np.ndarray: The local stiffness matrices (shape: M x 3 x 3).
        """
        gradients = self._gradients(mesh.vertices, areas)
        local_stiffnesses = np.einsum("nij,nkj->nik", gradients, gradients) * areas[:, None, None]
        return local_stiffnesses

    def apply_dirichlet_boundary_conditions(
        self, K: sparse.csr_matrix, F: np.ndarray, boundary_nodes: np.ndarray, value: float = 0.0
    ) -> tuple:
        """
        Apply Dirichlet boundary conditions (u = value) at specified nodes.

        Args:
            K (sparse.csr_matrix): The stiffness matrix (CSR format).
            F (np.ndarray): The load vector.
            boundary_nodes (np.ndarray): The indices of the nodes with Dirichlet boundary conditions.
            value (float): The prescribed value at the boundary (default: 0.0).

        Returns:
            tuple: The modified stiffness matrix and load vector after applying the boundary conditions.
        """

    def apply_dirichlet_boundary_conditions(
            self, K: sparse.csr_matrix, F: np.ndarray, boundary_nodes: np.ndarray, value: float = 0.0
    ) -> tuple:
        """
        Apply Dirichlet boundary conditions (u = value) at specified nodes.

        Args:
            K (sparse.csr_matrix): The stiffness matrix (CSR format).
            F (np.ndarray): The load vector.
            boundary_nodes (np.ndarray): The indices of the nodes with Dirichlet boundary conditions.
            value (float): The prescribed value at the boundary (default: 0.0).

        Returns:
            tuple: The modified stiffness matrix and load vector after applying the boundary conditions.
        """
        boundary_nodes = np.asarray(boundary_nodes).flatten()  # Convert to array and flatten

        # Ensure K is in CSR format
        if not isinstance(K, sparse.csr_matrix):
            K = K.tocsr()

        if not isinstance(K, sparse.csr_matrix):
            K = K.tocsr()

        K[boundary_nodes, :] = 0  # Set the entire row to 0
        K[:, boundary_nodes] = 0  # Set the entire column to 0

            # Step 2: Set the diagonal of boundary nodes to 1
        K[boundary_nodes, boundary_nodes] = 1  # Set the diagonal entries to 1


        # Step 3: Set the boundary values in the load vector
        F[boundary_nodes] = value

        return K, F

    def build_freespace_stiffness_poisson(self, mesh: 'Mesh', areas: np.ndarray) -> sparse.csr_matrix:
        """
        Build the stiffness matrix for the Poisson equation -∇²u = f in the free space (no boundary conditions).

        Args:
            mesh (Mesh): The mesh object containing the nodes and elements.
            areas (np.ndarray): The areas of the elements (shape: M,).

        Returns:
            sparse.csr_matrix: The sparse stiffness matrix in CSR format.
        """
        local_stiffnesses = self._compute_local_stiffnesses(mesh, areas).ravel()
        i_idx = np.repeat(mesh.elements, 3, axis=1).ravel()
        j_idx = np.tile(mesh.elements, (1, 3)).ravel()

        stiffness = sparse.coo_matrix((local_stiffnesses, (i_idx, j_idx)), shape=(mesh.numnd, mesh.numnd)).tocsr()
        return stiffness

    def build_load_vector(self, mesh: 'Mesh', areas: np.ndarray, f: callable) -> np.ndarray:
        """
        Build the load vector for the Poisson equation - div u = f using the finite element method.

        Args:
            mesh (Mesh): The mesh object containing the nodes and elements.
            areas (np.ndarray): The area of the elements.
            f (callable): The function representing the source term to project onto the mesh.

        Returns:
            np.ndarray: The global load vector (shape: N,).
        """
        nodes = mesh.nodes.copy()
        elements = mesh.elements.copy()
        num_nodes = mesh.numnd

        f_values = f(nodes[:, 0], nodes[:, 1])


        load_vector = np.zeros(num_nodes)
        np.add.at(load_vector, elements[:, 0], f_values[elements[:, 0]]*areas )
        np.add.at(load_vector, elements[:, 1], f_values[elements[:, 1]]*areas)
        np.add.at(load_vector, elements[:, 2], f_values[elements[:, 2]]*areas)

        return load_vector

    def apply_boundary_conditions(self, free_stiffness: sparse.csr_matrix, free_load: np.ndarray, mesh: 'Mesh', mode: str = 'dirichlet') -> tuple:
        """
        Apply boundary conditions to the free stiffness matrix and load vector.

        Args:
            free_stiffness (sparse.csr_matrix): The stiffness matrix (CSR format).
            free_load (np.ndarray): The load vector.
            mode (str): The type of boundary condition to apply (default: 'dirichlet').

        Returns:
            tuple: The modified stiffness matrix and load vector after applying the boundary conditions.
        """
        if mode == 'dirichlet':
            stiffness, load = self.apply_dirichlet_boundary_conditions(free_stiffness, free_load, mesh.boundary_nodes)
        else:
            raise NotImplementedError
        return stiffness, load

    def build_poisson_system(self, mesh: 'Mesh', load_function: callable, boundary_conditions: str) -> tuple:
        """
        Build the Poisson system (stiffness matrix and load vector).

        Args:
            mesh (Mesh): The mesh object containing the nodes and elements.
            load_function (callable): The source term function to project onto the mesh.
            boundary_conditions (str): The type of boundary conditions to apply.

        Returns:
            tuple: The stiffness matrix and load vector for the Poisson problem.
        """
        indices = np.arange(0, mesh.numel)
        areas = self._element_areas(mesh.vertices, indices)
        free_stiffness = self.build_freespace_stiffness_poisson(mesh, areas)
        free_load = self.build_load_vector(mesh, areas, load_function)
        A, f = self.apply_boundary_conditions(free_stiffness, free_load, mesh, mode=boundary_conditions)
        return A, f
