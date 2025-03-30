'''
This file contains the functions and classes related to the meshing of a squared domain

Author: Giuseppe Di Sciacca
'''
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Mesh:
    def __init__(self, nodes: np.ndarray, elements: np.ndarray) -> None:
        """
        Initialize the Mesh object with nodes and elements.

        Args:
            nodes (np.ndarray): Array of node coordinates with shape (num_nodes, 2).
            elements (np.ndarray): Array of elements (triangular connectivity) with shape (num_elements, 3).
        """
        self.nodes = nodes
        self.elements = elements
        self.numel = len(elements)
        self.numnd = len(nodes)
        self.boundary_nodes = self.find_boundary_nodes()
        self.vertices = self.nodes[self.elements]

    def visualise2D(self, f: np.ndarray = None) -> plt.Figure:
        """
        Visualize the mesh in 2D. Optionally, visualize a function on the mesh.

        Args:
            f (np.ndarray, optional): Array of function values at the mesh nodes. Defaults to None.

        Returns:
            plt.Figure: The created Matplotlib figure object.
        """
        fig, ax = plt.subplots()
        if f is not None:
            # Define a regular grid
            resolution = int(np.ceil(np.sqrt(self.numnd)))
            x_min, x_max = self.nodes[:, 0].min(), self.nodes[:, 0].max()
            y_min, y_max = self.nodes[:, 1].min(), self.nodes[:, 1].max()
            x_grid, y_grid = np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
            xx, yy = np.meshgrid(x_grid, y_grid)

            # Interpolate function values on the grid
            zz = griddata(self.nodes, f, (xx, yy), method='linear')

            # Plot using imshow
            im = ax.imshow(zz, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis', alpha=0.8)
            plt.colorbar(im, ax=ax)
            ax.set_title('Function Visualisation')
        else:
            # Otherwise, just plot the nodes without coloring
            ax.scatter(self.nodes[:, 0], self.nodes[:, 1], color='black')
            ax.scatter(self.nodes[self.boundary_nodes[:],0], self.nodes[self.boundary_nodes[:],1], color='yellow')
            ax.set_title('Mesh Visualisation')

            # Plot the mesh edges
            edges = np.column_stack([self.elements[:, [0, 1]], self.elements[:, [1, 2]], self.elements[:, [2, 0]]]).reshape(-1, 2)

            # Extract edge coordinates
            x_coords = self.nodes[edges, 0]
            y_coords = self.nodes[edges, 1]

            # Plot all edges at once
            ax.plot(x_coords.T, y_coords.T, 'g-', alpha=0.6)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
        plt.show(block=False)
        return fig

    def visualise3D(self) -> None:
        """
        3D visualization method (currently not implemented).

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError('Method not implemented')

    def find_boundary_nodes(self) -> list[int]:
        """
        Find boundary nodes of a triangular mesh using vectorized operations.

        Returns:
            list[int]: List of node indices that form the boundary of the mesh.
        """
        # Extract edges for all elements in a vectorized manner
        edges = np.sort(
            np.column_stack([self.elements[:, [0, 1]], self.elements[:, [1, 2]], self.elements[:, [2, 0]]])
            .reshape(-1, 2), axis=1)

        edges_tuple = np.ascontiguousarray(edges).view([('', edges.dtype)] * 2)
        unique_edges, counts = np.unique(edges_tuple, return_counts=True)
        boundary_edges = unique_edges[counts == 1].view(edges.dtype).reshape(-1, 2)
        boundary_nodes = np.unique(boundary_edges)

        return boundary_nodes.tolist()

    def _circum_incircle_ratio(self, nodes: np.ndarray, selected_elements: np.ndarray)->np.ndarray:
        """Compute the ratio of circumcircle radius to incircle radius for a given triangle.
        Args:
            nodes (np.array): nodes of the mesh
            selected_elements (np.ndarray): Indices of elements to be split.

        Returns:
            np.array containing the ratios of circumradii and inradii of the selected elements
        """
        A, B, C = nodes[selected_elements[:,0]],nodes[selected_elements[:,1]],nodes[selected_elements[:,2]]
        a, b, c = np.linalg.norm(B - C, axis = 1), np.linalg.norm(A - C, axis = 1), np.linalg.norm(A - B, axis = 1)
        s = (a + b + c) / 2
        inradius = np.sqrt((s - a) * (s - b) * (s - c) / s)
        circumradius = (a * b * c) / (4 * inradius * s)
        return circumradius / inradius

    def _edges_midpoint(self, nodes: np.ndarray, selected_elements: np.ndarray)->np.ndarray:
        """Compute the incentre for a given set of triangles identified by mesh nodes and selected elements.
        Args:
            nodes (np.array): nodes of the mesh
            selected_elements (np.ndarray): Indices of elements to be split.

        Returns:
            np.array containing incentre of the selected elements
        """
        A, B, C = nodes[selected_elements[:,0]],nodes[selected_elements[:,1]],nodes[selected_elements[:,2]]
        edges_mid = np.vstack([0.5 * (A + B), 0.5 * (A + C), 0.5 * (B + C)])
        return edges_mid

    def refine_mesh(self, elements_to_split: np.ndarray, tol = 1e-6) -> 'Mesh':
        """
        Refine specified elements by adding a new node at the centroid and splitting each triangle into three.
        Ensures the circumcircle-to-incircle ratio remains constant.

        Args:
            elements_to_split (np.ndarray): Indices of elements to be split.
            tol (float, optional): tolerance for the new ratio of incirce and circumcircle radii

        Returns:
            Mesh: A new refined Mesh instance.
        """
        nodes = self.nodes.copy()
        num_old_nodes = self.numnd
        if elements_to_split.ndim == 1:
            elements_to_split = elements_to_split.reshape((1,-1))
        selected_ratios = self._circum_incircle_ratio(nodes,elements_to_split)

        new_nodes = self._edges_midpoint(nodes, elements_to_split)
        nodes = np.vstack([nodes, new_nodes])
        new_indices = num_old_nodes + np.arange(0,len(new_nodes))

        new_elements = np.column_stack([
            elements_to_split[:, 0], new_indices[0::3], new_indices[1::3],  # First triangle
            elements_to_split[:, 1], new_indices[2::3], new_indices[0::3],  # Second triangle
            elements_to_split[:, 2], new_indices[1::3], new_indices[2::3],  # Third triangle
            new_indices[0::3], new_indices[1::3], new_indices[2::3]  # Center triangle
        ]).reshape(-1, 3)

        updated_ratios = self._circum_incircle_ratio(nodes, new_elements)
        augmented_ratios = np.repeat(selected_ratios, 4)

        # check that the ratio is kept constant
        idxs = np.abs( (augmented_ratios - updated_ratios)/updated_ratios)>tol
        if np.sum(idxs)>0:
            raise ValueError('A number of {} elements did not satisfy convergence theory'.format(np.sum(idxs)))

        # remove selected elements
        self.elements = np.delete(self.elements, np.where(np.isin(self.elements, elements_to_split).all(axis=1))[0],
                                  axis=0)
        elements = np.vstack([self.elements, new_elements])  # Add the new elements to the mesh

        return Mesh(nodes, elements)


def mesh_rectangle(rectangle_params: list, num_elements_x: int, num_elements_y: int) -> tuple:
    """
    Generate a triangular mesh for a rectangular domain.

    This function takes the extremes of a rectangular domain as a list `[a, b, c, d]`, where:
    - `a` is the lower x-boundary,
    - `b` is the upper x-boundary,
    - `c` is the lower y-boundary,
    - `d` is the upper y-boundary.
    The function then generates a grid of nodes with the specified number of elements along both x and y axes.
    Finally, it computes and returns the mesh nodes and the triangular elements within the rectangular domain.

    Args:
        rectangle_params (list): A list containing the rectangular domain boundaries `[a, b, c, d]`.
        num_elements_x (int): The number of elements in the x-direction.
        num_elements_y (int): The number of elements in the y-direction.

    Returns:
        tuple: A tuple containing two elements:
            - `nodes` (np.ndarray): The mesh nodes as an (N, 2) array, where N is the total number of nodes.
            - `elements` (np.ndarray): The triangular elements, represented by node indices for each triangle.
    """
    a, b, c, d = rectangle_params
    x = np.linspace(a, b, num_elements_x, endpoint=True)
    y = np.linspace(c, d, num_elements_y, endpoint=True)
    nodes = np.hstack([mg.reshape(-1, 1) for mg in np.meshgrid(x, y)])
    elements = find_triangular_elements_in_rect_domain([num_elements_y, num_elements_x])
    return nodes, elements


def find_triangular_elements_in_rect_domain(size: list) -> np.ndarray:
    """
    Find triangular elements in a rectangular domain.

    Given the size of the rectangular mesh (in terms of the number of elements in the x and y directions),
    this function divides the domain into square elements and then generates two triangles per square element.

    Args:
        size (list): A list containing the number of elements along the x and y directions `[num_elements_x, num_elements_y]`.

    Returns:
        np.ndarray: A (M, 3) array representing the triangular elements, where M is the total number of triangles.
    """
    square = divide_in_square(size)
    elements = np.hstack([
        np.array([square[0], square[1], square[2]]),
        np.array([square[0], square[2], square[3]])
    ]).T
    return elements


def divide_in_square(size: list) -> list:
    """
    Divide a rectangular grid into square elements.

    This function divides the grid defined by the given number of elements in the x and y directions into square elements
    and returns the indices of the four corners of each square.

    Args:
        size (list): A list containing the number of elements along the x and y directions `[num_elements_x, num_elements_y]`.

    Returns:
        list: A list containing the indices of the four corners of each square.
              Each element in the list corresponds to one of the corners of the square, and the corners are indexed
              in the following order: top-left, top-right, bottom-right, bottom-left.
    """
    idx = np.arange(0, size[0] * size[1]).reshape(size)
    square = [
        idx[:-1, :-1].reshape(-1),
        idx[1:, :-1].reshape(-1),
        idx[1:, 1:].reshape(-1),
        idx[:-1, 1:].reshape(-1)
    ]
    return square
