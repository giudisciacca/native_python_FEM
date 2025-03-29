'''
This file contains classes and functions for the solutions of matricial systems in the form A x = f

Author: Giuseppe Di Sciacca
'''
from scipy import sparse
import numpy as np
def sparse_conjugate_gradients_solver(A: sparse.csr_matrix, f: np.ndarray) -> np.ndarray:
    """
    Solve the linear system of equations Ax = f using the Conjugate Gradient method.

    Args:
        A (sparse.csr_matrix): The coefficient matrix in CSR format (shape: N x N).
        f (np.ndarray): The right-hand side vector (shape: N,).

    Returns:
        np.ndarray: The solution vector (shape: N,).
    """
    return sparse.linalg.cg(A, f)[0]
