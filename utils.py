'''
This file contains classes and functions for the validation of FEM solutions of the Poisson Equation

Author: Giuseppe Di Sciacca
'''
import numpy as np

class ValidateAgainstAnalytical:
    def __init__(self):
        """
        Initialize the TestAgainstAnalytical class.
        This class provides methods to compare numerical solutions with analytical solutions.
        """
        return

    @staticmethod
    def analytical_solution_1x1_square(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the analytical solution for a 1x1 square domain.

        Args:
            x (np.ndarray): The x-coordinates of the mesh points (shape: N,).
            y (np.ndarray): The y-coordinates of the mesh points (shape: N,).

        Returns:
            np.ndarray: The analytical solution at each mesh point (shape: N,).
        """
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    @staticmethod
    def load_function_analytical(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the analytical source term (load function) for a 1x1 square domain.

        Args:
            x (np.ndarray): The x-coordinates of the mesh points (shape: N,).
            y (np.ndarray): The y-coordinates of the mesh points (shape: N,).

        Returns:
            np.ndarray: The load function at each mesh point (shape: N,).
        """
        return 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
