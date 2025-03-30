# native python FEM
Set of code for the implementation of FEM on python native libraries

Requirements are:
numpy
scipy
matplotlib

The code leverages the fast meshing of domains that can result under 
the assumption of well-behaved rectangular domain

The approach is fully vectorised and makes use of the most common
functions in numpy and scipy (for solution of sparse systems and interpolation)

The code is subdivided in four modules:
- meshing, which takes care of meshing procedures and operations, including visualisation
- basis, which deals with building the FEM system
- solvers, which gathers utilities for the solution of linear systems in the form A x = b
- utils, utilities used to check the convergence of the FEM solution w.r.t. to known analytical solution

Running python main.py from terminal runs the main functions and provides and example of the capabilities of the techniques
Timing assessment has not been provided due to the constraints on the allowed libraries.

Future work should move towards meshing for any domain and towards the solution of other PDE,
starting from parabolic equations and anisotropic diffusion.


