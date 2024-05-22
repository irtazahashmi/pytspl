"""This module is the main module of the library.

It imports all the necessary modules and classes to be used by the user.
"""

# data reader module
from sclibrary.io import (
    dataset_loader,
    generate_random_simplicial_complex,
    read_B1,
    read_csv,
    read_tntp,
)

# plotting module
from sclibrary.plot import SCPlot

# simplicial complex module
from sclibrary.simplicial_complex import SimplicialComplex
