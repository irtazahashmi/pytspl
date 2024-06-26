"""This module is the main module of the library.

It imports all the necessary modules and classes to be used by the user.
"""

# data reader module
from sclibrary.io import (
    generate_random_simplicial_complex,
    read_B1_B2,
    read_csv,
    read_tntp,
)

# plotting module
from sclibrary.io.dataset_loader import dataset_loader
from sclibrary.plot import SCPlot

# simplicial complex module
from sclibrary.simplicial_complex import SimplicialComplex
