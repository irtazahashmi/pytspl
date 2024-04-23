"""This module is the main module of the library.

It imports all the necessary modules and classes to be used by the user.
"""

# data reader module
from sclibrary.io import (
    generate_random_simplicial_complex,
    get_coordinates,
    read_csv,
    read_incidence_matrix,
    read_tntp,
)

# plotting module
from sclibrary.plot import SCPlot

# simplicial complex module
from sclibrary.simplicial_complex import SimplicialComplexNetwork
