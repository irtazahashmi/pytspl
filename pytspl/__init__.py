"""This module is the main module of the library.

It imports all the necessary modules and classes to be used by the user.
"""

# plotting module
# data reader module
from pytspl.io import (
    generate_random_simplicial_complex,
    list_datasets,
    load_dataset,
    read_B1_B2,
    read_csv,
    read_tntp,
)
from pytspl.plot import SCPlot

# simplicial complex module
from pytspl.simplicial_complex import SimplicialComplex
