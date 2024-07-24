"""This module is the main module of the library.

It imports all the necessary modules and classes to be used by the user.
"""

__version__ = "0.1.0"


# data reader module
from pytspl.io import (
    generate_random_simplicial_complex,
    list_datasets,
    load_dataset,
    read_B1_B2,
    read_csv,
    read_tntp,
)

# plotting module
from pytspl.plot import SCPlot

from . import decomposition  # noqa: F401
from . import filters  # noqa: F401
from . import hogde_gp  # noqa: F401
from . import io  # noqa: F401
from . import plot  # noqa: F401
from . import simplicial_complex  # noqa: F401

# # simplicial complex module
# from pytspl.simplicial_complex import SimplicialComplex
