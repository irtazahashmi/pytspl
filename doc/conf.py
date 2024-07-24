# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import pytspl

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))


project = "PyTSPL"
copyright = "2024, Irtaza Hashmi"
author = "Irtaza Hashmi"
version = pytspl.__version__
release = pytspl.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.mathjax",
]

# autoapi configurations
extensions.append("sphinx.ext.autodoc")
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "groupwise",
}

autoapi_type = "python"
autoapi_dirs = ["../pytspl"]
autoapi_template_dir = "_templates/autosummary"

# plotting
extensions.append("matplotlib.sphinxext.plot_directive")
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_working_directory = "."
plot_rcparams = {"figure.figsize": (5, 5)}
plot_pre_code = """
import numpy as np
import matplotlib.pyplot as plt
from pytspl import load_dataset, SCPlot
"""


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
pygments_style = "sphinx"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 2,
}
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
}
