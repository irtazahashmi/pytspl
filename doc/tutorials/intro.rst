Introduction to PyTSPL
=======================


This tutorial will show you the basic functionality of the toolbox. 
After installing the package with pip, start by opening a python shell, 
e.g. a Jupyter notebook, and import the PyTSPL.

Let's start by building a simplicial complex by using the built-in dataset 
loader. Once we load the dataset, we will get a summary of the simplcial 
complex (SC). Additionally, we will get the coordinates and the flow of the SC.


Loading a SC from a dataset
---------------------------

Before loading a dataset, we can list the available datasets that are 
currently available.

.. code-block:: python

    >>> from pytspl import list_datasets
    >>> list_datasets()


Now, let's load a dataset and get a summary of the simplicial complex.

.. code-block:: python

    >>> from pytspl import load_dataset
    >>> sc, coordinates, flow = load_dataset("paper")


We can plot the network using the SCPlot module.

.. code-block:: python

    >>> from pytspl import SCPlot
    >>> import matplotlib.pyplot as plt

    >>> fig = plt.figure(figsize = (5,5))
    >>> ax = fig.add_subplot(1, 1, 1)

    >>> scplot = SCPlot(simplical_complex=sc, coordinates=coordinates)
    >>> scplot.draw_network(ax=ax)

We can also plot the SC with its edge flow.


.. code-block:: python

    >>> fig = plt.figure(figsize = (6 ,5))
    >>> ax = fig.add_subplot(1, 1, 1)

    >>> scplot.draw_network(edge_flow=flow, ax=ax)

To retrive the properties of the SC, we can use the SimplicialComplex object. 
We can retrieve, for example, adjacency matrix, incidence matrices and the Hodge 
Laplacian matrices using the rank.

.. code-block:: python

    >>> sc.adjacency_matrix()
    >>> sc.incidence_matrices(rank=1)
    >>> sc.hodge_laplacian_matrices(rank=1)


Generate a random SC
------------------------------------

We can also generate a random SC in the following way.

.. code-block:: python

    >>> from pytspl import generate_random_simplicial_complex
    >>> sc, coordinates = generate_random_simplicial_complex(
    ...     num_of_nodes=7,
    ...     p=0.25,
    ...     seed=42,
    ...     dist_threshold=0.8
    ... )

    >>> scplot = SCPlot(sc, coordinates)
    >>> scplot.draw_network()

