Operations with Simplical Complexes
===================================

This tutorial will give an overview of the operations possible with 
simplicial complexes.

Simplicial Shifting
-------------------
In simplical shifting, we apply :math:`\mathbf{H}_k` to a 
:math:`k`-simplicial signal :math:`\mathbf{s}^k`, we shift the 
signal :math:`L` times over the lower or upper neighbourhoods. 
Let's go ahead and apply simplicial shifting to a simplical complex.

Let's start by loading a dataset and getting a summary of the simplicial complex.

.. plot::
    :context: close-figs

    >>> from pytspl import load_dataset
    >>> import matplotlib.pyplot as plt
    >>>
    >>> sc, coordinates, flow = load_dataset("paper")    

Now, we create a synthetic flow.

.. plot::
    :context: close-figs

    >>> synthetic_flow = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

In this example, we shift the signal :math:`L_1` times over the lower 
neighbourhoods by applying the function :func:`apply_lower_shifting`.

.. plot::
    :context: close-figs

    >>> from pytspl import SCPlot
    >>>
    >>> scplot = SCPlot(simplical_complex=sc, coordinates=coordinates)
    >>> fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    >>>
    >>> # plot indicator flow f
    >>> axs[0].set_title("Indicator flow $\mathbf{f}$")
    >>> scplot.draw_network(edge_flow=synthetic_flow, ax=axs[0])
    >>>
    >>> # apply one step lower shifting
    >>> steps = 1
    >>> flow = sc.apply_lower_shifting(synthetic_flow, steps=steps)
    >>>
    >>> axs[1].set_title(r"One step lower shifting - $L_{1, l} \ \ \mathbf{f}$")
    >>> scplot.draw_network(edge_flow=flow, ax=axs[1])
    >>>
    >>> # apply two steps lower shifting
    >>> steps = 2
    >>> flow = sc.apply_lower_shifting(synthetic_flow, steps=steps)
    >>>
    >>> axs[2].set_title(r"Two steps lower shifting - $L_{1, l}^2 \ \ \mathbf{f}$")
    >>> scplot.draw_network(edge_flow=flow, ax=axs[2])


Similiarly, we can apply the function :func:`apply_upper_shifting` to
shift the signal :math:`L_2` times over the upper neighbourhoods.

We can also apply :math:`k`-step simplicial shifting.

.. plot::
    :context: close-figs

    >>> # apply k-step shifting
    >>> k = 3
    >>> flow = sc.apply_k_step_shifting(synthetic_flow)
    >>>
    >>> fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    >>> ax.set_title(r"$k$-step shifting for $k = 3$")
    >>> scplot.draw_network(edge_flow=flow, ax=ax)



Simplicial Embeddings and Simplical Fourier Transform (SFT)
-----------------------------------------------------------

Given a flow :math:`\mathbf{f}`, we can extract the harmonic, curl, and gradient 
embeddings. Such embeddings represent a compressed representation of the edge flow.

The Simplicial Fourier Transform of flow :math:`\mathbf{f}` can be defined as 
:math:`\tilde{\mathbf{f}} = \left[ \tilde{\mathbf{f}}_H^\top, \tilde{\mathbf{f}}_G^\top, \tilde{\mathbf{f}}_C^\top \right]^\top`

Each of the embeddings represents the weight of the flow on the corresponding 
eigenvector.

.. plot::
    :context: close-figs

    >>> # define a synthetic flow
    >>> synthetic_flow = [0.03, 0.5, 2.38, 0.88, -0.53, -0.52, 1.08, 0.47, -1.17, 0.09] 

    >>> # get the simplicial embeddings for hamonic, curl and gradient
    >>> f_tilda_h, f_tilda_c, f_tilda_g = sc.get_simplicial_embeddings(synthetic_flow)
    >>>
    >>> print("embedding_h:", f_tilda_h)
    >>> print("embedding_g:", f_tilda_g)
    >>> print("embedding_c:", f_tilda_c)
    embedding_h: [-1.00084785]
    embedding_g: [-1.00061494 -1.00127703  1.00173495 -1.00287539  0.99531105  1.00412064]
    embedding_c: [-1.          0.99881597  0.99702056]