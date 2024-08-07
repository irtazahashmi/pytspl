=========================================================
Loading Custom Datasets and Building a Simplicial Complex
=========================================================

In this tutorial, we provide examples to read data and build a simplicial 
complex using data in different formats. These formats include CSV files,
TNTP files and incidence matrices. We also show how to load coordinates and
edge flow data from custom datasets.

CSV and TNTP Format
-------------------

Here, we load a CSV file containing edges defined over a network. The function 
processes the data to identify source and target nodes, extracts features, and 
builds a simplicial complex (SC). The summary of the SC is printed, providing
overview of the structure and properties of the SC, 
including the number of nodes, edges, and higher-dimensional simplices.


>>> from pytspl import read_csv
>>>
>>> # define the file path and the columns
>>> PAPER_DATA_FOLDER = "pytspl/data/paper_data"
>>> filename = f"{PAPER_DATA_FOLDER}/edges.csv"
>>> delimiter = " "
>>> src_col = "Source"
>>> dest_col = "Target"
>>> feature_cols = ["Distance"]
>>>
>>> # reading a csv file
>>> sc = read_csv(
>>>        filename=filename,
>>>        delimiter=delimiter,
>>>        src_col=src_col,
>>>        dest_col=dest_col,
>>>        feature_cols=feature_cols
>>>    ).to_simplicial_complex(condition="all")
>>>
>>> # print the summary
>>> sc.print_summary()
Num. of nodes: 7
Num. of edges: 10
Num. of triangles: 3
Shape: (7, 10, 3)
Max Dimension: 2

Similarly, the data can be loaded from a TNTP format using the :func:`read_tntp` 
function.


Incidence matrices 
------------------
The data can be directly loaded from incidence matrices :math:`\textbf{B}_1` 
and :math:`\textbf{B}_2`. The triangles (2-simplices) are extracted from 
the :math:`\textbf{B}_2` matrix.


An incidence matrix is a matrix that shows the relationship between 
two classes of objects. In the context of simplicial complexes, the 
incidence matrix :math:`\textbf{B}_1` represents the relationship 
between vertices and edges, that is, its is a node-to edge incidence
matrix. 

Similarly, the incidence matrix :math:`\textbf{B}_2` represents the 
relationship between edges and triangles (2-simplices), that is, its
is an edge-to-triangle incidence matrix.

>>> from pytspl import read_B1_B2
>>>
>>> B1_filename = f"{PAPER_DATA_FOLDER}/B1.csv"
>>> B2_filename = f"{PAPER_DATA_FOLDER}/B2t.csv"
>>>
>>> # extract the triangles
>>> scbuilder, triangles = read_B1_B2(
>>>     B1_filename=B1_filename,
>>>     B2_filename=B2_filename
>>> )

Now we can build the SC using the extracted triangles.

>>> sc = scbuilder.to_simplicial_complex(triangles=triangles)
>>> sc.print_summary()
Num. of nodes: 7
Num. of edges: 10
Num. of triangles: 3
Shape: (7, 10, 3)
Max Dimension: 2


Building a simplicial complex
-----------------------------

The *triangle-based* method finds all the triangles in the graph and 
considers them as 2-simplices. The *distance-based*
method finds all the triangles and only keeps the ones where 
the distance between the nodes is less than a threshold :math:`\epsilon`. 
By default, when we load a dataset using the :func:`load_dataset`
function, the SC is built using the triangle-based method. 

In this first example, we build the SC by finding all the 
triangles and consider them as 2-simplices.

>>> # build a SC using the triangle-based method
>>> sc = read_csv(
>>>        filename=filename,
>>>        delimiter=delimiter,
>>>        src_col=src_col,
>>>        dest_col=dest_col,
>>>        feature_cols=feature_cols
>>>      ).to_simplicial_complex(condition="all")
>>>
>>> sc.print_summary()
Num. of nodes: 7
Num. of edges: 10
Num. of triangles: 3
Shape: (7, 10, 3)
Max Dimension: 2

In the second example, we build a SC using the distance-based method
and define :math:`\epsilon`. In this case, we get one less triangle (2-simplex).

>>> # build a SC using the distance-based method
>>> sc = read_csv(
>>>        filename=filename,
>>>        delimiter=delimiter,
>>>        src_col=src_col,
>>>        dest_col=dest_col,
>>>        feature_cols=feature_cols
>>>    ).to_simplicial_complex(
>>>        condition="distance",
>>>        dist_col_name="Distance",
>>>        dist_threshold=1.5
>>>    )
>>>
>>> sc.print_summary()
Num. of nodes: 7
Num. of edges: 10
Num. of triangles: 2
Shape: (7, 10, 2)
Max Dimension: 2


Loading coordinates and edge flow from data
-------------------------------------------

We can also load coordinates the coordinates of the nodes
and the edge flow data from custom datasets. The following example
shows how to load the coordinates of the nodes


>>> from pytspl.io.network_reader import read_coordinates, read_flow
>>>
>>> coordinates_path = f"{PAPER_DATA_FOLDER}/coordinates.csv"
>>>
>>> # load the coordinates
>>> coordinates = read_coordinates(
>>>     filename=coordinates_path,
>>>     node_id_col="Id",
>>>     x_col="X",
>>>     y_col="Y",
>>>     delimiter=" "
>>> )
>>>
>>> print(coordinates)
{0: (0, 0.0), 1: (1, -0.5), 2: (0, -1.0), 3: (-1, -0.5), 4: (-1, -2.5), 
5: (0, -2.0), 6: (1, -2.5)}


To load the edge flow data, we can use the :func:`read_flow` function.
Here we define the source, target and, columns and the delimiter used
in the dataset file.

>>> # load the edge flow data
>>> flow_path = f"{PAPER_DATA_FOLDER}/flow.csv"
>>>
>>> flow = read_flow(
>>>     filename=flow_path,
>>>     src_col="Source",
>>>     dest_col="Target",
>>>     flow_col="Flow",
>>>     delimiter=" "
>>> )
>>>
>>> print(flow)
{0: 2.25, 1: 0.13, 2: 1.72, 3: -2.12, 4: 1.59, 5: 1.08, 6: -0.3, 7: -0.21, 8: 1.25, 9: 1.45}


References
----------

- Datasets by :cite:t:`Jia_2019`
