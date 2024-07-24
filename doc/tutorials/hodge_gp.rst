Hodge-Compostional Edge Gaussian Process
=========================================


Hodge-compositional Gaussian processes (Hodge-GP) are for modeling 
functions defined over the edge set of a simplical complexes. The goal of this 
tutorial is to demonstrate how to use the Hodge-GP to model functions over edge 
flows to learn flow-type data on networks where the edge flows can be charcterized
by discrete divergence and curl. They enable learning on different Hodge components
seperately allowing us to capture the behaviour of edge flows.

We will demonstrate an edge-based learning taks in foreign exchage market data.

>>> from pytspl import load_dataset
>>>
>>> # load the forex dataset
>>> sc, _, flow = load_dataset("forex")
>>>
>>> # get the flow from the dict and convert it to numpy array
>>> y = np.fromiter(flow.values(), dtype=float)
WARNING: No coordinates found.
Generating coordinates using spring layout.
Num. of nodes: 25
Num. of edges: 210
Num. of triangles: 770
Shape: (25, 210, 770)
Max Dimension: 2
Coordinates: 25
Flow: 210

Next, we create a Hodge-GP model and fit it to the data. We pass in the simplicial complex 
and the flow data as input to the model.

>>> from pytspl.hodge_gp import HodgeGPTrainer
>>>
>>> # create the trainer object
>>> hogde_gp = HodgeGPTrainer(sc=sc, y=y)

Set the model parameters and split the data into training and testing sets.

>>> # set the training ratio
>>> train_ratio = 0.2
>>> # set the data normalization
>>> data_normalization = False
>>> 
>>> # get the eigenpairs
>>> eigenpairs = hogde_gp.get_eigenpairs()
>>>
>>> # split the data into training and testing sets
>>> x_train, y_train, x_test, y_test, x, y  = hogde_gp.train_test_split(train_ratio=train_ratio, data_normalization=data_normalization)
x_train: (42,)
x_test: (168,)
y_train: (42,)
y_test: (168,)

Now, we serialize a kernel type using the eigenpairs. The kernels encode prior knowledge 
about the unknown function and can be often difficult to choose. The available kernel types
can be found in the `pytspl.hodge_gp.kernels` module.

>>> from pytspl.hogde_gp.kernel_serializer import KernelSerializer
>>>
>>> # set the kernel parameters
>>> kernel_type = "matern" # kernel type
>>> data_name = "forex" # dataset name
>>>
>>> # serialize the kernel
>>> kernel = KernelSerializer().serialize(
>>>    eigenpairs=eigenpairs, 
>>>    kernel_type=kernel_type, 
>>>    data_name=data_name
>>> )


Specify the likelihood and the mean function for the model.

>>> import gpytorch
>>> from pytspl.hogde_gp import ExactGPModel
>>>
>>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
>>> model = ExactGPModel(x_train, y_train, likelihood, kernel, mean_function=None)


We can specify the output device for the model and likelihood.

>>> import torch
>>> output_device = "cpu"
>>>
>>> if torch.cuda.is_available():
>>>    model = model.to(output_device)
>>>    likelihood = likelihood.to(output_device)


Next, we can print the model parameters and their values.

>>> for param_name, param in model.named_parameters():
>>>    print(f'Parameter name: {param_name:50} value = {param.item()}')


Now, we can train the model using the training data.

>>> # train the model
>>> model.train()
>>> likelihood.train()
>>> hogde_gp.train(model, likelihood, x_train, y_train)
Iteration 1/1000 - Loss: 5.387 
Iteration 2/1000 - Loss: 4.940 
Iteration 3/1000 - Loss: 4.554 
Iteration 4/1000 - Loss: 4.221 
...
Iteration 997/1000 - Loss: -0.170 
Iteration 998/1000 - Loss: -0.170 
Iteration 999/1000 - Loss: -0.171 
Iteration 1000/1000 - Loss: -0.171 


To evaluate the model, we can use the testing data.

>>> # evaluate the model
>>> hogde_gp.predict(model, likelihood, x_test, y_test)
Test MAE: 5.07415461470373e-05
Test MSE: 4.291597299754812e-09
Test R2: 1.0
Test MLSS: -3.288797616958618
Test NLPD: -3.5441062450408936