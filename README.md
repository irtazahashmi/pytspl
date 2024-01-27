# Simplical Complex Graph Library

## Using the library

### Create a virtual environment

Create a virtual enviroment to make sure the dependencies of each project is kept seperate. To create a virtual environment _my_venv_, run the following command:

```console
$ python3 -m venv my_venv
```

Activate the vitual environment _my_venv_ using the following command:

```console
$ source ./my_venv/bin/activate
```

### Install project requirements

To download the project requirements, run the following command:

```console
$ python3 -m pip install -r requirements.txt
```

## Running unit tests

To run all the **unit tests**, run the following command:

```console
$ python3 -m pytest tests/
```

To get the test coverage report, run the following command:

```console
$ coverage run --source=sclibrary -m pytest -v tests && coverage report -m
```
