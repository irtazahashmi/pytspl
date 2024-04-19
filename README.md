# Simplical Complex Graph Library

## Contributing to the library

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

## Quality assurance

### Unit tests

To run all the **unit tests**, run the following command:

```console
$ python3 -m pytest tests/
```

To get the test coverage report, run the following command:

```console
$ coverage run --source=sclibrary -m pytest -v tests && coverage report -m
```

### Static code analysis (flake8)

To run the flake8 linter, run the following command:

```console
$ python3 -m flake8 sclibrary
```
