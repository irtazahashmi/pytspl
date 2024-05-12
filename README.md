[![Build](https://github.com/irtazahashmi/sc-graph-library/actions/workflows/onpush.yml/badge.svg)](https://github.com/irtazahashmi/sc-graph-library/actions/workflows/onpush.yml)
[![codecov](https://codecov.io/gh/irtazahashmi/sc-graph-library/graph/badge.svg?token=7KQ0U8FW70)](https://codecov.io/gh/irtazahashmi/sc-graph-library)
[![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python)](https://www.python.org/)

# SCLibrary - Simplicial Complexes Library

A Python package for higher-interactions, specifically simplical complexes.

## Contributing to the library

### Install the dependencies

To install the project dependencies, run the following command:

```console
$ poetry install
```

## Quality assurance

### Unit tests

To run all the **unit tests**, run the following command:

```console
$ poetry run pytest
```

To get the test coverage report, run the following command:

```console
$ coverage run -m pytest && coverage report -m
```

### Static code analysis (flake8)

To run the flake8 linter, run the following command:

```console
$ flake8 sclibrary
```

## Documentation

The code documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/). After making changes to the code/documentation, go to the _docs_ folder and run the following commands to regenreate the documentation:

```console
$ make clean html
$ make html
```
