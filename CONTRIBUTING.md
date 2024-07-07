# Contributing

Contributions are welcome, and they are greatly appreciated! The development of this package takes place on [GitHub](https://github.com/irtazahashmi/pytspl/tree/dev). Issues, bugs, and feature requests should be reported [there](https://github.com/irtazahashmi/pytspl/issues). Code and documentation can be improved by submitting a [pull request](https://github.com/irtazahashmi/pytspl/pulls). Please add documentation and tests for any new code.

### How to Contribute

To contribute, follow these steps:

1. **Fork the repository**: Start by forking the repository to your own GitHub account.

2. **Clone the repository**: Clone your forked repository to your local machine.

```console
$ git clone https://github.com/your-username/pytspl.git
```

3. **Create a new branch**: Create a new branch for your changes.

```console
$ git clone https://github.com/your-username/pytspl.git
```

4. **Make your changes**: Improve or add functionality in the **`pytspl`** folder, along with corresponding unit tests in **`pytspl/tests/test_*.py`** (with reasonable coverage). If you have a nice example to demonstrate the use of the introduced functionality, please consider adding a tutorial in **`notebooks`**.

5. **Update documentation**: Update **`README.md`** and **`CHANGELOG.md`** if applicable.

6. **Check the style and run tests**: Ensure your code meets the style guidelines and all tests pass. You can do this by running the following commands:

```console
$ poetry run pflake8 pytspl
$ poetry run pytest
```

7. **Build the documentation**: Generate the documentation to ensure it builds correctly.

```console
$ make html
```

8. **Check coverage**: Check the generated coverage report at **`htmlcov/index.html`** to make sure the tests reasonably cover the changes you've introduced.

9. **Submit a pull request**: Go to the original repository on GitHub and submit a pull request. Ensure you fill in the pull request template with all relevant details.

## Making a release

Todo.

Thank you for contributing!
