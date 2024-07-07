# Contributing

Contributions are welcome, and they are greatly appreciated! The development of this package takes place on [GitHub](https://github.com/irtazahashmi/pytspl/tree/dev). Issues, bugs, and feature requests should be reported [there](https://github.com/irtazahashmi/pytspl/issues). Code and documentation can be improved by submitting a [pull request](https://github.com/irtazahashmi/pytspl/pulls). Please add documentation and tests for any new code.

You can improve or add functionality in the **`pytspl`**. folder, along with corresponding unit tests in **`pytspl/tests/test_*.py`** (with reasonable coverage). If you have a nice example to demonstrate the use of the introduced functionality, please consider adding a tutorial in **`notebooks`**.

Update **`README.md`** and **`CHANGELOG.md`** if applicable.

After making any change, please check the style, run the tests, and build the documentation with the following:

```console
$ poetry run pflake8 pytspl
$ poetry run pytest
$ make html
```

Check the generated coverage report at **`htmlcov/index.html`** to make sure the tests reasonably cover the changes you've introduced.

## Making a release

Todo.
