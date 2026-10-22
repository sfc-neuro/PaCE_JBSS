# PaCE

The reference implementation for Parallel Component Extraction (PaCE).

## Citing

When you use PaCE, please cite the paper:

    Silvia Cotroneo, Marijn van Vliet, and Riitta Salmelin (2026). "Parallel Component
    Extraction". Some Journal.


## Installation

This should also install the dependencies (they are listed below):

```bash
pip install -e .
```

### Dependencies

```
jax
scikit-learn
```

For building the documentation and running the unit tests, there are several more dependencies:

```
furo
pytest
pytest-cov
sphinx
```


## Building the documentation

Make sure the dependencies for the documentation are installed. They are listed above.

```bash
sphinx-build doc doc/_build/html
```


## Running the unit tests

Make sure the dependencies for running the unit tests are installed. They are listed above.

```bash
pytest
```
