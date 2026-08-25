# Graphtools

Graphon sampling and graph analysis based on PyTorch and PyTorch Geometric.

## Dependency management

The project uses two separate layers:

- `envs/environment.yml` creates a minimal Conda environment with Python and pip.
- `envs/environment-cuda.yml` adapts the minimal Conda environment to support CUDA.
- `pyproject.toml` defines the `graphtools` package and all its Python dependencies without overwriting the already installed torch version.

## First installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/lucafrattegiani/graphs.git
cd graphs
```

### CPU version

Create and activate the environment:

```bash
conda env create -f envs/environment.yml
conda activate graphs
```

Install the package in editable mode, including notebook support:

```bash
python -m pip install -e ".[notebook]"
```

### GPU (CUDA 12.6) version

Create and activate the environment:

```bash
conda env create -f envs/environment-cuda.yml
conda activate graphs
python -m pip install -e ".[notebook]"
```

## Import package

```python
from graphtools import graphon, metrics
```

## Package update

First activate the environment and retrieve the latest changes:

```bash
cd graphs
conda activate graphs
git pull
```

### CASE I: Changes in source files

The package does not need to be reinstalled. The editable installation points directly to `src/graphs`, so source changes are immediately available. 

### CASE II: Changes in `pyproject.toml`

Refresh the editable installation, so that pip keeps packages that already satisfy the requirements and installs or updates only what is necessary:

```bash
python -m pip install -e ".[notebook]"
```

### CASE III: Changes in the environment

#### CPU version

```bash
conda env update -f envs/environment.yml --prune
python -m pip install -e ".[notebook]"
```

#### GPU version

```bash
conda env update -f envs/environment-cuda.yml --prune
python -m pip install -e ".[notebook]"
```

If the major Python version or CUDA variant changes, recreating the environment
is generally safer than updating it in place.