# Graphtools

Graphon sampling and graph analysis based on PyTorch and PyTorch Geometric.

## Dependency management

The project uses two separate layers:

- `envs/environment.yml` creates a minimal Conda environment with Python and
  pip;
- `pyproject.toml` defines the `graphtools` package and all its Python dependencies.

The CUDA environment installs the CUDA-specific PyTorch build first. The subsequent package installation recognizes that `torch` is already available and installs only the missing dependencies.

## First installation on a new computer

Clone the repository and enter its directory:

```bash
git clone <REPOSITORY-URL>
cd graphs
```

### CPU

Create and activate the environment:

```bash
conda env create -f envs/environment.yml
conda activate graphs
```

Install the package in editable mode, including notebook support:

```bash
python -m pip install -e ".[notebook]"
```

### CUDA 12.6

Create the CUDA environment from the repository root:

```bash
conda env create -f envs/environment-cuda.yml
conda activate graphs
python -m pip install -e ".[notebook]"
```

Verify that PyTorch detects CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### Jupyter kernel

If the environment does not appear among the available kernels, register it
once:

```bash
python -m ipykernel install --user --name graphs --display-name "Python (graphs)"
```

Select the `Python (graphs)` kernel in notebooks. The package can then be
imported from any directory:

```python
from graphs import graphon, metrics
```

## Updating the project

First activate the environment and retrieve the latest changes:

```bash
cd graphs
conda activate graphs
git pull
```

### Only source files have changed

The package does not need to be reinstalled. The editable installation points
directly to `src/graphs`, so source changes are immediately available. In a
notebook, you may need to restart the kernel to clear previously imported
modules from memory.

### `pyproject.toml` has changed

Refresh the editable installation:

```bash
python -m pip install -e ".[notebook]"
```

pip keeps packages that already satisfy the requirements and installs or
updates only what is necessary.

### The CPU environment definition has changed

```bash
conda env update -f envs/environment.yml --prune
python -m pip install -e ".[notebook]"
```

### The CUDA environment definition has changed

```bash
conda env update -f envs/environment-cuda.yml --prune
python -m pip install -e ".[notebook]"
```

If the major Python version or CUDA variant changes, recreating the environment
is generally safer than updating it in place.

## Adding a dependency

Add regular Python libraries to `dependencies` in `pyproject.toml`, then install
them into the active environment with:

```bash
python -m pip install -e ".[notebook]"
```

The Conda files remain reserved for the Python version and installations that
require special configuration, such as PyTorch with CUDA.
