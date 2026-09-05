# Graphtools

PyTorch and PyTorch Geometric based package for:
- Graphon sampling (module ```graphon```)
- Network analysis (module ```stats```)
- Graph construction (module ```builder```)

## Dependency management

The project uses two separate layers:

- `envs/environment.yaml` creates a minimal Conda environment with Python and pip.
- `envs/environment-cuda.yaml` adapts the minimal Conda environment to support CUDA.
- `pyproject.toml` defines the `graphtools` package and all its Python dependencies without overwriting the already installed torch version.

## First installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/lucafrattegiani/graphs.git
cd graphs
```

### Installation on a new environment

#### CPU version

Create and activate the environment:

```bash
conda env create -f envs/environment.yaml
conda activate graphs
```

Install the package in editable mode, including notebook support:

```bash
python -m pip install -e ".[notebook]"
```

#### GPU (CUDA 12.6) version

Create and activate the environment:

```bash
conda env create -f envs/environment-cuda.yaml
conda activate graphs
python -m pip install -e ".[notebook]"
```

### Installation on a pre-existing environment

Activate the existing environment and install the package in editable mode,
including notebook support:

```bash
python -m pip install -e ".[notebook]"
```

Pip reads the package requirements from `pyproject.toml`, keeps installed
packages that already satisfy them, and installs any missing dependencies.
Checking each dependency manually is therefore not necessary.

If CUDA support is required and a suitable CUDA-enabled PyTorch build is not
already installed in the environment, install it before `graphtools`:

```bash
python -m pip install -r envs/requirements-torch-cuda.txt
python -m pip install -e ".[notebook]"
```

When a suitable PyTorch installation is already present, the first editable
installation command is sufficient.

## Import package

```python
from graphtools import graphon, stats, builder
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
conda env update -f envs/environment.yaml --prune
python -m pip install -e ".[notebook]"
```

#### GPU version

```bash
conda env update -f envs/environment-cuda.yaml --prune
python -m pip install -e ".[notebook]"
```

If the major Python version or CUDA variant changes, recreating the environment
is generally safer than updating it in place.
