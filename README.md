# Graphtools

PyTorch and PyTorch Geometric based package for:
- Graphon sampling ([graph sampling example](notebooks/graph_sampling.ipynb))
- Network analysis ([graph metrics example](notebooks/graph_metrics.ipynb))
- Graph construction ([graph building example](notebooks/graph_building.ipynb))
- Graph's optimal transport ([graph ot example](notebooks/graph_ot.ipynb))

## Dependency management

The project uses two separate layers:

- `envs/environment.yaml` creates a Conda environment with Python 3.11, pip and the CPU PyTorch build.
- `envs/environment-cuda.yaml` selects the CUDA 12.6 PyTorch build for supported Linux/Windows systems.
- `pyproject.toml` defines the package's dependency ranges. Pip retains an installed PyTorch build only when its version satisfies the declared range.
- `envs/requirements-torch-{cpu,cuda}.txt` select the PyTorch wheel index.

### Supported versions

| Dependency | Allowed range |
| --- | --- |
| matplotlib | `>=3.8.4,<4` |
| networkx | `>=2.8.8,<4` |
| numpy | `>=1.23.2,<3` |
| POT | `>=0.9.7,<0.10` |
| scipy | `>=1.13,<2` |
| shapely | `>=2.1,<3` |
| torch | `>=2.4.1,<3` |
| torch-geometric | `>=2.3,<3` |
| torch-kmeans | `>=0.2.0,<0.3` |
| tqdm | `>=4.65,<5` |
| ipykernel (notebook) | `>=6.17,<8` |
| ipython (notebook) | `>=8.6,<10` |
| ipywidgets (notebook) | `>=8,<9` |

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
from graphtools import graphon, metrics, builder, opt
```

## Test package modifications

Install the test and notebook extras, check dependency consistency, and run the
execution-contract tests (notebook extras enable the widget test):

```bash
python -m pip install -e ".[test,notebook]"
python -m pip check
python -m pytest tests -v
```

## Package update

First activate the environment and retrieve the latest changes:

```bash
cd graphs
conda activate graphs
git pull
```

### CASE I: Changes in source files

The package does not need to be reinstalled. The editable installation points directly to `src/graphtools`, so source changes are immediately available.

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
