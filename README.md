# Graphtools

PyTorch and PyTorch Geometric based package for:
- Graphon sampling (module ```graphon```)
- Network analysis (module ```metrics```)
- Graph construction (module ```builder```)
- Optimal transport between graphs (module ```opt```)

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
from graphtools import graphon, metrics, builder, opt
```

## Execution conventions

The same package supports CPU and Torch CUDA execution. Choosing a device does
not change the algorithm or automatically enable a different solver.

| Operations | Execution and output |
| --- | --- |
| Torch sampling | CPU by default; `device` selects where to create data. |
| Adjacency, Laplacian, random walk, clustering, spectral analysis | Follow `edge_index`, or an explicit `device`. Unused features and positions do not affect placement. |
| Homophily and informativeness | Follow `edge_index` and `x`. These must share a device unless an explicit `device` permits local transfers. |
| NetworkX centralities, inverse shortest path, Jaccard | CPU computation; output follows `edge_index` or an explicit `device`. Input graphs remain unchanged. |
| FGW conditional gradient (POT) | CPU preparation and solver; loss and info tensors follow `graph_1.edge_index` or an explicit `device`. |
| Voronoi builders | Geometry is computed on CPU. `build_graph` returns a new graph on the input positions' device; low-level builders use `device` for output placement. |
| Plotting | Rendering uses CPU data; GPU tensors are read through local CPU copies. |

For analysis functions, `device` selects the output device. Torch operations
also compute there when supported. CPU-only algorithms use local CPU data and
transfer the final result to the requested device, without moving the input graph:

```python
C_gpu = metrics.jaccard_similarity_matrix(graph, device="cuda")
loss_gpu, info_gpu = opt.fused_gromov_wasserstein_distance(g1, g2, device="cuda")
```

With `device=None`, topology functions follow `edge_index` (CPU if absent).
FGW follows the first graph's `edge_index`; the second graph is not moved.
The internal computation device can differ from the output device.

Feature-based metrics preserve floating input precision. FGW promotes the
floating dtypes of the features (when used), histograms and supplied structural
matrices to a common dtype. Integer-only analysis inputs and newly constructed
topological matrices use float32; indices and labels retain integer types.
Preserving a dtype does not imply that every backend supports every operation
in that dtype. Sampling without an input floating dtype retains its documented
defaults.

Analysis functions do not mutate input graphs: they do not overwrite attributes,
move graph storage, or attach caches. Local tensors may share storage with the
inputs, so input-derived values must not be modified in-place. A clone is needed
only when an operation actually writes to shared data. Creating and populating
a new graph is permitted.

Validation helpers are imported through `graphtools.validity`. Their
implementations live in `_validity/execution.py`, including device resolution,
floating dtype promotion, size checks and class probability validation. Checks
inspect only the attributes needed by the operation. An explicit device is
resolved immediately, so delegating wrappers do not repeat device-consistency
checks. Graph snapshots and immutability checks belong in tests, not in normal
execution. Preparation reuse and parallel FGW execution are subsequent steps.

Run the execution-contract tests with:

```bash
python -m pytest tests -v
```

The test suite is organized by contract:

| File | Checks |
| --- | --- |
| `tests/device_test.py` | Inferred and explicit output placement, CPU/CUDA transfers, mixed inputs and attributes that are not used by the operation. |
| `tests/precision_test.py` | Floating dtypes, known numerical results, dense/sparse matrices, spectral residuals and FGW costs and marginal constraints. |
| `tests/mutation_test.py` | Preservation of both graph values and tensor identities, metadata, integer inputs, optional histograms and structural matrices. |

`tests/utils.py` provides small example graphs, snapshots and explicit function
cases shared across the categories. Hard-label metrics receive one-hot features;
continuous metrics receive non-collinear features. Full covariance comparison
cases use `eps=1e-3` to control conditioning. The spectral tests use eigenpair
residuals and cluster properties, since eigenvector signs and cluster IDs can
differ without changing the result.

The current scope is the computational functions exposed by `metrics.py` and
FGW in `_opt/distance.py`. Plotting and the unimplemented non-parametric
informativeness method are excluded. CUDA cases are skipped when no CUDA device
is available; skipped cases do not constitute GPU validation. Each file can
also be run separately, for example `python -m pytest tests/device_test.py -v`.

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
