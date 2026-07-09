## Project
Temporary work on graphon distribution.

## Environment setup

Create the conda environment:

1) For CPU only pytorch place yourself in the root directory "graphs" and launch:
```bash
conda env create -f envs/environment.yml
conda activate graphs
```

2) For cuda (12.8) pytorch place yourself in the root directory "graphs" and launch:
```bash
conda env create -f envs/environment-cuda.yml
conda activate graphs
```