## Project
Temporary work on graphon distribution.

## Environment

### Set-up
Create the conda environment:

1) For CPU only pytorch place yourself in the root directory "graphs" and launch:
```bash
conda env create -f envs/environment.yml
```

2) For cuda (12.8) pytorch place yourself in the root directory "graphs" and launch:
```bash
conda env create -f envs/environment-cuda.yml
```

### Update:
Update an existing conda environment in case of modifications to the environment set-up

1) For new insertions/deletions with conda:
```bash
conda env update -f envs/environment.yml --prune
```
or for cuda:
```bash
conda env update -f envs/environment-cuda.yml --prune
```

2) For new insertions/deletions with pip:
```bash
pip install -r envs/requirements.txt
```
or for cuda:
```bash
pip install -r envs/requirements-torch-cuda.txt
```