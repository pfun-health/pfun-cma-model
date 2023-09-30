# pfun-cma-model

*CMA model microservice repo.*

## Environment Setup

## First

**Ref:** https://stackoverflow.com/a/74297233/1871569

```yaml
# environment.yml
---
dependencies:
  - python=3.10.*
  - mamba
  - pip
  - pip:
      - "poetry>=1.2" # poetry>=1.2 for conda + poetry
```

## First-time setup

You can avoid playing with the bootstrap env and simplify the example below if you have conda-lock, mamba and poetry already installed outside your target environment.

```bash
# Create a bootstrap env

conda create -p /tmp/bootstrap -c conda-forge 'python==3.10.*' mamba conda-lock poetry='1.*'
conda activate /tmp/bootstrap

# Create Conda lock file(s) from environment.yml

conda-lock --conda mamba

# Set up Poetry

poetry init --python=~3.10 # version spec should match the one from environment.yml

# Fix package versions installed by Conda to prevent upgrades

# perhaps not needed:

# poetry add --lock ...

# Add conda-lock (and other packages, as needed) to pyproject.toml and poetry.lock

poetry add --lock conda-lock

# Remove the bootstrap env

conda deactivate
rm -rf /tmp/bootstrap

# Add Conda spec and lock files

git add environment.yml virtual-packages.yml conda-linux-64.lock

# Add Poetry spec and lock files

git add pyproject.toml poetry.lock
git commit
```

## Usage

#### Creating the environment

```bash
conda create --name pfun-cma-model --file conda.lock
conda activate pfun-cma-model
poetry install
```

#### Activating the environment

```bash
conda activate pfun-cma-model
```

#### Updating the environment

```bash
# Update Conda packages based on environment.yml

conda env update

# Re-generate Conda lock file(s) based on environment.yml
conda-lock --conda mamba

# Update Conda packages based on re-generated lock file
mamba update --file conda.lock

# Update Poetry packages and re-generate poetry.lock
poetry update
```
