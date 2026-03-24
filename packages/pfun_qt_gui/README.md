# PFun Qt GUI

A PyQt6 frontend interface for generating and interacting with the PFun Health Tips Demo.

## Usage

### (Nix) devenv shell

##

    # Using nix develop ...
    nix develop --no-pure-eval

## Deploy

##

    # change to the **top-level** repo directory
    # e.g., PWD='~/Git/pfun-cma-model'

    # # First, generate the spec file (optional)
    # uv run pyside6-deploy

    # Then, deploy based on the configured specs
    uv run pyside6-deploy -c packages/pfun_qt_gui/pysidedeploy.spec
