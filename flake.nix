{
  description = "A development environment for pfun-cma-model";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        # As per pyproject.toml: >3.11,<3.13.
        python = pkgs.python312;

        # System-level dependencies for Python packages
        # that might be built from source by `uv`.
        # This ensures `uv sync` works smoothly.
        python_build_deps = with pkgs; [
          # for matplotlib, scipy
          freetype
          tk
          qhull

          # for scipy, numpy
          gfortran
          openblas

          # for pyarrow
          arrow-cpp

          # for numba (llvmlite)
          llvm

          # for pydantic (pydantic-core, which is a rust extension)
          rustc
          cargo

          # for paramiko (cryptography)
          openssl

          # for various packages that might be installed
          zlib

          # duckdb
          duckdb

          # data visualization
          #datasette  # currently isn't working due to dependency on pip module (should work once we have uv install the Python dependencies, but we want to be able to run `uv sync` without errors first)
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          name = "pfun-cma-model-dev";

          buildInputs = with pkgs; [
            python  # note that the version is defined above
            uv

            # General purpose build tools
            pkg-config
          ] ++ python_build_deps;

          shellHook = ''
            echo "Installing dev dependencies..."
            echo "...datasette plugins..."
            datasette install datasette-parquet
            datasette install datasette-plot
            echo "...done installing dev dependencies."

            echo "Welcome to the pfun-cma-model dev shell!"
            echo ""
            echo "This shell provides Python, uv."
            echo "The project's Python dependencies are defined in pyproject.toml."
            echo ""
            
            . ./.venv/bin/activate
            echo "Activated Python virtual environment from .venv/ directory."
          '';
        };
      }
    );
}
