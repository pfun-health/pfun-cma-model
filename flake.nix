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
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          name = "pfun-cma-model-dev";

          buildInputs = with pkgs; [
            python
            uv
            python312Packages.tox

            # General purpose build tools
            pkg-config
          ] ++ python_build_deps;

          shellHook = ''
            echo "Welcome to the pfun-cma-model dev shell!"
            echo ""
            echo "This shell provides Python, uv, and tox."
            echo "The project's Python dependencies are defined in pyproject.toml."
            echo ""
            echo "To get started (as per your README.md):"
            echo "1. Create a virtual environment: uv venv"
            echo "2. Activate it: source .venv/bin/activate"
            echo "3. Install dependencies: uv sync"
            echo ""
            echo "After that, you can run tests with 'uvx tox' or run the app."
          '';
        };
      }
    );
}
