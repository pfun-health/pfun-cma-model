{
  description = "A development environment for pfun-cma-model";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/25.11";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devenv.url = "github:cachix/devenv";
    nix2container = {
      url = "github:nlewo/nix2container";
      inputs = {
        nixpkgs.follows = "nixpkgs";
      };
    };
    mk-shell-bin.url = "github:rrbutani/nix-mk-shell-bin";
  };

  outputs =
    inputs@{
      self,
      flake-parts,
      nixpkgs,
      devenv,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {

      imports = [
        inputs.devenv.flakeModule
      ];
      systems = [ "x86_64-linux" ];

      perSystem =
        {
          config,
          self',
          inputs',
          pkgs,
          system,
          ...
        }:
        let
          # As per pyproject.toml: >3.11,<3.13.
          python = pkgs.python312;

          # System-level dependencies for Python packages
          # that might be built from source by `uv`.
          # This ensures `uv sync` works smoothly.
          python_build_deps = with pkgs; [
            python312Packages.pyside6 # for PyQt6
            python312Packages.pyqt6 # for PyQt6
          ];
        in
        {
          devenv.shells.default = {
            # https://devenv.sh/reference/options/
            name = "pfun-cma-model-dev";
            packages =
              with pkgs;
              [
                python # note that the version is defined above
                uv
                pkg-config # general-purpose build tool
                # for PyQt6
                libGLU
                libGL
                glib

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
              ]
              ++ python_build_deps;

            enterShell = ''
              echo "Entering the pfun-cma-model dev shell..."
              echo "Setting LD_LIBRARY_PATH..."
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib"
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

              ./scripts/uv-full-sync.sh
              echo "Synchronized Python dependencies using uv. You can now run Python scripts that depend on these packages without issues."
              echo "e.g., 'uv run pyside6-deploy -c packages/pfun_qt_gui/pysidedeploy.spec' to deploy the PyQt6 GUI application."
            '';
          };
        };
    };
}
