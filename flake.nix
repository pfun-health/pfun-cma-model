{
  description = "A development environment for pfun-cma-model";

  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devenv = {
      url = "github:cachix/devenv";
    };
    nix2container = {
      url = "github:nlewo/nix2container";
      inputs = {
        nixpkgs.follows = "nixpkgs";
      };
    };
    mk-shell-bin.url = "github:rrbutani/nix-mk-shell-bin";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs =
    inputs@{
      self,
      flake-parts,
      nixpkgs,
      devenv,
      ...
    }:
    let
      flakeOutputs = flake-parts.lib.mkFlake { inherit inputs; } {
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
            python = pkgs.python312;

            python_build_deps = with pkgs; [
              python312Packages.pyside6
              python312Packages.pyqt6
            ];
            devenv = inputs.devenv;
          in
          {
            devenv.shells.default = {
              name = "pfun-cma-model-dev";
              packages =
                with pkgs;
                [
                  python
                  uv

                  libxml2
                  pkg-config

                  libGLU
                  libGL
                  glib

                  freetype
                  tk
                  qhull

                  gfortran
                  openblas

                  arrow-cpp

                  llvm

                  rustc
                  cargo

                  openssl

                  zlib

                  duckdb
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

            packages.default = pkgs.mkShell {
              name = "pfun-cma-model-dev";
              packages =
                with pkgs;
                [
                  python
                  uv

                  libxml2
                  pkg-config

                  libGLU
                  libGL
                  glib

                  freetype
                  tk
                  qhull

                  gfortran
                  openblas

                  arrow-cpp

                  llvm

                  rustc
                  cargo

                  openssl

                  zlib

                  duckdb
                ]
                ++ python_build_deps;
            };
          };
      };

      defaultPackage = flakeOutputs.packages.x86_64-linux.default;

      nixosModules = {
        oci-containers =
          {
            pkgs,
            lib,
            config,
            ...
          }:
          {
            virtualisation.podman = {
              enable = true;
              autoPrune.enable = true;
              dockerCompat = true;
            };

            networking.firewall.interfaces =
              let
                matchAll = if !config.networking.nftables.enable then "podman+" else "podman*";
              in
              {
                "${matchAll}".allowedUDPPorts = [ 53 ];
              };

            virtualisation.oci-containers.backend = "podman";

            virtualisation.oci-containers.containers = {
              "pfun-cma-model" = {
                image = "localhost/compose2nix/pfun-cma-model";
                environmentFiles = [
                  "/home/robbiec/Git/pfun-cma-model/.env"
                ];
                ports = [
                  "8001:8001/tcp"
                ];
                cmd = [
                  "uv"
                  "run"
                  "uvicorn"
                  "pfun_cma_model.app:app"
                  "--proxy-headers"
                  "--host"
                  "0.0.0.0"
                  "--port"
                  "8001"
                  "--workers"
                  "2"
                ];
                dependsOn = [
                  "pfun-cma-model-redis"
                ];
                user = "nonroot:nonroot";
                log-driver = "journald";
                extraOptions = [
                  "--network-alias=pfun-cma-model"
                  "--network=pfun-cma-model_pfun-cma-network"
                ];
              };

              "pfun-cma-model-redis" = {
                image = "redis:7-alpine";
                environmentFiles = [
                  "/home/robbiec/Git/pfun-cma-model/.env"
                ];
                volumes = [
                  "pfun-cma-model_redis-data:/data:rw"
                ];
                cmd = [
                  "redis-server"
                  "--save"
                  "60"
                  "1"
                  "--loglevel"
                  "warning"
                  "--maxmemory"
                  "128mb"
                  "--maxmemory-policy"
                  "allkeys-lru"
                ];
                log-driver = "journald";
                extraOptions = [
                  "--cpus=0.5"
                  "--health-cmd=[\"redis-cli\", \"ping\"]"
                  "--health-interval=10s"
                  "--health-retries=3"
                  "--health-timeout=5s"
                  "--memory=268435456b"
                  "--network-alias=redis"
                  "--network=pfun-cma-model_pfun-cma-network"
                ];
              };

              "pfun-qt-gui" = {
                image = "localhost/pfun-qt-gui:latest";
                environment = {
                  "DISPLAY" = ":0";
                  "GROUP" = "nonroot";
                  "QT_X11_NO_MITSHM" = "1";
                  "USER" = "nonroot";
                };
                environmentFiles = [
                  "/home/robbiec/Git/pfun-cma-model/.env"
                ];
                volumes = [
                  "/home/robbiec/Git/pfun-cma-model/packages/pfun_qt_gui:/app:rw"
                  "/tmp/.X11-unix:/tmp/.X11-unix:rw"
                ];
                cmd = [ "./scripts/launch-qt-gui.sh" ];
                user = "nonroot:nonroot";
                log-driver = "journald";
                extraOptions = [
                  "--network=host"
                ];
              };
            };

            systemd.services =
              let
                mkPodmanService = name: {
                  serviceConfig = {
                    Restart = lib.mkOverride 90 "always";
                  };
                  partOf = [ "podman-compose-pfun-cma-model-root.target" ];
                  wantedBy = [ "podman-compose-pfun-cma-model-root.target" ];
                };
              in
              {
                "podman-pfun-cma-model" = mkPodmanService "pfun-cma-model" // {
                  after = [
                    "podman-network-pfun-cma-model_pfun-cma-network.service"
                  ];
                  requires = [
                    "podman-network-pfun-cma-model_pfun-cma-network.service"
                  ];
                };

                "podman-pfun-cma-model-redis" = mkPodmanService "pfun-cma-model-redis" // {
                  after = [
                    "podman-network-pfun-cma-model_pfun-cma-network.service"
                    "podman-volume-pfun-cma-model_redis-data.service"
                  ];
                  requires = [
                    "podman-network-pfun-cma-model_pfun-cma-network.service"
                    "podman-volume-pfun-cma-model_redis-data.service"
                  ];
                };

                "podman-pfun-qt-gui" = mkPodmanService "pfun-qt-gui";

                "podman-network-pfun-cma-model_pfun-cma-network" = {
                  path = [ pkgs.podman ];
                  serviceConfig = {
                    Type = "oneshot";
                    RemainAfterExit = true;
                    ExecStop = "podman network rm -f pfun-cma-model_pfun-cma-network";
                  };
                  script = ''
                    podman network inspect pfun-cma-model_pfun-cma-network || podman network create pfun-cma-model_pfun-cma-network --driver=bridge --subnet=172.20.0.0/16 --gateway=172.20.0.1
                  '';
                  partOf = [ "podman-compose-pfun-cma-model-root.target" ];
                  wantedBy = [ "podman-compose-pfun-cma-model-root.target" ];
                };

                "podman-volume-pfun-cma-model_redis-data" = {
                  path = [ pkgs.podman ];
                  serviceConfig = {
                    Type = "oneshot";
                    RemainAfterExit = true;
                  };
                  script = ''
                    podman volume inspect pfun-cma-model_redis-data || podman volume create pfun-cma-model_redis-data
                  '';
                  partOf = [ "podman-compose-pfun-cma-model-root.target" ];
                  wantedBy = [ "podman-compose-pfun-cma-model-root.target" ];
                };

                "podman-build-pfun-cma-model" = {
                  path = [
                    pkgs.podman
                    pkgs.git
                  ];
                  serviceConfig = {
                    Type = "oneshot";
                    TimeoutSec = 300;
                  };
                  script = ''
                    cd /home/robbiec/Git/pfun-cma-model
                    podman build -t compose2nix/pfun-cma-model .
                  '';
                };

                "podman-build-pfun-qt-gui" = {
                  path = [
                    pkgs.podman
                    pkgs.git
                  ];
                  serviceConfig = {
                    Type = "oneshot";
                    TimeoutSec = 300;
                  };
                  script = ''
                    cd /home/robbiec/Git/pfun-cma-model/packages/pfun_qt_gui
                    podman build -t pfun-qt-gui:latest .
                  '';
                };
              };

            systemd.targets."podman-compose-pfun-cma-model-root" = {
              unitConfig = {
                Description = "Root target for pfun-cma-model containers";
              };
              wantedBy = [ "multi-user.target" ];
            };
          };
      };
    in
    flakeOutputs // { inherit nixosModules; };
}
