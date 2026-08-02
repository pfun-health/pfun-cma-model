{
  description = "Nix deployment artifacts for pfun-cma-model";

  inputs = {
    # Pin the current nixos-24.05 nixpkgs revision and the current
    # nixos-generators revision directly in the flake so image builds stay
    # reproducible even without committing a generated flake.lock file.
    # nixpkgs rev b134951... was current on the nixos-24.05 branch when this
    # workflow was added, and nixos-generators rev 8946737... was the then-
    # current upstream HEAD used to produce qcow images.
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
    nixos-generators = {
      url = "github:nix-community/nixos-generators/7c60ba4bc8d6aa2ba3e5b0f6ceb9fc07bc261565";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      nixos-generators,
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
      appName = pyproject.project.name;
      appVersion = pyproject.project.version;
      appModule = builtins.replaceStrings [ "-" ] [ "_" ] appName;
      defaultAppDir = "/var/lib/pfun-cma-model";
      sourceTree = builtins.path {
        path = ./.;
        name = "${appName}-source-tree";
      };
      appSource = pkgs.runCommand "${appName}-source" { } ''
        mkdir -p "$out"
        cp -R ${sourceTree}/. "$out/"
        chmod -R u+w "$out"
      '';
      runtimeInputs = [
        pkgs.bash
        pkgs.coreutils
        pkgs.git
        pkgs.gnumake
        pkgs.pkg-config
        pkgs.portaudio
        pkgs.python312
        pkgs.stdenv.cc.cc
        pkgs.uv
      ];
      startScript = pkgs.writeShellApplication {
        name = "start-pfun-cma-model";
        inherit runtimeInputs;
        text = ''
          APP_SOURCE=${appSource}
          APP_DIR="''${APP_DIR:-${defaultAppDir}}"
          export HOME="''${HOME:-/tmp}"
          export PYTHONUNBUFFERED=1
          export UV_PROJECT_ENVIRONMENT=".venv"

          mkdir -p "$APP_DIR"
          if [ ! -f "$APP_DIR/pyproject.toml" ] || [ ! -f "$APP_DIR/${appModule}/__init__.py" ]; then
            cp -R "$APP_SOURCE"/. "$APP_DIR"/
            chmod -R u+w "$APP_DIR"
          fi

          cd "$APP_DIR"

          if [ ! -d .venv ]; then
            # Copy mode keeps the runtime environment self-contained instead of
            # relying on symlinks back into transient build locations, which is
            # important because the app directory is materialized at runtime.
            uv sync --frozen --no-dev --link-mode copy
          fi

          exec uv run uvicorn pfun_cma_model.main:app --host 0.0.0.0 --port "''${PORT:-8001}"
        '';
      };
      ociImage = pkgs.dockerTools.buildLayeredImage {
        name = appName;
        tag = appVersion;
        contents = runtimeInputs ++ [
          appSource
          startScript
        ];
        config = {
          Cmd = [ "${startScript}/bin/start-pfun-cma-model" ];
          Env = [
            "APP_DIR=${defaultAppDir}"
            "HOME=/tmp"
            "PORT=8001"
            "PYTHONUNBUFFERED=1"
            "UV_PROJECT_ENVIRONMENT=.venv"
          ];
          ExposedPorts = {
            "8001/tcp" = { };
          };
          WorkingDir = defaultAppDir;
        };
      };
      vmImage = nixos-generators.nixosGenerate {
        inherit system;
        format = "qcow";
        modules = [
          (
            { ... }:
            {
              system.stateVersion = "24.05";
              nixpkgs.hostPlatform = system;
              networking.hostName = "pfun-cma-model";
              networking.firewall.allowedTCPPorts = [ 8001 ];

              # Non-root service user with sudo access
              users.mutableUsers = false;
              users.users.pfun = {
                isNormalUser = true;
                description = "pfun-cma-model service account";
                # wheel gives sudo access; add any other groups needed here
                extraGroups = [ "wheel" "audio" ];
                initialPassword = "";
                home = "/home/pfun";
                createHome = true;
              };

              # Allow wheel members to sudo without a password
              security.sudo.wheelNeedsPassword = false;

              # Autologin pfun on tty1 (console)
              services.getty.autologinUser = "pfun";

              environment.systemPackages = runtimeInputs;

              systemd.services.pfun-cma-model = {
                description = "pfun-cma-model API";
                wantedBy = [ "multi-user.target" ];
                after = [ "network-online.target" ];
                wants = [ "network-online.target" ];
                serviceConfig = {
                  Type = "simple";
                  Restart = "on-failure";
                  # Run as the non-root pfun user
                  User = "pfun";
                  Group = "users";
                  WorkingDirectory = "/var/lib/pfun-cma-model";
                  StateDirectory = "pfun-cma-model";
                  StateDirectoryMode = "0750";
                  Environment = [
                    "APP_DIR=/var/lib/pfun-cma-model"
                    "HOME=/var/lib/pfun-cma-model"
                    "PORT=8001"
                    "PYTHONUNBUFFERED=1"
                    "UV_PROJECT_ENVIRONMENT=.venv"
                  ];
                  ExecStart = "${startScript}/bin/start-pfun-cma-model";
                };
              };
            }
          )
        ];
      };
      # Script that builds every package output and places each symlink under
      # ./result/<output-name> instead of the default ./result.
      buildAllScript = pkgs.writeShellApplication {
        name = "build-all";
        runtimeInputs = [ pkgs.nix ];
        text = ''
          set -euo pipefail
          FLAKE_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)"
          cd "$FLAKE_ROOT"
          mkdir -p result
          nix build .#oci-image --out-link result/oci-image "$@"
          nix build .#vm-image  --out-link result/vm-image  "$@"
        '';
      };
    in
    {
      packages.${system} = {
        default = ociImage;
        oci-image = ociImage;
        vm-image = vmImage;
      };

      # Run `nix run .#build-all` (or `build-all` inside `nix develop`) to
      # build every output and place symlinks at:
      #   ./result/oci-image -> /nix/store/…-pfun-cma-model-<ver>.tar.gz
      #   ./result/vm-image  -> /nix/store/…-nixos-…-qcow2
      apps.${system}.build-all = {
        type = "app";
        program = "${buildAllScript}/bin/build-all";
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [ buildAllScript ] ++ runtimeInputs;
        shellHook = ''
          echo "Run 'build-all' to build all outputs into ./result/<output>"
        '';
      };
    };
}
