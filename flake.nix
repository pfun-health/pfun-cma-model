{
  description = "Nix deployment artifacts for pfun-cma-model";

  inputs = {
    nixpkgs.url = "git+https://github.com/NixOS/nixpkgs?ref=nixos-24.05&rev=b134951a4c9f3c995fd7be05f3243f8ecd65d798";
    nixos-generators.url = "git+https://github.com/nix-community/nixos-generators?rev=8946737ff703382fda7623b9fab071d037e897d5";
    nixos-generators.inputs.nixpkgs.follows = "nixpkgs";
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
          APP_DIR="''${APP_DIR:-/tmp/pfun-cma-model}"
          export HOME="''${HOME:-/tmp}"
          export PYTHONUNBUFFERED=1
          export UV_PROJECT_ENVIRONMENT=".venv"

          mkdir -p "$APP_DIR"
          if [ ! -f "$APP_DIR/pyproject.toml" ] || [ ! -f "$APP_DIR/pfun_cma_model/main.py" ]; then
            cp -R "$APP_SOURCE"/. "$APP_DIR"/
            chmod -R u+w "$APP_DIR"
          fi

          cd "$APP_DIR"

          if [ ! -d .venv ]; then
            # Copy mode keeps the runtime environment self-contained instead of
            # relying on symlinks back into transient build locations.
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
            "APP_DIR=/tmp/pfun-cma-model"
            "HOME=/tmp"
            "PORT=8001"
            "PYTHONUNBUFFERED=1"
            "UV_PROJECT_ENVIRONMENT=.venv"
          ];
          ExposedPorts = {
            "8001/tcp" = { };
          };
          WorkingDir = "/tmp/pfun-cma-model";
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

              environment.systemPackages = runtimeInputs;

              systemd.services.pfun-cma-model = {
                description = "pfun-cma-model API";
                wantedBy = [ "multi-user.target" ];
                after = [ "network-online.target" ];
                wants = [ "network-online.target" ];
                serviceConfig = {
                  Type = "simple";
                  Restart = "on-failure";
                  WorkingDirectory = "/var/lib/pfun-cma-model";
                  StateDirectory = "pfun-cma-model";
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
    in
    {
      packages.${system} = {
        default = ociImage;
        oci-image = ociImage;
        vm-image = vmImage;
      };
    };
}
