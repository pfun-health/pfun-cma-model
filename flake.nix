{
  description = "Nix deployment artifacts for Vite application";

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
      
      packageJson = builtins.fromJSON (builtins.readFile ./package.json);
      appName = packageJson.name;
      appVersion = packageJson.version;
      
      # Use the specific Node version matching your pipeline
      nodePkg = pkgs.nodejs_20; 
      pnpmPkg = pkgs.pnpm;

      # Filter source to avoid invalidating the build on minor text edits
      sourceTree = pkgs.lib.cleanSource ./.;

      # Step 1: Fetch dependencies and compile Vite static assets
      # This handles your typescript/pnpm build step reproducibly.
      appBundle = pkgs.stdenv.mkDerivation {
        pname = appName;
        version = appVersion;
        src = sourceTree;

        nativeBuildInputs = [ nodePkg pnpmPkg ];

        # Fetch pnpm store out-of-band for sandboxed builds
        pnpmDeps = pnpmPkg.fetchDeps {
          pname = "${appName}-pnpm-deps";
          inherit appVersion;
          src = sourceTree;
          hash = pkgs.lib.fakeHash; # Run `nix build` once, copy the real hash here when it fails
        };

        buildPhase = ''
          export HOME = "/tmp"
          pnpm config set store-dir $pnpmDeps
          pnpm install --frozen-lockfile --offline
          pnpm build
        '';

        installPhase = ''
          mkdir -p "$out"
          cp -R dist/. "$out/"
        '';
      };

      # A production runner script to serve the built static site
      # Uses a fast, small Rust binary instead of spinning up a whole Node runtime for static files
      serverPkg = pkgs.static-web-server;
      startScript = pkgs.writeShellApplication {
        name = "start-${appName}";
        runtimeInputs = [ serverPkg ];
        text = ''
          exec static-web-server \
            --port "''${PORT:-8001}" \
            --host 0.0.0.0 \
            --root ${appBundle} \
            --assets-bypass-extensions html
        '';
      };

      # Step 2: Container Image Config
      ociImage = pkgs.dockerTools.buildLayeredImage {
        name = appName;
        tag = appVersion;
        contents = [ startScript ];
        config = {
          Cmd = [ "${startScript}/bin/start-${appName}" ];
          Env = [ "PORT=8001" ];
          ExposedPorts = { "8001/tcp" = { }; };
        };
      };

      # Step 3: VM Config
      vmImage = nixos-generators.nixosGenerate {
        inherit system;
        format = "qcow";
        modules = [
          (
            { ... }:
            {
              system.stateVersion = "24.05";
              nixpkgs.hostPlatform = system;
              networking.hostName = appName;
              networking.firewall.allowedTCPPorts = [ 8001 ];

              systemd.services.${appName} = {
                description = "${appName} Web Server";
                wantedBy = [ "multi-user.target" ];
                after = [ "network-online.target" ];
                wants = [ "network-online.target" ];
                serviceConfig = {
                  Type = "simple";
                  Restart = "on-failure";
                  ExecStart = "${startScript}/bin/start-${appName}";
                  DynamicUser = true; # Security bonus: runs as an unprivileged ephemeral user
                  Environment = [ "PORT=8001" ];
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
        static-assets = appBundle;
      };
    };
}
