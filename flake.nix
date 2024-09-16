{
  description = "Flake for the pfun-cma-model project";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      packages.${system} = rec {
        pfun-cma-model = pkgs.stdenv.mkDerivation {
          name = "pfun-cma-model";
          src = self;
          buildInputs = with pkgs; [ gcc ];
          buildPhase = ''
            g++ -std=c++17 -O2 -o pfun-cma-model src/*.cc
          '';
          installPhase = ''
            mkdir -p $out/bin
            cp pfun-cma-model $out/bin/
          '';
        };
        default = pfun-cma-model;
      };

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [ gcc ];
      };
    };
}

