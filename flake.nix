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
            g++ -std=c++17 -O2 -I./src/includes -o pfun-cma-model ./src/*.cc
          '';
          installPhase = ''
            mkdir -p $out/bin
            cp pfun-cma-model $out/bin/
          '';
        };
        pfun-cma-model-wasm = pkgs.stdenv.mkDerivation {
          name = "pfun-cma-model-wasm";
          src = self;
          buildInputs = with pkgs; [ emscripten nodejs ];
          buildPhase = ''
            # emscripten needs a HOME directory to work
            export HOME=$(mktemp -d)
            echo "HOME=$HOME"

            # quickfix for emscripten cache (get around read-only filesystem error)
            # ref: https://github.com/NixOS/nixpkgs/issues/139943#issuecomment-930432045
            export EMSCRIPTEN_ROOT="$(dirname $(dirname $(which emcc)))/share/emscripten"
            cp -r $EMSCRIPTEN_ROOT/cache \
              $HOME/.emscripten_cache && \
            chmod u+rwX -R $HOME/.emscripten_cache
            export EM_CACHE=$HOME/.emscripten_cache

            emcc -std=c++17 -O2 -I./src/includes \
              -s WASM=1 \
              -s EXPORTED_FUNCTIONS='["_run_calc"]' \
              -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]' \
              -o pfun-cma-model.html \
              ./src/*.cc
          '';
          installPhase = ''
            mkdir -p $out/lib
            cp pfun-cma-model.js pfun-cma-model.wasm $out/lib/
          '';
          testInputs = with pkgs; [ python3 ];
          testPhase = ''
            echo "Launching python server locally (http://localhost:8000)..."
            python3 -m http.server
          '';
        };
        default = pkgs.symlinkJoin {
          name = "pfun-cma-model-all";
          paths = [ pfun-cma-model pfun-cma-model-wasm ];
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        # quickfix for emscripten cache (get around read-only filesystem error)
        # ref: https://github.com/NixOS/nixpkgs/issues/139943#issuecomment-1753113985
        EM_CONFIG = pkgs.writeText ".emscripten" ''
            EMSCRIPTEN_ROOT = '${pkgs.emscripten}/share/emscripten'
            LLVM_ROOT = '${pkgs.emscripten.llvmEnv}/bin'
            BINARYEN_ROOT = '${pkgs.binaryen}'
            NODE_JS = '${pkgs.nodejs-18_x}/bin/node'
            CACHE = '${toString ./.cache}'
          '';
        buildInputs = with pkgs; [ gcc emscripten ];
      };
    };
}

