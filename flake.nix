{
  description = "Flake for the pfun-cma-model project";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: {

    packages.x86_64-linux.pfun-cma-model = nixpkgs.legacyPackages.x86_64-linux.pfun-cma-model;

    packages.x86_64-linux.default = self.packages.x86_64-linux.pfun-cma-model;

  };
}
