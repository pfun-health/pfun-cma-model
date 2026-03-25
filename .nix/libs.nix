let
  pkgs = import (builtins.getFlake "nixpkgs") { };
in
[
  pkgs.gcc.cc
  pkgs.glibc
  pkgs.zlib
  pkgs.OpenGL
  pkgs.libxml2
  pkgs.libGLU
  pkgs.libGL
  pkgs.glib
  pkgs.pkg-config
]
