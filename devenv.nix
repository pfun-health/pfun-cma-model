{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  # Enable the .env file for the project pfun-cma-model
  dotenv.enable = true;
  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages = [ pkgs.git ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;
  languages.python.enable = true;

  # Processes:
  # https://devenv.sh/processes/
  processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";
  processes = {
    server = {
      exec = "./scripts/serve_dev.sh";
      cwd = builtins.getEnv "PWD";
    };
  };

  # https://devenv.sh/services/
  services.postgres.enable = true;

  # https://devenv.sh/scripts/
  # scripts.hello.exec = ''
  #   echo hello from $GREET
  # '';

  # https://devenv.sh/basics/
  enterShell = ''
    git --version # Use packages
    pfun-cma-model version         # Run scripts directly
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
    ./scripts/run-tests.sh
    echo -e "\n#################################################"
    echo -e "#################################################\n"
    ./scripts/run-benchmarks.sh
  '';

  # https://devenv.sh/git-hooks/
  git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
