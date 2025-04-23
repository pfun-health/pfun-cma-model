# meson.build - Top-level Meson build file for Python package with Minpack dependency

project(
  'pfun_cma_model',
  ['fortran', 'c'],
  version: '0.1.0',
  meson_version: '>=0.55',
  default_options: ['buildtype=release']
)

# ---- Python setup ----
py_mod = import('python')
python = py_mod.find_installation('python3', required: true)
py_dep = python.dependency()

# ---- Minpack subproject ----
# Configure Minpack with Python bindings (if supported by Minpack's Meson)
minpack_opts = {
  'python': 'true',
  'python_version': python.path()
}

minpack_proj = subproject('minpack', default_options: minpack_opts)
minpack_dep = minpack_proj.get_variable('minpack_dep')  # fallback for demonstration

# ---- Install pure Python code ----
# Installs all Python files in your_package/ to the correct site-packages dir
install_subdir(
  'pfun_cma_model',
  install_dir: python.get_install_dir()
)

# Optionally install package metadata, scripts, etc.
install_data('pyproject.toml')
install_data('README.md')
install_subdir('examples/data', install_dir: join_paths(python.get_install_dir(), 'examples/data'))
install_subdir('examples/output', install_dir: join_paths(python.get_install_dir(), 'examples/output'))