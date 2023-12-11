import importlib
from pathlib import Path
import sys

root_path = str(Path(__file__).parents[1])
if root_path not in sys.path:
    sys.path.insert(0, root_path)
test_fit_model = importlib.import_module(
    "pfun_cma_model.runtime.tests.test_fit_model")


def plot():
    res = test_fit_model.interactive_plot()
    print(res)
    return res


if __name__ == '__main__':
    out = plot()
