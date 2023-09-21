import sys
from typing import Any, Optional
import os


def get_lib_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def append_path(path: Optional[str | os.PathLike] = None) -> list[str]:
    if path is None:
        path = get_lib_path()
    if path not in sys.path:
        sys.path.insert(0, str(path))
    return sys.path


#: automatically append the repo root to sys.path when imported
append_path()

if __name__ == '__main__':
    print(get_lib_path())
    print(sys.path)