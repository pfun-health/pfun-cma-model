import zipfile
import shutil
import os
from pathlib import Path

repo_root = Path(__file__).parents[2]

if __name__ == '__main__':
    shutil.move(repo_root / 'sdk', repo_root / 'sdk.zip')
    with zipfile.ZipFile(repo_root / 'sdk.zip') as zfile:
        zfile.extractall(repo_root / 'runtime/chalicelib/www')
    os.remove(repo_root / 'sdk.zip')
