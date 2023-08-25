#!/usr/bin/env python3
import sys
from pathlib import Path
root_path = Path(__file__).parents[2]
for pth in [root_path, ]:
    pth = str(pth)
    if pth not in sys.path:
        sys.path.insert(0, pth)
try:
    from aws_cdk import core as cdk
except ImportError:
    import aws_cdk as cdk
try:
    from stacks.chaliceapp import ChaliceApp
except (ImportError, ModuleNotFoundError):
    import importlib
    ChaliceApp = importlib.import_module('.chaliceapp', package='infrastructure.stacks').ChaliceApp

app = cdk.App()
ChaliceApp(app, 'pfun-cma-model')

app.synth()
