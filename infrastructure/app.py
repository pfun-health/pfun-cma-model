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
    from stacks.cloudfrontapp import CloudFrontApp
except (ImportError, ModuleNotFoundError):
    import importlib
    ChaliceApp = importlib.import_module('.chaliceapp', package='infrastructure.stacks').ChaliceApp
    CloudFrontApp = importlib.import_module('.cloudfrontapp', package='infrastructure.stacks').CloudFrontApp

environment = cdk.Environment(account='860311922912', region='us-east-1')
app = cdk.App()
chalice_stack = ChaliceApp(app, 'pfun-cma-model', env=environment)
cloudfront_stack = CloudFrontApp(app, 'PFunCMAEndpointDistribution', chalice_stack, env=environment)

app.synth()
