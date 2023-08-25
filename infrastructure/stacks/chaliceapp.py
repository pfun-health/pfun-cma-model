import os

try:
    from aws_cdk import core as cdk
except ImportError:
    import aws_cdk as cdk
from aws_cdk import aws_autoscaling as autoscaling

from chalice.cdk import Chalice

import sys
from pathlib import Path
root_path = Path(__file__).parents[2]
for pth in [root_path, ]:
    pth = str(pth)
    if pth not in sys.path:
        sys.path.insert(0, pth)
RUNTIME_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), os.pardir, 'runtime')


class ChaliceApp(cdk.Stack):

    def __init__(self, scope, id, **kwargs):
        super().__init__(scope, id, **kwargs)
        self.chalice = Chalice(
            self, 'PFunCMAModelChaliceApp', source_dir=RUNTIME_SOURCE_DIR,
            stage_config={
                'environment_variables': {
                }
            }
        )

        self.chalice.source_repository = 'https://github.com/rocapp/pfun-cma-model'
        self.chalice.stage_config['name'] = 'dev'
        self.chalice.stage_config['lambda_memory_size'] = 256
        self.chalice.stage_config['lambda_timeout'] = 15

        launch_configuration = autoscaling.CfnLaunchConfiguration(
            self, "PFunCMAModelLaunchConfiguration",
            image_id='ami-02675d30b814d1daa',
            instance_type='m5.large',
            # other configuration options
        )

        autoscaling_group = autoscaling.CfnAutoScalingGroup(
            self, "PFunCMAModelScalingGroup",
            min_size='1',
            max_size='10',
            desired_capacity='5',
            launch_configuration_name=launch_configuration.ref,
            availability_zones=[
                'us-east-1a',
                'us-east-1b',
                'us-east-1c',
            ]
        )
