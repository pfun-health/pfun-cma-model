import os

try:
    from aws_cdk import core as cdk
except ImportError:
    import aws_cdk as cdk
from aws_cdk import aws_autoscaling as autoscaling

from chalice.cdk import Chalice


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
            image_id='ami-041c209db5829dfb9',
            instance_type='m5.large',
            # other configuration options
        )

        autoscaling_group = autoscaling.CfnAutoScalingGroup(
            self, "PFunCMAModelScalingGroup",
            min_size='1',
            max_size='10',
            desired_capacity='5',
            launch_configuration_name=launch_configuration.ref,
        )
