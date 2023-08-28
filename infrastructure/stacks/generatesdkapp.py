import aws_cdk as core
import boto3
import sys
from pathlib import Path
import importlib
root_path = Path(__file__).parents[2]
for pth in [root_path, ]:
    pth = str(pth)
    if pth not in sys.path:
        sys.path.insert(0, pth)
get_output_value = \
    importlib.import_module(
        '.cloudfrontapp', package='infrastructure.stacks').get_output_value


class GenerateSDKApp(core.Stack):
    def __init__(self, scope, id: str, chalice_stack, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        self.add_dependency(chalice_stack)
        
        rest_api_id = get_output_value(chalice_stack, 'RestAPIId')
        response = boto3.client('apigateway').get_sdk(
            restApiId=rest_api_id,
            stageName='api',
            sdkType='javascript',
        )
        sdk_stream = response['body']
        sdk_bytes = sdk_stream.read()

        # Output the generated SDK
        core.CfnOutput(
            self, "SDK",
            value=sdk_bytes
        )
