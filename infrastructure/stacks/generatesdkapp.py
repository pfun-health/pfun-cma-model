from aws_cdk import aws_cloudformation as cfn, aws_lambda as lambda_
import aws_cdk as core
from aws_cdk.aws_lambda_python_alpha import PythonFunction


class GenerateSDKApp(core.Stack):
    def __init__(self, scope, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Define the custom resource
        sdk_generation_resource = cfn.CfnCustomResource(
            self, "SDKGenerationResource",
            service_token=PythonFunction(
                self, "SDKGenerationFunction",
                entry="lambda",
                runtime=lambda_.Runtime.PYTHON_3_10
            ).function_arn
        )

        # Output the generated SDK URL
        core.CfnOutput(
            self, "SDK",
            value=sdk_generation_resource.get_att("SDK").to_string(),
        )
