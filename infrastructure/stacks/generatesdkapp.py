from aws_cdk import aws_cloudformation as cfn, aws_lambda as lambda_
import aws_cdk as core


class GenerateSDKApp(core.Stack):
    def __init__(self, scope, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Define the custom resource
        sdk_generation_resource = cfn.CfnCustomResource(
            self, "SDKGenerationResource",
            service_token=lambda_.SingletonFunction(
                self, "SDKGenerationFunction",
                uuid="d4b8e4a9-3f7c-4d3a-bc22-eeb541558b5f",
                code=lambda_.Code.from_asset("lambda"),
                handler="index.handler",
                runtime=lambda_.Runtime.PYTHON_3_9,
            ).function_arn,
        )

        # Output the generated SDK URL
        core.CfnOutput(
            self, "SDK",
            value=sdk_generation_resource.get_att("SDK").to_string(),
        )
