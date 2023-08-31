import aws_cdk as cdk
from constructs import Construct
from stacks.chaliceapp import ChaliceApp


class PFuncMAModelPipelineAppStage(cdk.Stage):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        chaliceStack = ChaliceApp(self, "ChaliceApp")