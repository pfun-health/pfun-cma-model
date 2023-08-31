import aws_cdk as cdk
from constructs import Construct
from aws_cdk.pipelines import (
    CodePipeline, CodePipelineSource, ShellStep,
    ManualApprovalStep
)
from stacks.appstage import PFuncMAModelPipelineAppStage


class PFunCMAModelPipelineStack(cdk.Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        pipeline = CodePipeline(self, "Pipeline",
                                pipeline_name="PFunCMAModelPipeline",
                                synth=ShellStep("Synth",
                                                input=CodePipelineSource.git_hub("rocapp/pfun-cma-model", "main"),
                                                install_commands=[
                                                    "cd ${CODEBUILD_SRC_DIR}",
                                                    "npm install -g aws-cdk",
                                                    "pip install -r requirements.txt"],
                                                commands=["cdk synth"]
                                                )
                                )
        pipeline.add_stage(
            PFuncMAModelPipelineAppStage(
                self, "LocalTestAppStage", env=kwargs['env'])
        )

        localTestWave = pipeline.add_wave("LocalTest")

        localTestWave.add_pre(
            ShellStep(
                "PreLocalTestShellStep-GenerateSDK",
                commands=[
                    "cd ${CODEBUILD_SRC_DIR}",
                    "./runtime/scripts/generate-sdk.sh"
                ]
            )
        )

        localTestWave.add_post(
            ManualApprovalStep('ApproveLocalTest')
        )
