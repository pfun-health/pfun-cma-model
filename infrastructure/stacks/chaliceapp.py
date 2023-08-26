import json
import os

try:
    from aws_cdk import core as cdk
except ImportError:
    import aws_cdk as cdk
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_iam as iam

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

        attach_policy_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                'apigateway:POST',
                'lambda:CreateFunction',
                'lambda:CreateAlias',
                'iam:AttachRolePolicy'
            ],
            resources=[
                'arn:aws:apigateway:*::/*',
                'arn:aws:lambda:*:*:function:*',
                'arn:aws:iam::*:role/pfun-cma-model-dev',
                'arn:aws:sts::860311922912:assumed-role/pfun-cma-model-*'
            ]
        )

        iam_role = iam.Role(
            self, 'PFunCMAModelRole',
            assumed_by=iam.ServicePrincipal('lambda.amazonaws.com'),)
        iam_role.grant_assume_role(
            iam.ServicePrincipal('apigateway.amazonaws.com'))
        iam_role.grant_assume_role(
            iam.ServicePrincipal('lambda.amazonaws.com'))
        iam_role.grant_assume_role(iam.ServicePrincipal('iam.amazonaws.com'))
        iam_role.add_to_policy(attach_policy_statement)

        trust_policy = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=["sts:AssumeRole", "iam:AttachRolePolicy"],
            resources=["*"],
        )

        sts_role = iam.LazyRole(
            self, "PFunSTSLazyRole",
            assumed_by=iam.ServicePrincipal("sts.amazonaws.com")
        )
        sts_role.add_to_policy(trust_policy)

        pfun_cma_model_dev_role = iam.Role(
            self, 'PFunDevSTSRole', role_name='pfun-cma-model-dev',
            assumed_by=iam.ServicePrincipal("sts.amazonaws.com"),
        )

        sts_role.grant_assume_role(pfun_cma_model_dev_role)

        pfun_cma_model_dev_lambda_role = iam.Role(
            self, "PFunCMAModelDevLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )

        pfun_cma_model_dev_lambda_role.add_to_policy(trust_policy)

        statements = json.loads(
            open(
                os.path.join(
                    RUNTIME_SOURCE_DIR,
                    'gateway-assume-role-policy.json'), 'r').read())['Statement']
        policy_effect = iam.Effect.ALLOW if statements[0]['Effect'] == 'Allow' \
            else iam.Effect.DENY
        policy_doc = iam.PolicyDocument(statements=[
            iam.PolicyStatement(
                actions=statements[0]['Action'],
                effect=policy_effect,
                resources=statements[0]['Resource']
            )
        ])
        apihandler_policy = \
            iam.Policy(
                self, id='PFunCMAModel-APIHandler-Policy',
                document=policy_doc, force=True
            )
        apihandler_policy.attach_to_role(iam_role)  # type: ignore

        iam.CfnRolePolicy(self, 'PFunCMAModel-APIHandler-Policy-Role',
                          role_name=iam_role.role_name,
                          policy_name=apihandler_policy.policy_name,
                          policy_document=apihandler_policy.document)
