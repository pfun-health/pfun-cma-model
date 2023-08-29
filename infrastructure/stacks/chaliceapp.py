import json
import os

try:
    from aws_cdk import core as cdk
except ImportError:
    import aws_cdk as cdk
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_iam as iam
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import (
    aws_ec2 as ec2,
    aws_route53_targets as targets,
    aws_route53 as route53,
    aws_elasticloadbalancingv2_targets as elbv2_targets,
    aws_lambda as lambda_
)

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


def get_lambda_function(stack, name):
    """get lambda functions from chalice"""
    fun = stack.chalice.get_function(name)
    return lambda_.Function.from_function_attributes(
        stack, fun.node.path, function_arn=fun.function_arn,
        same_environment=True)


class ChaliceApp(cdk.Stack):

    def __init__(self, scope, id, **kwargs):
        super().__init__(scope, id, **kwargs)
        self.chalice = Chalice(
            self, 'PFunCMAModelChaliceApp', source_dir=RUNTIME_SOURCE_DIR,
        )
        self.chalice.source_repository = 'https://github.com/rocapp/pfun-cma-model'

        # Create a VPC
        vpc = ec2.Vpc(self, "PFunCMAModelVPC",
                      cidr="10.0.0.3/16",
                      max_azs=2,
                      nat_gateways=1,
                      subnet_configuration=[
                          ec2.SubnetConfiguration(
                              name="public",
                              subnet_type=ec2.SubnetType.PUBLIC
                          )
                      ]
                      )

        al_image = ec2.AmazonLinuxImage(
            generation=ec2.AmazonLinuxGeneration.AMAZON_LINUX_2)
        al_inst = ec2.InstanceType.of(
            ec2.InstanceClass.BURSTABLE3, ec2.InstanceSize.MEDIUM)

        # Configure the ASG as a target for the ALB
        alb = elbv2.ApplicationLoadBalancer(
            self, "PFunCMAModelLoadBalancer", vpc=vpc, internet_facing=True)
        alb.add_redirect(source_port=443, target_port=80)

        listener = alb.add_listener("PFunCMAModelListener", port=80, open=True)

        #: lambda function targets
        chalice_lambda_functions = (
            'RunModel', 'FitModel', 'RunAtTime', 'FakeAuth', 'APIHandler')

        # Configure permissions
        for function_name in chalice_lambda_functions:
            func = self.chalice.get_function(function_name)
            func.add_permission(
                f'Invoke{function_name}FromLoadBalancer',
                principal=iam.ServicePrincipal(
                    'elasticloadbalancing.amazonaws.com'),
                action='lambda:InvokeFunction',
                source_arn=alb.load_balancer_arn
            )
            func.add_permission(
                f'Invoke{function_name}APIHandler',
                principal=iam.ServicePrincipal('apigateway.amazonaws.com'),
                action='lambda:InvokeFunction')
            func.add_permission(
                f'Invoke{function_name}APIHandler',
                principal=iam.ServicePrincipal(
                    'apigatewayv2.amazonaws.com'),
                action='lambda:InvokeFunction')
        fake_auth = self.chalice.get_function('FakeAuth')
        fake_auth.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=['sts:AssumeRole', 'sts:GetCallerIdentity',
                         'secretsmanager:GetSecretValue'],
                resources=['*']
            )
        )

        # Create a listener rule to forward requests to the Chalice API
        listener.add_targets("PFunCMAModelAPIHandlerTarget",
                             targets=[
                                 elbv2_targets.LambdaTarget(
                                     get_lambda_function(self, 'APIHandler')),
                             ],
                             health_check=elbv2.HealthCheck(
                                 interval=cdk.Duration.minutes(5)),
                             )

        # Output the ALB DNS name
        cdk.CfnOutput(self, "PFunAlbDNSName", value=alb.load_balancer_dns_name)
