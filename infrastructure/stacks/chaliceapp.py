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
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_certificatemanager as acm,
    aws_apigateway as apigw,
    aws_apigatewayv2 as apigwv2,
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
                      cidr="10.0.0.0/16",
                      enable_dns_hostnames=True,
                      enable_dns_support=True,
                      create_internet_gateway=True,
                      max_azs=2,
                      nat_gateways=1,
                      subnet_configuration=[
                          ec2.SubnetConfiguration(
                              name="public",
                              subnet_type=ec2.SubnetType.PUBLIC
                          )
                      ]
                      )

        # Configure the ASG as a target for the ALB
        alb = elbv2.ApplicationLoadBalancer(
            self, "PFunCMAModelLoadBalancer", vpc=vpc, internet_facing=True,
            ip_address_type=elbv2.IpAddressType.IPV4)
        alb.add_security_group(ec2.SecurityGroup(
            self, "PFunAlbSecurityGroup", vpc=vpc,
            allow_all_outbound=True,
        ))
        certificate = acm.Certificate.from_certificate_arn(
            self, 'PFunCMAModelListenerCert',
            'arn:aws:acm:us-east-1:860311922912:certificate/01704bec-f302-4d8a-a1ae-b211d880a9d6'
        )
        listener = alb.add_listener("PFunCMAModelListener", port=443, open=True)
        listener.add_certificates("PFunCMAModelListenerCert", certificates=[
            certificate])

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
            func.add_to_role_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        's3:Get*',
                        's3:List*',
                    ],
                    resources=['*']
                )
            )
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
        alb.add_redirect(source_protocol=elbv2.ApplicationProtocol.HTTP,
                         target_protocol=elbv2.ApplicationProtocol.HTTPS,
                         source_port=80, target_port=443)
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

        # Create the Custom Domain Name
        domain_name_raw = 'dev.pfun.app'

        # Associate the Custom Domain Name with the HTTP API
        rest_api = self.chalice.get_resource('RestAPI')
        #: ref: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-apigateway-restapi.html#aws-resource-apigateway-restapi-return-values
        #: ref: https://stackoverflow.com/a/65709106/1871569
        http_api = apigw.RestApi.from_rest_api_id(
            self, 'PFunDevCMAModelHttpApi',
            rest_api_id=rest_api.get_att('RestApiId').to_string()
        )

        # Create the HttpApiMapping
        apigwv2.CfnApiMapping(self, 'PFunDevCMAModelHttpApiMapping',
                              api_id=http_api.rest_api_id,
                              domain_name=domain_name_raw,
                              stage='api')

        # Output the Custom Domain Name
        cdk.CfnOutput(self, 'PFunDevCMAModelCustomDomainNameOutput',
                      value=domain_name_raw)
