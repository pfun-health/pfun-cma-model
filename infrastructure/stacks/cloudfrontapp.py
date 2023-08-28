try:
    from aws_cdk import core as cdk
except ImportError:
    import aws_cdk as cdk
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_certificatemanager as acm
import boto3


def get_output_value(stack, output_key):
    return next(out['OutputValue'] for out in
                boto3.resource('cloudformation').
                Stack(stack.stack_name).outputs
                if out['OutputKey'] == output_key)


class CloudFrontApp(cdk.Stack):
    def __init__(self, scope, id: str, chalice_stack, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        existing_certificate_arn = 'arn:aws:acm:us-east-1:860311922912:certificate/01704bec-f302-4d8a-a1ae-b211d880a9d6'
        certificate = acm.Certificate.from_certificate_arn(
            self, "DevPFunCertificate", existing_certificate_arn)

        # Define the CloudFront distribution
        viewer_cert = cloudfront.ViewerCertificate.from_acm_certificate(
            certificate, aliases=['dev.pfun.app'],
            security_policy=cloudfront.SecurityPolicyProtocol.TLS_V1,  # default
            ssl_method=cloudfront.SSLMethod.SNI
        )

        #: add dependency on chalice stack
        self.add_dependency(chalice_stack)

        #: get the origin URL from chalice stack
        endpoint_url = get_output_value(chalice_stack, 'EndpointURL')

        custom_origin_config = cloudfront.CustomOriginConfig(
            domain_name=endpoint_url
        )
        distribution = cloudfront.CloudFrontWebDistribution(
            self, 'PFunCMAEndpointDistribution',
            viewer_certificate=viewer_cert,
            origin_configs=[
                cloudfront.SourceConfiguration(
                    custom_origin_source=custom_origin_config,
                    behaviors=[
                        cloudfront.Behavior(is_default_behavior=True),
                        cloudfront.Behavior(path_pattern="/*")
                    ]
                ),
            ]
        )

        # Output the CloudFront distribution domain name
        cdk.CfnOutput(
            self, 'DistributionDomainName',
            value=distribution.distribution_domain_name
        )
