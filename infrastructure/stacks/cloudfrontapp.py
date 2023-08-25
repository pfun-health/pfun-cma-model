try:
    from aws_cdk import core as cdk
except ImportError:
    import aws_cdk as cdk
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_certificatemanager as acm


class CloudFrontApp(cdk.Stack):
    def __init__(self, scope, id: str, chalice_stack, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        certificate = acm.Certificate(
            self, id='44cae873-0fb0-44cf-8679-9534ba12b25b',
            domain_name='dev.pfun.app',
            certificate_name='dev.pfun.app',
        )

        # Define the CloudFront distribution
        viewer_cert = cloudfront.ViewerCertificate.from_acm_certificate(
            certificate, aliases=['dev.pfun.app'],
            security_policy=cloudfront.SecurityPolicyProtocol.TLS_V1,  # default
            ssl_method=cloudfront.SSLMethod.SNI
        )
        custom_origin_config = cloudfront.CustomOriginConfig(
            domain_name=f"{chalice_stack.stack_name.lower()}.{cdk.Aws.REGION}.amazonaws.com"
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
