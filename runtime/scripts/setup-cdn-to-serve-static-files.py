import boto3
import uuid
import json


def create_s3_bucket(bucket_name):
    try:
        s3_client = boto3.client('s3')
        s3_client.create_bucket(Bucket=bucket_name)
    except Exception as e:
        print(e)


def create_s3_http_api(bucket_name, api_name, role_arn, object_key='{{proxy}}'):
    # Create an Amazon API Gateway REST API
    apigateway_client = boto3.client('apigateway')

    try:
        api_response = apigateway_client.create_rest_api(
            name=api_name,
            endpointConfiguration={
                'types': ['REGIONAL']
            }
        )
    except Exception as e:
        api_response = {}
        for rest_api in reversed(apigateway_client.get_rest_apis()['items']):
            if rest_api['name'] == api_name:
                api_response = rest_api
                break
    print('API ID: %s' % api_response['id'])

    # Retrieve the API ID from the response
    api_id = str(api_response['id'])

    # Get the resources for the API
    response = apigateway_client.get_resources(
        restApiId=api_id  # Replace with your API identifier
    )

    # Find the root resource
    resource_response = next(
        (resource for resource in response['items'] if resource['path'] == '/'),
        None
    )

    # Retrieve the resource ID from the response
    resource_id = resource_response['id']
    print('Resource ID: %s' % resource_id)

    method_response = apigateway_client.put_method(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='GET',
        authorizationType='NONE'
    )

    # Set up an integration with the S3 bucket
    region = boto3.Session().region_name
    integration_response = apigateway_client.put_integration(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='GET',
        integrationHttpMethod='GET',
        type='AWS',
        uri=f'arn:aws:apigateway:{region}:s3:path/{bucket_name}/{object_key}',
        credentials=role_arn
    )

    # Deploy the API
    deployment_response = apigateway_client.create_deployment(
        restApiId=api_id,
        stageName='api'
    )

    return api_id


def configure_s3_bucket_policy(bucket_name, cloudfront_domains):
    _domains = []
    for cloudfront_domain in cloudfront_domains:
        _domains.extend([
            f'http://{cloudfront_domain}/*',
            f'https://{cloudfront_domain}/*'])
    bucket_policy = {
        'Version': '2012-10-17',
        'Statement': [{
            'Sid': 'AllowCloudFrontAccess',
            'Effect': 'Deny',
            'Principal': '*',
            'Action': 's3:*',
            'Resource': f'arn:aws:s3:::{bucket_name}/*',
            'Condition': {
                'StringNotLike': {
                    'aws:Referer': cloudfront_domains
                }
            }
        }]
    }

    s3_client = boto3.client('s3')
    s3_client.put_bucket_policy(
        Bucket=bucket_name,
        Policy=json.dumps(bucket_policy)
    )


def create_cloudfront_distribution(bucket_name, unique_caller_reference,
                                   acm_certificate_arn, certificate_id,
                                   cloudfront_domains,
                                   api_domain_name, api_origin_path='/api'):
    cloudfront_client = boto3.client('cloudfront')
    distribution_config = {
        'CallerReference': unique_caller_reference,
        'Comment': 'CloudFront distribution for static files',
        'Origins': {
            'Quantity': 2,
            'Items': [
                {
                    'Id': 'APIOrigin',
                    'DomainName': api_domain_name,
                    'OriginPath': api_origin_path,
                    'CustomHeaders': {
                        'Quantity': 0,
                        'Items': []
                    },
                    'CustomOriginConfig': {
                        'HTTPPort': 80,
                        'HTTPSPort': 443,
                        'OriginProtocolPolicy': 'https-only',
                                                'OriginSslProtocols': {
                                                    'Quantity': 3,
                                                    'Items': ['TLSv1', 'TLSv1.1', 'TLSv1.2']
                                                }
                    }
                },
                {
                    'Id': 'S3Origin',
                    'DomainName': f'{bucket_name}.s3-website-us-east-1.amazonaws.com',
                    'S3OriginConfig': {
                        'OriginAccessIdentity': ''
                    }
                }
            ]
        },
        'DefaultRootObject': 'index.template.html',
        'DefaultCacheBehavior': {
            'TargetOriginId': 'S3Origin',
            'ForwardedValues': {
                'QueryString': False,
                'Cookies': {
                    'Forward': 'none'
                }
            },
            'TrustedSigners': {
                'Enabled': False,
                'Quantity': 0
            },
            'ViewerProtocolPolicy': 'redirect-to-https',
            'MinTTL': 0,
            'AllowedMethods': {
                'Quantity': 2,
                'Items': ['GET', 'HEAD'],
                'CachedMethods': {
                    'Quantity': 2,
                    'Items': ['GET', 'HEAD']
                }
            },
            'Compress': True
        },
        'PriceClass': 'PriceClass_All',
        'Enabled': True,
        'ViewerCertificate': {
            'ACMCertificateArn': acm_certificate_arn,
            'SSLSupportMethod': 'sni-only',
            'MinimumProtocolVersion': 'TLSv1.2_2019',
            'Certificate': certificate_id
        },
        'Aliases': {
            'Quantity': len(cloudfront_domains),
            'Items': cloudfront_domains
        }
    }

    cloudfront_client.create_distribution(
        DistributionConfig=distribution_config)


def get_acm_certificate_info(certificate_name):
    acm_client = boto3.client('acm')
    response = acm_client.list_certificates()

    for certificate in response['CertificateSummaryList']:
        if certificate['DomainName'] == certificate_name:
            certificate_arn = certificate['CertificateArn']
            certificate_id = certificate_arn.split('/')[-1]
            return certificate_arn, certificate_id

    return None, None


def get_integration_details(rest_api_id, resource_path, http_method, region='us-east-1'):
    # Create the API Gateway client
    apigateway_client = boto3.client('apigateway')

    try:
        # Get the resource details
        resource_response = apigateway_client.get_resources(
            restApiId=rest_api_id
        )

        if 'items' in resource_response:
            resource_id = None
            for resource in resource_response['items']:
                if resource['path'] == resource_path:
                    resource_id = resource['id']
                    break

            if resource_id:
                # Get the method details
                method_response = apigateway_client.get_method(
                    restApiId=rest_api_id,
                    resourceId=resource_id,
                    httpMethod=http_method
                )

                if 'methodIntegration' in method_response:
                    integration = method_response['methodIntegration']
                    integration_type = integration['type']
                    integration_uri = integration['uri']

                    if integration_type and integration_uri:
                        base_url = f"https://{rest_api_id}.execute-api.{region}.amazonaws.com/prod"
                        http_url = construct_http_url(base_url, resource_path)

                        return integration_type, integration_uri, http_url
                    return integration_type, integration_uri, None
                else:
                    return None, None

            else:
                return None, None

        else:
            return None, None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None


def construct_http_url(base_url, resource_path):
    if base_url.endswith('/'):
        base_url = base_url[:-1]  # Remove trailing slash if present

    if resource_path.startswith('/'):
        resource_path = resource_path[1:]  # Remove leading slash if present

    return f"{base_url}/{resource_path}"


if __name__ == '__main__':
    # Usage
    certificate_name = '*.dev.pfun.app'
    acm_certificate_arn, certificate_id = get_acm_certificate_info(
        certificate_name)

    print(f"ACM Certificate ARN: {acm_certificate_arn}")
    print(f"Certificate ID: {certificate_id}")

    # Usage example
    bucket_name = 'pfun-cma-model-www'
    cloudfront_domains = [
        'static.dev.pfun.app',
        'pfun-cma-model-www.s3-us-east-1.amazonaws.com',
        'pfun-cma-model-www.s3-website-us-east-1.amazonaws.com',
        'pfun-cma-model-www.s3-website.us-east-1.amazonaws.com',
        'ryj5p4wfwe.execute-api.us-east-1.amazonaws.com'
        '*.cloudfront.net',
        '69.149.120.126'
    ]
    custom_domains = [
        'static.dev.pfun.app',
    ]

    # Step 1: Create S3 bucket
    create_s3_bucket(bucket_name)

    # Step 2: Configure S3 bucket policy
    configure_s3_bucket_policy(bucket_name, cloudfront_domains)

    #: Step 2.1: Create S3 HTTP API
    api_name = 'PFunCMADevStaticFilesHttpApi'
    role_policy_doc = \
        json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "cloudfront:GetDistribution",
                            "cloudfront:GetDistributionConfig",
                            "cloudfront:ListDistributions",
                            "cloudfront:ListDistributionsByWebACLId"
                        ],
                        "Resource": "*"
                    }
                ]
            }
        )
    trust_policy_doc = \
        json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "apigateway.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
        )
    iam_client = boto3.client('iam')
    try:
        role_policy_resp = iam_client.create_policy(
            PolicyName='PFunCMADevStaticFilesHttpApiPolicy',
            PolicyDocument=role_policy_doc
        )
    except:
        role_policy_resp = None
        for policy in iam_client.list_policies()['Policies']:
            if policy['PolicyName'] == \
                    'PFunCMADevStaticFilesHttpApiPolicy':
                role_policy_resp = {'Policy': policy}
                break
    try:
        role_resp = iam_client.create_role(
            RoleName='PFunCMADevStaticFilesHttpApiRole',
            AssumeRolePolicyDocument=trust_policy_doc,
            Description='Role for PFunCMADevStaticFilesHttpApi',
        )
    except:
        role_resp = iam_client.get_role(RoleName='PFunCMADevStaticFilesHttpApiRole')
    role_arn = role_resp['Role']['Arn']
    iam_client.attach_role_policy(
        RoleName='PFunCMADevStaticFilesHttpApiRole',
        PolicyArn=role_policy_resp['Policy']['Arn']
    )

    api_id = create_s3_http_api(bucket_name, api_name, role_arn)

    deets = get_integration_details(api_id, '/', 'GET')
    print('Deets:', deets)
    integration_type, integration_uri, http_url = deets

    # Step 3: Create CloudFront distribution
    unique_caller_reference = f'cloudfront-distribution-{uuid.uuid4()}'
    create_cloudfront_distribution(
        bucket_name,
        unique_caller_reference,
        acm_certificate_arn,
        certificate_id,
        custom_domains,
        http_url.replace('https://', '').split('/')[0]
    )
