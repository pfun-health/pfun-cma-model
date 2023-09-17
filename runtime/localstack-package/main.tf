provider "aws" {
  region                      = "us-east-1"
  shared_config_files      = ["/.aws/config"]
  shared_credentials_files = ["/.aws/credentials"]
  profile                     = "localstack"
  s3_use_path_style         = true
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  endpoints {
    apigateway     = "https://localhost.localstack.cloud:31566"
    lambda         = "https://localhost.localstack.cloud:31566"
    s3             = "https://localhost.localstack.cloud:31566"
    sts            = "https://localhost.localstack.cloud:31566"
    iam            = "https://localhost.localstack.cloud:31566"
    dynamodb       = "https://localhost.localstack.cloud:31566"
    sqs            = "https://localhost.localstack.cloud:31566"
    sns            = "https://localhost.localstack.cloud:31566"
    ec2            = "https://localhost.localstack.cloud:31566"
    apigatewayv2   = "https://localhost.localstack.cloud:31566"
    cloudformation = "https://localhost.localstack.cloud:31566"
    secretsmanager = "https://localhost.localstack.cloud:31566"
  }
}