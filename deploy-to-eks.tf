variable "create_helm_repo" {
  description = "Toggle to create Helm repo on S3"
  default     = false
}

resource "aws_s3_bucket" "helm_repo" {
    count  = var.create_helm_repo ? 1 : 0
    bucket = "my-helm-repo"
    acl    = "private"

    versioning {
        enabled = true
    }

    # Ensure that S3 bucket has a Public Access block
    block_public_acls       = true
    block_public_policy     = true
    ignore_public_acls      = true
    restrict_public_buckets = true

    # Ensure that an S3 bucket has a lifecycle configuration
    lifecycle {
        prevent_destroy = true
        rule {
            id      = "example-rule"
            status  = "Enabled"
            prefix  = "example-prefix"
            enabled = true
            expiration {
                days = 90
            }
        }
    }

    # Ensure S3 buckets should have event notifications enabled
    event_notification {
        queue {
            queue_arn = aws_sqs_queue.s3_bucket_queue.arn
            events    = ["s3:ObjectCreated:*"]
        }
        filter_prefix = "prefix/"
        filter_suffix = ".txt"
    }

    # Ensure that S3 bucket has cross-region replication enabled
    replication_configuration {
        role = aws_iam_role.replication.arn

        rule {
            id      = "example-rule"
            status  = "Enabled"
            prefix  = "example-prefix"
            destination {
                bucket        = "arn:aws:s3:::destination-bucket"
                storage_class = "STANDARD"
            }
        }
    }

    # Ensure that S3 buckets are encrypted with KMS by default
    server_side_encryption_configuration {
        rule {
            apply_server_side_encryption_by_default {
                sse_algorithm = "aws:kms"
            }
        }
    }

    # Ensure the S3 bucket has access logging enabled
    logging {
        target_bucket = "my-log-bucket"
        target_prefix = "my-log-prefix"
    }
}


resource "null_resource" "init_helm_repo" {
  count = var.create_helm_repo ? 1 : 0

  provisioner "local-exec" {
    command = "helm s3 init s3://my-helm-repo/charts"

    environment = {
      AWS_ACCESS_KEY_ID     = var.aws_access_key_id
      AWS_SECRET_ACCESS_KEY = var.aws_secret_access_key
      AWS_DEFAULT_REGION    = var.aws_default_region
    }
  }

  triggers = {
    always_run = "${timestamp()}"
  }
}

resource "null_resource" "deploy_to_eks" {
  depends_on = [null_resource.test_local_z38s, null_resource.test_local_d2bd]

  provisioner "local-exec" {
    command = <<EOL
      # Package the Helm chart
      helm package ./path/to/helm/chart

      # Upload the Helm chart to your chart repository (e.g., AWS S3)
      helm s3 push ./path/to/helm/chart/package.tgz my-helm-repo

      # Update the Helm repo
      helm repo update

      # Deploy the Helm chart to the EKS cluster
      helm upgrade --install pfun-cma-model my-helm-repo/pfun-cma-model --namespace your-namespace
    EOL

    environment = {
      AWS_ACCESS_KEY_ID     = var.aws_access_key_id
      AWS_SECRET_ACCESS_KEY = var.aws_secret_access_key
      AWS_DEFAULT_REGION    = var.aws_default_region
      KUBECONFIG            = var.kubeconfig_path
    }
  }

  triggers = {
    always_run = "${timestamp()}"
  }
}

variable "aws_access_key_id" {
  description = "AWS Access Key ID"
}

variable "aws_secret_access_key" {
  description = "AWS Secret Access Key"
}

variable "aws_default_region" {
  description = "AWS Default Region"
  default     = "us-west-2"
}

variable "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  default     = "~/.kube/config"
}
