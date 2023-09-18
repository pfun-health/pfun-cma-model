provider "aws" {
  region = "us-west-2"
}

locals {
  cluster_name = "pfun-eks-cluster"
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  name    = "pfun-eks-vpc"
  cidr    = "10.0.0.0/16"
  azs     = ["us-west-2a", "us-west-2b", "us-west-2c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.4.0/24", "10.0.5.0/24", "10.0.6.0/24"]
}

module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version = "~> 17.0"  # specify the version
  cluster_name    = local.cluster_name
  cluster_version = "1.25"
  subnets         = module.vpc.private_subnets
  tags = {
    Terraform   = "true"
    Environment = "dev"
  }

  vpc_id = module.vpc.vpc_id
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  value = module.eks.cluster_security_group_id
}

output "cluster_arn" {
  value = module.eks.cluster_arn
}
