# Staging Environment
# Terraform configuration for staging deployment

terraform {
  required_version = ">= 1.0"
  
  backend "s3" {
    bucket = "dharmamind-terraform-state"
    key    = "staging/terraform.tfstate"
    region = "us-west-2"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Staging environment variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "staging"
}

variable "aws_region" {
  description = "AWS region for staging"
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "dharmamind-staging"
}

variable "replicas" {
  description = "Number of replicas for each service"
  type = object({
    backend   = number
    dharmallm = number
    frontend  = number
  })
  default = {
    backend   = 2
    dharmallm = 2
    frontend  = 2
  }
}

variable "instance_types" {
  description = "EC2 instance types for different workloads"
  type = object({
    general = list(string)
    gpu     = list(string)
  })
  default = {
    general = ["t3.medium", "t3.large"]
    gpu     = ["g4dn.xlarge", "g4dn.2xlarge"]
  }
}

# Staging-specific resources
resource "aws_eks_cluster" "staging" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids = aws_subnet.staging[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_policy,
    aws_iam_role_policy_attachment.service_policy,
  ]
}

# Staging environment outputs
output "cluster_info" {
  description = "Staging cluster information"
  value = {
    cluster_name     = aws_eks_cluster.staging.name
    cluster_endpoint = aws_eks_cluster.staging.endpoint
    cluster_version  = aws_eks_cluster.staging.version
  }
}

output "service_urls" {
  description = "Staging service URLs"
  value = {
    backend_url   = "https://api-staging.dharmamind.ai"
    dharmallm_url = "https://llm-staging.dharmamind.ai"
    frontend_url  = "https://staging.dharmamind.ai"
    monitoring    = "https://monitoring-staging.dharmamind.ai"
  }
}

# Staging environment notes
locals {
  staging_notes = {
    purpose = "Pre-production testing and validation"
    features = [
      "Production-like environment",
      "Automated deployments from main branch",
      "Performance testing",
      "Security scanning",
      "Load testing capabilities"
    ]
    access = [
      "Internal team access only",
      "VPN required for some services",
      "Monitoring dashboards available",
      "Log aggregation enabled"
    ]
  }
}