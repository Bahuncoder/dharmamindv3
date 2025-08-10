# 📤 Terraform Outputs
# Infrastructure resource outputs

# ================================
# 🌐 NETWORKING OUTPUTS
# ================================
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "nat_gateway_ids" {
  description = "List of IDs of the NAT Gateways"
  value       = module.vpc.natgw_ids
}

# ================================
# ☸️ EKS CLUSTER OUTPUTS
# ================================
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_primary_security_group_id" {
  description = "Cluster security group that was created by Amazon EKS for the cluster"
  value       = module.eks.cluster_primary_security_group_id
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if enabled"
  value       = module.eks.oidc_provider_arn
}

# ================================
# 🔐 NODE GROUP OUTPUTS
# ================================
output "eks_managed_node_groups" {
  description = "Map of attribute maps for all EKS managed node groups created"
  value       = module.eks.eks_managed_node_groups
}

output "eks_managed_node_groups_autoscaling_group_names" {
  description = "List of the autoscaling group names created by EKS managed node groups"
  value       = module.eks.eks_managed_node_groups_autoscaling_group_names
}

# ================================
# 💾 DATABASE OUTPUTS
# ================================
output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = module.db.db_instance_endpoint
  sensitive   = false
}

output "db_instance_name" {
  description = "RDS instance name"
  value       = module.db.db_instance_name
}

output "db_instance_username" {
  description = "RDS instance root username"
  value       = module.db.db_instance_username
  sensitive   = true
}

output "db_instance_port" {
  description = "RDS instance port"
  value       = module.db.db_instance_port
}

output "db_subnet_group_name" {
  description = "RDS subnet group name"
  value       = module.db.db_subnet_group_name
}

output "db_parameter_group_name" {
  description = "RDS parameter group name"
  value       = module.db.db_parameter_group_name
}

# ================================
# 🗄️ CACHE OUTPUTS
# ================================
output "redis_cluster_id" {
  description = "Redis cluster ID"
  value       = module.redis.cluster_id
}

output "redis_primary_endpoint" {
  description = "Redis primary endpoint"
  value       = module.redis.primary_endpoint
}

output "redis_port" {
  description = "Redis port"
  value       = module.redis.port
}

# ================================
# 📁 STORAGE OUTPUTS
# ================================
output "s3_bucket_app_storage_id" {
  description = "ID of the S3 bucket for application storage"
  value       = aws_s3_bucket.app_storage.id
}

output "s3_bucket_app_storage_arn" {
  description = "ARN of the S3 bucket for application storage"
  value       = aws_s3_bucket.app_storage.arn
}

output "s3_bucket_backups_id" {
  description = "ID of the S3 bucket for backups"
  value       = aws_s3_bucket.backups.id
}

output "s3_bucket_backups_arn" {
  description = "ARN of the S3 bucket for backups"
  value       = aws_s3_bucket.backups.arn
}

# ================================
# 🔐 SECRETS OUTPUTS
# ================================
output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret"
  value       = aws_secretsmanager_secret.app_secrets.arn
}

output "secrets_manager_secret_name" {
  description = "Name of the Secrets Manager secret"
  value       = aws_secretsmanager_secret.app_secrets.name
}

# ================================
# 🎯 LOAD BALANCER OUTPUTS
# ================================
output "alb_id" {
  description = "ID of the Application Load Balancer"
  value       = module.alb.lb_id
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = module.alb.lb_arn
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = module.alb.lb_dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = module.alb.lb_zone_id
}

output "alb_target_group_arns" {
  description = "ARNs of the target groups"
  value       = module.alb.target_group_arns
}

# ================================
# 🔒 SSL CERTIFICATE OUTPUTS
# ================================
output "acm_certificate_arn" {
  description = "ARN of the ACM certificate"
  value       = aws_acm_certificate.main.arn
}

output "acm_certificate_domain_name" {
  description = "Domain name of the ACM certificate"
  value       = aws_acm_certificate.main.domain_name
}

output "acm_certificate_status" {
  description = "Status of the ACM certificate"
  value       = aws_acm_certificate.main.status
}

# ================================
# 🌐 DNS OUTPUTS
# ================================
output "route53_zone_id" {
  description = "Route53 hosted zone ID"
  value       = data.aws_route53_zone.main.zone_id
}

output "route53_name_servers" {
  description = "Route53 hosted zone name servers"
  value       = data.aws_route53_zone.main.name_servers
}

output "domain_name" {
  description = "Domain name"
  value       = var.domain_name
}

# ================================
# 📊 MONITORING OUTPUTS
# ================================
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.app_logs.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.app_logs.arn
}

# ================================
# 🏷️ RESOURCE INFORMATION
# ================================
output "aws_caller_identity" {
  description = "AWS caller identity information"
  value = {
    account_id = data.aws_caller_identity.current.account_id
    arn        = data.aws_caller_identity.current.arn
    user_id    = data.aws_caller_identity.current.user_id
  }
}

output "availability_zones" {
  description = "List of availability zones used"
  value       = slice(data.aws_availability_zones.available.names, 0, 3)
}

# ================================
# 🚀 DEPLOYMENT INFORMATION
# ================================
output "deployment_info" {
  description = "Deployment information for CI/CD"
  value = {
    cluster_name    = module.eks.cluster_id
    cluster_region  = var.aws_region
    database_url    = "postgresql://${module.db.db_instance_username}@${module.db.db_instance_endpoint}:${module.db.db_instance_port}/${module.db.db_instance_name}"
    redis_url       = "redis://${module.redis.primary_endpoint}:${module.redis.port}/0"
    app_url         = "https://${var.domain_name}"
    api_url         = "https://api.${var.domain_name}"
  }
  sensitive = true
}

# ================================
# 🔧 KUBECTL CONFIGURATION
# ================================
output "kubectl_config" {
  description = "kubectl configuration for accessing the cluster"
  value = {
    cluster_name     = module.eks.cluster_id
    cluster_endpoint = module.eks.cluster_endpoint
    cluster_ca_data  = module.eks.cluster_certificate_authority_data
    region          = var.aws_region
  }
  sensitive = false
}

# ================================
# 💰 COST INFORMATION
# ================================
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown (approximate)"
  value = {
    message = "Cost estimates are approximate and may vary based on usage"
    eks_cluster = "~$73/month"
    rds_instance = "~$200-500/month (depending on instance class)"
    load_balancer = "~$25/month"
    nat_gateway = "~$45/month"
    data_transfer = "Variable based on usage"
    storage = "Variable based on usage"
    note = "Use AWS Cost Explorer for accurate cost tracking"
  }
}
