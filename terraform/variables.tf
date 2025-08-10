# 🔧 Terraform Variables
# Infrastructure configuration variables

# ================================
# 🌟 PROJECT CONFIGURATION
# ================================
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "dharmamind"
}

variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be one of: production, staging, development."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "dharmamind.ai"
}

# ================================
# 🌐 NETWORKING
# ================================
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

# ================================
# ☸️ KUBERNETES
# ================================
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_group_instance_types" {
  description = "Instance types for EKS node groups"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "node_group_min_size" {
  description = "Minimum number of nodes in the node group"
  type        = number
  default     = 2
}

variable "node_group_max_size" {
  description = "Maximum number of nodes in the node group"
  type        = number
  default     = 10
}

variable "node_group_desired_size" {
  description = "Desired number of nodes in the node group"
  type        = number
  default     = 3
}

# ================================
# 💾 DATABASE
# ================================
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "Initial allocated storage for RDS"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS"
  type        = number
  default     = 1000
}

variable "db_backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

# ================================
# 🗄️ CACHE
# ================================
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 1
}

# ================================
# 🔐 SECURITY
# ================================
variable "jwt_secret" {
  description = "JWT secret key"
  type        = string
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  sensitive   = true
}

variable "anthropic_api_key" {
  description = "Anthropic API key"
  type        = string
  sensitive   = true
}

variable "google_api_key" {
  description = "Google AI API key"
  type        = string
  sensitive   = true
}

# ================================
# 📊 MONITORING
# ================================
variable "enable_container_insights" {
  description = "Enable CloudWatch Container Insights"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# ================================
# 💰 COST OPTIMIZATION
# ================================
variable "enable_spot_instances" {
  description = "Use Spot instances for cost optimization"
  type        = bool
  default     = true
}

variable "spot_instance_interruption_behavior" {
  description = "Behavior when Spot instance is interrupted"
  type        = string
  default     = "terminate"
  
  validation {
    condition     = contains(["terminate", "stop", "hibernate"], var.spot_instance_interruption_behavior)
    error_message = "Spot instance interruption behavior must be one of: terminate, stop, hibernate."
  }
}

# ================================
# 🚀 SCALING
# ================================
variable "auto_scaling_enabled" {
  description = "Enable auto scaling for node groups"
  type        = bool
  default     = true
}

variable "scale_up_cooldown" {
  description = "Scale up cooldown period in seconds"
  type        = number
  default     = 300
}

variable "scale_down_cooldown" {
  description = "Scale down cooldown period in seconds"
  type        = number
  default     = 300
}

# ================================
# 🌍 MULTI-REGION
# ================================
variable "enable_multi_region" {
  description = "Enable multi-region deployment"
  type        = bool
  default     = false
}

variable "backup_regions" {
  description = "List of backup regions"
  type        = list(string)
  default     = ["us-west-2"]
}

# ================================
# 🔄 BACKUP & DISASTER RECOVERY
# ================================
variable "enable_point_in_time_recovery" {
  description = "Enable point-in-time recovery for RDS"
  type        = bool
  default     = true
}

variable "backup_schedule" {
  description = "Cron expression for automated backups"
  type        = string
  default     = "0 2 * * *"  # Daily at 2 AM UTC
}

variable "cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = true
}

# ================================
# 🏷️ RESOURCE TAGGING
# ================================
variable "tags" {
  description = "Additional tags for all resources"
  type        = map(string)
  default = {
    Project     = "DharmaMind"
    Team        = "Platform"
    Owner       = "DevOps"
    CostCenter  = "Engineering"
  }
}

# ================================
# 🔧 FEATURE FLAGS
# ================================
variable "enable_waf" {
  description = "Enable AWS WAF for application protection"
  type        = bool
  default     = true
}

variable "enable_shield" {
  description = "Enable AWS Shield Advanced for DDoS protection"
  type        = bool
  default     = false
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty for threat detection"
  type        = bool
  default     = true
}

variable "enable_config" {
  description = "Enable AWS Config for compliance monitoring"
  type        = bool
  default     = true
}

# ================================
# 📈 PERFORMANCE
# ================================
variable "db_performance_insights" {
  description = "Enable RDS Performance Insights"
  type        = bool
  default     = true
}

variable "enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS"
  type        = bool
  default     = true
}

variable "monitoring_interval" {
  description = "Enhanced monitoring interval in seconds"
  type        = number
  default     = 60
  
  validation {
    condition     = contains([0, 1, 5, 10, 15, 30, 60], var.monitoring_interval)
    error_message = "Monitoring interval must be one of: 0, 1, 5, 10, 15, 30, 60."
  }
}
