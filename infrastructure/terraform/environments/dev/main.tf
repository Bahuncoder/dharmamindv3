# Development Environment
# Terraform configuration for local/development deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Local development variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "development"
}

variable "replicas" {
  description = "Number of replicas for each service"
  type = object({
    backend   = number
    dharmallm = number
    frontend  = number
  })
  default = {
    backend   = 1
    dharmallm = 1
    frontend  = 1
  }
}

variable "resources" {
  description = "Resource limits for development"
  type = object({
    backend = object({
      cpu    = string
      memory = string
    })
    dharmallm = object({
      cpu    = string
      memory = string
    })
  })
  default = {
    backend = {
      cpu    = "500m"
      memory = "512Mi"
    }
    dharmallm = {
      cpu    = "1000m"
      memory = "2Gi"
    }
  }
}

# Development-specific outputs
output "services" {
  description = "Development service URLs"
  value = {
    backend_url   = "http://localhost:8000"
    dharmallm_url = "http://localhost:8001"
    frontend_url  = "http://localhost:3000"
    docs_url      = "http://localhost:3001"
  }
}

# Local development notes
locals {
  development_notes = {
    setup_instructions = [
      "1. Install Docker Desktop",
      "2. Enable Kubernetes in Docker Desktop",
      "3. Run: kubectl apply -f ../../../infrastructure/k8s/",
      "4. Access services at localhost ports"
    ]
    debugging = [
      "Use kubectl logs <pod-name> for debugging",
      "Port-forward services: kubectl port-forward svc/<service> <port>",
      "Access Prometheus: http://localhost:9090",
      "Access Grafana: http://localhost:3000"
    ]
  }
}