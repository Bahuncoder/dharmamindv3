# 🏗️ Infrastructure Documentation

This directory contains all infrastructure, deployment, and DevOps configurations for the DharmaMind platform.

## 📁 Directory Structure

```
infrastructure/
├── k8s/                    # Kubernetes configurations
│   ├── backend/           # Authentication service deployments
│   ├── dharmallm/         # AI/LLM service deployments
│   ├── frontends/         # Frontend application deployments
│   ├── shared/            # Shared K8s resources (ingress, secrets, etc.)
│   └── ingress/           # Load balancer and routing configurations
├── terraform/             # Infrastructure as Code
│   ├── environments/      # Environment-specific configurations
│   │   ├── dev/          # Development environment
│   │   ├── staging/      # Staging environment
│   │   └── production/   # Production environment
│   ├── modules/          # Reusable Terraform modules
│   └── providers/        # Cloud provider configurations
├── monitoring/           # Observability and monitoring
│   ├── prometheus/       # Metrics collection configuration
│   ├── grafana/         # Dashboard configurations
│   ├── alerts/          # Alert rules and notifications
│   └── logs/            # Log aggregation and analysis
├── nginx/               # Load balancer and reverse proxy configs
└── scripts/             # DevOps automation scripts
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Kubernetes cluster (local or cloud)
- Terraform >= 1.0
- kubectl configured

### Local Development

```bash
# Start all services locally
./infrastructure/scripts/deploy_local.sh

# Check service health
./infrastructure/scripts/health_check.sh
```

### Staging Deployment

```bash
# Deploy to staging environment
cd infrastructure/terraform/environments/staging
terraform init && terraform apply
```

### Production Deployment

```bash
# Deploy to production (requires approval)
cd infrastructure/terraform/environments/production
terraform init && terraform plan
# Review plan, then apply
terraform apply
```

## 📊 Monitoring

- **Prometheus**: Metrics collection at `:9090`
- **Grafana**: Dashboards at `:3000`
- **Alert Manager**: Alert routing at `:9093`

## 🔧 Configuration Management

### Environment Variables

Each environment has its own variable files:

- `dev.tfvars` - Development settings
- `staging.tfvars` - Staging settings
- `production.tfvars` - Production settings

### Secrets Management

Secrets are managed through:

- Kubernetes secrets for runtime
- Terraform for infrastructure secrets
- External secret managers for sensitive data

## 📈 Scaling

### Horizontal Pod Autoscaling

- Backend: CPU-based scaling (50-80% threshold)
- DharmaLLM: GPU utilization scaling (60-90% threshold)
- Frontends: Request-based scaling

### Infrastructure Scaling

- Use Terraform modules for consistent scaling
- Environment-specific resource limits
- Cost optimization through right-sizing

## 🛡️ Security

### Network Security

- Network policies for pod-to-pod communication
- Ingress with TLS termination
- Service mesh for internal communication

### Secret Management

- Encrypted secrets at rest
- Rotation policies for sensitive credentials
- Least privilege access principles

## 📚 Related Documentation

- [Deployment Guide](../docs/deployment/)
- [Architecture Overview](../docs/architecture/)
- [Development Setup](../docs/development/)

---

For questions or issues, contact the DevOps team or create an issue in the repository.
