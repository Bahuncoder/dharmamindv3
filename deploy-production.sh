# 🚀 One-Click Production Setup Script
# Complete infrastructure deployment for DharmaMind

#!/bin/bash

set -euo pipefail

# ================================
# 🎨 COLORS AND FORMATTING
# ================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis
SUCCESS="✅"
ERROR="❌"
WARNING="⚠️"
INFO="ℹ️"
ROCKET="🚀"
GEAR="⚙️"
CLOUD="☁️"
LOCK="🔒"
MONEY="💰"

# ================================
# 📋 CONFIGURATION
# ================================
PROJECT_NAME="dharmamind"
AWS_REGION="${AWS_REGION:-us-east-1}"
DOMAIN_NAME="${DOMAIN_NAME:-dharmamind.ai}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Infrastructure configuration
CLUSTER_NAME="${PROJECT_NAME}-cluster"
NODE_INSTANCE_TYPE="${NODE_INSTANCE_TYPE:-t3.large}"
MIN_NODES="${MIN_NODES:-2}"
MAX_NODES="${MAX_NODES:-10}"
DESIRED_NODES="${DESIRED_NODES:-3}"

# Database configuration
DB_INSTANCE_CLASS="${DB_INSTANCE_CLASS:-db.t3.medium}"
DB_STORAGE="${DB_STORAGE:-100}"

# Paths
TERRAFORM_DIR="./terraform"
K8S_DIR="./k8s"
SCRIPTS_DIR="./scripts"

# ================================
# 🛠️ UTILITY FUNCTIONS
# ================================
log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}${SUCCESS} $1${NC}"
}

error() {
    echo -e "${RED}${ERROR} $1${NC}"
}

warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

info() {
    echo -e "${BLUE}${INFO} $1${NC}"
}

# Progress indicator
show_progress() {
    local task="$1"
    local duration="${2:-3}"
    
    echo -n "$task "
    for ((i=0; i<duration; i++)); do
        echo -n "."
        sleep 1
    done
    echo " Done!"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ================================
# 🔍 PREREQUISITES CHECK
# ================================
check_prerequisites() {
    log "${GEAR} Checking prerequisites..."
    
    local missing_tools=()
    local required_tools=("aws" "terraform" "kubectl" "docker" "jq" "curl")
    
    for tool in "${required_tools[@]}"; do
        if command_exists "$tool"; then
            success "$tool is installed"
        else
            missing_tools+=("$tool")
            error "$tool is not installed"
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Please install missing tools: ${missing_tools[*]}"
        echo
        info "Installation commands:"
        echo "📦 AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        echo "🏗️  Terraform: https://learn.hashicorp.com/tutorials/terraform/install-cli"
        echo "☸️  kubectl: https://kubernetes.io/docs/tasks/tools/"
        echo "🐳 Docker: https://docs.docker.com/get-docker/"
        echo "🔧 jq: sudo apt-get install jq (Ubuntu) or brew install jq (macOS)"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &>/dev/null; then
        error "AWS credentials not configured"
        info "Run: aws configure"
        exit 1
    fi
    
    local aws_account=$(aws sts get-caller-identity --query Account --output text)
    local aws_user=$(aws sts get-caller-identity --query Arn --output text)
    success "AWS credentials configured"
    info "Account: $aws_account"
    info "User: $aws_user"
    
    # Check Terraform version
    local tf_version=$(terraform version -json | jq -r '.terraform_version')
    success "Terraform version: $tf_version"
    
    # Check kubectl
    local kubectl_version=$(kubectl version --client -o json 2>/dev/null | jq -r '.clientVersion.gitVersion' || echo "unknown")
    success "kubectl version: $kubectl_version"
    
    success "All prerequisites satisfied!"
}

# ================================
# 🔐 SETUP ENVIRONMENT
# ================================
setup_environment() {
    log "${GEAR} Setting up environment..."
    
    # Check if .env file exists
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.production.template" ]]; then
            warning ".env file not found, copying from template"
            cp .env.production.template .env
            warning "Please edit .env file with your actual values before continuing"
            warning "Required: JWT_SECRET, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY"
            warning "Press Enter when ready, or Ctrl+C to exit"
            read -r
        else
            error ".env file not found and no template available"
            exit 1
        fi
    fi
    
    # Source environment variables
    set -a
    source .env
    set +a
    
    # Validate required variables
    local required_vars=(
        "JWT_SECRET"
        "OPENAI_API_KEY"
        "ANTHROPIC_API_KEY"
        "GOOGLE_API_KEY"
        "DB_PASSWORD"
        "REDIS_PASSWORD"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing required environment variables: ${missing_vars[*]}"
        error "Please update your .env file"
        exit 1
    fi
    
    success "Environment variables configured"
}

# ================================
# 🏗️ TERRAFORM INFRASTRUCTURE
# ================================
deploy_infrastructure() {
    log "${CLOUD} Deploying AWS infrastructure with Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    # Initialize Terraform
    log "Initializing Terraform..."
    terraform init
    
    # Create terraform.tfvars
    cat > terraform.tfvars <<EOF
project_name = "$PROJECT_NAME"
environment = "$ENVIRONMENT"
aws_region = "$AWS_REGION"
domain_name = "$DOMAIN_NAME"

# EKS Configuration
kubernetes_version = "1.28"
node_group_instance_types = ["$NODE_INSTANCE_TYPE"]
node_group_min_size = $MIN_NODES
node_group_max_size = $MAX_NODES
node_group_desired_size = $DESIRED_NODES

# Database Configuration
db_instance_class = "$DB_INSTANCE_CLASS"
db_allocated_storage = $DB_STORAGE

# Secrets
jwt_secret = "$JWT_SECRET"
openai_api_key = "$OPENAI_API_KEY"
anthropic_api_key = "$ANTHROPIC_API_KEY"
google_api_key = "$GOOGLE_API_KEY"
EOF
    
    # Plan
    log "Creating Terraform plan..."
    terraform plan -out=tfplan
    
    # Apply
    log "Applying Terraform configuration..."
    warning "This will create AWS resources that incur costs!"
    info "Estimated monthly cost: $200-500 depending on usage"
    warning "Press Enter to continue, or Ctrl+C to cancel"
    read -r
    
    terraform apply tfplan
    
    # Save outputs
    terraform output -json > ../terraform-outputs.json
    
    success "Infrastructure deployed successfully!"
    
    cd ..
}

# ================================
# ☸️ KUBERNETES SETUP
# ================================
setup_kubernetes() {
    log "${GEAR} Setting up Kubernetes cluster..."
    
    # Update kubeconfig
    aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"
    
    # Verify cluster connection
    kubectl cluster-info
    
    # Install essential add-ons
    log "Installing cluster add-ons..."
    
    # AWS Load Balancer Controller
    log "Installing AWS Load Balancer Controller..."
    kubectl apply -f https://github.com/kubernetes-sigs/aws-load-balancer-controller/releases/download/v2.7.2/v2_7_2_full.yaml
    
    # Metrics Server
    log "Installing Metrics Server..."
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    
    # Cluster Autoscaler
    log "Installing Cluster Autoscaler..."
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.28.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/$CLUSTER_NAME
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
        env:
        - name: AWS_REGION
          value: $AWS_REGION
EOF
    
    success "Kubernetes cluster setup completed!"
}

# ================================
# 📊 MONITORING SETUP
# ================================
setup_monitoring() {
    log "${GEAR} Setting up monitoring stack..."
    
    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Install Prometheus
    log "Installing Prometheus..."
    kubectl create configmap prometheus-config --from-file=monitoring/prometheus.yml -n monitoring --dry-run=client -o yaml | kubectl apply -f -
    kubectl create configmap prometheus-alerts --from-file=monitoring/alert_rules.yml -n monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: alerts
          mountPath: /etc/prometheus/rules
        command:
        - '/bin/prometheus'
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--storage.tsdb.retention.time=200h'
        - '--web.enable-lifecycle'
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: alerts
        configMap:
          name: prometheus-alerts
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: LoadBalancer
EOF
    
    # Install Grafana
    log "Installing Grafana..."
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "$GRAFANA_PASSWORD"
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: monitoring
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: LoadBalancer
EOF
    
    success "Monitoring stack deployed!"
}

# ================================
# 🚀 APPLICATION DEPLOYMENT
# ================================
deploy_application() {
    log "${ROCKET} Deploying DharmaMind application..."
    
    # Make deployment script executable
    chmod +x "$SCRIPTS_DIR/deploy-production.sh"
    
    # Run deployment
    IMAGE_TAG="${IMAGE_TAG:-latest}" \
    CLUSTER_NAME="$CLUSTER_NAME" \
    AWS_REGION="$AWS_REGION" \
    NAMESPACE="dharmamind-production" \
    "$SCRIPTS_DIR/deploy-production.sh"
    
    success "Application deployed successfully!"
}

# ================================
# 🔒 SSL CERTIFICATE SETUP
# ================================
setup_ssl() {
    log "${LOCK} Setting up SSL certificates..."
    
    # Wait for Route53 records to propagate
    log "Waiting for DNS propagation..."
    sleep 60
    
    # Verify ACM certificate
    local cert_arn=$(jq -r '.acm_certificate_arn.value' terraform-outputs.json)
    local cert_status=$(aws acm describe-certificate --certificate-arn "$cert_arn" --query 'Certificate.Status' --output text)
    
    info "Certificate status: $cert_status"
    
    if [[ "$cert_status" == "ISSUED" ]]; then
        success "SSL certificate is ready!"
    else
        warning "SSL certificate is still being validated"
        info "This may take a few minutes. Check ACM console for status."
    fi
}

# ================================
# 🧪 SMOKE TESTS
# ================================
run_smoke_tests() {
    log "${GEAR} Running smoke tests..."
    
    # Get load balancer URLs from terraform outputs
    local backend_url=$(jq -r '.deployment_info.value.api_url' terraform-outputs.json 2>/dev/null || echo "")
    local frontend_url=$(jq -r '.deployment_info.value.app_url' terraform-outputs.json 2>/dev/null || echo "")
    
    if [[ -n "$backend_url" ]]; then
        log "Testing backend health..."
        if curl -f -s "$backend_url/health" > /dev/null; then
            success "Backend health check passed"
        else
            warning "Backend health check failed (may still be starting up)"
        fi
    fi
    
    if [[ -n "$frontend_url" ]]; then
        log "Testing frontend..."
        if curl -f -s "$frontend_url" > /dev/null; then
            success "Frontend health check passed"
        else
            warning "Frontend health check failed (may still be starting up)"
        fi
    fi
    
    # Test Kubernetes services
    log "Checking Kubernetes services..."
    kubectl get services -n dharmamind-production
    kubectl get pods -n dharmamind-production
    
    success "Smoke tests completed!"
}

# ================================
# 📊 COST ESTIMATION
# ================================
show_cost_estimate() {
    log "${MONEY} Cost Estimation (Monthly)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}💰 EKS Cluster:${NC} ~$73/month"
    echo -e "${BLUE}💾 RDS PostgreSQL:${NC} ~$200-400/month"
    echo -e "${BLUE}🗄️  ElastiCache Redis:${NC} ~$50-100/month"
    echo -e "${BLUE}🌐 Load Balancer:${NC} ~$25/month"
    echo -e "${BLUE}🔗 NAT Gateway:${NC} ~$45/month"
    echo -e "${BLUE}📁 S3 Storage:${NC} ~$10-50/month"
    echo -e "${BLUE}📊 CloudWatch:${NC} ~$20-100/month"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}💰 Total Estimated:${NC} $423-793/month"
    echo -e "${YELLOW}⚠️  Actual costs may vary based on usage${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ================================
# 📋 DEPLOYMENT SUMMARY
# ================================
show_deployment_summary() {
    log "${ROCKET} Deployment Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Extract URLs from terraform outputs
    local app_url=$(jq -r '.deployment_info.value.app_url' terraform-outputs.json 2>/dev/null || echo "Pending...")
    local api_url=$(jq -r '.deployment_info.value.api_url' terraform-outputs.json 2>/dev/null || echo "Pending...")
    local prometheus_lb=$(kubectl get service prometheus -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "Pending...")
    local grafana_lb=$(kubectl get service grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "Pending...")
    
    echo -e "${GREEN}🌟 DharmaMind Application:${NC}"
    echo -e "   🌐 Frontend: $app_url"
    echo -e "   🔗 API: $api_url"
    echo
    echo -e "${BLUE}📊 Monitoring:${NC}"
    echo -e "   📈 Prometheus: http://$prometheus_lb:9090"
    echo -e "   📊 Grafana: http://$grafana_lb:3000"
    echo
    echo -e "${PURPLE}☸️  Kubernetes:${NC}"
    echo -e "   🎯 Cluster: $CLUSTER_NAME"
    echo -e "   🌍 Region: $AWS_REGION"
    echo -e "   📁 Namespace: dharmamind-production"
    echo
    echo -e "${CYAN}🔧 Management:${NC}"
    echo -e "   📊 AWS Console: https://console.aws.amazon.com/"
    echo -e "   ☸️  Kubernetes Dashboard: kubectl proxy"
    echo -e "   📝 Logs: kubectl logs -f deployment/dharmamind-backend -n dharmamind-production"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    info "DNS propagation and SSL certificate validation may take 5-15 minutes"
    info "Use 'kubectl get all -n dharmamind-production' to check deployment status"
    warning "Remember to destroy resources when not needed: terraform destroy"
}

# ================================
# 🚀 MAIN DEPLOYMENT FUNCTION
# ================================
main() {
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}    🌟 DharmaMind Production Deployment    ${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    log "${ROCKET} Starting complete production deployment..."
    log "Domain: $DOMAIN_NAME"
    log "Region: $AWS_REGION"  
    log "Environment: $ENVIRONMENT"
    echo
    
    local start_time=$(date +%s)
    
    # Deployment steps
    check_prerequisites
    setup_environment
    show_cost_estimate
    
    warning "This will deploy production infrastructure to AWS"
    warning "Press Enter to continue, or Ctrl+C to cancel"
    read -r
    
    deploy_infrastructure
    setup_kubernetes
    setup_monitoring
    deploy_application
    setup_ssl
    run_smoke_tests
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo
    success "${ROCKET} Deployment completed successfully in $((duration / 60))m $((duration % 60))s!"
    echo
    
    show_deployment_summary
    
    # Final success message
    echo
    echo -e "${GREEN}🎉 DharmaMind is now live in production! 🎉${NC}"
    echo -e "${BLUE}📖 Documentation: https://docs.dharmamind.ai${NC}"
    echo -e "${PURPLE}💬 Support: https://github.com/dharmamind/dharmamind${NC}"
    echo
}

# ================================
# 🎯 SCRIPT EXECUTION
# ================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
