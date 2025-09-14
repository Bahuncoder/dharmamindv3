# 🚀 DharmaMind Complete Deployment Guide

## 🎯 **Deployment Strategy Overview**

You're right to be concerned about deploying such a comprehensive system! DharmaMind is an enterprise-grade project with multiple components. Here's your complete deployment strategy:

## 📊 **System Architecture Analysis**

### **🏗️ Component Breakdown:**

1. **Backend API** (`/backend/`) - FastAPI with 24,000+ lines of code
2. **Frontend Brand Website** (`/Brand_Webpage/`) - Next.js application
3. **Community Platform** (`/DhramaMind_Community/`) - Community features
4. **DharmaLLM** (`/dharmallm/`) - Custom LLM training system
5. **Monitoring Stack** (`/monitoring/`) - Prometheus + Grafana
6. **Kubernetes Deployment** (`/k8s/`) - Production orchestration
7. **Testing Infrastructure** (`/backend/tests/`) - Comprehensive test suite

## 🔧 **Deployment Options (Choose Your Path)**

### **Option 1: 🚀 Quick Development Deployment**

_Best for: Development, testing, demo_

### **Option 2: 🐳 Docker Compose Deployment**

_Best for: Local production, small scale_

### **Option 3: ☸️ Kubernetes Deployment**

_Best for: Production scale, enterprise_

---

## 🚀 **Option 1: Quick Development Deployment**

### **Step 1: Environment Setup**

```bash
# Clone and navigate to project
cd /media/rupert/New\ Volume/FinalTesting/DharmaMind-chat-master

# Set up Python environment for backend
cd backend
python -m venv dharmamind_env
source dharmamind_env/bin/activate
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### **Step 2: Database Setup**

```bash
# Install PostgreSQL locally
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres createuser --interactive dharmamind_user
sudo -u postgres createdb dharmamind_db -O dharmamind_user
```

### **Step 3: Start Development Services**

```bash
# Terminal 1: Backend API
cd backend
source dharmamind_env/bin/activate
python run_server.py

# Terminal 2: Brand Website
cd Brand_Webpage
npm install
npm run dev

# Terminal 3: Community Platform
cd DhramaMind_Community
npm install
npm run dev
```

---

## 🐳 **Option 2: Docker Compose Deployment (Recommended)**

### **Complete Production Stack**

I've created a comprehensive Docker Compose configuration that handles all services:

```yaml
# docker-compose.yml includes:
- PostgreSQL Database
- Redis Cache
- Backend API (FastAPI)
- Brand Website (Next.js)
- Community Platform
- DharmaLLM Service
- Prometheus Monitoring
- Grafana Dashboards
- Nginx Reverse Proxy
```

### **Step 1: Environment Configuration**

```bash
# Copy production environment template
cp .env.production .env

# Edit .env with your actual values:
# - Database credentials
# - API keys
# - Security secrets
# - Domain configuration
```

### **Step 2: Build and Deploy**

```bash
# Build all services
docker-compose build

# Start the complete stack
docker-compose up -d

# Check service status
docker-compose ps
```

### **Step 3: Initialize Database**

```bash
# Run database migrations
docker-compose exec backend python -m alembic upgrade head

# Optional: Load initial data
docker-compose exec backend python -c "from app.core.init_db import init_db; init_db()"
```

### **Step 4: Verify Deployment**

```bash
# Check health endpoints
curl http://localhost/health          # Nginx health
curl http://localhost/api/health      # Backend health
curl http://localhost:3001/health     # Brand website
curl http://localhost:3002/health     # Community platform

# Access services
echo "Main Website: http://localhost"
echo "API Docs: http://localhost/api/docs"
echo "Community: http://localhost:3002"
echo "Monitoring: http://localhost:3000"  # Grafana
echo "Metrics: http://localhost:9090"     # Prometheus
```

---

## 🔧 **Nginx Reverse Proxy Configuration**

The complete Nginx configuration provides:

- **Load Balancing**: Distributes traffic across service instances
- **SSL Termination**: Ready for HTTPS certificates
- **Rate Limiting**: Protects against abuse and DDoS
- **Caching**: Optimizes static asset delivery
- **Security Headers**: Implements security best practices
- **Health Checks**: Monitors service availability

### **Nginx Route Configuration**

```nginx
# Main routing
/ → Brand Website (Next.js)
/api/ → Backend API (FastAPI)
/ws → WebSocket connections
community.dharmamind.ai → Community Platform
monitoring.dharmamind.ai → Monitoring Dashboard
```

### **SSL/HTTPS Setup (Production)**

1. **Obtain SSL Certificate** (Let's Encrypt recommended):

```bash
# Install certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d dharmamind.ai -d www.dharmamind.ai
```

2. **Uncomment HTTPS Configuration** in `nginx/nginx.conf`

3. **Update Environment Variables**:

```bash
# Update .env.production
SECURE_SSL_REDIRECT=true
SECURE_PROXY_SSL_HEADER=HTTP_X_FORWARDED_PROTO,https
```

---

## ☸️ **Option 3: Kubernetes Deployment**

### **Production-Scale Deployment**

```bash
# Your existing Kubernetes manifests are in /k8s/
cd k8s/

# Deploy to Kubernetes cluster
kubectl apply -f .

# Check deployment status
kubectl get pods -A
kubectl get services
kubectl get ingress
```

### **Scaling Configuration**

```bash
# Scale backend instances
kubectl scale deployment backend-api --replicas=3

# Auto-scaling setup
kubectl autoscale deployment backend-api --cpu-percent=70 --min=2 --max=10
```

---

## 🧪 **Deployment Validation & Testing**

### **1. Pre-Deployment Testing**

```bash
# Validate Docker Compose configuration
docker-compose -f docker-compose.yml config

# Check environment variables
docker-compose -f docker-compose.yml config --services

# Test individual services
docker-compose up postgres redis  # Start dependencies first
docker-compose up backend  # Test backend
docker-compose up brand_website  # Test frontend
```

### **2. Health Check Validation**

```bash
# After full deployment, test all health endpoints
curl http://localhost/health  # Nginx health
curl http://localhost/api/health  # Backend health
curl http://localhost:3001/api/health  # Brand website health
curl http://localhost:3002/health  # Community health
```

### **3. Monitoring Validation**

```bash
# Check Prometheus metrics
curl http://localhost:9090/metrics

# Check Grafana dashboard
open http://localhost:3000  # Default: admin/admin
```

### **4. Testing Infrastructure Integration**

```bash
# Run complete test suite against deployed services
cd backend
python -m pytest tests/ -v --tb=short --cov=app

# Test API endpoints
python -m pytest tests/test_integration/ -v

# Load testing
python -m pytest tests/test_load/ -v
```

---

## 📊 **Production Monitoring & Maintenance**

### **Log Management**

```bash
# View service logs
docker-compose logs -f backend
docker-compose logs -f nginx
docker-compose logs -f postgres

# Rotate logs (setup logrotate)
sudo nano /etc/logrotate.d/dharmamind
```

### **Performance Monitoring**

- **Grafana Dashboards**: Pre-configured system and application metrics
- **Prometheus Alerts**: Automated alerting for critical issues
- **Health Checks**: Continuous service availability monitoring

### **Backup Strategy**

```bash
# Database backup
docker-compose exec postgres pg_dump -U dharmamind_user dharmamind_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Volume backup
docker run --rm -v dharmamind_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
```

---

## 🔄 **Deployment Troubleshooting**

### **Common Issues & Solutions**

1. **Port Conflicts**

```bash
# Check port usage
sudo netstat -tlnp | grep :80
sudo netstat -tlnp | grep :3000

# Kill conflicting processes
sudo fuser -k 80/tcp
```

2. **Database Connection Issues**

```bash
# Check database connectivity
docker-compose exec backend python -c "import asyncpg; print('PostgreSQL driver available')"

# Test database connection
docker-compose exec postgres psql -U dharmamind_user -d dharmamind_db -c "SELECT version();"
```

3. **Memory Issues**

```bash
# Monitor resource usage
docker stats

# Adjust memory limits in docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

4. **SSL Certificate Issues**

```bash
# Test SSL configuration
nginx -t

# Renew Let's Encrypt certificates
sudo certbot renew --dry-run
```

---

## 🚀 **Production Deployment Checklist**

### **Pre-Deployment**

- [ ] Environment variables configured in `.env.production`
- [ ] Database credentials secured
- [ ] SSL certificates obtained (if using HTTPS)
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Monitoring alerts configured

### **Deployment**

- [ ] Services start successfully
- [ ] Health checks pass
- [ ] Database migrations complete
- [ ] Static files served correctly
- [ ] WebSocket connections working
- [ ] Monitoring dashboards accessible

### **Post-Deployment**

- [ ] Performance testing completed
- [ ] Security scan passed
- [ ] Backup restoration tested
- [ ] Log rotation configured
- [ ] Alert notifications working
- [ ] Documentation updated

---

## 🎯 **Final Deployment Summary**

Your DharmaMind system is now equipped with:

✅ **Complete Docker Compose Stack** - All services containerized and orchestrated
✅ **Production Environment Template** - Secure configuration management
✅ **Nginx Reverse Proxy** - Load balancing, SSL, security headers, rate limiting
✅ **Comprehensive Testing Infrastructure** - 36 tests covering all components
✅ **Monitoring Integration** - Prometheus metrics and Grafana dashboards
✅ **Database Management** - PostgreSQL with backup and migration strategies
✅ **Troubleshooting Guides** - Solutions for common deployment issues

**Recommended deployment path for your use case:**

1. Start with **Docker Compose deployment** (Option 2)
2. Use the provided `.env.production` template
3. Follow the validation checklist
4. Scale to Kubernetes when needed

This deployment strategy transforms your complex multi-component system into a manageable, production-ready deployment with full testing integration.
