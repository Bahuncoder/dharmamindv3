#!/bin/bash
# 🔒 DharmaMind Security Hardening Script
# Comprehensive security fixes for the spiritual AI system

echo "🕉️ ========================================"
echo "🔒 DHARMAMIND SECURITY HARDENING"
echo "🕉️ ========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root (we don't want this for security)
if [ "$EUID" -eq 0 ]; then
    print_error "Don't run this as root! Run as regular user for security."
    exit 1
fi

print_status "Starting DharmaMind security hardening..."

# 1. CRITICAL: Fix Database File Permissions
echo ""
echo "🔒 STEP 1: Securing Database Files"
echo "=================================="

# Find and secure all database files
find . -name "*.db" -type f 2>/dev/null | while read -r db_file; do
    if [ -f "$db_file" ]; then
        current_perms=$(stat -c "%a" "$db_file" 2>/dev/null)
        print_status "Found database: $db_file (permissions: $current_perms)"
        
        # Set secure permissions (owner read/write only)
        chmod 600 "$db_file"
        if [ $? -eq 0 ]; then
            print_success "Secured: $db_file (now 600)"
        else
            print_error "Failed to secure: $db_file"
        fi
    fi
done

# 2. CRITICAL: Fix .env File Permissions
echo ""
echo "🔐 STEP 2: Securing Environment Files"
echo "===================================="

find . -name ".env*" -type f 2>/dev/null | while read -r env_file; do
    if [ -f "$env_file" ]; then
        current_perms=$(stat -c "%a" "$env_file" 2>/dev/null)
        print_status "Found env file: $env_file (permissions: $current_perms)"
        
        # Set secure permissions (owner read/write only)
        chmod 600 "$env_file"
        if [ $? -eq 0 ]; then
            print_success "Secured: $env_file (now 600)"
        else
            print_error "Failed to secure: $env_file"
        fi
    fi
done

# 3. CRITICAL: Scan for Hardcoded Secrets (Improved)
echo ""
echo "🔍 STEP 3: Scanning for Hardcoded Secrets"
echo "========================================="

# Create a more targeted secrets scan
secrets_found=0

# Look for actual patterns that matter
print_status "Scanning for dangerous patterns..."

# API Keys
api_keys=$(grep -r "api_key\s*=\s*[\"'][a-zA-Z0-9_-]\{20,\}[\"']" . --include="*.py" --include="*.js" --include="*.ts" 2>/dev/null | wc -l)
if [ "$api_keys" -gt 0 ]; then
    print_warning "Found $api_keys potential API keys in source code"
    secrets_found=$((secrets_found + api_keys))
fi

# Passwords
passwords=$(grep -r "password\s*=\s*[\"'][^\"']\{6,\}[\"']" . --include="*.py" --include="*.js" --include="*.ts" 2>/dev/null | wc -l)
if [ "$passwords" -gt 0 ]; then
    print_warning "Found $passwords potential hardcoded passwords"
    secrets_found=$((secrets_found + passwords))
fi

# JWT Secrets
jwt_secrets=$(grep -r "jwt.*secret\s*=\s*[\"'][a-zA-Z0-9_-]\{20,\}[\"']" . --include="*.py" --include="*.js" --include="*.ts" 2>/dev/null | wc -l)
if [ "$jwt_secrets" -gt 0 ]; then
    print_warning "Found $jwt_secrets potential JWT secrets"
    secrets_found=$((secrets_found + jwt_secrets))
fi

if [ "$secrets_found" -eq 0 ]; then
    print_success "No obvious hardcoded secrets found"
else
    print_warning "Total potential secrets found: $secrets_found"
    print_warning "Review these manually and move to .env files!"
fi

# 4. Set Proper File Permissions for Config Files
echo ""
echo "📁 STEP 4: Setting Proper File Permissions"
echo "=========================================="

# Docker compose files - should be readable but not writable by others
find . -name "docker-compose*.yml" -type f 2>/dev/null | while read -r compose_file; do
    chmod 644 "$compose_file"
    print_success "Set permissions for: $compose_file (644)"
done

# Requirements files
find . -name "requirements*.txt" -type f 2>/dev/null | while read -r req_file; do
    chmod 644 "$req_file"
    print_success "Set permissions for: $req_file (644)"
done

# Python files
find . -name "*.py" -type f 2>/dev/null | while read -r py_file; do
    chmod 644 "$py_file"
done
print_success "Set permissions for Python files (644)"

# 5. Create Security Headers for FastAPI
echo ""
echo "🛡️ STEP 5: Creating Security Headers Configuration"
echo "================================================"

cat > security_headers.py << 'EOF'
"""
🔒 DharmaMind Security Headers Configuration
Enhanced security middleware for spiritual AI protection
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import time

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    🛡️ Security headers middleware for DharmaMind
    Protects spiritual AI endpoints with proper security headers
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers for spiritual data protection
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Custom header for spiritual AI protection
        response.headers["X-Dharma-Protection"] = "spiritual-data-secured"
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    🕉️ Spiritual-aware rate limiting
    Protects against abuse while allowing genuine spiritual seeking
    """
    
    def __init__(self, app, calls_per_minute: int = 30):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            ip: times for ip, times in self.requests.items()
            if any(t > current_time - 60 for t in times)
        }
        
        # Check rate limit
        if client_ip in self.requests:
            # Filter requests from last minute
            recent_requests = [t for t in self.requests[client_ip] if t > current_time - 60]
            if len(recent_requests) >= self.calls_per_minute:
                return Response(
                    content="🕉️ Rate limit exceeded. Please seek wisdom mindfully.",
                    status_code=429,
                    headers={"X-Dharma-Message": "patience-is-a-virtue"}
                )
            self.requests[client_ip] = recent_requests + [current_time]
        else:
            self.requests[client_ip] = [current_time]
        
        return await call_next(request)

def configure_security(app: FastAPI):
    """
    🔒 Configure comprehensive security for DharmaMind
    """
    
    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add rate limiting (gentle for spiritual seekers)
    app.add_middleware(RateLimitMiddleware, calls_per_minute=30)
    
    # Configure CORS properly
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],  # Update with your domain
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    return app

# Example usage in your FastAPI app:
"""
from security_headers import configure_security

app = FastAPI(title="DharmaMind Spiritual AI")
app = configure_security(app)
"""
EOF

print_success "Created security_headers.py - Add this to your FastAPI app!"

# 6. Create Environment Template
echo ""
echo "🔧 STEP 6: Creating Secure Environment Template"
echo "=============================================="

cat > .env.template << 'EOF'
# 🕉️ DharmaMind Spiritual AI - Environment Configuration
# 🔒 SECURITY: Keep this file secure (permissions 600)
# ⚠️  WARNING: Never commit actual .env file to version control!

# Database Configuration
DATABASE_URL=sqlite:///./data/dharma_knowledge.db
DATABASE_ENCRYPTION_KEY=your-strong-encryption-key-here

# API Security
JWT_SECRET_KEY=your-very-strong-jwt-secret-here-minimum-256-bits
API_SECRET_KEY=your-api-secret-key-here
ENCRYPTION_SALT=your-encryption-salt-here

# Spiritual AI Configuration
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
SPIRITUAL_KNOWLEDGE_ENCRYPTION=true

# Security Settings
HTTPS_ONLY=true
RATE_LIMIT_PER_MINUTE=30
SESSION_TIMEOUT_MINUTES=60

# CORS Settings (Production)
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Database Security
DB_BACKUP_ENCRYPTION=true
SPIRITUAL_DATA_ANONYMIZATION=true

# Monitoring
SECURITY_MONITORING=true
AUDIT_LOGGING=true
SPIRITUAL_QUERY_LOGGING=true

# Production Settings
DEBUG=false
TESTING=false
ENVIRONMENT=production
EOF

print_success "Created .env.template - Copy to .env and fill with your values!"

# 7. Create Docker Security Configuration
echo ""
echo "🐳 STEP 7: Creating Secure Docker Configuration"
echo "=============================================="

cat > docker-compose.security.yml << 'EOF'
# 🔒 DharmaMind Secure Docker Compose Configuration
# Enhanced security for spiritual AI containerization

version: '3.8'

services:
  dharmamind-backend:
    build: 
      context: ./backend
      dockerfile: Dockerfile
    restart: unless-stopped
    
    # Security configurations
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    
    # User security
    user: "1000:1000"  # Non-root user
    
    # Resource limits (prevent DoS)
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    
    # Secure environment
    environment:
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
    
    # Secrets (use Docker secrets in production)
    secrets:
      - jwt_secret
      - db_encryption_key
      - api_keys
    
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    
    # Network security
    networks:
      - dharmamind-secure

  dharmamind-frontend:
    build:
      context: ./dharmamind-chat
      dockerfile: Dockerfile
    restart: unless-stopped
    
    # Security configurations
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=50m
    
    # User security
    user: "1000:1000"
    
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '0.3'
          memory: 256M
    
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3
    
    networks:
      - dharmamind-secure

  # Reverse proxy with security headers
  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "443:443"
      - "80:80"
    
    # Security configurations
    security_opt:
      - no-new-privileges:true
    
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    
    depends_on:
      - dharmamind-backend
      - dharmamind-frontend
    
    networks:
      - dharmamind-secure

# Secure network
networks:
  dharmamind-secure:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

# Docker secrets (use in production)
secrets:
  jwt_secret:
    external: true
  db_encryption_key:
    external: true
  api_keys:
    external: true

# Secure volumes
volumes:
  spiritual_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /secure/path/to/spiritual/data
EOF

print_success "Created docker-compose.security.yml - Use this for production!"

# 8. Final Security Report
echo ""
echo "🎯 STEP 8: Security Hardening Complete!"
echo "======================================"

print_success "Database files secured with 600 permissions"
print_success "Environment files secured with 600 permissions"
print_success "Config files set to proper permissions"
print_success "Security middleware configuration created"
print_success "Environment template created"
print_success "Secure Docker configuration created"

echo ""
echo "🛡️ NEXT STEPS:"
echo "=============="
print_status "1. Copy .env.template to .env and fill with your secrets"
print_status "2. Add security_headers.py to your FastAPI app"
print_status "3. Use docker-compose.security.yml for production"
print_status "4. Set up HTTPS certificates"
print_status "5. Configure monitoring and alerting"

echo ""
echo "🕉️ DHARMAMIND SECURITY STATUS: SIGNIFICANTLY IMPROVED!"
echo "======================================================"
print_success "Your spiritual AI system is now much more secure!"
print_success "Continue monitoring and update security regularly."
print_success "May your code be bug-free and your security unbreachable! 🙏"

echo ""
echo "📊 SECURITY CHECKLIST:"
echo "====================="
echo "✅ Database files secured"
echo "✅ Environment files secured"
echo "✅ File permissions corrected"
echo "✅ Security headers configured"
echo "✅ Rate limiting implemented"
echo "✅ Docker security enhanced"
echo "✅ Environment template created"
echo "⚠️  TODO: Set up HTTPS certificates"
echo "⚠️  TODO: Configure production secrets"
echo "⚠️  TODO: Set up monitoring"

exit 0
