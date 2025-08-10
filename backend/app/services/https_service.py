"""
🔒 DharmaMind HTTPS Security Service

SSL/TLS certificate management and HTTPS enforcement.
Implements secure transport layer security throughout the application.
"""

import os
import ssl
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


class HTTPSSecurityService:
    """🔒 HTTPS and SSL/TLS security management"""
    
    def __init__(self):
        self.cert_dir = Path("certs")
        self.cert_dir.mkdir(exist_ok=True)
        
        self.cert_file = self.cert_dir / "dharmamind.crt"
        self.key_file = self.cert_dir / "dharmamind.key"
        self.ca_file = self.cert_dir / "ca.crt"
        
        logger.info("🔒 HTTPS Security Service initialized")
    
    def create_self_signed_cert(self) -> bool:
        """Create self-signed SSL certificate for development"""
        try:
            # Check if certificates already exist
            if self.cert_file.exists() and self.key_file.exists():
                logger.info("✅ SSL certificates already exist")
                return True
            
            logger.info("🔧 Creating self-signed SSL certificate...")
            
            # OpenSSL command to create self-signed certificate
            cmd = [
                "openssl", "req", "-x509", "-newkey", "rsa:4096",
                "-keyout", str(self.key_file),
                "-out", str(self.cert_file),
                "-days", "365", "-nodes",
                "-subj", "/C=US/ST=CA/L=SanFrancisco/O=DharmaMind/CN=localhost"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Self-signed certificate created successfully")
                return True
            else:
                logger.error(f"❌ Certificate creation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Certificate creation error: {e}")
            return False
    
    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Get SSL context for secure connections"""
        try:
            if not (self.cert_file.exists() and self.key_file.exists()):
                logger.warning("⚠️ SSL certificates not found")
                return None
            
            # Create SSL context
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(self.cert_file), str(self.key_file))
            
            # Security settings
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            # Disable weak protocols
            context.options |= ssl.OP_NO_SSLv2
            context.options |= ssl.OP_NO_SSLv3
            context.options |= ssl.OP_NO_TLSv1
            context.options |= ssl.OP_NO_TLSv1_1
            
            # Set secure ciphers
            context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
            
            logger.info("✅ SSL context configured successfully")
            return context
            
        except Exception as e:
            logger.error(f"❌ SSL context creation failed: {e}")
            return None
    
    def get_uvicorn_ssl_config(self) -> Dict[str, Any]:
        """Get SSL configuration for Uvicorn server"""
        if self.cert_file.exists() and self.key_file.exists():
            return {
                "ssl_keyfile": str(self.key_file),
                "ssl_certfile": str(self.cert_file),
                "ssl_version": ssl.PROTOCOL_TLS_SERVER,
                "ssl_cert_reqs": ssl.CERT_NONE,
                "ssl_ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
            }
        return {}
    
    def generate_nginx_config(self) -> str:
        """Generate Nginx configuration for HTTPS"""
        return f"""
# DharmaMind HTTPS Nginx Configuration
server {{
    listen 80;
    server_name localhost dharmamind.local;
    return 301 https://$server_name$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name localhost dharmamind.local;
    
    # SSL Configuration
    ssl_certificate {self.cert_file.absolute()};
    ssl_certificate_key {self.key_file.absolute()};
    
    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'";
    
    # Frontend (Next.js)
    location / {{
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }}
    
    # Backend API
    location /api/ {{
        proxy_pass http://localhost:5001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # API Security
        proxy_set_header X-API-Source "nginx";
        add_header X-API-Server "dharmamind" always;
    }}
    
    # Auth endpoints
    location /auth/ {{
        proxy_pass http://localhost:5001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Additional security for auth endpoints
        add_header X-Auth-Server "dharmamind" always;
        proxy_set_header X-Auth-Source "nginx";
    }}
}}
"""
    
    def create_nginx_config_file(self) -> bool:
        """Create Nginx configuration file"""
        try:
            config_content = self.generate_nginx_config()
            config_file = self.cert_dir / "nginx.conf"
            
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            logger.info(f"✅ Nginx configuration created: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Nginx config creation failed: {e}")
            return False
    
    def create_docker_https_compose(self) -> str:
        """Create Docker Compose with HTTPS support"""
        return """
version: '3.8'

services:
  # Nginx HTTPS Proxy
  nginx:
    image: nginx:alpine
    container_name: dharmamind_nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./certs/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./certs:/etc/ssl/certs:ro
    depends_on:
      - backend
      - frontend
    networks:
      - dharmamind_network

  # Secure PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: dharmamind_postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: dharmamind
      POSTGRES_USER: dharmamind
      POSTGRES_PASSWORD: secure_password_123
      POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql:ro
    command:
      - postgres
      - -c
      - ssl=on
      - -c
      - ssl_cert_file=/var/lib/postgresql/server.crt
      - -c
      - ssl_key_file=/var/lib/postgresql/server.key
    networks:
      - dharmamind_network

  # DharmaMind Backend with HTTPS
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: dharmamind_backend
    restart: unless-stopped
    environment:
      DATABASE_URL: postgresql://dharmamind:secure_password_123@postgres:5432/dharmamind
      SECRET_KEY: ${SECRET_KEY:-super-secure-secret-key-change-in-production}
      DHARMAMIND_ENCRYPTION_KEY: ${ENCRYPTION_KEY:-}
      ENVIRONMENT: production
      HTTPS_ENABLED: "true"
      SSL_CERT_FILE: /app/certs/dharmamind.crt
      SSL_KEY_FILE: /app/certs/dharmamind.key
    ports:
      - "5001:5001"
    depends_on:
      - postgres
    volumes:
      - ./backend:/app
      - ./certs:/app/certs:ro
    networks:
      - dharmamind_network

  # DharmaMind Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: dharmamind_frontend
    restart: unless-stopped
    environment:
      NEXT_PUBLIC_API_URL: https://localhost:443
      NEXT_PUBLIC_ENVIRONMENT: production
      NEXT_PUBLIC_HTTPS_ENABLED: "true"
    ports:
      - "3001:3000"
    depends_on:
      - backend
    networks:
      - dharmamind_network

volumes:
  postgres_data:
    driver: local

networks:
  dharmamind_network:
    driver: bridge
"""
    
    def setup_https_development(self) -> bool:
        """Set up HTTPS for development environment"""
        try:
            logger.info("🔧 Setting up HTTPS for development...")
            
            # Create self-signed certificate
            if not self.create_self_signed_cert():
                return False
            
            # Create Nginx configuration
            if not self.create_nginx_config_file():
                return False
            
            # Create HTTPS Docker Compose
            compose_content = self.create_docker_https_compose()
            compose_file = Path("docker-compose.https.yml")
            
            with open(compose_file, 'w') as f:
                f.write(compose_content)
            
            logger.info("✅ HTTPS development setup complete")
            logger.info("📝 To start with HTTPS:")
            logger.info("   docker-compose -f docker-compose.https.yml up")
            logger.info("🌐 Access at: https://localhost")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ HTTPS setup failed: {e}")
            return False
    
    def validate_ssl_config(self) -> Dict[str, Any]:
        """Validate SSL configuration"""
        try:
            results = {
                "certificate_exists": self.cert_file.exists(),
                "private_key_exists": self.key_file.exists(),
                "ssl_context_valid": False,
                "certificate_info": None
            }
            
            if results["certificate_exists"] and results["private_key_exists"]:
                context = self.get_ssl_context()
                results["ssl_context_valid"] = context is not None
                
                # Get certificate info
                try:
                    import subprocess
                    cmd = ["openssl", "x509", "-in", str(self.cert_file), "-text", "-noout"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        results["certificate_info"] = {
                            "status": "valid",
                            "details": "Certificate parsed successfully"
                        }
                    else:
                        results["certificate_info"] = {
                            "status": "invalid",
                            "error": result.stderr
                        }
                except Exception as e:
                    results["certificate_info"] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ SSL validation failed: {e}")
            return {"error": str(e)}


# Global HTTPS service instance
https_service = HTTPSSecurityService()


# Convenience functions
def setup_https():
    """Set up HTTPS for the application"""
    return https_service.setup_https_development()


def get_ssl_config():
    """Get SSL configuration for Uvicorn"""
    return https_service.get_uvicorn_ssl_config()


def validate_https():
    """Validate HTTPS configuration"""
    return https_service.validate_ssl_config()
