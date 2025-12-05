"""
🔑 DharmaMind Secure Environment Configuration

Production-ready environment configuration with secure secret management:
- Secure random secret generation
- Environment variable validation
- Production-ready defaults
- Security best practices
"""

import os
import secrets
import string
from typing import Dict, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SecretManager:
    """🔐 Secure secret and environment management"""
    
    def __init__(self):
        self.secrets_cache = {}
        self.required_secrets = {
            "SECRET_KEY": "JWT and session signing key",
            "DHARMAMIND_ENCRYPTION_KEY": "Data encryption key",
            "DATABASE_URL": "Database connection string",
            "REDIS_URL": "Redis cache connection",
            "OPENAI_API_KEY": "OpenAI API access key",
            "ANTHROPIC_API_KEY": "Anthropic API access key",
            "GOOGLE_CLIENT_ID": "Google OAuth client ID",
            "GOOGLE_CLIENT_SECRET": "Google OAuth client secret",
            "STRIPE_SECRET_KEY": "Stripe payment processing",
            "STRIPE_WEBHOOK_SECRET": "Stripe webhook validation",
            "SENDGRID_API_KEY": "Email service API key",
            "SENTRY_DSN": "Error monitoring service"
        }
    
    def generate_secure_secret(self, length: int = 64) -> str:
        """Generate cryptographically secure random secret"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_encryption_key(self) -> str:
        """Generate Fernet-compatible encryption key"""
        from cryptography.fernet import Fernet
        return Fernet.generate_key().decode()
    
    def get_secret(self, key: str, default: Optional[str] = None) -> str:
        """Get secret from environment with caching"""
        if key in self.secrets_cache:
            return self.secrets_cache[key]
        
        value = os.getenv(key, default)
        
        if value is None:
            # Generate secure default for critical secrets
            if key == "SECRET_KEY":
                value = self.generate_secure_secret(64)
                logger.warning(f"Generated temporary {key}. Set environment variable for production!")
            elif key == "DHARMAMIND_ENCRYPTION_KEY":
                value = self.generate_encryption_key()
                logger.warning(f"Generated temporary {key}. Set environment variable for production!")
            else:
                logger.warning(f"Missing environment variable: {key}")
                return ""
        
        self.secrets_cache[key] = value
        return value
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate all required environment variables"""
        validation_results = {
            "valid": True,
            "missing_secrets": [],
            "weak_secrets": [],
            "recommendations": []
        }
        
        for secret_key, description in self.required_secrets.items():
            value = os.getenv(secret_key)
            
            if not value:
                validation_results["missing_secrets"].append({
                    "key": secret_key,
                    "description": description
                })
                validation_results["valid"] = False
            else:
                # Check for weak/default secrets
                if self._is_weak_secret(secret_key, value):
                    validation_results["weak_secrets"].append({
                        "key": secret_key,
                        "reason": "Using default or weak value"
                    })
                    validation_results["valid"] = False
        
        # Add recommendations
        if validation_results["missing_secrets"]:
            validation_results["recommendations"].append(
                "Set missing environment variables before production deployment"
            )
        
        if validation_results["weak_secrets"]:
            validation_results["recommendations"].append(
                "Replace weak/default secrets with secure generated values"
            )
        
        return validation_results
    
    def _is_weak_secret(self, key: str, value: str) -> bool:
        """Check if secret is weak or uses default values"""
        weak_patterns = [
            "changeme", "default", "secret", "password", "key",
            "test", "dev", "development", "localhost", "example"
        ]
        
        value_lower = value.lower()
        
        # Check for common weak patterns
        for pattern in weak_patterns:
            if pattern in value_lower:
                return True
        
        # Check minimum length requirements
        min_lengths = {
            "SECRET_KEY": 32,
            "DHARMAMIND_ENCRYPTION_KEY": 32,
            "OPENAI_API_KEY": 20,
            "ANTHROPIC_API_KEY": 20,
            "STRIPE_SECRET_KEY": 20,
            "STRIPE_WEBHOOK_SECRET": 20
        }
        
        if key in min_lengths and len(value) < min_lengths[key]:
            return True
        
        return False
    
    def create_production_env_template(self) -> str:
        """Create .env template for production deployment"""
        template_lines = [
            "# 🔑 DharmaMind Production Environment Configuration",
            "# Copy this file to .env and fill in your production values",
            "",
            "# 🛡️ Security Configuration",
            f"SECRET_KEY={self.generate_secure_secret(64)}",
            f"DHARMAMIND_ENCRYPTION_KEY={self.generate_encryption_key()}",
            "",
            "# 🗄️ Database Configuration",
            "DATABASE_URL=postgresql://username:password@localhost:5432/dharmamind",
            "REDIS_URL=redis://localhost:6379/0",
            "",
            "# 🤖 AI Service Configuration",
            "OPENAI_API_KEY=sk-your-openai-api-key-here",
            "ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here",
            "",
            "# 🔐 OAuth Configuration",
            "GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com",
            "GOOGLE_CLIENT_SECRET=your-google-client-secret",
            "",
            "# 💳 Payment Configuration",
            "STRIPE_SECRET_KEY=sk_live_your-stripe-secret-key",
            "STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret",
            "",
            "# 📧 Email Configuration",
            "SENDGRID_API_KEY=SG.your-sendgrid-api-key",
            "",
            "# 📊 Monitoring Configuration",
            "SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id",
            "",
            "# 🚀 Deployment Configuration",
            "ENVIRONMENT=production",
            "DEBUG=false",
            "LOG_LEVEL=INFO",
            "",
            "# 🌐 Server Configuration",
            "HOST=0.0.0.0",
            "PORT=8000",
            "WORKERS=4",
            "",
            "# 🔒 HTTPS Configuration",
            "SSL_CERTIFICATE_PATH=/path/to/certificate.pem",
            "SSL_PRIVATE_KEY_PATH=/path/to/private-key.pem",
            ""
        ]
        
        return "\n".join(template_lines)
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get comprehensive environment status"""
        validation = self.validate_environment()
        
        return {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug_mode": os.getenv("DEBUG", "true").lower() == "true",
            "secrets_configured": len(validation["missing_secrets"]) == 0,
            "secrets_secure": len(validation["weak_secrets"]) == 0,
            "validation": validation,
            "total_secrets": len(self.required_secrets),
            "configured_secrets": len(self.required_secrets) - len(validation["missing_secrets"]),
            "security_score": self._calculate_security_score(validation)
        }
    
    def _calculate_security_score(self, validation: Dict[str, Any]) -> float:
        """Calculate security score based on environment configuration"""
        total_secrets = len(self.required_secrets)
        missing_count = len(validation["missing_secrets"])
        weak_count = len(validation["weak_secrets"])
        
        configured_secrets = total_secrets - missing_count
        secure_secrets = configured_secrets - weak_count
        
        if total_secrets == 0:
            return 100.0
        
        return (secure_secrets / total_secrets) * 100


# Global secret manager instance
secret_manager = SecretManager()


def get_secret(key: str, default: Optional[str] = None) -> str:
    """Convenience function to get secrets"""
    return secret_manager.get_secret(key, default)


def validate_production_environment() -> bool:
    """Validate if environment is ready for production"""
    validation = secret_manager.validate_environment()
    return validation["valid"]


def create_env_file(file_path: str = ".env.production.template") -> str:
    """Create environment file template"""
    template = secret_manager.create_production_env_template()
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    return file_path


if __name__ == "__main__":
    # Demo the secret manager
    print("🔑 DharmaMind Secret Manager")
    print("=" * 50)
    
    status = secret_manager.get_environment_status()
    print(f"Environment: {status['environment']}")
    print(f"Debug Mode: {status['debug_mode']}")
    print(f"Security Score: {status['security_score']:.1f}%")
    print(f"Secrets Configured: {status['configured_secrets']}/{status['total_secrets']}")
    
    if not status['secrets_configured']:
        print("\n⚠️ Missing secrets detected!")
        for secret in status['validation']['missing_secrets']:
            print(f"  - {secret['key']}: {secret['description']}")
    
    if not status['secrets_secure']:
        print("\n⚠️ Weak secrets detected!")
        for secret in status['validation']['weak_secrets']:
            print(f"  - {secret['key']}: {secret['reason']}")
    
    # Create template
    template_path = create_env_file()
    print(f"\n✅ Created environment template: {template_path}")
