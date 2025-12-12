"""
🔐 Production Environment Variable Validator

Ensures all required security-critical environment variables are properly
configured before the application starts in production.
"""

import os
import sys
import secrets
from typing import List, Tuple


class EnvValidationError(Exception):
    """Raised when environment validation fails."""
    pass


def validate_production_env() -> Tuple[bool, List[str]]:
    """
    Validate all security-critical environment variables for production.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    warnings = []
    
    env = os.getenv("ENVIRONMENT", "development").lower()
    is_production = env in ("production", "prod")
    
    # ============================================
    # CRITICAL: JWT Secret Key
    # ============================================
    jwt_secret = os.getenv("JWT_SECRET_KEY", "")
    
    if not jwt_secret:
        if is_production:
            errors.append("JWT_SECRET_KEY is required in production")
        else:
            warnings.append("JWT_SECRET_KEY not set - using generated key (OK for dev)")
            # Generate secure key for development
            os.environ["JWT_SECRET_KEY"] = secrets.token_urlsafe(32)
    elif len(jwt_secret) < 32:
        errors.append(f"JWT_SECRET_KEY must be at least 32 characters (got {len(jwt_secret)})")
    elif jwt_secret in ("your-secret-key", "change-me", "secret", "dharmamind-secure-key-change-in-production-2024"):
        errors.append("JWT_SECRET_KEY is using an insecure default value")
    
    # ============================================
    # CRITICAL: Database URL
    # ============================================
    db_url = os.getenv("DATABASE_URL", "")
    
    if is_production:
        if not db_url:
            errors.append("DATABASE_URL is required in production")
        elif "localhost" in db_url or "127.0.0.1" in db_url:
            warnings.append("DATABASE_URL points to localhost in production")
        elif "password" in db_url.lower() and ("password123" in db_url or "admin" in db_url):
            errors.append("DATABASE_URL contains weak credentials")
    
    # ============================================
    # CRITICAL: Redis URL (for sessions)
    # ============================================
    redis_url = os.getenv("REDIS_URL", "")
    
    if is_production and not redis_url:
        warnings.append("REDIS_URL not set - session management may be limited")
    
    # ============================================
    # IMPORTANT: Stripe Keys (for payments)
    # ============================================
    stripe_key = os.getenv("STRIPE_SECRET_KEY", "")
    stripe_webhook = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    
    if stripe_key:
        if stripe_key.startswith("sk_test_") and is_production:
            errors.append("STRIPE_SECRET_KEY is using test key in production")
        elif not stripe_key.startswith("sk_"):
            errors.append("STRIPE_SECRET_KEY format is invalid")
    
    if is_production and stripe_key and not stripe_webhook:
        warnings.append("STRIPE_WEBHOOK_SECRET not set - webhook verification disabled")
    
    # ============================================
    # IMPORTANT: CORS Configuration
    # ============================================
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    
    if is_production and cors_origins == "*":
        errors.append("CORS_ORIGINS should not be '*' in production")
    
    # ============================================
    # IMPORTANT: Debug Mode
    # ============================================
    debug_mode = os.getenv("DEBUG", "false").lower()
    
    if is_production and debug_mode in ("true", "1", "yes"):
        errors.append("DEBUG mode should be disabled in production")
    
    # ============================================
    # OPTIONAL: Email Configuration
    # ============================================
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_user = os.getenv("SMTP_USER", "")
    
    if is_production and not smtp_host:
        warnings.append("SMTP_HOST not configured - email features disabled")
    
    # ============================================
    # OPTIONAL: Google OAuth
    # ============================================
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
    
    if google_client_id and not google_client_secret:
        errors.append("GOOGLE_CLIENT_SECRET required when GOOGLE_CLIENT_ID is set")
    
    # ============================================
    # Security Headers
    # ============================================
    if is_production:
        allowed_hosts = os.getenv("ALLOWED_HOSTS", "")
        if not allowed_hosts:
            warnings.append("ALLOWED_HOSTS not set - using default")
    
    return len(errors) == 0, errors, warnings


def print_validation_report(is_valid: bool, errors: List[str], warnings: List[str]):
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("🔐 DHARMAMIND ENVIRONMENT VALIDATION REPORT")
    print("=" * 60)
    
    env = os.getenv("ENVIRONMENT", "development")
    print(f"\n📍 Environment: {env.upper()}")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for error in errors:
            print(f"   • {error}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"   • {warning}")
    
    if is_valid and not warnings:
        print("\n✅ All environment variables are properly configured!")
    elif is_valid:
        print("\n✅ Validation passed with warnings")
    else:
        print("\n🛑 VALIDATION FAILED - Fix errors before starting production")
    
    print("\n" + "=" * 60 + "\n")


def require_valid_env():
    """
    Validate environment and exit if production validation fails.
    Call this at application startup.
    """
    is_valid, errors, warnings = validate_production_env()
    
    # Always print report in development
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env != "testing":
        print_validation_report(is_valid, errors, warnings)
    
    # Only exit on errors in production
    if not is_valid and env in ("production", "prod"):
        print("🛑 Application startup blocked due to configuration errors.")
        print("   Please fix the above errors and restart.")
        sys.exit(1)
    
    return is_valid


# Example .env template
ENV_TEMPLATE = """
# ==================================================
# DHARMAMIND PRODUCTION ENVIRONMENT TEMPLATE
# ==================================================
# Copy this to .env and fill in your values

# Environment (development, staging, production)
ENVIRONMENT=production

# CRITICAL - Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
JWT_SECRET_KEY=

# Database
DATABASE_URL=postgresql://user:password@host:5432/dharmamind

# Redis (for sessions and caching)
REDIS_URL=redis://localhost:6379/0

# Stripe (Payment Processing)
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# CORS (comma-separated origins)
CORS_ORIGINS=https://dharmamind.com,https://dharmamind.ai,https://dharmamind.org

# Disable debug in production
DEBUG=false

# Email (Optional)
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=

# Google OAuth (Optional)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=

# Allowed Hosts
ALLOWED_HOSTS=dharmamind.com,dharmamind.ai,dharmamind.org
"""


if __name__ == "__main__":
    # Run validation when script is executed directly
    require_valid_env()
