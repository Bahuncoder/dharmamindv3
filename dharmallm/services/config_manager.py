"""
Secure Configuration Manager
Loads and validates environment variables
"""

import os
import secrets
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class SecurityConfig:
    """Security-related configuration"""
    
    # JWT Settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(
        os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")
    )
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(
        os.getenv("RATE_LIMIT_PER_MINUTE", "60")
    )
    RATE_LIMIT_BURST: int = int(os.getenv("RATE_LIMIT_BURST", "100"))
    
    # Session Settings
    SESSION_TIMEOUT_SECONDS: int = int(
        os.getenv("SESSION_TIMEOUT_SECONDS", "3600")
    )
    MAX_LOGIN_ATTEMPTS: int = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    LOCKOUT_DURATION_MINUTES: int = int(
        os.getenv("LOCKOUT_DURATION_MINUTES", "15")
    )
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000"
    ).split(",")
    ALLOWED_METHODS: List[str] = os.getenv(
        "ALLOWED_METHODS", "GET,POST"
    ).split(",")
    ALLOWED_HEADERS: List[str] = os.getenv(
        "ALLOWED_HEADERS", "Authorization,Content-Type"
    ).split(",")
    
    @classmethod
    def validate(cls) -> None:
        """Validate critical security settings"""
        errors = []
        
        # Validate JWT Secret
        if not cls.JWT_SECRET_KEY:
            errors.append(
                "JWT_SECRET_KEY is not set. Generate one with: "
                "python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        elif len(cls.JWT_SECRET_KEY) < 32:
            errors.append(
                f"JWT_SECRET_KEY is too short ({len(cls.JWT_SECRET_KEY)} chars). "
                "Must be at least 32 characters for security."
            )
        
        # Validate Origins
        if "*" in cls.ALLOWED_ORIGINS:
            logger.warning(
                "⚠️ CORS is set to allow all origins (*). "
                "This is insecure for production!"
            )
        
        if errors:
            error_msg = "\n".join(f"  - {err}" for err in errors)
            raise ValueError(
                f"Security configuration errors:\n{error_msg}"
            )
        
        logger.info("✓ Security configuration validated successfully")
    
    @classmethod
    def generate_secret_key(cls) -> str:
        """Generate a secure random secret key"""
        return secrets.token_urlsafe(32)


class DatabaseConfig:
    """Database configuration"""
    
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./data/dharma.db"
    )
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    
    @classmethod
    def validate(cls) -> None:
        """Validate database settings"""
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL is not set")
        
        # Warn if using SQLite in production
        if "sqlite" in cls.DATABASE_URL.lower():
            logger.warning(
                "⚠️ Using SQLite database. Consider PostgreSQL for production."
            )
        
        logger.info("✓ Database configuration validated")


class AppConfig:
    """Application configuration"""
    
    APP_NAME: str = os.getenv("APP_NAME", "DharmaMind")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/dharmamind.log")
    
    @classmethod
    def validate(cls) -> None:
        """Validate application settings"""
        if cls.DEBUG and cls.ENVIRONMENT == "production":
            logger.warning(
                "⚠️ DEBUG mode is enabled in production! This is insecure."
            )
        
        # Create log directory if it doesn't exist
        log_dir = Path(cls.LOG_FILE).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✓ Application configuration validated")


class Config:
    """Main configuration class"""
    
    security = SecurityConfig
    database = DatabaseConfig
    app = AppConfig
    
    @classmethod
    def validate_all(cls) -> None:
        """Validate all configuration sections"""
        logger.info("🔒 Validating configuration...")
        
        try:
            cls.security.validate()
            cls.database.validate()
            cls.app.validate()
            logger.info("✅ All configuration validated successfully")
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    @classmethod
    def print_config(cls, hide_secrets: bool = True) -> None:
        """Print current configuration (for debugging)"""
        print("\n" + "="*60)
        print(f"🕉️  {cls.app.APP_NAME} Configuration")
        print("="*60)
        
        print("\n[Application]")
        print(f"  Version: {cls.app.APP_VERSION}")
        print(f"  Environment: {cls.app.ENVIRONMENT}")
        print(f"  Debug: {cls.app.DEBUG}")
        print(f"  Log Level: {cls.app.LOG_LEVEL}")
        
        print("\n[Security]")
        if hide_secrets:
            print(f"  JWT Secret: {'*' * 20} (hidden)")
        else:
            print(f"  JWT Secret: {cls.security.JWT_SECRET_KEY}")
        print(f"  JWT Algorithm: {cls.security.JWT_ALGORITHM}")
        print(f"  Token Expire: {cls.security.JWT_ACCESS_TOKEN_EXPIRE_MINUTES}min")
        print(f"  Rate Limit: {cls.security.RATE_LIMIT_PER_MINUTE}/min")
        print(f"  Max Login Attempts: {cls.security.MAX_LOGIN_ATTEMPTS}")
        
        print("\n[Database]")
        db_url = cls.database.DATABASE_URL
        if hide_secrets and "@" in db_url:
            # Hide password in connection string
            parts = db_url.split("@")
            db_url = f"{parts[0].split(':')[0]}:***@{parts[1]}"
        print(f"  URL: {db_url}")
        
        print("\n[CORS]")
        print(f"  Allowed Origins: {', '.join(cls.security.ALLOWED_ORIGINS)}")
        print(f"  Allowed Methods: {', '.join(cls.security.ALLOWED_METHODS)}")
        
        print("="*60 + "\n")


# Validate configuration on import
try:
    Config.validate_all()
except Exception as e:
    logger.error(f"Failed to validate configuration: {e}")
    # In development, we might want to continue with defaults
    if os.getenv("ENVIRONMENT") == "production":
        raise
    else:
        logger.warning("⚠️ Using default configuration (not secure for production)")
