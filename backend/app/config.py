"""
🕉️ DharmaMind Backend Configuration - Advanced System Settings

Centralized configuration management for the complete DharmaMind backend services.
Uses environment variables with sensible defaults and advanced features:

- Multi-environment support (dev, staging, prod)
- Advanced security configurations
- Comprehensive monitoring settings
- AI/ML model configurations
- D            # Validate subscription and payment secrets in production
            subscription_key = values.get("SUBSCRIPTION_ENCRYPTION_KEY", "")
            payment_secret = values.get("PAYMENT_WEBHOOK_SECRET", "")
            
            if not subscription_key or len(subscription_key) < 32:
                raise ValueError("SUBSCRIPTION_ENCRYPTION_KEY must be at least 32 characters in production")
            
            if not payment_secret or len(payment_secret) < 32:
                raise ValueError("PAYMENT_WEBHOOK_SECRET must be at least 32 characters in production")
            
            # Validate CORS settings in production
            allowed_hosts = values.get("ALLOWED_HOSTS", [])
            if "*" in allowed_hosts:
                raise ValueError("Wildcard hosts (*) not allowed in production")
            
            # Basic validation - gateway URL should be configured for external LLM featuresarmic compliance settings
- Performance optimization parameters
"""

import os
import logging
from typing import List, Optional, Dict, Any, Union
try:
    from pydantic_settings import BaseSettings
    from pydantic import field_validator, model_validator, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Create mock classes for when pydantic_settings is not installed
    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def model_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def Field(*args, **kwargs):
        return None
    
    PYDANTIC_AVAILABLE = False

from pathlib import Path
from enum import Enum
import json
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass
    DOTENV_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class VectorDBType(str, Enum):
    """Supported vector database types"""
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMA = "chroma"
    FAISS = "faiss"

class AIProvider(str, Enum):
    """AI service providers"""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    DHARMALLM = "dharmallm"
    GATEWAY = "gateway"  # External LLMs via gateway service

class Settings(BaseSettings):
    """🕉️ Advanced DharmaMind Application Settings"""
    
    # ===============================
    # CORE APPLICATION SETTINGS
    # ===============================
    APP_NAME: str = "DharmaMind Complete API"
    VERSION: str = "2.0.0"
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = False
    
    # ===============================
    # SECURITY & AUTHENTICATION
    # ===============================
    SECRET_KEY: str = Field(default="", env="SECRET_KEY")
    JWT_SECRET_KEY: str = Field(default="", env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # API Security
    API_KEY_HEADER: str = "X-API-Key"
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # seconds
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # CORS Configuration
    CORS_ORIGINS: Union[List[str], str] = Field(
        default=[
            "http://localhost:3000",  # Development frontend
            "http://localhost:3002",  # Brand website
            "http://localhost:3003",  # Chat application
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3002",
            "http://127.0.0.1:3003"
        ],
        env="CORS_ORIGINS",
        description="Allowed CORS origins - configure specific domains for production"
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "PATCH"]
    CORS_ALLOW_HEADERS: List[str] = [
        "Authorization",
        "Content-Type", 
        "X-API-Key",
        "X-Requested-With"
    ]
    
    # Trusted Hosts - Restrictive by default
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "dharmamind.com", "*.dharmamind.com"]
    
    # ===============================
    # SUBSCRIPTION & PAYMENT SETTINGS
    # ===============================
    # Payment Providers
    STRIPE_PUBLISHABLE_KEY: Optional[str] = None
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    
    PAYPAL_CLIENT_ID: Optional[str] = None
    PAYPAL_SECRET_KEY: Optional[str] = None
    PAYPAL_WEBHOOK_ID: Optional[str] = None
    
    # Subscription Configuration
    SUBSCRIPTION_ENCRYPTION_KEY: str = Field(default="", env="SUBSCRIPTION_ENCRYPTION_KEY")
    PAYMENT_WEBHOOK_SECRET: str = Field(default="", env="PAYMENT_WEBHOOK_SECRET")
    
    # Billing Settings
    DEFAULT_CURRENCY: str = "USD"
    TRIAL_PERIOD_DAYS: int = 14
    BILLING_GRACE_PERIOD_DAYS: int = 3
    
    # Usage Limits (Free Plan)
    FREE_PLAN_MONTHLY_CHATS: int = 50
    FREE_PLAN_API_CALLS: int = 0
    FREE_PLAN_STORAGE_GB: int = 1
    
    # Pro Plan Limits
    PRO_PLAN_MONTHLY_CHATS: int = -1  # Unlimited
    PRO_PLAN_API_CALLS: int = 1000
    PRO_PLAN_STORAGE_GB: int = 10
    
    # Max Plan Limits  
    MAX_PLAN_MONTHLY_CHATS: int = -1  # Unlimited
    MAX_PLAN_API_CALLS: int = 10000
    MAX_PLAN_STORAGE_GB: int = 100
    
    # Enterprise Plan Limits
    ENTERPRISE_PLAN_MONTHLY_CHATS: int = -1  # Unlimited
    ENTERPRISE_PLAN_API_CALLS: int = -1  # Unlimited
    ENTERPRISE_PLAN_STORAGE_GB: int = -1  # Unlimited
    
    # Payment Security
    PCI_COMPLIANCE_LEVEL: str = "Level 1"
    PAYMENT_DATA_RETENTION_DAYS: int = 2555  # 7 years
    ENABLE_PAYMENT_FRAUD_DETECTION: bool = True
    REQUIRE_3D_SECURE: bool = True
    
    # Subscription Analytics
    ENABLE_SUBSCRIPTION_ANALYTICS: bool = True
    USAGE_TRACKING_ENABLED: bool = True
    BILLING_ANALYTICS_ENABLED: bool = True
    
    # ===============================
    # DATABASE CONFIGURATION
    # ===============================
    # Primary Database
    DATABASE_URL: str = Field(
        default="sqlite:///./dharmamind_dev.db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_ECHO: bool = False
    
    # Redis Cache & Sessions
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 10
    REDIS_RETRY_ON_TIMEOUT: bool = True
    
    # Vector Database Configuration
    VECTOR_DB_TYPE: VectorDBType = VectorDBType.QDRANT
    VECTOR_DB_URL: str = "http://localhost:6333"
    VECTOR_DB_API_KEY: Optional[str] = None
    VECTOR_DB_COLLECTION: str = "dharmamind_vectors"
    VECTOR_DIMENSION: int = 1536  # OpenAI embedding dimension
    VECTOR_BATCH_SIZE: int = 100
    
    # ===============================
    # AI/ML MODEL CONFIGURATION
    # ===============================
    # Primary AI Provider
    PRIMARY_AI_PROVIDER: AIProvider = AIProvider.DHARMALLM
    FALLBACK_AI_PROVIDER: AIProvider = AIProvider.LOCAL
    
    # LLM Gateway Configuration (Separate Microservice)
    LLM_GATEWAY_URL: str = "http://localhost:8003"
    LLM_GATEWAY_API_KEY: str = Field(default="", env="LLM_GATEWAY_API_KEY")
    
    # HuggingFace Configuration
    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_MODEL: str = "microsoft/DialoGPT-large"
    HUGGINGFACE_ENDPOINT: Optional[str] = None
    
    # DharmaLLM Configuration (Local Model)
    DHARMALLM_MODEL_PATH: str = "./models/dharmallm-7b"
    DHARMALLM_MAX_LENGTH: int = 2048
    DHARMALLM_TEMPERATURE: float = 0.7
    DHARMALLM_TOP_P: float = 0.9
    DHARMALLM_DEVICE: str = "auto"  # auto, cpu, cuda
    DHARMALLM_LOAD_IN_8BIT: bool = False
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "auto"
    EMBEDDING_BATCH_SIZE: int = 32
    
    # ===============================
    # MONITORING & LOGGING
    # ===============================
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/dharmamind.log"
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 60  # seconds
    
    # Performance Monitoring
    ENABLE_TRACING: bool = False
    JAEGER_ENDPOINT: Optional[str] = None
    PERFORMANCE_LOGGING: bool = True
    
    # ===============================
    # DHARMIC COMPLIANCE SETTINGS
    # ===============================
    # Dharmic Validation
    ENABLE_DHARMIC_VALIDATION: bool = True
    DHARMIC_CONFIDENCE_THRESHOLD: float = 0.8
    DHARMIC_PRINCIPLES: List[str] = [
        "ahimsa",  # non-violence
        "satya",   # truthfulness
        "asteya",  # non-stealing
        "brahmacharya",  # moderation
        "aparigraha"  # non-possessiveness
    ]
    
    # Content Filtering
    ENABLE_CONTENT_FILTER: bool = True
    BLOCKED_TOPICS: List[str] = [
        "violence", "hate_speech", "harmful_content",
        "illegal_activities", "extremism"
    ]
    
    # Wisdom Sources
    SCRIPTURE_SOURCES: List[str] = [
        "bhagavad_gita", "upanishads", "vedas", 
        "dharma_shastras", "puranas", "buddhist_texts"
    ]
    
    # ===============================
    # ADVANCED FEATURES
    # ===============================
    # Memory & Context Management
    MAX_CONVERSATION_HISTORY: int = 20
    CONTEXT_WINDOW_SIZE: int = 8192
    MEMORY_CLEANUP_INTERVAL: int = 3600  # seconds
    ENABLE_LONG_TERM_MEMORY: bool = True
    
    # Response Generation
    ENABLE_RESPONSE_SCORING: bool = True
    MIN_CONFIDENCE_THRESHOLD: float = 0.7
    MAX_RESPONSE_LENGTH: int = 2048
    RESPONSE_TIMEOUT: int = 30
    
    # Module Configuration
    MODULE_CONFIG_PATH: str = "./modules"
    CHAKRA_MODULE_PATH: str = "./chakra_modules"
    SPIRITUAL_MODULE_PATH: str = "./spiritual_modules"
    
    # Caching
    ENABLE_RESPONSE_CACHE: bool = True
    CACHE_TTL: int = 3600  # seconds
    CACHE_MAX_SIZE: int = 1000
    
    # ===============================
    # DEVELOPMENT & TESTING
    # ===============================
    # Development Features
    ENABLE_DEBUG_ENDPOINTS: bool = False
    ENABLE_SWAGGER_UI: bool = True
    ENABLE_REDOC: bool = True
    
    # Testing Configuration
    TEST_DATABASE_URL: Optional[str] = None
    TEST_REDIS_URL: Optional[str] = None
    
    # Feature Flags
    FEATURE_FLAGS: Dict[str, bool] = {
        "advanced_analytics": True,
        "experimental_features": False,
        "beta_modules": False,
        "premium_features": False
    }
    
    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting"""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                return Environment.DEVELOPMENT
        return v
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def validate_cors_origins(cls, v):
        """Parse CORS origins from string if needed"""
        if isinstance(v, str):
            # Handle comma-separated string
            origins = [origin.strip() for origin in v.split(",")]
            return origins
        elif isinstance(v, list):
            return v
        return v
    
    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def validate_allowed_hosts(cls, v):
        """Parse allowed hosts from string if needed"""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @field_validator("FEATURE_FLAGS", mode="before")
    @classmethod
    def validate_feature_flags(cls, v):
        """Parse feature flags from JSON string if needed"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v or {}
    
    @model_validator(mode='after')
    @classmethod
    def validate_ai_configuration(cls, values):
        """Validate AI provider configuration"""
        if isinstance(values, dict):
            primary_provider = values.get("PRIMARY_AI_PROVIDER")
            environment = values.get("ENVIRONMENT")
    @model_validator(mode='before')
    @classmethod
    def validate_settings(cls, values):
        """Validate critical security settings"""
        
        # Validate environment-specific settings
        environment = values.get("ENVIRONMENT", Environment.DEVELOPMENT)
        
        # Check for production security requirements
        if environment == Environment.PRODUCTION:
            secret_key = values.get("SECRET_KEY", "")
            jwt_secret = values.get("JWT_SECRET_KEY", "")
            
            if not secret_key or len(secret_key) < 32:
                raise ValueError("SECRET_KEY must be at least 32 characters in production")
            
            if not jwt_secret or len(jwt_secret) < 32:
                raise ValueError("JWT_SECRET_KEY must be at least 32 characters in production")
            
            # Validate database URL in production
            db_url = values.get("DATABASE_URL", "")
            if "sqlite" in db_url.lower():
                raise ValueError("SQLite not recommended for production. Use PostgreSQL.")
            
            # Check debug mode
            debug = values.get("DEBUG", False)
            if debug:
                raise ValueError("DEBUG must be False in production")
        
        # Basic validation - gateway URL should be configured for external LLM features
        return values
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.ENVIRONMENT == Environment.TESTING
    
    def get_database_url(self, test: bool = False) -> str:
        """Get appropriate database URL"""
        if test and self.TEST_DATABASE_URL:
            return self.TEST_DATABASE_URL
        return self.DATABASE_URL
    
    def get_redis_url(self, test: bool = False) -> str:
        """Get appropriate Redis URL"""
        if test and self.TEST_REDIS_URL:
            return self.TEST_REDIS_URL
        return self.REDIS_URL
    
    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get feature flag value"""
        return self.FEATURE_FLAGS.get(flag_name, default)
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "use_enum_values": True,
        "extra": "ignore"
    }

# Global settings instance
settings = Settings()


# ===============================
# UTILITY FUNCTIONS
# ===============================

def is_production() -> bool:
    """Convenience function to check if running in production"""
    return settings.is_production

def get_env_file_path() -> str:
    """Get path to environment file based on environment"""
    env = os.getenv("ENVIRONMENT", "development")
    env_files = {
        "development": ".env.dev",
        "staging": ".env.staging", 
        "production": ".env.prod",
        "testing": ".env.test"
    }
    
    env_file = env_files.get(env, ".env")
    return env_file if os.path.exists(env_file) else ".env"


def load_settings() -> Settings:
    """Load settings with appropriate environment file"""
    env_file = get_env_file_path()
    return Settings(_env_file=env_file)


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    DEBUG: bool = True
    LOG_LEVEL: LogLevel = LogLevel.DEBUG
    ENABLE_DEBUG_ENDPOINTS: bool = True
    ENABLE_TRACING: bool = True


class ProductionSettings(Settings):
    """Production environment settings"""
    DEBUG: bool = False
    LOG_LEVEL: LogLevel = LogLevel.INFO
    ENABLE_DEBUG_ENDPOINTS: bool = False
    WORKERS: int = 4
    

class TestingSettings(Settings):
    """Testing environment settings"""
    DEBUG: bool = True
    LOG_LEVEL: LogLevel = LogLevel.DEBUG
    DATABASE_URL: str = "sqlite:///./test.db"
    REDIS_URL: str = "redis://localhost:6379/1"


def get_settings_for_environment(env: Environment) -> Settings:
    """Get settings class for specific environment"""
    settings_map = {
        Environment.DEVELOPMENT: DevelopmentSettings,
        Environment.PRODUCTION: ProductionSettings,
        Environment.TESTING: TestingSettings,
        Environment.STAGING: Settings  # Use base settings for staging
    }
    
    settings_class = settings_map.get(env, Settings)
    return settings_class()


# Export the global settings instance
__all__ = [
    "Settings",
    "Environment", 
    "LogLevel",
    "VectorDBType",
    "AIProvider",
    "settings",
    "is_production",
    "get_settings_for_environment",
    "load_settings"
]
