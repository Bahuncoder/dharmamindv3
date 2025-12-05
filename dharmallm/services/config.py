"""
Application Configuration Settings
===================================

Central configuration for the DharmaMind backend application.
"""

from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings"""
    
    # Application
    APP_NAME: str = "DharmaMind Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Security
    SECRET_KEY: str = "dharma-secret-key-change-in-production"
    JWT_SECRET_KEY: str = "jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/dharma_knowledge.db"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    USE_FAKE_REDIS: bool = True  # Use fakeredis for development
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Performance
    ENABLE_PERFORMANCE_MONITORING: bool = True
    ENABLE_DISTRIBUTED_TRACING: bool = False
    METRICS_ENABLED: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # LLM Configuration
    LLM_CACHE_ENABLED: bool = True
    LLM_CACHE_TTL: int = 3600
    LLM_TIMEOUT: int = 30
    
    # Dharmic Settings
    DEFAULT_TRADITION: str = "universal"
    DEFAULT_WISDOM_LEVEL: str = "intermediate"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
