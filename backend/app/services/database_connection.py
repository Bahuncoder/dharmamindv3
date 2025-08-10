"""
Database Connection Service for DharmaMind
Integrates secure database with authentication and email services
"""

import asyncio
import asyncpg
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import os
from datetime import datetime

# Import our enterprise modules
from .secure_database import SecureDatabaseManager
from .email_service import EmailService
from .google_oauth import GoogleOAuthService, OAuthStateManager

logger = logging.getLogger(__name__)


class DatabaseConnectionService:
    """Main database connection service that orchestrates all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_manager = None
        self.email_service = None
        self.oauth_service = None
        self.oauth_state_manager = None
        self.connection_pool = None
        
    async def initialize(self):
        """Initialize all services with database connection"""
        try:
            # Database configuration
            db_config = {
                "host": self.config.get("db_host", "localhost"),
                "port": self.config.get("db_port", 5432),
                "database": self.config.get("db_name", "dharmamind"),
                "user": self.config.get("db_user", "postgres"),
                "password": self.config.get("db_password", "password"),
                "min_size": self.config.get("db_min_pool_size", 5),
                "max_size": self.config.get("db_max_pool_size", 20),
                "command_timeout": self.config.get("db_timeout", 60)
            }
            
            # Create connection pool
            self.connection_pool = await asyncpg.create_pool(**db_config)
            logger.info("Database connection pool created successfully")
            
            # Initialize secure database manager
            encryption_key = self.config.get("encryption_key")
            if not encryption_key:
                logger.warning("No encryption key provided, generating new one")
                from cryptography.fernet import Fernet
                encryption_key = Fernet.generate_key().decode()
            
            self.db_manager = SecureDatabaseManager(
                pool=self.connection_pool,
                encryption_key=encryption_key
            )
            
            # Initialize database tables
            await self.db_manager.initialize_database()
            logger.info("Database tables initialized")
            
            # Initialize email service
            email_config = {
                "smtp_host": self.config.get("smtp_host", "smtp.gmail.com"),
                "smtp_port": self.config.get("smtp_port", 587),
                "smtp_username": self.config.get("smtp_username"),
                "smtp_password": self.config.get("smtp_password"),
                "sender_email": self.config.get("sender_email"),
                "sender_name": self.config.get("sender_name", "DharmaMind AI"),
                "base_url": self.config.get("base_url", "http://localhost:3005")
            }
            
            self.email_service = EmailService(
                config=email_config,
                database_manager=self.db_manager
            )
            logger.info("Email service initialized")
            
            # Initialize OAuth state manager
            self.oauth_state_manager = OAuthStateManager(
                database_manager=self.db_manager
            )
            
            # Initialize Google OAuth service
            if all(key in self.config for key in ["google_client_id", "google_client_secret", "google_redirect_uri"]):
                self.oauth_service = GoogleOAuthService(
                    client_id=self.config["google_client_id"],
                    client_secret=self.config["google_client_secret"],
                    redirect_uri=self.config["google_redirect_uri"],
                    database_manager=self.db_manager,
                    email_service=self.email_service
                )
                logger.info("Google OAuth service initialized")
            else:
                logger.warning("Google OAuth configuration incomplete")
            
            # Start background tasks
            await self._start_background_tasks()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection service: {e}")
            return False
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        try:
            # Schedule periodic cleanup tasks
            asyncio.create_task(self._periodic_cleanup())
            logger.info("Background tasks started")
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired data"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if self.db_manager:
                    # Clean up expired sessions
                    await self.db_manager.cleanup_expired_sessions()
                    
                    # Clean up expired password reset tokens
                    await self.db_manager.cleanup_expired_tokens()
                    
                    # Clean up old rate limit records
                    await self.db_manager.cleanup_old_rate_limits()
                
                if self.oauth_state_manager:
                    # Clean up expired OAuth states
                    await self.oauth_state_manager.cleanup_expired_states()
                
                logger.info("Periodic cleanup completed")
                
            except Exception as e:
                logger.error(f"Periodic cleanup error: {e}")
    
    async def close(self):
        """Close all connections and cleanup"""
        try:
            if self.connection_pool:
                await self.connection_pool.close()
                logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.connection_pool:
            raise RuntimeError("Database not initialized")
        
        async with self.connection_pool.acquire() as conn:
            yield conn
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        health = {
            "database": False,
            "email_service": False,
            "oauth_service": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Check database connection
            if self.connection_pool:
                async with self.connection_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                    health["database"] = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        # Check email service
        if self.email_service:
            health["email_service"] = True  # Could add SMTP connection test
        
        # Check OAuth service
        if self.oauth_service:
            health["oauth_service"] = True
        
        return health


class AuthenticationAPI:
    """Complete authentication API using all services"""
    
    def __init__(self, db_service: DatabaseConnectionService):
        self.db_service = db_service
        self.db = db_service.db_manager
        self.email = db_service.email_service
        self.oauth = db_service.oauth_service
        self.oauth_state = db_service.oauth_state_manager
    
    async def register_user(self, user_data: Dict[str, Any], 
                          ip_address: str = None) -> Dict[str, Any]:
        """Register new user with email verification"""
        try:
            # Validate input data
            required_fields = ["email", "password", "first_name", "last_name"]
            for field in required_fields:
                if field not in user_data or not user_data[field]:
                    return {
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }
            
            # Create user
            result = await self.db.create_user({
                **user_data,
                "ip_address": ip_address,
                "auth_provider": "email"
            })
            
            if result["success"]:
                # Send verification email
                verification_result = await self.email.send_verification_email({
                    "user_id": result["user_id"],
                    "email": user_data["email"],
                    "first_name": user_data["first_name"]
                })
                
                return {
                    "success": True,
                    "user_id": result["user_id"],
                    "message": "Registration successful. Please check your email for verification.",
                    "email_sent": verification_result.get("success", False)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"User registration error: {e}")
            return {
                "success": False,
                "error": "Registration failed"
            }
    
    async def login_user(self, email: str, password: str, 
                        ip_address: str = None) -> Dict[str, Any]:
        """Authenticate user login"""
        try:
            result = await self.db.authenticate_user(email, password, ip_address)
            
            if result["success"]:
                # Create session
                session_result = await self.db.create_session(
                    user_id=result["user_id"],
                    ip_address=ip_address,
                    user_agent="API Client"
                )
                
                if session_result["success"]:
                    return {
                        "success": True,
                        "user": result["user"],
                        "session_token": session_result["session_token"],
                        "expires_at": session_result["expires_at"]
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"User login error: {e}")
            return {
                "success": False,
                "error": "Login failed"
            }
    
    async def initiate_google_oauth(self, redirect_url: str = None) -> Dict[str, Any]:
        """Initiate Google OAuth flow"""
        try:
            if not self.oauth:
                return {
                    "success": False,
                    "error": "OAuth not configured"
                }
            
            # Create state token
            state = await self.oauth_state.create_state(redirect_url=redirect_url)
            
            # Generate OAuth URL
            auth_url, _ = self.oauth.generate_auth_url(state)
            
            return {
                "success": True,
                "auth_url": auth_url,
                "state": state
            }
            
        except Exception as e:
            logger.error(f"OAuth initiation error: {e}")
            return {
                "success": False,
                "error": "OAuth initiation failed"
            }
    
    async def complete_google_oauth(self, code: str, state: str, 
                                  ip_address: str = None) -> Dict[str, Any]:
        """Complete Google OAuth flow"""
        try:
            if not self.oauth:
                return {
                    "success": False,
                    "error": "OAuth not configured"
                }
            
            # Validate state
            state_result = await self.oauth_state.validate_and_consume_state(state)
            if not state_result["valid"]:
                return {
                    "success": False,
                    "error": "Invalid or expired state"
                }
            
            # Exchange code for tokens
            token_result = await self.oauth.exchange_code_for_tokens(code, state)
            if not token_result["success"]:
                return token_result
            
            # Verify ID token and get user info
            id_token_result = await self.oauth.verify_id_token(token_result["id_token"])
            if not id_token_result["success"]:
                return id_token_result
            
            # Authenticate or create user
            auth_result = await self.oauth.authenticate_or_create_user(
                id_token_result["user_info"], ip_address
            )
            
            if auth_result["success"]:
                # Create session
                session_result = await self.db.create_session(
                    user_id=auth_result["user"]["user_id"],
                    ip_address=ip_address,
                    user_agent="OAuth Client"
                )
                
                if session_result["success"]:
                    return {
                        "success": True,
                        "action": auth_result["action"],
                        "user": auth_result["user"],
                        "session_token": session_result["session_token"],
                        "expires_at": session_result["expires_at"]
                    }
            
            return auth_result
            
        except Exception as e:
            logger.error(f"OAuth completion error: {e}")
            return {
                "success": False,
                "error": "OAuth completion failed"
            }
    
    async def verify_email(self, token: str) -> Dict[str, Any]:
        """Verify user email with token"""
        try:
            return await self.db.verify_email(token)
        except Exception as e:
            logger.error(f"Email verification error: {e}")
            return {
                "success": False,
                "error": "Email verification failed"
            }
    
    async def validate_session(self, session_token: str, 
                             ip_address: str = None) -> Dict[str, Any]:
        """Validate user session"""
        try:
            return await self.db.validate_session(session_token, ip_address)
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return {
                "success": False,
                "error": "Session validation failed"
            }
    
    async def logout_user(self, session_token: str) -> Dict[str, Any]:
        """Logout user and invalidate session"""
        try:
            return await self.db.invalidate_session(session_token)
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return {
                "success": False,
                "error": "Logout failed"
            }


# Configuration loader
def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    return {
        # Database
        "db_host": os.getenv("DB_HOST", "localhost"),
        "db_port": int(os.getenv("DB_PORT", "5432")),
        "db_name": os.getenv("DB_NAME", "dharmamind"),
        "db_user": os.getenv("DB_USER", "postgres"),
        "db_password": os.getenv("DB_PASSWORD", "password"),
        "db_min_pool_size": int(os.getenv("DB_MIN_POOL_SIZE", "5")),
        "db_max_pool_size": int(os.getenv("DB_MAX_POOL_SIZE", "20")),
        "db_timeout": int(os.getenv("DB_TIMEOUT", "60")),
        
        # Encryption
        "encryption_key": os.getenv("ENCRYPTION_KEY"),
        
        # Email
        "smtp_host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "smtp_username": os.getenv("SMTP_USERNAME"),
        "smtp_password": os.getenv("SMTP_PASSWORD"),
        "sender_email": os.getenv("SENDER_EMAIL"),
        "sender_name": os.getenv("SENDER_NAME", "DharmaMind AI"),
        "base_url": os.getenv("BASE_URL", "http://localhost:3005"),
        
        # Google OAuth
        "google_client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "google_client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "google_redirect_uri": os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5000/auth/google/callback")
    }


# Main service instance
db_connection_service = None

async def initialize_services():
    """Initialize all services"""
    global db_connection_service
    
    config = load_config_from_env()
    db_connection_service = DatabaseConnectionService(config)
    
    success = await db_connection_service.initialize()
    if success:
        logger.info("All services initialized successfully")
        return db_connection_service
    else:
        logger.error("Failed to initialize services")
        return None

async def get_auth_api() -> AuthenticationAPI:
    """Get authentication API instance"""
    global db_connection_service
    
    if not db_connection_service:
        db_connection_service = await initialize_services()
    
    if db_connection_service:
        return AuthenticationAPI(db_connection_service)
    else:
        raise RuntimeError("Authentication services not available")
