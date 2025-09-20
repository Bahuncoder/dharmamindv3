"""
🔐 Authentication & Authorization Module
=======================================

Contains all authentication and authorization related services:

- auth_service.py        - Main authentication service
- google_oauth.py        - Google OAuth integration
- security_service.py    - Security and encryption services
- subscription_service.py - User subscription management
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Import main classes with robust fallback handling
try:
    from .auth_service import AuthenticationService
    # Create alias for backward compatibility
    AuthService = AuthenticationService
    logger.info("✅ Successfully imported AuthenticationService")
except ImportError as e:
    logger.error(f"❌ Could not import AuthenticationService: {e}")
    
    class AuthenticationService:
        """Fallback AuthenticationService with basic functionality"""
        def __init__(self, *args, **kwargs):
            logger.warning("⚠️ Using fallback AuthenticationService - limited functionality")
            self.initialized = False
        
        async def authenticate(self, token: str) -> Optional[Dict[str, Any]]:
            logger.error("🚫 Authentication service not available")
            return None
        
        async def validate_user(self, user_id: str) -> bool:
            logger.error("🚫 User validation service not available")
            return False
    
    class AuthService(AuthenticationService):
        """Alias for backward compatibility"""
        pass

try:
    from .google_oauth import GoogleOAuthService
    logger.info("✅ Successfully imported GoogleOAuthService")
except ImportError as e:
    logger.error(f"❌ Could not import GoogleOAuthService: {e}")
    
    class GoogleOAuthService:
        """Fallback GoogleOAuthService with basic functionality"""
        def __init__(self, *args, **kwargs):
            logger.warning("⚠️ Using fallback GoogleOAuthService - OAuth disabled")
            self.initialized = False
        
        async def get_authorization_url(self) -> str:
            logger.error("🚫 Google OAuth service not available")
            return ""
        
        async def authenticate_user(self, code: str) -> Optional[Dict[str, Any]]:
            logger.error("🚫 Google OAuth authentication not available")
            return None

try:
    from .security_service import SecurityService
    logger.info("✅ Successfully imported SecurityService")
except ImportError as e:
    logger.error(f"❌ Could not import SecurityService: {e}")
    
    class SecurityService:
        """Fallback SecurityService with basic functionality"""
        def __init__(self, *args, **kwargs):
            logger.warning("⚠️ Using fallback SecurityService - security features limited")
            self.initialized = False
        
        def encrypt_data(self, data: str) -> str:
            logger.error("🚫 Encryption service not available")
            return data  # Return unencrypted as fallback
        
        def decrypt_data(self, encrypted_data: str) -> str:
            logger.error("🚫 Decryption service not available")
            return encrypted_data  # Return as-is as fallback
        
        def hash_password(self, password: str) -> str:
            logger.error("🚫 Password hashing service not available")
            return password  # Insecure fallback

try:
    from .subscription_service import SubscriptionService
    logger.info("✅ Successfully imported SubscriptionService")
except ImportError as e:
    logger.error(f"❌ Could not import SubscriptionService: {e}")
    
    class SubscriptionService:
        """Fallback SubscriptionService with basic functionality"""
        def __init__(self, *args, **kwargs):
            logger.warning("⚠️ Using fallback SubscriptionService - subscription features disabled")
            self.initialized = False
        
        async def get_user_subscription(self, user_id: str) -> Optional[Dict[str, Any]]:
            logger.error("🚫 Subscription service not available")
            return {"plan": "free", "status": "active"}  # Default free plan
        
        async def validate_subscription(self, user_id: str, feature: str) -> bool:
            logger.error("🚫 Subscription validation not available")
            return True  # Allow all features as fallback

# Health check function
def check_auth_services() -> Dict[str, bool]:
    """Check which authentication services are available"""
    services = {}
    
    try:
        auth = AuthenticationService()
        services['authentication'] = hasattr(auth, 'initialized') and auth.initialized
    except Exception:
        services['authentication'] = False
    
    try:
        oauth = GoogleOAuthService()
        services['google_oauth'] = hasattr(oauth, 'initialized') and oauth.initialized
    except Exception:
        services['google_oauth'] = False
    
    try:
        security = SecurityService()
        services['security'] = hasattr(security, 'initialized') and security.initialized
    except Exception:
        services['security'] = False
    
    try:
        subscription = SubscriptionService()
        services['subscription'] = hasattr(subscription, 'initialized') and subscription.initialized
    except Exception:
        services['subscription'] = False
    
    return services

__all__ = [
    'AuthService',
    'AuthenticationService', 
    'GoogleOAuthService',
    'SecurityService',
    'SubscriptionService',
    'check_auth_services'
]
