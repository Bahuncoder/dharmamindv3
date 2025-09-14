"""
🔐 Authentication & Authorization Module
=======================================

Contains all authentication and authorization related services:

- auth_service.py        - Main authentication service
- google_oauth.py        - Google OAuth integration
- security_service.py    - Security and encryption services
- subscription_service.py - User subscription management
"""

# Import main classes with fallback handling
try:
    from .auth_service import AuthenticationService
    # Create alias for backward compatibility
    AuthService = AuthenticationService
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import AuthenticationService: {e}")
    class AuthenticationService:
        pass
    class AuthService:
        pass

try:
    from .google_oauth import GoogleOAuthService
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import GoogleOAuthService: {e}")
    class GoogleOAuthService:
        pass

try:
    from .security_service import SecurityService
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import SecurityService: {e}")
    class SecurityService:
        pass

try:
    from .subscription_service import SubscriptionService
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import SubscriptionService: {e}")
    class SubscriptionService:
        pass

__all__ = [
    'AuthService',
    'AuthenticationService', 
    'GoogleOAuthService',
    'SecurityService',
    'SubscriptionService'
]
