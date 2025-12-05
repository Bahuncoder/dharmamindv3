"""
🔌 External Integrations Module
===============================

Contains all external service integrations:

- email_service.py        - Email service integration
- notification_service.py - Push notification service
- secret_manager.py       - External secret management
"""

# Import main classes with fallback handling
try:
    from .email_service import EmailService
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import EmailService: {e}")
    class EmailService:
        pass

try:
    from .notification_service import NotificationService
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import NotificationService: {e}")
    class NotificationService:
        pass

try:
    from .secret_manager import SecretManager, get_secret
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import SecretManager: {e}")
    class SecretManager:
        pass
    def get_secret(key):
        return None

try:
    from .https_service import HTTPSService
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import HTTPSService: {e}")
    class HTTPSService:
        pass

__all__ = [
    'EmailService',
    'NotificationService',
    'SecretManager',
    'get_secret',
    'HTTPSService'
]
