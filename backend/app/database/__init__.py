"""
🗄️ Database Module
==================

Contains all database-related services and repositories:

- connection.py        - Database connection management
- database_service.py  - Main database service
- repositories/        - Data access layer repositories
- migrations/         - Database migration scripts
"""

# Import main classes with fallback handling
try:
    from .database_service import SecureDatabaseService
    # Create alias for backward compatibility
    DatabaseService = SecureDatabaseService
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import SecureDatabaseService: {e}")
    class SecureDatabaseService:
        pass
    class DatabaseService:
        pass

try:
    from .connection import DatabaseConnection
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import DatabaseConnection: {e}")
    class DatabaseConnection:
        pass

__all__ = [
    'DatabaseService',
    'DatabaseConnection'
]
