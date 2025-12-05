"""
🗄️ Database Module
==================

Contains all database-related services and repositories:

- connection.py        - Database connection management
- database_service.py  - Main database service
- repositories/        - Data access layer repositories
- migrations/         - Database migration scripts
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Import main classes with robust fallback handling
try:
    from .database_service import SecureDatabaseService
    # Create alias for backward compatibility
    DatabaseService = SecureDatabaseService
    logger.info("✅ Successfully imported SecureDatabaseService")
except ImportError as e:
    logger.error(f"❌ Could not import SecureDatabaseService: {e}")
    
    class SecureDatabaseService:
        """Fallback SecureDatabaseService with basic functionality"""
        def __init__(self, *args, **kwargs):
            logger.warning("⚠️ Using fallback SecureDatabaseService - database features limited")
            self.initialized = False
            self.connected = False
        
        async def connect(self) -> bool:
            logger.error("🚫 Database connection service not available")
            return False
        
        async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
            logger.error("🚫 Database query execution not available")
            return []
        
        async def health_check(self) -> Dict[str, Any]:
            return {
                "status": "unhealthy",
                "error": "Database service not initialized",
                "connected": False
            }
    
    class DatabaseService(SecureDatabaseService):
        """Alias for backward compatibility"""
        pass

try:
    from .connection import DatabaseConnection
    logger.info("✅ Successfully imported DatabaseConnection")
except ImportError as e:
    logger.error(f"❌ Could not import DatabaseConnection: {e}")
    
    class DatabaseConnection:
        """Fallback DatabaseConnection with basic functionality"""
        def __init__(self, *args, **kwargs):
            logger.warning("⚠️ Using fallback DatabaseConnection - connection features limited")
            self.initialized = False
            self.connection = None
        
        async def connect(self) -> bool:
            logger.error("🚫 Database connection not available")
            return False
        
        async def disconnect(self) -> bool:
            logger.error("🚫 Database disconnection not available")
            return False
        
        def is_connected(self) -> bool:
            return False

# Health check function
def check_database_services() -> Dict[str, bool]:
    """Check which database services are available"""
    services = {}
    
    try:
        db_service = DatabaseService()
        services['database_service'] = hasattr(db_service, 'initialized') and db_service.initialized
    except Exception:
        services['database_service'] = False
    
    try:
        db_connection = DatabaseConnection()
        services['database_connection'] = hasattr(db_connection, 'initialized') and db_connection.initialized
    except Exception:
        services['database_connection'] = False
    
    return services

__all__ = [
    'DatabaseService',
    'SecureDatabaseService',
    'DatabaseConnection',
    'check_database_services'
]
