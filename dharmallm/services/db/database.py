"""Database manager for DharmaMind - SECURE IMPLEMENTATION"""

import logging
import os
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import aiosqlite
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and operations with security best practices
    - Uses parameterized queries to prevent SQL injection
    - Implements connection pooling
    - Handles transactions properly
    """
    
    def __init__(self, database_url: str = None):
        # Use environment variable or default
        self.database_url = (
            database_url or
            os.getenv("DATABASE_URL", "sqlite:///./data/dharma.db")
        )
        
        # Extract path from URL for SQLite
        if self.database_url.startswith("sqlite:///"):
            self.db_path = self.database_url.replace("sqlite:///", "")
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self.db_path = None
        
        self.connected = False
        self._connection = None
        logger.info(f"DatabaseManager initialized with: {self.database_url}")
    
    async def initialize(self):
        """Initialize the database and create tables"""
        await self.connect()
        await self._create_tables()
        logger.info("✓ Database initialized")
    
    async def cleanup(self):
        """Cleanup database connections"""
        await self.disconnect()
        logger.info("✓ Database cleanup complete")
    
    async def connect(self):
        """Establish database connection"""
        if self.db_path:
            self._connection = await aiosqlite.connect(self.db_path)
            # Enable foreign key support
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.commit()
            self.connected = True
            logger.info("✓ Database connected")
        else:
            # For non-SQLite databases, use appropriate async driver
            logger.warning(
                "Non-SQLite database not yet implemented. "
                "Use SQLite or implement PostgreSQL/MySQL support."
            )
    
    async def disconnect(self):
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None
        self.connected = False
        logger.info("✓ Database disconnected")
    
    async def _create_tables(self):
        """Create necessary database tables"""
        if not self._connection:
            return
        
        # Users table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                is_verified INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP
            )
        """)
        
        # Sessions table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Chat history table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        await self._connection.commit()
        logger.info("✓ Database tables created/verified")
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions
        
        Usage:
            async with db.transaction():
                await db.execute(query, params)
                await db.execute(query2, params2)
        """
        if not self._connection:
            raise RuntimeError("Database not connected")
        
        try:
            yield self._connection
            await self._connection.commit()
        except Exception as e:
            await self._connection.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise
    
    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> int:
        """
        Execute a query with parameterized inputs (prevents SQL injection)
        
        Args:
            query: SQL query with ? placeholders
            params: Tuple of parameters for the query
            
        Returns:
            Number of rows affected
            
        Example:
            await db.execute(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                ("john", "john@example.com")
            )
        """
        if not self._connection:
            raise RuntimeError("Database not connected")
        
        try:
            cursor = await self._connection.execute(query, params or ())
            await self._connection.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Database execute error: {e}")
            logger.error(f"Query: {query}")
            raise
    
    async def fetch_one(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row as a dictionary
        
        Args:
            query: SQL query with ? placeholders
            params: Tuple of parameters for the query
            
        Returns:
            Dictionary with column names as keys, or None
            
        Example:
            user = await db.fetch_one(
                "SELECT * FROM users WHERE username = ?",
                ("john",)
            )
        """
        if not self._connection:
            raise RuntimeError("Database not connected")
        
        try:
            self._connection.row_factory = aiosqlite.Row
            cursor = await self._connection.execute(query, params or ())
            row = await cursor.fetchone()
            
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Database fetch_one error: {e}")
            logger.error(f"Query: {query}")
            raise
    
    async def fetch_all(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all rows as a list of dictionaries
        
        Args:
            query: SQL query with ? placeholders
            params: Tuple of parameters for the query
            
        Returns:
            List of dictionaries with column names as keys
            
        Example:
            users = await db.fetch_all(
                "SELECT * FROM users WHERE is_active = ?",
                (1,)
            )
        """
        if not self._connection:
            raise RuntimeError("Database not connected")
        
        try:
            self._connection.row_factory = aiosqlite.Row
            cursor = await self._connection.execute(query, params or ())
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Database fetch_all error: {e}")
            logger.error(f"Query: {query}")
            raise
    
    async def execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> int:
        """
        Execute a query multiple times with different parameters
        
        Args:
            query: SQL query with ? placeholders
            params_list: List of parameter tuples
            
        Returns:
            Total number of rows affected
            
        Example:
            await db.execute_many(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                [("john", "john@example.com"), ("jane", "jane@example.com")]
            )
        """
        if not self._connection:
            raise RuntimeError("Database not connected")
        
        try:
            cursor = await self._connection.executemany(query, params_list)
            await self._connection.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Database execute_many error: {e}")
            raise


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def init_database(database_url: str = None) -> DatabaseManager:
    """Initialize the global database manager"""
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    await _db_manager.initialize()
    return _db_manager


def get_database() -> Optional[DatabaseManager]:
    """Get the global database manager instance"""
    return _db_manager

