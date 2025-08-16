"""
🗄️ DharmaMind Secure Database Service

PostgreSQL integration with encryption for sensitive data storage.
Implements secure database operations with field-level encryption.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from services.security_service import (
    encrypt_data,
    decrypt_data,
    sanitize_input,
    hash_password,
    SecurityLevel
)

logger = logging.getLogger(__name__)


class SecureDatabaseService:
    """🔐 Secure PostgreSQL database service with encryption"""
    
    def __init__(self):
        self.database_url = self._get_database_url()
        self.engine = None
        self.async_session = None
        
        # Define which fields should be encrypted
        self.encrypted_fields = {
            'users': ['phone', 'personal_notes'],
            'user_profiles': ['address', 'emergency_contact'],
            'sessions': ['ip_address'],
            'security_events': ['details']
        }
        
        logger.info("🗄️ Secure Database Service initialized")
    
    def _get_database_url(self) -> str:
        """Get database URL from environment or use default"""
        # Try environment variables first
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url
        
        # Build from individual components
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        name = os.getenv('DB_NAME', 'dharmamind')
        user = os.getenv('DB_USER', 'dharmamind')
        password = os.getenv('DB_PASSWORD', 'secure_password_123')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
    
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.database_url.replace('postgresql://', 'postgresql+asyncpg://'),
                echo=False,  # Set to True for SQL debugging
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Create session factory
            async_session = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            self.async_session = async_session
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info("✅ Database connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    async def _create_tables(self):
        """Create database tables with proper schema"""
        create_sql = """
        -- Users table with encrypted fields
        CREATE TABLE IF NOT EXISTS users (
            user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            phone TEXT,  -- encrypted
            subscription_plan VARCHAR(50) DEFAULT 'free',
            status VARCHAR(50) DEFAULT 'active',
            email_verified BOOLEAN DEFAULT FALSE,
            auth_provider VARCHAR(50) DEFAULT 'email',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            login_count INTEGER DEFAULT 0,
            personal_notes TEXT  -- encrypted
        );
        
        -- User profiles table
        CREATE TABLE IF NOT EXISTS user_profiles (
            profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
            timezone VARCHAR(50) DEFAULT 'UTC',
            marketing_consent BOOLEAN DEFAULT FALSE,
            profile_picture TEXT,
            address TEXT,  -- encrypted
            emergency_contact TEXT,  -- encrypted
            preferences JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Sessions table
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_token VARCHAR(255) UNIQUE NOT NULL,
            user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
            ip_address TEXT,  -- encrypted
            user_agent TEXT,
            expires_at TIMESTAMP NOT NULL,
            remember_me BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Security events table
        CREATE TABLE IF NOT EXISTS security_events (
            event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            event_type VARCHAR(100) NOT NULL,
            user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
            severity VARCHAR(50) DEFAULT 'medium',
            ip_address INET,
            details JSONB,  -- encrypted
            action_taken VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Password reset tokens table
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            token_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            token VARCHAR(255) UNIQUE NOT NULL,
            user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
            expires_at TIMESTAMP NOT NULL,
            used BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
        CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security_events(user_id);
        CREATE INDEX IF NOT EXISTS idx_reset_tokens_token ON password_reset_tokens(token);
        
        -- Update timestamp triggers
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        DROP TRIGGER IF EXISTS update_users_updated_at ON users;
        CREATE TRIGGER update_users_updated_at 
            BEFORE UPDATE ON users 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        
        DROP TRIGGER IF EXISTS update_profiles_updated_at ON user_profiles;
        CREATE TRIGGER update_profiles_updated_at 
            BEFORE UPDATE ON user_profiles 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """
        
        async with self.engine.begin() as conn:
            for statement in create_sql.split(';'):
                if statement.strip():
                    await conn.execute(text(statement.strip()))
        
        logger.info("✅ Database tables created successfully")
    
    def _encrypt_user_data(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in user data"""
        if table not in self.encrypted_fields:
            return data
        
        encrypted_data = data.copy()
        
        for field in self.encrypted_fields[table]:
            if field in encrypted_data and encrypted_data[field]:
                try:
                    encrypted_data[field] = encrypt_data(str(encrypted_data[field]))
                except Exception as e:
                    logger.error(f"Encryption failed for {field}: {e}")
                    # Don't store unencrypted sensitive data
                    encrypted_data[field] = None
        
        return encrypted_data
    
    def _decrypt_user_data(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in user data"""
        if table not in self.encrypted_fields:
            return data
        
        decrypted_data = data.copy()
        
        for field in self.encrypted_fields[table]:
            if field in decrypted_data and decrypted_data[field]:
                try:
                    decrypted_data[field] = decrypt_data(decrypted_data[field])
                except Exception as e:
                    logger.warning(f"Decryption failed for {field}: {e}")
                    # Field might not be encrypted or corrupted
                    decrypted_data[field] = None
        
        return decrypted_data
    
    async def create_user(self, user_data: Dict[str, Any]) -> Optional[str]:
        """Create new user with encrypted sensitive data"""
        try:
            # Sanitize inputs
            user_data['email'] = sanitize_input(user_data['email'])
            user_data['first_name'] = sanitize_input(user_data['first_name'])
            user_data['last_name'] = sanitize_input(user_data['last_name'])
            
            # Encrypt sensitive fields
            encrypted_data = self._encrypt_user_data('users', user_data)
            
            # Hash password securely
            encrypted_data['password_hash'] = hash_password(user_data['password'])
            
            # Remove plain password
            if 'password' in encrypted_data:
                del encrypted_data['password']
            
            async with self.async_session() as session:
                # Check if user already exists
                result = await session.execute(
                    text("SELECT user_id FROM users WHERE email = :email"),
                    {"email": encrypted_data['email']}
                )
                existing_user = result.fetchone()
                
                if existing_user:
                    logger.warning(f"User creation failed: email already exists")
                    return None
                
                # Insert new user
                insert_sql = """
                INSERT INTO users (email, password_hash, first_name, last_name, 
                                 phone, subscription_plan, status, email_verified, 
                                 auth_provider, login_count)
                VALUES (:email, :password_hash, :first_name, :last_name,
                        :phone, :subscription_plan, :status, :email_verified,
                        :auth_provider, :login_count)
                RETURNING user_id
                """
                
                result = await session.execute(text(insert_sql), {
                    'email': encrypted_data['email'],
                    'password_hash': encrypted_data['password_hash'],
                    'first_name': encrypted_data['first_name'],
                    'last_name': encrypted_data['last_name'],
                    'phone': encrypted_data.get('phone'),
                    'subscription_plan': encrypted_data.get('subscription_plan', 'free'),
                    'status': encrypted_data.get('status', 'active'),
                    'email_verified': encrypted_data.get('email_verified', True),
                    'auth_provider': encrypted_data.get('auth_provider', 'email'),
                    'login_count': 0
                })
                
                user_id = result.fetchone()[0]
                await session.commit()
                
                logger.info(f"✅ User created successfully: {encrypted_data['email']}")
                return str(user_id)
                
        except Exception as e:
            logger.error(f"❌ User creation failed: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email with decrypted sensitive data"""
        try:
            email = sanitize_input(email)
            
            async with self.async_session() as session:
                result = await session.execute(
                    text("""
                    SELECT user_id, email, password_hash, first_name, last_name,
                           phone, subscription_plan, status, email_verified,
                           auth_provider, created_at, updated_at, last_login,
                           login_count, personal_notes
                    FROM users WHERE email = :email
                    """),
                    {"email": email}
                )
                
                row = result.fetchone()
                if not row:
                    return None
                
                # Convert to dict
                user_data = dict(row._mapping)
                
                # Decrypt sensitive fields
                decrypted_data = self._decrypt_user_data('users', user_data)
                
                return decrypted_data
                
        except Exception as e:
            logger.error(f"❌ Get user failed: {e}")
            return None
    
    async def update_user_login(self, user_id: str, ip_address: str = None):
        """Update user login statistics"""
        try:
            # Encrypt IP address
            encrypted_ip = encrypt_data(ip_address) if ip_address else None
            
            async with self.async_session() as session:
                await session.execute(
                    text("""
                    UPDATE users 
                    SET last_login = CURRENT_TIMESTAMP,
                        login_count = login_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = :user_id
                    """),
                    {"user_id": user_id}
                )
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Update login failed: {e}")
    
    async def create_session(self, session_data: Dict[str, Any]) -> bool:
        """Create user session with encrypted IP"""
        try:
            # Encrypt sensitive fields
            encrypted_data = self._encrypt_user_data('sessions', session_data)
            
            async with self.async_session() as session:
                insert_sql = """
                INSERT INTO user_sessions (session_token, user_id, ip_address,
                                         user_agent, expires_at, remember_me)
                VALUES (:session_token, :user_id, :ip_address, :user_agent,
                        :expires_at, :remember_me)
                """
                
                await session.execute(text(insert_sql), {
                    'session_token': encrypted_data['session_token'],
                    'user_id': encrypted_data['user_id'],
                    'ip_address': encrypted_data.get('ip_address'),
                    'user_agent': encrypted_data.get('user_agent'),
                    'expires_at': encrypted_data['expires_at'],
                    'remember_me': encrypted_data.get('remember_me', False)
                })
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Session creation failed: {e}")
            return False
    
    async def get_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get session with decrypted data"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    text("""
                    SELECT s.*, u.email, u.status 
                    FROM user_sessions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.session_token = :token 
                    AND s.expires_at > CURRENT_TIMESTAMP
                    """),
                    {"token": session_token}
                )
                
                row = result.fetchone()
                if not row:
                    return None
                
                session_data = dict(row._mapping)
                
                # Decrypt sensitive fields
                decrypted_data = self._decrypt_user_data('sessions', session_data)
                
                return decrypted_data
                
        except Exception as e:
            logger.error(f"❌ Get session failed: {e}")
            return None
    
    async def log_security_event(self, event_data: Dict[str, Any]):
        """Log security event with encrypted details"""
        try:
            # Encrypt sensitive details
            if 'details' in event_data:
                event_data['details'] = encrypt_data(json.dumps(event_data['details']))
            
            async with self.async_session() as session:
                insert_sql = """
                INSERT INTO security_events (event_type, user_id, severity,
                                           ip_address, details, action_taken)
                VALUES (:event_type, :user_id, :severity, :ip_address,
                        :details, :action_taken)
                """
                
                await session.execute(text(insert_sql), {
                    'event_type': event_data.get('event_type'),
                    'user_id': event_data.get('user_id'),
                    'severity': event_data.get('severity', 'medium'),
                    'ip_address': event_data.get('ip_address'),
                    'details': event_data.get('details'),
                    'action_taken': event_data.get('action_taken', 'logged')
                })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Security event logging failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Database health check"""
        try:
            async with self.async_session() as session:
                # Test connection
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
                
                # Get basic stats
                stats_result = await session.execute(text("""
                SELECT 
                    (SELECT COUNT(*) FROM users) as total_users,
                    (SELECT COUNT(*) FROM user_sessions WHERE expires_at > CURRENT_TIMESTAMP) as active_sessions,
                    (SELECT COUNT(*) FROM security_events WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours') as recent_events
                """))
                
                stats = dict(stats_result.fetchone()._mapping)
                
                return {
                    "status": "healthy",
                    "database": "postgresql",
                    "encryption": "enabled",
                    "stats": stats,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("🔒 Database connections closed")


# Global database service instance
db_service = SecureDatabaseService()


# Convenience functions for easy integration
async def init_database():
    """Initialize database service"""
    return await db_service.initialize()


async def get_db_health():
    """Get database health status"""
    return await db_service.health_check()


async def close_database():
    """Close database connections"""
    await db_service.close()
