"""
Enterprise-Grade Database Schema for DharmaMind
Secure user management with encryption, verification, and audit trails
"""

import asyncpg
import bcrypt
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import logging
from cryptography.fernet import Fernet
import os

logger = logging.getLogger(__name__)

class UserStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"

class AuthProvider(Enum):
    EMAIL = "email"
    GOOGLE = "google"
    MICROSOFT = "microsoft"

class SubscriptionPlan(Enum):
    FREE = "free"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class DatabaseManager:
    def __init__(self, database_url: str, encryption_key: Optional[str] = None):
        self.database_url = database_url
        self.pool = None
        # Generate or use provided encryption key for sensitive data
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'application_name': 'dharmamind_auth',
                    'jit': 'off'
                }
            )
            
            await self.create_tables()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def create_tables(self):
        """Create all necessary tables with proper indexes and constraints"""
        async with self.pool.acquire() as conn:
            # Enable required extensions
            await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";")
            
            # Users table with comprehensive security features
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    email VARCHAR(255) UNIQUE NOT NULL,
                    email_verified BOOLEAN DEFAULT FALSE,
                    email_verification_token VARCHAR(255),
                    email_verification_expires TIMESTAMP,
                    
                    -- Authentication
                    password_hash TEXT,
                    auth_provider VARCHAR(50) DEFAULT 'email',
                    google_id VARCHAR(255),
                    microsoft_id VARCHAR(255),
                    
                    -- Profile information (encrypted)
                    first_name_encrypted TEXT,
                    last_name_encrypted TEXT,
                    phone_encrypted TEXT,
                    
                    -- Account status and security
                    status VARCHAR(20) DEFAULT 'pending',
                    subscription_plan VARCHAR(50) DEFAULT 'free',
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,
                    last_login TIMESTAMP,
                    password_reset_token VARCHAR(255),
                    password_reset_expires TIMESTAMP,
                    
                    -- Two-factor authentication
                    two_factor_enabled BOOLEAN DEFAULT FALSE,
                    two_factor_secret_encrypted TEXT,
                    backup_codes_encrypted TEXT,
                    
                    -- Audit trail
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by_ip INET,
                    last_activity TIMESTAMP,
                    
                    -- Privacy and compliance
                    terms_accepted_at TIMESTAMP,
                    privacy_accepted_at TIMESTAMP,
                    marketing_consent BOOLEAN DEFAULT FALSE,
                    data_retention_expires TIMESTAMP,
                    
                    CONSTRAINT valid_status CHECK (status IN ('pending', 'active', 'suspended', 'deleted')),
                    CONSTRAINT valid_subscription CHECK (subscription_plan IN ('free', 'professional', 'enterprise')),
                    CONSTRAINT valid_auth_provider CHECK (auth_provider IN ('email', 'google', 'microsoft'))
                );
            """)
            
            # User sessions for secure session management
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    session_token VARCHAR(255) UNIQUE NOT NULL,
                    refresh_token VARCHAR(255) UNIQUE,
                    device_fingerprint TEXT,
                    user_agent TEXT,
                    ip_address INET,
                    location_data JSONB,
                    
                    expires_at TIMESTAMP NOT NULL,
                    refresh_expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    is_active BOOLEAN DEFAULT TRUE,
                    revoked_at TIMESTAMP,
                    revoked_reason VARCHAR(100)
                );
            """)
            
            # Payment information (PCI DSS compliant - tokenized)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS payment_methods (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    
                    -- Tokenized payment data (never store actual card numbers)
                    payment_provider VARCHAR(50) NOT NULL, -- stripe, paypal, etc.
                    provider_customer_id VARCHAR(255),
                    provider_payment_method_id VARCHAR(255),
                    
                    -- Safe to store metadata
                    card_last_four VARCHAR(4),
                    card_brand VARCHAR(20),
                    card_exp_month INTEGER,
                    card_exp_year INTEGER,
                    billing_country VARCHAR(2),
                    
                    is_default BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Subscription management
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    payment_method_id UUID REFERENCES payment_methods(id),
                    
                    plan VARCHAR(50) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    
                    -- Billing cycle
                    current_period_start TIMESTAMP NOT NULL,
                    current_period_end TIMESTAMP NOT NULL,
                    trial_start TIMESTAMP,
                    trial_end TIMESTAMP,
                    
                    -- Payment provider data
                    provider_subscription_id VARCHAR(255),
                    provider_customer_id VARCHAR(255),
                    
                    -- Pricing
                    amount INTEGER, -- in cents
                    currency VARCHAR(3) DEFAULT 'USD',
                    interval_type VARCHAR(20), -- monthly, yearly
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    canceled_at TIMESTAMP,
                    cancellation_reason TEXT
                );
            """)
            
            # Security audit log
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_audit_log (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                    
                    event_type VARCHAR(100) NOT NULL,
                    event_details JSONB,
                    
                    ip_address INET,
                    user_agent TEXT,
                    session_id UUID,
                    
                    severity VARCHAR(20) DEFAULT 'info',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Compliance fields
                    compliance_category VARCHAR(50),
                    retention_expires TIMESTAMP
                );
            """)
            
            # Email verification and communication log
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS email_communications (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    
                    email_type VARCHAR(50) NOT NULL, -- verification, password_reset, marketing, etc.
                    email_address VARCHAR(255) NOT NULL,
                    
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    delivered_at TIMESTAMP,
                    opened_at TIMESTAMP,
                    clicked_at TIMESTAMP,
                    
                    status VARCHAR(20) DEFAULT 'sent',
                    provider_message_id VARCHAR(255),
                    
                    template_id VARCHAR(100),
                    template_data JSONB
                );
            """)
            
            # API rate limiting and usage tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    
                    endpoint VARCHAR(255) NOT NULL,
                    method VARCHAR(10) NOT NULL,
                    
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_status INTEGER,
                    response_time_ms INTEGER,
                    
                    ip_address INET,
                    user_agent TEXT,
                    
                    -- Rate limiting
                    rate_limit_key VARCHAR(255),
                    rate_limit_remaining INTEGER
                );
            """)
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
                "CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);",
                "CREATE INDEX IF NOT EXISTS idx_users_subscription ON users(subscription_plan);",
                "CREATE INDEX IF NOT EXISTS idx_users_auth_provider ON users(auth_provider);",
                "CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);",
                "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);",
                "CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at);",
                "CREATE INDEX IF NOT EXISTS idx_payment_methods_user ON payment_methods(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_audit_user_id ON security_audit_log(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON security_audit_log(timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON security_audit_log(event_type);",
                "CREATE INDEX IF NOT EXISTS idx_email_comm_user ON email_communications(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_api_usage_user ON api_usage(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);",
            ]
            
            for index_sql in indexes:
                await conn.execute(index_sql)
            
            logger.info("Database tables and indexes created successfully")

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not data:
            return None
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not encrypted_data:
            return None
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user with encrypted sensitive data"""
        async with self.pool.acquire() as conn:
            # Hash password
            password_hash = None
            if user_data.get('password'):
                password_hash = bcrypt.hashpw(
                    user_data['password'].encode('utf-8'), 
                    bcrypt.gensalt()
                ).decode('utf-8')
            
            # Encrypt sensitive fields
            first_name_encrypted = self.encrypt_data(user_data.get('first_name', ''))
            last_name_encrypted = self.encrypt_data(user_data.get('last_name', ''))
            phone_encrypted = self.encrypt_data(user_data.get('phone', ''))
            
            # Generate verification token
            verification_token = secrets.token_urlsafe(32)
            verification_expires = datetime.utcnow() + timedelta(hours=24)
            
            user_id = await conn.fetchval("""
                INSERT INTO users (
                    email, password_hash, auth_provider,
                    first_name_encrypted, last_name_encrypted, phone_encrypted,
                    email_verification_token, email_verification_expires,
                    google_id, microsoft_id, created_by_ip,
                    terms_accepted_at, privacy_accepted_at, marketing_consent
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                ) RETURNING id;
            """, 
                user_data['email'].lower(),
                password_hash,
                user_data.get('auth_provider', 'email'),
                first_name_encrypted,
                last_name_encrypted,
                phone_encrypted,
                verification_token,
                verification_expires,
                user_data.get('google_id'),
                user_data.get('microsoft_id'),
                user_data.get('ip_address'),
                datetime.utcnow() if user_data.get('accept_terms') else None,
                datetime.utcnow() if user_data.get('accept_privacy') else None,
                user_data.get('marketing_consent', False)
            )
            
            # Log user creation
            await self.log_security_event(
                user_id=user_id,
                event_type="user_created",
                event_details={
                    "auth_provider": user_data.get('auth_provider', 'email'),
                    "email": user_data['email'],
                    "ip_address": user_data.get('ip_address')
                },
                ip_address=user_data.get('ip_address')
            )
            
            return {
                "user_id": str(user_id),
                "email_verification_token": verification_token,
                "verification_expires": verification_expires.isoformat()
            }

    async def verify_email(self, token: str) -> bool:
        """Verify user email with token"""
        async with self.pool.acquire() as conn:
            user_id = await conn.fetchval("""
                UPDATE users 
                SET email_verified = TRUE, 
                    email_verification_token = NULL,
                    email_verification_expires = NULL,
                    status = 'active',
                    updated_at = CURRENT_TIMESTAMP
                WHERE email_verification_token = $1 
                    AND email_verification_expires > CURRENT_TIMESTAMP
                    AND status = 'pending'
                RETURNING id;
            """, token)
            
            if user_id:
                await self.log_security_event(
                    user_id=user_id,
                    event_type="email_verified",
                    event_details={"verification_token": token}
                )
                return True
            return False

    async def authenticate_user(self, email: str, password: str, ip_address: str = None) -> Optional[Dict[str, Any]]:
        """Authenticate user with rate limiting and account lockout"""
        async with self.pool.acquire() as conn:
            # Check if account is locked
            user_record = await conn.fetchrow("""
                SELECT id, email, password_hash, status, failed_login_attempts, locked_until,
                       email_verified, subscription_plan, last_login
                FROM users 
                WHERE email = $1;
            """, email.lower())
            
            if not user_record:
                await self.log_security_event(
                    event_type="login_failed",
                    event_details={"email": email, "reason": "user_not_found", "ip_address": ip_address},
                    ip_address=ip_address
                )
                return None
            
            user_id = user_record['id']
            
            # Check if account is locked
            if user_record['locked_until'] and user_record['locked_until'] > datetime.utcnow():
                await self.log_security_event(
                    user_id=user_id,
                    event_type="login_blocked",
                    event_details={"reason": "account_locked", "ip_address": ip_address},
                    ip_address=ip_address,
                    severity="warning"
                )
                return None
            
            # Check if account is active
            if user_record['status'] != 'active':
                await self.log_security_event(
                    user_id=user_id,
                    event_type="login_blocked",
                    event_details={"reason": "account_inactive", "status": user_record['status'], "ip_address": ip_address},
                    ip_address=ip_address,
                    severity="warning"
                )
                return None
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), user_record['password_hash'].encode('utf-8')):
                # Increment failed attempts
                failed_attempts = user_record['failed_login_attempts'] + 1
                locked_until = None
                
                # Lock account after 5 failed attempts
                if failed_attempts >= 5:
                    locked_until = datetime.utcnow() + timedelta(minutes=30)
                
                await conn.execute("""
                    UPDATE users 
                    SET failed_login_attempts = $1, locked_until = $2
                    WHERE id = $3;
                """, failed_attempts, locked_until, user_id)
                
                await self.log_security_event(
                    user_id=user_id,
                    event_type="login_failed",
                    event_details={
                        "reason": "invalid_password",
                        "failed_attempts": failed_attempts,
                        "account_locked": locked_until is not None,
                        "ip_address": ip_address
                    },
                    ip_address=ip_address,
                    severity="warning" if failed_attempts >= 3 else "info"
                )
                return None
            
            # Successful login - reset failed attempts and update last login
            await conn.execute("""
                UPDATE users 
                SET failed_login_attempts = 0, 
                    locked_until = NULL,
                    last_login = CURRENT_TIMESTAMP,
                    last_activity = CURRENT_TIMESTAMP
                WHERE id = $1;
            """, user_id)
            
            await self.log_security_event(
                user_id=user_id,
                event_type="login_successful",
                event_details={"ip_address": ip_address},
                ip_address=ip_address
            )
            
            return {
                "user_id": str(user_id),
                "email": user_record['email'],
                "status": user_record['status'],
                "subscription_plan": user_record['subscription_plan'],
                "email_verified": user_record['email_verified'],
                "last_login": user_record['last_login']
            }

    async def create_session(self, user_id: uuid.UUID, session_data: Dict[str, Any]) -> Dict[str, str]:
        """Create a secure session with tokens"""
        async with self.pool.acquire() as conn:
            session_token = secrets.token_urlsafe(32)
            refresh_token = secrets.token_urlsafe(32)
            
            session_expires = datetime.utcnow() + timedelta(hours=24)
            refresh_expires = datetime.utcnow() + timedelta(days=30)
            
            session_id = await conn.fetchval("""
                INSERT INTO user_sessions (
                    user_id, session_token, refresh_token,
                    device_fingerprint, user_agent, ip_address,
                    expires_at, refresh_expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id;
            """, 
                user_id, session_token, refresh_token,
                session_data.get('device_fingerprint'),
                session_data.get('user_agent'),
                session_data.get('ip_address'),
                session_expires, refresh_expires
            )
            
            await self.log_security_event(
                user_id=user_id,
                event_type="session_created",
                event_details={
                    "session_id": str(session_id),
                    "ip_address": session_data.get('ip_address'),
                    "user_agent": session_data.get('user_agent')
                },
                session_id=session_id,
                ip_address=session_data.get('ip_address')
            )
            
            return {
                "session_token": session_token,
                "refresh_token": refresh_token,
                "expires_at": session_expires.isoformat(),
                "refresh_expires_at": refresh_expires.isoformat()
            }

    async def log_security_event(self, event_type: str, event_details: Dict[str, Any], 
                                user_id: uuid.UUID = None, ip_address: str = None,
                                session_id: uuid.UUID = None, severity: str = "info"):
        """Log security events for audit trail"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO security_audit_log (
                    user_id, event_type, event_details, ip_address, 
                    session_id, severity, compliance_category,
                    retention_expires
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8);
            """, 
                user_id, event_type, json.dumps(event_details), ip_address,
                session_id, severity, "authentication",
                datetime.utcnow() + timedelta(days=2555)  # 7 years retention
            )

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions and tokens"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE user_sessions 
                SET is_active = FALSE, revoked_at = CURRENT_TIMESTAMP, revoked_reason = 'expired'
                WHERE expires_at < CURRENT_TIMESTAMP AND is_active = TRUE;
            """)
            
            await conn.execute("""
                DELETE FROM user_sessions 
                WHERE revoked_at < CURRENT_TIMESTAMP - INTERVAL '30 days';
            """)

    async def get_user_by_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get user data by valid session token"""
        async with self.pool.acquire() as conn:
            user_data = await conn.fetchrow("""
                SELECT u.id, u.email, u.status, u.subscription_plan, u.email_verified,
                       u.first_name_encrypted, u.last_name_encrypted,
                       s.session_token, s.expires_at, s.ip_address as session_ip
                FROM users u
                JOIN user_sessions s ON u.id = s.user_id
                WHERE s.session_token = $1 
                    AND s.expires_at > CURRENT_TIMESTAMP 
                    AND s.is_active = TRUE
                    AND u.status = 'active';
            """, session_token)
            
            if not user_data:
                return None
            
            # Update last used timestamp
            await conn.execute("""
                UPDATE user_sessions 
                SET last_used = CURRENT_TIMESTAMP 
                WHERE session_token = $1;
            """, session_token)
            
            # Decrypt sensitive data
            first_name = self.decrypt_data(user_data['first_name_encrypted']) if user_data['first_name_encrypted'] else None
            last_name = self.decrypt_data(user_data['last_name_encrypted']) if user_data['last_name_encrypted'] else None
            
            return {
                "user_id": str(user_data['id']),
                "email": user_data['email'],
                "status": user_data['status'],
                "subscription_plan": user_data['subscription_plan'],
                "email_verified": user_data['email_verified'],
                "first_name": first_name,
                "last_name": last_name,
                "session_ip": str(user_data['session_ip']) if user_data['session_ip'] else None
            }

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
