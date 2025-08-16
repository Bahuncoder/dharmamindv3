"""
Database Setup Script for DharmaMind Enterprise Authentication
Creates PostgreSQL database and tables for production deployment
"""

import asyncio
import asyncpg
import os
import sys
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "dharmamind"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password")
}

# SQL scripts for table creation
CREATE_TABLES_SQL = """
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table with enhanced security
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name_encrypted TEXT,
    last_name_encrypted TEXT,
    subscription_plan VARCHAR(20) DEFAULT 'free' CHECK (subscription_plan IN ('free', 'premium', 'enterprise')),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'suspended', 'deleted')),
    email_verified BOOLEAN DEFAULT FALSE,
    auth_provider VARCHAR(20) DEFAULT 'email' CHECK (auth_provider IN ('email', 'google', 'demo')),
    google_id VARCHAR(255) UNIQUE,
    login_count INTEGER DEFAULT 0,
    last_login TIMESTAMP WITH TIME ZONE,
    last_activity TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    
    -- Email verification
    email_verification_token VARCHAR(255),
    email_verification_expires TIMESTAMP WITH TIME ZONE,
    
    -- Password reset
    password_reset_token VARCHAR(255),
    password_reset_expires TIMESTAMP WITH TIME ZONE,
    
    -- Terms and privacy
    terms_accepted_at TIMESTAMP WITH TIME ZONE,
    privacy_accepted_at TIMESTAMP WITH TIME ZONE,
    marketing_consent BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    registration_ip INET,
    user_agent TEXT
);

-- User profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    phone_encrypted TEXT,
    timezone VARCHAR(50) DEFAULT 'UTC',
    profile_picture_url TEXT,
    bio_encrypted TEXT,
    website_url TEXT,
    
    -- Preferences
    preferences JSONB DEFAULT '{}',
    notification_settings JSONB DEFAULT '{"email": true, "push": true, "sms": false}',
    privacy_settings JSONB DEFAULT '{"profile_public": false, "show_email": false}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table for authentication tokens
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    remember_me BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Security events table for audit logging
CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,
    event_details JSONB,
    ip_address INET,
    user_agent TEXT,
    severity VARCHAR(20) DEFAULT 'info' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Rate limiting table
CREATE TABLE IF NOT EXISTS rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL, -- IP address or user ID
    action VARCHAR(100) NOT NULL, -- login, register, password_reset, etc.
    attempts INTEGER DEFAULT 1,
    first_attempt TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_attempt TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    blocked_until TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(identifier, action)
);

-- OAuth states table for secure OAuth flows
CREATE TABLE IF NOT EXISTS oauth_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    state_token VARCHAR(255) UNIQUE NOT NULL,
    state_data JSONB,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Payment information table (PCI DSS compliant tokenization)
CREATE TABLE IF NOT EXISTS payment_methods (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    payment_token VARCHAR(255) NOT NULL, -- Tokenized payment method
    payment_provider VARCHAR(50) NOT NULL, -- stripe, paypal, etc.
    payment_type VARCHAR(20) NOT NULL, -- card, bank_account, etc.
    last_four VARCHAR(4),
    brand VARCHAR(20), -- visa, mastercard, etc.
    expires_month INTEGER,
    expires_year INTEGER,
    is_default BOOLEAN DEFAULT FALSE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Subscription history table
CREATE TABLE IF NOT EXISTS subscription_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan VARCHAR(20) NOT NULL,
    price_amount DECIMAL(10,2),
    price_currency VARCHAR(3) DEFAULT 'USD',
    billing_period VARCHAR(20), -- monthly, yearly
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'cancelled', 'expired', 'suspended')),
    payment_method_id UUID REFERENCES payment_methods(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Email communications table
CREATE TABLE IF NOT EXISTS email_communications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    email_address VARCHAR(255) NOT NULL,
    email_type VARCHAR(50) NOT NULL, -- verification, welcome, password_reset, etc.
    subject VARCHAR(255),
    template_name VARCHAR(100),
    template_data JSONB,
    sent_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE,
    opened_at TIMESTAMP WITH TIME ZONE,
    clicked_at TIMESTAMP WITH TIME ZONE,
    bounced_at TIMESTAMP WITH TIME ZONE,
    complained_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'delivered', 'bounced', 'complained')),
    provider_message_id VARCHAR(255),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security_events(user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_created_at ON security_events(created_at);

CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier);
CREATE INDEX IF NOT EXISTS idx_rate_limits_action ON rate_limits(action);

CREATE INDEX IF NOT EXISTS idx_oauth_states_token ON oauth_states(state_token);
CREATE INDEX IF NOT EXISTS idx_oauth_states_expires_at ON oauth_states(expires_at);

CREATE INDEX IF NOT EXISTS idx_payment_methods_user_id ON payment_methods(user_id);
CREATE INDEX IF NOT EXISTS idx_subscription_history_user_id ON subscription_history(user_id);
CREATE INDEX IF NOT EXISTS idx_email_communications_user_id ON email_communications(user_id);
CREATE INDEX IF NOT EXISTS idx_email_communications_status ON email_communications(status);

-- Triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_payment_methods_updated_at BEFORE UPDATE ON payment_methods
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
"""

# Function to setup database
async def setup_database():
    """Setup PostgreSQL database and tables"""
    try:
        logger.info("Connecting to PostgreSQL...")
        
        # Try to connect to the specific database first
        try:
            conn = await asyncpg.connect(**DB_CONFIG)
            logger.info(f"Connected to database: {DB_CONFIG['database']}")
        except asyncpg.InvalidCatalogNameError:
            # Database doesn't exist, create it
            logger.info(f"Database {DB_CONFIG['database']} doesn't exist, creating...")
            
            # Connect to default postgres database to create our database
            temp_config = DB_CONFIG.copy()
            temp_config["database"] = "postgres"
            
            temp_conn = await asyncpg.connect(**temp_config)
            await temp_conn.execute(f'CREATE DATABASE "{DB_CONFIG["database"]}"')
            await temp_conn.close()
            
            # Now connect to our new database
            conn = await asyncpg.connect(**DB_CONFIG)
            logger.info(f"Created and connected to database: {DB_CONFIG['database']}")
        
        # Execute table creation script
        logger.info("Creating tables and indexes...")
        await conn.execute(CREATE_TABLES_SQL)
        logger.info("Database setup completed successfully!")
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        logger.info("Created tables:")
        for table in tables:
            logger.info(f"  - {table['table_name']}")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False

# Function to create sample data
async def create_sample_data():
    """Create sample data for testing"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("Creating sample data...")
        
        # Insert sample users
        sample_users = [
            {
                "email": "admin@dharmamind.ai",
                "password_hash": "hashed_password_admin",
                "first_name": "Admin",
                "last_name": "User",
                "subscription_plan": "enterprise",
                "status": "active",
                "email_verified": True,
                "auth_provider": "email"
            },
            {
                "email": "demo@dharmamind.ai",
                "password_hash": "hashed_password_demo",
                "first_name": "Demo",
                "last_name": "User",
                "subscription_plan": "premium",
                "status": "active",
                "email_verified": True,
                "auth_provider": "email"
            }
        ]
        
        for user in sample_users:
            await conn.execute("""
                INSERT INTO users (email, password_hash, first_name_encrypted, last_name_encrypted, 
                                 subscription_plan, status, email_verified, auth_provider, 
                                 terms_accepted_at, privacy_accepted_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (email) DO NOTHING
            """, user["email"], user["password_hash"], user["first_name"], 
                user["last_name"], user["subscription_plan"], user["status"], 
                user["email_verified"], user["auth_provider"])
        
        logger.info("Sample data created successfully!")
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Sample data creation failed: {e}")
        return False

# Function to test database connection
async def test_connection():
    """Test database connection and basic operations"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Test basic query
        result = await conn.fetchval("SELECT COUNT(*) FROM users")
        logger.info(f"Database connection test successful. Users count: {result}")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

# Main function
async def main():
    """Main setup function"""
    logger.info("Starting DharmaMind Database Setup...")
    logger.info(f"Database config: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    
    # Setup database
    if not await setup_database():
        logger.error("Database setup failed!")
        sys.exit(1)
    
    # Test connection
    if not await test_connection():
        logger.error("Database connection test failed!")
        sys.exit(1)
    
    # Ask if user wants to create sample data
    create_samples = input("Create sample data? (y/N): ").lower().strip() == 'y'
    if create_samples:
        await create_sample_data()
    
    logger.info("Database setup completed successfully!")
    logger.info("You can now start the enhanced authentication backend.")

if __name__ == "__main__":
    asyncio.run(main())
