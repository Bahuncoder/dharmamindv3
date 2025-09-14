"""
🧪 DharmaMind Test Configuration - Fixtures & Setup
==================================================

Central configuration for all tests including:
- Database test fixtures
- Authentication fixtures  
- Redis test setup
- Spiritual AI test data
- Mock configurations
"""

import pytest
import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import event
from alembic.config import Config
from alembic import command
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import your app modules
from app.main import app, get_db_manager
from app.config import settings, Settings
from app.db.database import DatabaseManager
from app.models.user import User, Base
from app.auth.auth_service import AuthenticationService
from app.chakra_modules.system_orchestrator import SystemOrchestrator

# ================================
# 🔧 Test Environment Setup
# ================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")  
def test_settings() -> Settings:
    """Test environment settings with safe defaults."""
    return Settings(
        ENVIRONMENT="testing",
        DEBUG=True,
        DATABASE_URL="postgresql+asyncpg://test_user:test_pass@localhost:5432/dharmamind_test",
        REDIS_URL="redis://localhost:6379/15",  # Use DB 15 for testing
        SECRET_KEY="test_secret_key_for_testing_only",
        JWT_SECRET_KEY="test_jwt_secret_key",
        LLM_GATEWAY_API_KEY="test_llm_key",
        ENABLE_RATE_LIMITING=False,  # Disable in tests
        ENABLE_SECURITY_MIDDLEWARE=False,  # Disable in tests
        LOG_LEVEL="INFO"
    )


# ================================
# 🗄️ Database Test Fixtures
# ================================

@pytest.fixture(scope="function")
async def test_db_engine(test_settings):
    """Create test database engine."""
    engine = create_async_engine(
        test_settings.DATABASE_URL,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True,
        pool_recycle=300
    )
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session with transaction rollback."""
    connection = await test_db_engine.connect()
    transaction = await connection.begin()
    
    # Create session
    async_session = async_sessionmaker(
        bind=connection, 
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    session = async_session()
    
    try:
        yield session
    finally:
        await session.close()
        await transaction.rollback()
        await connection.close()


# ================================  
# 🔴 Redis Test Fixtures
# ================================

@pytest.fixture
async def redis_client(test_settings) -> AsyncGenerator[redis.Redis, None]:
    """Test Redis client with cleanup."""
    client = redis.from_url(test_settings.REDIS_URL)
    
    # Clear test database
    await client.flushdb()
    
    yield client
    
    # Cleanup
    await client.flushdb()
    await client.close()


# ================================
# 🌐 FastAPI Test Client Fixtures
# ================================

@pytest.fixture
def override_get_settings(test_settings):
    """Override app settings for testing."""
    def get_test_settings():
        return test_settings
    return get_test_settings


@pytest.fixture  
def override_get_db_session(db_session):
    """Override database manager for testing."""
    async def get_test_db_manager():
        # Create a mock database manager that uses our test session
        mock_db_manager = MagicMock(spec=DatabaseManager)
        mock_db_manager.get_connection = AsyncMock(return_value=db_session)
        return mock_db_manager
    return get_test_db_manager


@pytest.fixture
def test_app(override_get_settings, override_get_db_session):
    """FastAPI test application with overrides."""
    # Override dependencies
    app.dependency_overrides[get_db_manager] = override_get_db_session
    
    yield app
    
    # Clear overrides
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_app) -> Generator[TestClient, None, None]:
    """Synchronous test client."""
    with TestClient(test_app) as c:
        yield c


@pytest.fixture
async def async_client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Asynchronous test client."""
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        yield ac


# ================================
# 👤 Authentication Test Fixtures
# ================================

@pytest.fixture
async def test_user(db_session) -> User:
    """Create a test user."""
    user = User(
        email="test@dharmamind.ai",
        username="testuser",
        hashed_password="$2b$12$test_hashed_password",
        full_name="Test User",
        is_active=True,
        is_verified=True,
        spiritual_level="intermediate",
        dharmic_path="mindfulness",
        meditation_experience=365
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user


@pytest.fixture
async def test_admin_user(db_session) -> User:
    """Create a test admin user."""
    admin = User(
        email="admin@dharmamind.ai",
        username="admin",
        hashed_password="$2b$12$admin_hashed_password",
        full_name="Admin User", 
        is_active=True,
        is_verified=True,
        is_superuser=True,
        spiritual_level="advanced",
        dharmic_path="all_paths"
    )
    
    db_session.add(admin)
    await db_session.commit()
    await db_session.refresh(admin)
    
    return admin


@pytest.fixture
def auth_headers(test_user) -> dict:
    """Generate authentication headers for test user."""
    # This would use your actual auth service to generate tokens
    fake_token = "fake_jwt_token_for_testing"
    return {
        "Authorization": f"Bearer {fake_token}",
        "Content-Type": "application/json"
    }


@pytest.fixture
def admin_auth_headers(test_admin_user) -> dict:
    """Generate authentication headers for admin user."""
    fake_admin_token = "fake_admin_jwt_token_for_testing"  
    return {
        "Authorization": f"Bearer {fake_admin_token}",
        "Content-Type": "application/json",
        "X-Admin-Access": "true"
    }


# ================================
# 🧘 Spiritual AI Test Fixtures  
# ================================

@pytest.fixture
def mock_system_orchestrator():
    """Mock System Orchestrator for spiritual AI tests."""
    mock_orchestrator = AsyncMock(spec=SystemOrchestrator)
    
    # Mock successful spiritual guidance response
    mock_orchestrator.process_spiritual_guidance.return_value = {
        "guidance": "Focus on your breath and let go of attachments",
        "dharmic_validation": True,
        "confidence": 0.95,
        "spiritual_context": "mindfulness_meditation",
        "suggested_practices": ["breathing_meditation", "loving_kindness"]
    }
    
    # Mock consciousness analysis
    mock_orchestrator.analyze_consciousness.return_value = {
        "consciousness_level": "aware", 
        "emotional_state": "calm",
        "mental_clarity": 0.8,
        "spiritual_readiness": True
    }
    
    return mock_orchestrator


@pytest.fixture
def spiritual_test_data():
    """Sample spiritual guidance test data."""
    return {
        "simple_query": {
            "message": "I am feeling anxious about work",
            "context": {"mood": "anxious", "situation": "work_stress"}
        },
        "deep_query": {
            "message": "What is the nature of consciousness?",
            "context": {"topic": "consciousness", "depth": "philosophical"}
        },
        "practice_query": {
            "message": "How should I meditate?",
            "context": {"experience": "beginner", "time_available": "10_minutes"}
        },
        "dharmic_query": {
            "message": "What does the Bhagavad Gita say about duty?",
            "context": {"scripture": "bhagavad_gita", "topic": "dharma"}
        }
    }


@pytest.fixture
def expected_spiritual_responses():
    """Expected responses for spiritual queries."""
    return {
        "anxiety_response": {
            "contains": ["breath", "present moment", "impermanence"],
            "dharmic_validation": True,
            "confidence_min": 0.7
        },
        "consciousness_response": {
            "contains": ["awareness", "observer", "witness"],
            "dharmic_validation": True,
            "confidence_min": 0.8
        },
        "meditation_response": {
            "contains": ["posture", "breath", "attention"],
            "dharmic_validation": True,
            "suggested_practices": True
        }
    }


# ================================
# 🔒 Security Test Fixtures
# ================================

@pytest.fixture
def malicious_payloads():
    """Common security attack payloads for testing."""
    return {
        "xss_payloads": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ],
        "sql_injection": [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "' UNION SELECT * FROM users --"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ],
        "command_injection": [
            "; cat /etc/passwd",
            "| whoami",
            "$(cat /etc/passwd)"
        ]
    }


@pytest.fixture
def rate_limit_test_client(test_app):
    """Test client configured for rate limiting tests."""
    # Enable rate limiting for specific tests
    original_setting = settings.ENABLE_RATE_LIMITING
    settings.ENABLE_RATE_LIMITING = True
    
    with TestClient(test_app) as client:
        yield client
    
    # Restore setting
    settings.ENABLE_RATE_LIMITING = original_setting


# ================================
# 🎯 Utility Test Functions
# ================================

@pytest.fixture
def time_machine():
    """Freeze time for testing time-dependent functionality."""
    from freezegun import freeze_time
    
    def freeze_at(timestamp):
        return freeze_time(timestamp)
    
    return freeze_at


@pytest.fixture  
def capture_logs():
    """Capture logs during tests for verification."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    
    logger = logging.getLogger("dharmamind")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    def get_logs():
        return log_capture.getvalue()
    
    yield get_logs
    
    logger.removeHandler(handler)


# ================================
# 🧪 Test Data Factories
# ================================

class UserFactory:
    """Factory for creating test users."""
    
    @staticmethod
    def create_user_data(**kwargs):
        """Create user data with defaults."""
        default_data = {
            "email": "user@example.com",
            "username": "testuser",
            "password": "SecurePassword123!",
            "full_name": "Test User",
            "spiritual_level": "beginner",
            "dharmic_path": "mindfulness"
        }
        default_data.update(kwargs)
        return default_data
    
    @staticmethod  
    def create_multiple_users(count: int):
        """Create multiple test users."""
        users = []
        for i in range(count):
            users.append(UserFactory.create_user_data(
                email=f"user{i}@example.com",
                username=f"testuser{i}",
                full_name=f"Test User {i}"
            ))
        return users


@pytest.fixture
def user_factory():
    """User factory fixture."""
    return UserFactory


# ================================
# 🏁 Session Cleanup
# ================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    
    # Any cleanup needed after each test
    # This runs automatically after every test
    pass


# ================================
# 📊 Performance Test Fixtures  
# ================================

@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time
    
    return Timer


# Performance thresholds for tests
PERFORMANCE_THRESHOLDS = {
    "api_response_time": 1.0,  # 1 second max
    "db_query_time": 0.5,      # 500ms max
    "spiritual_processing": 2.0, # 2 seconds max for AI processing
    "authentication": 0.2,     # 200ms max for auth
}
