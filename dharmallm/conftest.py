"""
Pytest Configuration for DharmaMind

Configures pytest with:
- Async test support
- Coverage tracking
- Fixtures for common test resources
- Database test isolation
- Mock services
- Performance profiling

Usage:
    pytest                          # Run all tests
    pytest tests/unit               # Run unit tests only
    pytest tests/integration        # Run integration tests
    pytest --cov=services           # With coverage
    pytest -v --tb=short            # Verbose with short tracebacks
    pytest -k "test_auth"           # Run specific tests
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, uses real components)"
    )
    config.addinivalue_line(
        "markers", "security: Security and penetration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "asyncio: Async tests"
    )


# ============================================================================
# Async Support
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Test Directories
# ============================================================================

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp(prefix="dharmamind_test_"))
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def test_data_dir(temp_dir: Path) -> Path:
    """Create test data directory"""
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(scope="function")
def test_backup_dir(temp_dir: Path) -> Path:
    """Create test backup directory"""
    backup_dir = temp_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="function")
async def test_db_path(test_data_dir: Path) -> AsyncGenerator[str, None]:
    """Create a temporary test database"""
    import aiosqlite
    
    db_path = test_data_dir / "test_database.db"
    
    # Create database with basic schema
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        await db.commit()
    
    yield str(db_path)
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


# ============================================================================
# Security Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def test_user_data() -> dict:
    """Sample user data for testing"""
    return {
        "username": "test_user",
        "email": "test@example.com",
        "password": "SecurePassword123!",
        "full_name": "Test User"
    }


@pytest.fixture(scope="function")
def test_malicious_inputs() -> dict:
    """Common malicious input patterns for security testing"""
    return {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ],
        "xss": [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(`XSS`)'>"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//....//etc/passwd"
        ],
        "command_injection": [
            "; ls -la",
            "| cat /etc/passwd",
            "`whoami`",
            "$(uname -a)"
        ]
    }


# ============================================================================
# Authentication Fixtures
# ============================================================================

@pytest.fixture(scope="function")
async def auth_service():
    """Create authentication service for testing"""
    from services.security.authentication import AuthenticationService
    
    service = AuthenticationService()
    await service.initialize()
    yield service


@pytest.fixture(scope="function")
async def jwt_manager():
    """Create JWT manager for testing"""
    from services.security.jwt_manager import JWTManager
    
    manager = JWTManager(
        secret_key="test_secret_key_for_testing_only",
        algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7
    )
    await manager.initialize()
    yield manager
    await manager.cleanup()


# ============================================================================
# Backup Fixtures
# ============================================================================

@pytest.fixture(scope="function")
async def backup_manager(test_backup_dir: Path):
    """Create backup manager for testing"""
    from services.backup.backup_manager import BackupManager
    
    manager = BackupManager(
        backup_dir=str(test_backup_dir),
        retention_days=7,
        max_backups=10
    )
    await manager.initialize()
    yield manager
    
    # Cleanup
    if backup_manager.scheduler_task:
        await manager.stop_scheduled_backups()


# ============================================================================
# Health Check Fixtures
# ============================================================================

@pytest.fixture(scope="function")
async def health_checker():
    """Create health checker for testing"""
    try:
        from services.monitoring.health_checks import HealthChecker
        
        checker = HealthChecker()
        await checker.initialize()
        yield checker
    except ImportError:
        pytest.skip("Health checker not available")


# ============================================================================
# FastAPI Test Client
# ============================================================================

@pytest.fixture(scope="function")
async def test_client():
    """Create FastAPI test client"""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    
    app = FastAPI()
    
    # Import and include routers
    try:
        from services.backup.backup_routes import router as backup_router
        app.include_router(backup_router, prefix="/api/v1")
    except ImportError:
        pass
    
    try:
        from services.monitoring.health_checks import router as health_router
        app.include_router(health_router)
    except ImportError:
        pass
    
    client = TestClient(app)
    yield client


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_llm_response():
    """Mock LLM response for testing"""
    return {
        "response": "This is a test response",
        "confidence": 0.95,
        "dharmic_score": 0.87,
        "tokens_used": 150,
        "processing_time": 0.5
    }


@pytest.fixture(scope="function")
def mock_training_data():
    """Mock training data for testing"""
    return [
        {
            "text": "Sample dharmic text from sacred scriptures",
            "source": "test_source",
            "category": "philosophy"
        },
        {
            "text": "Another sample text for testing purposes",
            "source": "test_source",
            "category": "ethics"
        }
    ]


# ============================================================================
# Performance Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def performance_timer():
    """Timer for performance testing"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Cleanup test artifacts after session"""
    yield
    
    # Clean up any test files
    test_patterns = [
        "test_*.db",
        "test_backup_*",
        "*_test.log"
    ]
    
    for pattern in test_patterns:
        for file in Path(".").glob(pattern):
            try:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
            except Exception:
                pass


# ============================================================================
# Parametrize Helpers
# ============================================================================

# Common test cases for parametrization
VALID_EMAILS = [
    "user@example.com",
    "test.user@example.co.uk",
    "user+tag@example.com"
]

INVALID_EMAILS = [
    "invalid",
    "@example.com",
    "user@",
    "user @example.com"
]

VALID_PASSWORDS = [
    "SecurePass123!",
    "Another$ecure456",
    "Complex&Pass789"
]

INVALID_PASSWORDS = [
    "short",
    "nouppercase123!",
    "NOLOWERCASE123!",
    "NoSpecialChar123"
]


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_test_user(index: int = 0) -> dict:
    """Generate test user data"""
    return {
        "username": f"test_user_{index}",
        "email": f"test{index}@example.com",
        "password": f"SecurePass{index}!",
        "full_name": f"Test User {index}"
    }


def generate_test_users(count: int = 5) -> list:
    """Generate multiple test users"""
    return [generate_test_user(i) for i in range(count)]
