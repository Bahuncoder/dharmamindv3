"""
🧪 DharmaMind Test Configuration - Simplified Fixtures
======================================================
"""

import pytest
import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import app
from app.main import app

# ================================
# 🔧 Test Environment Setup
# ================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ================================
# 🌐 FastAPI Test Client Fixtures
# ================================

@pytest.fixture
def test_app():
    """FastAPI test application."""
    yield app


@pytest.fixture
def client(test_app) -> Generator[TestClient, None, None]:
    """Synchronous test client."""
    with TestClient(test_app) as c:
        yield c


@pytest.fixture
async def async_client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Async test client for async endpoint testing."""
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        yield ac


# ================================
# 🔐 Authentication Test Fixtures
# ================================

@pytest.fixture
def test_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@dharmamind.ai",
        "username": "testuser",
        "password": "TestPassword123!",
        "full_name": "Test User"
    }


@pytest.fixture
def auth_headers():
    """Mock authentication headers."""
    return {
        "Authorization": "Bearer test_jwt_token_for_testing"
    }


# ================================
# 🕉️ Spiritual Content Test Fixtures
# ================================

@pytest.fixture
def sample_dharmic_query():
    """Sample dharmic query for testing."""
    return {
        "message": "What is the meaning of dharma in daily life?",
        "user_id": "test-user-123",
        "context": {
            "tradition": "vedantic",
            "language": "en"
        }
    }


@pytest.fixture
def sample_meditation_data():
    """Sample meditation practice data."""
    return {
        "practice_type": "dhyana",
        "duration_minutes": 20,
        "tradition": "yoga",
        "guidance_level": "intermediate"
    }


# ================================
# 📊 Mock Response Fixtures
# ================================

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "response": "Dharma represents one's duty and righteous path in life.",
        "confidence": 0.95,
        "sources": ["Bhagavad Gita", "Dharmashastra"],
        "model_used": "dharmallm-7b"
    }


@pytest.fixture
def mock_chat_response():
    """Mock chat response."""
    return {
        "message": "Thank you for your question about dharma.",
        "conversation_id": "conv-123",
        "message_id": "msg-456",
        "timestamp": datetime.now().isoformat()
    }
