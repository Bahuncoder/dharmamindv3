"""
🌐 API Endpoint Tests
====================

Comprehensive tests for all DharmaMind API endpoints:
- Chat endpoints
- Authentication endpoints
- User profile endpoints
- Spiritual guidance API
- Admin endpoints
"""

import pytest
import json
from unittest.mock import AsyncMock, patch
from fastapi import status
from httpx import AsyncClient


@pytest.mark.api
class TestChatEndpoints:
    """Test chat-related API endpoints."""
    
    async def test_chat_message_success(self, async_client, auth_headers, spiritual_test_data):
        """Test successful chat message processing."""
        chat_data = {
            "message": spiritual_test_data["simple_query"]["message"],
            "context": spiritual_test_data["simple_query"]["context"]
        }
        
        response = await async_client.post(
            "/api/v1/chat/message",
            json=chat_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        result = response.json()
        assert "response" in result
        assert "confidence" in result
        assert "dharmic_validation" in result
        assert "session_id" in result
        
        # Validate response quality
        assert len(result["response"]) > 10
        assert result["confidence"] >= 0.5
        assert result["dharmic_validation"] is True
    
    async def test_chat_message_unauthorized(self, async_client):
        """Test chat endpoint without authentication."""
        chat_data = {
            "message": "How should I meditate?",
            "context": {}
        }
        
        response = await async_client.post(
            "/api/v1/chat/message",
            json=chat_data
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    async def test_chat_message_validation_error(self, async_client, auth_headers):
        """Test chat endpoint with invalid data."""
        # Empty message
        invalid_data = {
            "message": "",
            "context": {}
        }
        
        response = await async_client.post(
            "/api/v1/chat/message", 
            json=invalid_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_chat_history(self, async_client, auth_headers):
        """Test retrieving chat history.""" 
        response = await async_client.get(
            "/api/v1/chat/history",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        history = response.json()
        assert "messages" in history
        assert "total_count" in history
        assert "page" in history
        assert isinstance(history["messages"], list)
    
    async def test_chat_session_create(self, async_client, auth_headers):
        """Test creating a new chat session."""
        session_data = {
            "title": "Morning Meditation Guidance",
            "spiritual_focus": "mindfulness"
        }
        
        response = await async_client.post(
            "/api/v1/chat/session",
            json=session_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        
        session = response.json()
        assert "session_id" in session
        assert "title" in session
        assert "created_at" in session
        assert session["title"] == session_data["title"]
    
    async def test_spiritual_guidance_endpoint(self, async_client, auth_headers):
        """Test dedicated spiritual guidance endpoint."""
        guidance_request = {
            "query": "I'm struggling with anger. How can dharma help?",
            "context": {
                "emotional_state": "angry",
                "spiritual_level": "beginner",
                "situation": "interpersonal_conflict"
            }
        }
        
        response = await async_client.post(
            "/api/v1/spiritual/guidance",
            json=guidance_request,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        guidance = response.json()
        assert "guidance" in guidance
        assert "practices" in guidance
        assert "dharmic_principles" in guidance
        assert "confidence" in guidance
        
        # Should address anger specifically
        guidance_text = guidance["guidance"].lower()
        assert any(word in guidance_text for word in 
                  ["anger", "patience", "compassion", "forgive"])


@pytest.mark.api
class TestAuthenticationEndpoints:
    """Test authentication API endpoints."""
    
    async def test_user_registration_success(self, async_client, user_factory):
        """Test successful user registration."""
        user_data = user_factory.create_user_data()
        
        response = await async_client.post(
            "/auth/register",
            json=user_data
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        
        result = response.json()
        assert "user" in result
        assert "message" in result
        assert result["user"]["email"] == user_data["email"]
        assert result["user"]["username"] == user_data["username"]
        assert "password" not in result["user"]  # Password not returned
    
    async def test_user_registration_duplicate_email(self, async_client, test_user, user_factory):
        """Test registration with existing email."""
        user_data = user_factory.create_user_data(email=test_user.email)
        
        response = await async_client.post(
            "/auth/register",
            json=user_data
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        error = response.json()
        assert "detail" in error
        assert "email" in error["detail"].lower()
    
    async def test_user_login_success(self, async_client, test_user):
        """Test successful user login."""
        # Mock verified user with known password
        login_data = {
            "username": test_user.email,
            "password": "TestPassword123!"
        }
        
        with patch('app.auth.auth_service.AuthService.authenticate_user') as mock_auth:
            mock_auth.return_value = test_user
            
            response = await async_client.post(
                "/auth/login",
                data=login_data  # Form data for OAuth2
            )
        
        assert response.status_code == status.HTTP_200_OK
        
        tokens = response.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert "token_type" in tokens
        assert tokens["token_type"] == "bearer"
    
    async def test_user_login_invalid_credentials(self, async_client):
        """Test login with invalid credentials."""
        login_data = {
            "username": "nonexistent@example.com",
            "password": "wrongpassword"
        }
        
        response = await async_client.post(
            "/auth/login",
            data=login_data
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        error = response.json()
        assert "detail" in error
    
    async def test_refresh_token(self, async_client, test_user):
        """Test token refresh functionality."""
        # Mock valid refresh token
        refresh_data = {
            "refresh_token": "valid_refresh_token"
        }
        
        with patch('app.auth.auth_service.AuthService.verify_refresh_token') as mock_verify:
            mock_verify.return_value = {"sub": str(test_user.id)}
            
            with patch('app.auth.auth_service.AuthService.create_access_token') as mock_create:
                mock_create.return_value = "new_access_token"
                
                response = await async_client.post(
                    "/auth/refresh",
                    json=refresh_data
                )
        
        assert response.status_code == status.HTTP_200_OK
        
        tokens = response.json()
        assert "access_token" in tokens
        assert tokens["access_token"] == "new_access_token"
    
    async def test_user_profile(self, async_client, auth_headers, test_user):
        """Test retrieving user profile."""
        with patch('app.auth.auth_service.AuthService.get_current_user') as mock_user:
            mock_user.return_value = test_user
            
            response = await async_client.get(
                "/auth/profile",
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        
        profile = response.json()
        assert "email" in profile
        assert "username" in profile
        assert "spiritual_level" in profile
        assert "dharmic_path" in profile
        assert profile["email"] == test_user.email
    
    async def test_update_profile(self, async_client, auth_headers, test_user):
        """Test updating user profile."""
        update_data = {
            "spiritual_level": "advanced",
            "dharmic_path": "zen_buddhism",
            "bio": "Dedicated meditation practitioner"
        }
        
        with patch('app.auth.auth_service.AuthService.get_current_user') as mock_user:
            mock_user.return_value = test_user
            
            with patch('app.auth.auth_service.AuthService.update_user_profile') as mock_update:
                updated_user = test_user
                updated_user.spiritual_level = update_data["spiritual_level"]
                mock_update.return_value = updated_user
                
                response = await async_client.put(
                    "/auth/profile",
                    json=update_data,
                    headers=auth_headers
                )
        
        assert response.status_code == status.HTTP_200_OK
        
        result = response.json()
        assert result["spiritual_level"] == update_data["spiritual_level"]


@pytest.mark.api
class TestKnowledgeEndpoints:
    """Test spiritual knowledge API endpoints."""
    
    async def test_search_teachings(self, async_client, auth_headers):
        """Test searching spiritual teachings."""
        search_params = {
            "query": "meditation techniques",
            "category": "meditation", 
            "limit": 10
        }
        
        response = await async_client.get(
            "/api/v1/knowledge/search",
            params=search_params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        results = response.json()
        assert "teachings" in results
        assert "total_count" in results
        assert isinstance(results["teachings"], list)
        
        # Validate teaching structure
        if results["teachings"]:
            teaching = results["teachings"][0]
            assert "title" in teaching
            assert "content" in teaching
            assert "source" in teaching
            assert "relevance_score" in teaching
    
    async def test_get_practice_details(self, async_client, auth_headers):
        """Test getting practice instructions."""
        practice_id = "mindfulness_meditation"
        
        response = await async_client.get(
            f"/api/v1/knowledge/practice/{practice_id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        practice = response.json()
        assert "name" in practice
        assert "instructions" in practice
        assert "benefits" in practice
        assert "duration" in practice
        assert isinstance(practice["instructions"], list)
    
    async def test_get_dharmic_concepts(self, async_client, auth_headers):
        """Test retrieving dharmic concepts."""
        response = await async_client.get(
            "/api/v1/knowledge/concepts",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        concepts = response.json()
        assert "concepts" in concepts
        assert isinstance(concepts["concepts"], list)
        
        # Should include key dharmic concepts
        concept_names = [c["name"].lower() for c in concepts["concepts"]]
        expected_concepts = ["dharma", "karma", "mindfulness", "compassion"]
        assert any(concept in concept_names for concept in expected_concepts)


@pytest.mark.api
class TestFeedbackEndpoints:
    """Test feedback and rating endpoints."""
    
    async def test_submit_feedback(self, async_client, auth_headers):
        """Test submitting feedback for a response."""
        feedback_data = {
            "session_id": "test_session_123",
            "message_id": "msg_456", 
            "rating": 5,
            "feedback_text": "Very helpful spiritual guidance!",
            "categories": ["helpful", "dharmic", "compassionate"]
        }
        
        response = await async_client.post(
            "/api/v1/feedback/submit",
            json=feedback_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        
        result = response.json()
        assert "feedback_id" in result
        assert "message" in result
    
    async def test_get_feedback_stats(self, async_client, admin_auth_headers):
        """Test retrieving feedback statistics (admin only)."""
        response = await async_client.get(
            "/api/v1/feedback/stats",
            headers=admin_auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        stats = response.json()
        assert "average_rating" in stats
        assert "total_feedback" in stats
        assert "rating_distribution" in stats
        assert "category_breakdown" in stats


@pytest.mark.api
class TestAdminEndpoints:
    """Test admin-only API endpoints."""
    
    async def test_admin_dashboard(self, async_client, admin_auth_headers):
        """Test admin dashboard data."""
        response = await async_client.get(
            "/api/v1/admin/dashboard",
            headers=admin_auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        dashboard = response.json()
        assert "user_stats" in dashboard
        assert "system_health" in dashboard
        assert "recent_activity" in dashboard
        
        # Validate user stats
        user_stats = dashboard["user_stats"]
        assert "total_users" in user_stats
        assert "active_users" in user_stats
        assert "new_registrations" in user_stats
    
    async def test_admin_access_required(self, async_client, auth_headers):
        """Test that admin endpoints require admin access."""
        response = await async_client.get(
            "/api/v1/admin/dashboard",
            headers=auth_headers  # Regular user, not admin
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    async def test_system_health_check(self, async_client, admin_auth_headers):
        """Test system health monitoring endpoint."""
        response = await async_client.get(
            "/api/v1/admin/health",
            headers=admin_auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        health = response.json()
        assert "database" in health
        assert "redis" in health
        assert "spiritual_ai" in health
        assert "llm_gateway" in health
        
        # Each component should have status
        for component, status_info in health.items():
            assert "status" in status_info
            assert "response_time" in status_info


@pytest.mark.api
class TestErrorHandling:
    """Test API error handling and edge cases."""
    
    async def test_invalid_json(self, async_client, auth_headers):
        """Test handling of invalid JSON data."""
        # Send malformed JSON
        response = await async_client.post(
            "/api/v1/chat/message",
            data="invalid json{{{",
            headers={**auth_headers, "content-type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_missing_required_fields(self, async_client, auth_headers):
        """Test validation of required fields."""
        # Missing 'message' field
        incomplete_data = {
            "context": {}
        }
        
        response = await async_client.post(
            "/api/v1/chat/message",
            json=incomplete_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        error = response.json()
        assert "detail" in error
        # Should mention the missing field
        assert any("message" in str(err).lower() for err in error["detail"])
    
    async def test_oversized_request(self, async_client, auth_headers):
        """Test handling of oversized requests."""
        # Create very large message
        oversized_message = "x" * 50000  # 50KB message
        
        large_data = {
            "message": oversized_message,
            "context": {}
        }
        
        response = await async_client.post(
            "/api/v1/chat/message",
            json=large_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    
    async def test_rate_limiting(self, async_client, auth_headers):
        """Test API rate limiting."""
        # Make many requests quickly
        responses = []
        for i in range(20):  # Exceed rate limit
            response = await async_client.post(
                "/api/v1/chat/message",
                json={"message": f"Test message {i}", "context": {}},
                headers=auth_headers
            )
            responses.append(response)
        
        # Should get rate limited
        rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS 
                          for r in responses)
        assert rate_limited
    
    async def test_internal_server_error_handling(self, async_client, auth_headers):
        """Test handling of internal server errors."""
        # Mock an internal error
        with patch('app.services.llm_router.LLMRouter.process_message') as mock_process:
            mock_process.side_effect = Exception("Simulated internal error")
            
            response = await async_client.post(
                "/api/v1/chat/message",
                json={"message": "Test message", "context": {}},
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        error = response.json()
        assert "detail" in error
        assert "internal server error" in error["detail"].lower()
        # Should not expose internal details in production


@pytest.mark.integration
@pytest.mark.api
class TestAPIIntegrationFlow:
    """Integration tests for API workflows."""
    
    async def test_complete_user_journey(self, async_client, user_factory):
        """Test complete user journey from registration to spiritual guidance."""
        # 1. Register new user
        user_data = user_factory.create_user_data()
        register_response = await async_client.post("/auth/register", json=user_data)
        assert register_response.status_code == status.HTTP_201_CREATED
        
        # 2. Login
        login_response = await async_client.post("/auth/login", data={
            "username": user_data["email"],
            "password": user_data["password"]
        })
        assert login_response.status_code == status.HTTP_200_OK
        tokens = login_response.json()
        
        # 3. Get spiritual guidance
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        guidance_response = await async_client.post(
            "/api/v1/chat/message",
            json={
                "message": "How should I start meditating?",
                "context": {"experience": "beginner"}
            },
            headers=headers
        )
        assert guidance_response.status_code == status.HTTP_200_OK
        
        # 4. Submit feedback
        guidance_result = guidance_response.json()
        feedback_response = await async_client.post(
            "/api/v1/feedback/submit",
            json={
                "session_id": guidance_result["session_id"],
                "rating": 5,
                "feedback_text": "Very helpful!"
            },
            headers=headers
        )
        assert feedback_response.status_code == status.HTTP_201_CREATED
    
    async def test_concurrent_api_requests(self, async_client, auth_headers):
        """Test handling of concurrent API requests."""
        import asyncio
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            task = async_client.post(
                "/api/v1/chat/message",
                json={
                    "message": f"Spiritual question {i}",
                    "context": {"sequence": i}
                },
                headers=auth_headers
            )
            tasks.append(task)
        
        # Execute concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed
        successful = [r for r in responses if not isinstance(r, Exception) and r.status_code == 200]
        assert len(successful) >= 8  # At least 80% success rate
