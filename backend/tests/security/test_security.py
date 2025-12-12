"""
🔒 Security Tests
================

Comprehensive security testing for DharmaMind backend:
- Authentication & authorization testing
- Input validation & sanitization
- Rate limiting tests
- XSS & injection prevention
- Security headers validation
- Session security
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from fastapi import status
from httpx import AsyncClient


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""
    
    @pytest.mark.parametrize("malicious_input", [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>", 
        "<svg onload=alert('xss')>",
        "data:text/html,<script>alert('xss')</script>",
        "&#x3C;script&#x3E;alert('xss')&#x3C;/script&#x3E;",
    ])
    async def test_xss_prevention(self, async_client, auth_headers, malicious_input):
        """Test XSS attack prevention."""
        chat_data = {
            "message": f"Hello {malicious_input} world",
            "context": {"test": malicious_input}
        }
        
        response = await async_client.post(
            "/api/v1/chat/message",
            json=chat_data,
            headers=auth_headers
        )
        
        # Should either reject the input or sanitize it
        if response.status_code == status.HTTP_200_OK:
            result = response.json()
            # Response should not contain the malicious script
            response_text = str(result.get("response", "")).lower()
            assert "<script>" not in response_text
            assert "javascript:" not in response_text
            assert "onerror=" not in response_text
        else:
            # Should be a validation error
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ]
    
    @pytest.mark.parametrize("sql_injection", [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "' UNION SELECT * FROM users --",
        "admin'--",
        "admin' /*",
        "' OR 1=1#",
        "'; EXEC sp_configure 'show advanced options', 1--"
    ])
    async def test_sql_injection_prevention(self, async_client, auth_headers, sql_injection):
        """Test SQL injection prevention."""
        # Test in various input fields
        test_data = {
            "message": f"Tell me about {sql_injection}",
            "context": {"search": sql_injection}
        }
        
        response = await async_client.post(
            "/api/v1/chat/message",
            json=test_data,
            headers=auth_headers
        )
        
        # Should not cause database errors
        assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
        
        if response.status_code == status.HTTP_200_OK:
            result = response.json()
            # Response should not indicate SQL injection success
            response_text = str(result).lower()
            assert "syntax error" not in response_text
            assert "mysql" not in response_text
            assert "postgresql" not in response_text
    
    @pytest.mark.parametrize("command_injection", [
        "; cat /etc/passwd",
        "| whoami", 
        "$(cat /etc/passwd)",
        "`whoami`",
        "; ls -la /",
        "& ping google.com",
        "|| echo vulnerable"
    ])
    async def test_command_injection_prevention(self, async_client, auth_headers, command_injection):
        """Test command injection prevention."""
        test_data = {
            "message": f"Execute this: {command_injection}",
            "context": {}
        }
        
        response = await async_client.post(
            "/api/v1/chat/message",
            json=test_data,
            headers=auth_headers
        )
        
        # Should handle safely without executing commands
        assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
        
        if response.status_code == status.HTTP_200_OK:
            result = response.json()
            # Should not contain system information
            response_text = str(result).lower()
            assert "root:" not in response_text  # /etc/passwd content
            assert "uid=" not in response_text   # whoami output
    
    async def test_oversized_input_handling(self, async_client, auth_headers):
        """Test handling of oversized inputs."""
        # Create oversized message (>100KB)
        oversized_message = "A" * 100000
        
        test_data = {
            "message": oversized_message,
            "context": {}
        }
        
        response = await async_client.post(
            "/api/v1/chat/message",
            json=test_data,
            headers=auth_headers
        )
        
        # Should reject oversized input
        assert response.status_code in [
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    async def test_unicode_handling(self, async_client, auth_headers):
        """Test proper Unicode character handling."""
        unicode_inputs = [
            "नमस्ते 🙏 How to meditate?",  # Hindi + emoji
            "Здравствуйте! Что такое медитация?",  # Russian
            "مرحبا! كيف أتأمل؟",  # Arabic
            "你好！如何冥想？",  # Chinese
            "🧘‍♂️🕉️☸️ Spiritual practice",  # Emojis only
        ]
        
        for unicode_input in unicode_inputs:
            test_data = {
                "message": unicode_input,
                "context": {}
            }
            
            response = await async_client.post(
                "/api/v1/chat/message",
                json=test_data,
                headers=auth_headers
            )
            
            # Should handle Unicode properly
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert "response" in result
            assert len(result["response"]) > 0


@pytest.mark.security
class TestAuthenticationSecurity:
    """Test authentication and authorization security."""
    
    async def test_jwt_token_expiry(self, async_client, test_user):
        """Test JWT token expiration handling."""
        # Create expired token
        import jwt
        from datetime import datetime, timedelta
        from app.config import settings
        
        expired_payload = {
            "sub": str(test_user.id),
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
        }
        
        expired_token = jwt.encode(
            expired_payload,
            settings.JWT_SECRET_KEY,
            algorithm="HS256"
        )
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        
        response = await async_client.get("/auth/profile", headers=headers)
        
        # Should reject expired token
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    async def test_invalid_jwt_token(self, async_client):
        """Test handling of invalid JWT tokens."""
        invalid_tokens = [
            "invalid_token",
            "Bearer invalid.token.here",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid",
            "",
            "null"
        ]
        
        for invalid_token in invalid_tokens:
            headers = {"Authorization": f"Bearer {invalid_token}"}
            
            response = await async_client.get("/auth/profile", headers=headers)
            
            # Should reject invalid tokens
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    async def test_token_reuse_prevention(self, async_client, test_user):
        """Test prevention of token reuse after logout.""" 
        # Mock successful login
        with patch('app.auth.auth_service.AuthService.authenticate_user') as mock_auth:
            mock_auth.return_value = test_user
            
            # Login to get token
            login_response = await async_client.post("/auth/login", data={
                "username": test_user.email,
                "password": "test_password"
            })
            
            tokens = login_response.json()
            access_token = tokens["access_token"]
            headers = {"Authorization": f"Bearer {access_token}"}
            
            # Use token successfully
            profile_response = await async_client.get("/auth/profile", headers=headers)
            assert profile_response.status_code == status.HTTP_200_OK
            
            # Logout (invalidate token)
            logout_response = await async_client.post("/auth/logout", headers=headers)
            assert logout_response.status_code == status.HTTP_200_OK
            
            # Try to use token after logout
            post_logout_response = await async_client.get("/auth/profile", headers=headers)
            assert post_logout_response.status_code == status.HTTP_401_UNAUTHORIZED
    
    async def test_brute_force_protection(self, async_client, test_user):
        """Test brute force attack protection."""
        # Attempt multiple failed logins
        failed_attempts = []
        
        for i in range(10):  # Exceed brute force threshold
            response = await async_client.post("/auth/login", data={
                "username": test_user.email,
                "password": f"wrong_password_{i}"
            })
            failed_attempts.append(response)
        
        # Should start blocking after several attempts
        later_attempts = failed_attempts[-3:]  # Last 3 attempts
        
        # Should get rate limited or account locked
        blocked = any(r.status_code in [
            status.HTTP_429_TOO_MANY_REQUESTS,
            status.HTTP_423_LOCKED
        ] for r in later_attempts)
        
        assert blocked, "Brute force protection not working"
    
    async def test_session_fixation_prevention(self, async_client, test_user):
        """Test session fixation attack prevention."""
        # Get initial session
        initial_response = await async_client.get("/auth/status")
        initial_session_id = initial_response.headers.get("set-cookie", "")
        
        # Login
        with patch('app.auth.auth_service.AuthService.authenticate_user') as mock_auth:
            mock_auth.return_value = test_user
            
            login_response = await async_client.post("/auth/login", data={
                "username": test_user.email,
                "password": "test_password"
            })
        
        # Session ID should change after login
        login_session_id = login_response.headers.get("set-cookie", "")
        assert initial_session_id != login_session_id


@pytest.mark.security 
class TestRateLimiting:
    """Test rate limiting security measures."""
    
    async def test_chat_endpoint_rate_limiting(self, async_client, auth_headers):
        """Test rate limiting on chat endpoints."""
        # Make many rapid requests
        responses = []
        
        for i in range(25):  # Exceed rate limit
            response = await async_client.post(
                "/api/v1/chat/message",
                json={
                    "message": f"Test message {i}",
                    "context": {}
                },
                headers=auth_headers
            )
            responses.append(response)
        
        # Should get rate limited
        rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS 
                          for r in responses)
        assert rate_limited
        
        # Rate limit response should include retry-after header
        rate_limit_response = next(
            r for r in responses 
            if r.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        )
        assert "retry-after" in rate_limit_response.headers
    
    async def test_auth_endpoint_rate_limiting(self, async_client):
        """Test stricter rate limiting on auth endpoints."""
        responses = []
        
        # Auth endpoints should have stricter limits
        for i in range(10):
            response = await async_client.post("/auth/login", data={
                "username": f"test{i}@example.com",
                "password": "wrong_password"
            })
            responses.append(response)
        
        # Should get rate limited faster than regular endpoints
        rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS 
                          for r in responses)
        assert rate_limited
    
    async def test_ip_based_rate_limiting(self, async_client):
        """Test IP-based rate limiting."""
        # Simulate different IPs making requests
        headers_list = [
            {"X-Forwarded-For": "192.168.1.1"},
            {"X-Forwarded-For": "192.168.1.2"},
            {"X-Forwarded-For": "192.168.1.3"}
        ]
        
        # Each IP should have separate rate limits
        for headers in headers_list:
            responses = []
            for i in range(15):
                response = await async_client.post(
                    "/auth/login",
                    data={"username": "test@example.com", "password": "wrong"},
                    headers=headers
                )
                responses.append(response)
            
            # This IP should get rate limited
            rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS 
                              for r in responses)
            assert rate_limited


@pytest.mark.security
class TestSecurityHeaders:
    """Test security headers in responses.""" 
    
    async def test_security_headers_present(self, async_client):
        """Test presence of essential security headers."""
        response = await async_client.get("/")
        
        expected_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection", 
            "strict-transport-security",
            "referrer-policy",
            "content-security-policy"
        ]
        
        for header in expected_headers:
            assert header in response.headers, f"Missing security header: {header}"
    
    async def test_csp_header_configuration(self, async_client):
        """Test Content Security Policy header configuration."""
        response = await async_client.get("/")
        
        csp = response.headers.get("content-security-policy", "").lower()
        
        # Should have restrictive CSP
        assert "default-src 'self'" in csp
        assert "script-src" in csp
        assert "style-src" in csp
        
        # Should not allow unsafe-eval in production
        assert "'unsafe-eval'" not in csp or "test" in response.headers.get("environment", "")
    
    async def test_hsts_header(self, async_client):
        """Test HTTP Strict Transport Security header."""
        response = await async_client.get("/")
        
        hsts = response.headers.get("strict-transport-security", "")
        
        # Should enforce HTTPS
        assert "max-age=" in hsts
        assert "includeSubDomains" in hsts
    
    async def test_no_server_info_leakage(self, async_client):
        """Test that server information is not leaked."""
        response = await async_client.get("/")
        
        # Should not reveal server details
        server_header = response.headers.get("server", "").lower()
        assert "apache" not in server_header
        assert "nginx" not in server_header
        assert "microsoft" not in server_header
        
        # Should have custom server identifier
        x_powered_by = response.headers.get("x-powered-by", "")
        if x_powered_by:
            assert "dharmamind" in x_powered_by.lower()


@pytest.mark.security
class TestDataProtection:
    """Test data protection and privacy measures."""
    
    async def test_password_not_in_responses(self, async_client, user_factory):
        """Test that passwords never appear in API responses."""
        user_data = user_factory.create_user_data()
        
        # Registration response should not contain password
        response = await async_client.post("/auth/register", json=user_data)
        if response.status_code == status.HTTP_201_CREATED:
            result = response.json()
            assert "password" not in str(result).lower()
            assert user_data["password"] not in str(result)
    
    async def test_user_data_isolation(self, async_client, test_user, auth_headers):
        """Test that users can only access their own data."""
        # Try to access another user's data
        other_user_id = test_user.id + 1000  # Different user ID
        
        response = await async_client.get(
            f"/api/v1/user/{other_user_id}/profile",
            headers=auth_headers
        )
        
        # Should be forbidden
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    async def test_sensitive_error_handling(self, async_client, auth_headers):
        """Test that error messages don't leak sensitive information.""" 
        # Try to trigger various errors
        error_scenarios = [
            ("/api/v1/nonexistent/endpoint", {}),
            ("/api/v1/chat/message", {"invalid": "data"}),
            ("/api/v1/user/999999/profile", {})
        ]
        
        for endpoint, data in error_scenarios:
            if data:
                response = await async_client.post(endpoint, json=data, headers=auth_headers)
            else:
                response = await async_client.get(endpoint, headers=auth_headers)
            
            if response.status_code >= 400:
                error_content = str(response.json()).lower()
                
                # Should not leak sensitive paths or internal details
                sensitive_terms = [
                    "/etc/passwd", "database", "postgresql", "redis",
                    "secret", "password", "token", "key", "traceback",
                    "internal server error", "exception", "stack trace"
                ]
                
                for term in sensitive_terms:
                    assert term not in error_content
    
    async def test_log_injection_prevention(self, async_client, capture_logs):
        """Test prevention of log injection attacks."""
        malicious_inputs = [
            "test\n[ERROR] Fake error message",
            "user input\r\nINJECTED: Malicious log entry",
            "normal input\x00\x01\x02binary injection"
        ]
        
        for malicious_input in malicious_inputs:
            # Send malicious input
            await async_client.post("/auth/login", data={
                "username": malicious_input,
                "password": "test"
            })
        
        # Check logs don't contain injected content
        log_content = capture_logs()
        assert "[ERROR] Fake error message" not in log_content
        assert "INJECTED: Malicious log entry" not in log_content


@pytest.mark.security
@pytest.mark.slow
class TestSecurityIntegration:
    """Integration tests for security measures."""
    
    async def test_comprehensive_security_scan(self, async_client, malicious_payloads):
        """Run comprehensive security payload tests."""
        endpoints = [
            "/api/v1/chat/message",
            "/auth/register", 
            "/api/v1/feedback/submit",
            "/api/v1/knowledge/search"
        ]
        
        all_payloads = []
        for category, payloads in malicious_payloads.items():
            all_payloads.extend(payloads)
        
        # Test each payload against each endpoint
        for endpoint in endpoints:
            for payload in all_payloads:
                test_data = {
                    "message": payload,
                    "context": {"test": payload},
                    "query": payload
                }
                
                response = await async_client.post(endpoint, json=test_data)
                
                # Should not cause server errors
                assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
                
                # If successful, response should be sanitized
                if response.status_code == status.HTTP_200_OK:
                    content = str(response.json()).lower()
                    
                    # Should not contain dangerous script tags or SQL
                    dangerous_patterns = [
                        "<script>", "javascript:", "drop table",
                        "union select", "exec ", "system("
                    ]
                    
                    for pattern in dangerous_patterns:
                        assert pattern not in content
    
    async def test_concurrent_security_attacks(self, async_client, malicious_payloads):
        """Test security under concurrent malicious requests."""
        tasks = []
        
        # Create concurrent malicious requests
        for i in range(20):
            payload = malicious_payloads["xss_payloads"][i % len(malicious_payloads["xss_payloads"])]
            
            task = async_client.post("/auth/login", data={
                "username": f"attacker{i}@evil.com{payload}",
                "password": f"password{payload}"
            })
            tasks.append(task)
        
        # Execute concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # System should remain stable
        server_errors = sum(1 for r in responses 
                           if hasattr(r, 'status_code') and 
                           r.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Should have minimal server errors (< 5%)
        assert server_errors < len(responses) * 0.05
    
    async def test_security_monitoring_alerts(self, async_client, capture_logs):
        """Test that security events are properly logged."""
        # Trigger security events
        security_events = [
            ("login_failure", "/auth/login", {"username": "admin", "password": "wrong"}),
            ("rate_limit", "/api/v1/chat/message", {"message": "test"}),
            ("invalid_token", "/auth/profile", {}),
            ("xss_attempt", "/api/v1/chat/message", {"message": "<script>alert('xss')</script>"})
        ]
        
        for event_type, endpoint, data in security_events:
            # Multiple attempts to trigger alerts
            for i in range(5):
                await async_client.post(endpoint, data=data if data else None, json=data if not data else None)
        
        # Check that security events are logged
        logs = capture_logs()
        assert len(logs) > 0
        
        # Should contain security-related log entries
        security_logs = [log for log in logs.split('\n') 
                        if any(term in log.lower() 
                              for term in ['security', 'auth', 'failed', 'blocked'])]
        
        assert len(security_logs) > 0, "Security events not properly logged"
