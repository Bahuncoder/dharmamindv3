"""
🔐 Authentication System Unit Tests
==================================

Comprehensive tests for DharmaMind authentication including:
- User registration and login
- JWT token management  
- Password security
- Email verification
- MFA authentication
- OAuth integration
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi import HTTPException, status
from jose import jwt
from passlib.context import CryptContext

from app.auth.auth_service import AuthService
from app.models.user import User, UserCreate, UserResponse
from app.config import settings


class TestAuthService:
    """Test authentication service functionality."""
    
    @pytest.fixture
    def auth_service(self, db_session, redis_client):
        """Create auth service instance."""
        return AuthService(db_session, redis_client)
    
    @pytest.fixture 
    def pwd_context(self):
        """Password hashing context."""
        return CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    async def test_create_user_success(self, auth_service, user_factory):
        """Test successful user creation."""
        user_data = UserCreate(**user_factory.create_user_data())
        
        created_user = await auth_service.create_user(user_data)
        
        assert created_user.email == user_data.email
        assert created_user.username == user_data.username
        assert created_user.full_name == user_data.full_name
        assert created_user.is_active is True
        assert created_user.is_verified is False  # Email not verified yet
        assert created_user.spiritual_level == user_data.spiritual_level
    
    async def test_create_user_duplicate_email(self, auth_service, test_user, user_factory):
        """Test user creation with duplicate email."""
        user_data = UserCreate(**user_factory.create_user_data(
            email=test_user.email  # Use existing email
        ))
        
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.create_user(user_data)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "email already registered" in str(exc_info.value.detail).lower()
    
    async def test_create_user_duplicate_username(self, auth_service, test_user, user_factory):
        """Test user creation with duplicate username."""
        user_data = UserCreate(**user_factory.create_user_data(
            username=test_user.username  # Use existing username
        ))
        
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.create_user(user_data)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "username already taken" in str(exc_info.value.detail).lower()
    
    async def test_authenticate_user_success(self, auth_service, test_user, pwd_context):
        """Test successful user authentication."""
        # Set known password
        plain_password = "TestPassword123!"
        test_user.hashed_password = pwd_context.hash(plain_password)
        
        authenticated_user = await auth_service.authenticate_user(
            test_user.email, plain_password
        )
        
        assert authenticated_user is not None
        assert authenticated_user.id == test_user.id
        assert authenticated_user.email == test_user.email
    
    async def test_authenticate_user_wrong_password(self, auth_service, test_user):
        """Test authentication with wrong password."""
        wrong_password = "WrongPassword123!"
        
        authenticated_user = await auth_service.authenticate_user(
            test_user.email, wrong_password
        )
        
        assert authenticated_user is None
    
    async def test_authenticate_user_nonexistent(self, auth_service):
        """Test authentication with nonexistent user."""
        authenticated_user = await auth_service.authenticate_user(
            "nonexistent@example.com", "password123"
        )
        
        assert authenticated_user is None
    
    async def test_authenticate_user_inactive(self, auth_service, test_user, pwd_context):
        """Test authentication with inactive user."""
        # Make user inactive
        test_user.is_active = False
        
        plain_password = "TestPassword123!"
        test_user.hashed_password = pwd_context.hash(plain_password)
        
        authenticated_user = await auth_service.authenticate_user(
            test_user.email, plain_password
        )
        
        assert authenticated_user is None


class TestJWTTokens:
    """Test JWT token creation and validation."""
    
    @pytest.fixture
    def auth_service(self, db_session, redis_client):
        return AuthService(db_session, redis_client)
    
    async def test_create_access_token(self, auth_service, test_user):
        """Test access token creation."""
        token_data = {"sub": str(test_user.id), "email": test_user.email}
        
        access_token = await auth_service.create_access_token(token_data)
        
        assert access_token is not None
        assert isinstance(access_token, str)
        
        # Decode and verify token
        payload = jwt.decode(
            access_token, 
            settings.JWT_SECRET_KEY, 
            algorithms=["HS256"]
        )
        
        assert payload["sub"] == str(test_user.id)
        assert payload["email"] == test_user.email
        assert "exp" in payload  # Has expiration
    
    async def test_create_refresh_token(self, auth_service, test_user):
        """Test refresh token creation.""" 
        refresh_token = await auth_service.create_refresh_token(test_user.id)
        
        assert refresh_token is not None
        assert isinstance(refresh_token, str)
        
        # Verify token structure
        payload = jwt.decode(
            refresh_token,
            settings.JWT_SECRET_KEY,
            algorithms=["HS256"]
        )
        
        assert payload["sub"] == str(test_user.id)
        assert payload["type"] == "refresh"
        assert "exp" in payload
    
    async def test_verify_token_valid(self, auth_service, test_user):
        """Test token verification with valid token."""
        # Create token
        token_data = {"sub": str(test_user.id), "email": test_user.email}
        access_token = await auth_service.create_access_token(token_data)
        
        # Verify token
        payload = await auth_service.verify_token(access_token)
        
        assert payload is not None
        assert payload["sub"] == str(test_user.id)
        assert payload["email"] == test_user.email
    
    async def test_verify_token_expired(self, auth_service, test_user):
        """Test token verification with expired token."""
        # Create expired token
        expired_time = datetime.utcnow() - timedelta(hours=1)
        token_data = {
            "sub": str(test_user.id),
            "email": test_user.email, 
            "exp": expired_time
        }
        
        expired_token = jwt.encode(
            token_data, 
            settings.JWT_SECRET_KEY, 
            algorithm="HS256"
        )
        
        # Verify should return None for expired token
        payload = await auth_service.verify_token(expired_token)
        assert payload is None
    
    async def test_verify_token_invalid_signature(self, auth_service):
        """Test token verification with invalid signature."""
        # Create token with wrong secret
        fake_token = jwt.encode(
            {"sub": "123", "email": "test@example.com"},
            "wrong_secret_key",
            algorithm="HS256"
        )
        
        # Verify should return None
        payload = await auth_service.verify_token(fake_token)
        assert payload is None


class TestPasswordSecurity:
    """Test password hashing and security."""
    
    @pytest.fixture
    def auth_service(self, db_session, redis_client):
        return AuthService(db_session, redis_client)
    
    def test_hash_password(self, auth_service):
        """Test password hashing."""
        plain_password = "SecurePassword123!"
        
        hashed = auth_service.hash_password(plain_password)
        
        assert hashed != plain_password
        assert hashed.startswith("$2b$")  # bcrypt format
        assert len(hashed) > 50  # Reasonable hash length
    
    def test_verify_password_correct(self, auth_service):
        """Test password verification with correct password."""
        plain_password = "SecurePassword123!"
        hashed_password = auth_service.hash_password(plain_password)
        
        is_valid = auth_service.verify_password(plain_password, hashed_password)
        
        assert is_valid is True
    
    def test_verify_password_incorrect(self, auth_service):
        """Test password verification with incorrect password."""
        plain_password = "SecurePassword123!"
        wrong_password = "WrongPassword123!"
        hashed_password = auth_service.hash_password(plain_password)
        
        is_valid = auth_service.verify_password(wrong_password, hashed_password)
        
        assert is_valid is False
    
    @pytest.mark.parametrize("weak_password", [
        "123456",
        "password", 
        "qwerty",
        "abc123",
        "password123",
        "12345678",
    ])
    async def test_weak_password_rejection(self, auth_service, user_factory, weak_password):
        """Test rejection of weak passwords."""
        user_data = UserCreate(**user_factory.create_user_data(
            password=weak_password
        ))
        
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.create_user(user_data)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "password" in str(exc_info.value.detail).lower()


class TestEmailVerification:
    """Test email verification functionality."""
    
    @pytest.fixture
    def auth_service(self, db_session, redis_client):
        return AuthService(db_session, redis_client)
    
    async def test_generate_verification_token(self, auth_service, test_user):
        """Test email verification token generation."""
        verification_token = await auth_service.generate_verification_token(test_user.email)
        
        assert verification_token is not None
        assert isinstance(verification_token, str)
        assert len(verification_token) > 20  # Reasonable token length
    
    async def test_verify_email_success(self, auth_service, test_user):
        """Test successful email verification."""
        # Generate verification token
        token = await auth_service.generate_verification_token(test_user.email)
        
        # Verify email
        result = await auth_service.verify_email(token)
        
        assert result is True
        # User should now be verified
        # Note: This would require fetching user from DB to verify
    
    async def test_verify_email_invalid_token(self, auth_service):
        """Test email verification with invalid token."""
        invalid_token = "invalid_token_12345"
        
        result = await auth_service.verify_email(invalid_token)
        
        assert result is False
    
    async def test_verify_email_expired_token(self, auth_service, test_user):
        """Test email verification with expired token.""" 
        # This would require mocking time or creating an expired token
        with patch('app.auth.auth_service.datetime') as mock_datetime:
            # Mock current time to be after token expiration
            mock_datetime.utcnow.return_value = datetime.utcnow() + timedelta(hours=25)
            
            token = await auth_service.generate_verification_token(test_user.email)
            result = await auth_service.verify_email(token)
            
            assert result is False


class TestMFAAuthentication:
    """Test Multi-Factor Authentication."""
    
    @pytest.fixture
    def auth_service(self, db_session, redis_client):
        return AuthService(db_session, redis_client)
    
    async def test_enable_mfa(self, auth_service, test_user):
        """Test enabling MFA for user."""
        secret = await auth_service.enable_mfa(test_user.id)
        
        assert secret is not None
        assert isinstance(secret, str)
        assert len(secret) >= 16  # TOTP secret length
    
    async def test_verify_mfa_code_valid(self, auth_service, test_user):
        """Test MFA code verification with valid code."""
        # Enable MFA first
        secret = await auth_service.enable_mfa(test_user.id)
        
        # Generate valid TOTP code (this would use pyotp in real implementation)
        with patch('pyotp.TOTP') as mock_totp:
            mock_totp_instance = MagicMock()
            mock_totp.return_value = mock_totp_instance
            mock_totp_instance.verify.return_value = True
            
            is_valid = await auth_service.verify_mfa_code(test_user.id, "123456")
            
            assert is_valid is True
    
    async def test_verify_mfa_code_invalid(self, auth_service, test_user):
        """Test MFA code verification with invalid code."""
        # Enable MFA first
        await auth_service.enable_mfa(test_user.id)
        
        # Mock invalid TOTP code
        with patch('pyotp.TOTP') as mock_totp:
            mock_totp_instance = MagicMock()
            mock_totp.return_value = mock_totp_instance
            mock_totp_instance.verify.return_value = False
            
            is_valid = await auth_service.verify_mfa_code(test_user.id, "000000")
            
            assert is_valid is False
    
    async def test_disable_mfa(self, auth_service, test_user):
        """Test disabling MFA for user."""
        # Enable MFA first
        await auth_service.enable_mfa(test_user.id)
        
        # Disable MFA
        result = await auth_service.disable_mfa(test_user.id)
        
        assert result is True


class TestOAuthIntegration:
    """Test OAuth integration (Google, etc.)."""
    
    @pytest.fixture
    def auth_service(self, db_session, redis_client):
        return AuthService(db_session, redis_client)
    
    async def test_google_oauth_success(self, auth_service):
        """Test successful Google OAuth authentication."""
        # Mock Google token validation
        mock_google_user_info = {
            "email": "user@gmail.com",
            "name": "Google User",
            "picture": "https://example.com/avatar.jpg",
            "email_verified": True
        }
        
        with patch('app.services.google_oauth.verify_google_token') as mock_verify:
            mock_verify.return_value = mock_google_user_info
            
            user = await auth_service.authenticate_with_google("mock_google_token")
            
            assert user is not None
            assert user.email == mock_google_user_info["email"]
            assert user.full_name == mock_google_user_info["name"]
            assert user.is_verified is True  # Google emails are pre-verified
    
    async def test_google_oauth_invalid_token(self, auth_service):
        """Test Google OAuth with invalid token."""
        with patch('app.services.google_oauth.verify_google_token') as mock_verify:
            mock_verify.return_value = None  # Invalid token
            
            user = await auth_service.authenticate_with_google("invalid_token")
            
            assert user is None


class TestUserProfileManagement:
    """Test user profile and spiritual data management."""
    
    @pytest.fixture
    def auth_service(self, db_session, redis_client):
        return AuthService(db_session, redis_client)
    
    async def test_update_spiritual_profile(self, auth_service, test_user):
        """Test updating user's spiritual profile."""
        update_data = {
            "spiritual_level": "advanced",
            "dharmic_path": "zen_buddhism",
            "meditation_experience": 1000,
            "bio": "Dedicated practitioner of mindfulness meditation"
        }
        
        updated_user = await auth_service.update_user_profile(test_user.id, update_data)
        
        assert updated_user.spiritual_level == update_data["spiritual_level"]
        assert updated_user.dharmic_path == update_data["dharmic_path"]  
        assert updated_user.meditation_experience == update_data["meditation_experience"]
        assert updated_user.bio == update_data["bio"]
    
    async def test_get_user_by_id(self, auth_service, test_user):
        """Test retrieving user by ID."""
        retrieved_user = await auth_service.get_user_by_id(test_user.id)
        
        assert retrieved_user is not None
        assert retrieved_user.id == test_user.id
        assert retrieved_user.email == test_user.email
    
    async def test_get_user_by_email(self, auth_service, test_user):
        """Test retrieving user by email."""
        retrieved_user = await auth_service.get_user_by_email(test_user.email)
        
        assert retrieved_user is not None
        assert retrieved_user.email == test_user.email
        assert retrieved_user.id == test_user.id


@pytest.mark.integration
class TestAuthenticationFlow:
    """Integration tests for complete authentication flows."""
    
    async def test_complete_registration_flow(self, async_client, user_factory):
        """Test complete user registration flow."""
        # 1. Register user
        user_data = user_factory.create_user_data()
        response = await async_client.post("/auth/register", json=user_data)
        
        assert response.status_code == 201
        created_user = response.json()
        assert created_user["email"] == user_data["email"]
        assert created_user["is_verified"] is False
        
        # 2. Login should fail before email verification
        login_response = await async_client.post("/auth/login", data={
            "username": user_data["email"],
            "password": user_data["password"]
        })
        assert login_response.status_code == 400  # Unverified email
        
        # 3. Verify email (mock)
        # In real test, this would involve email verification
        
        # 4. Login after verification
        # This would succeed after email verification
    
    async def test_complete_login_flow(self, async_client, test_user, pwd_context):
        """Test complete login flow with valid user."""
        # Set up user with known password
        plain_password = "TestPassword123!"
        test_user.hashed_password = pwd_context.hash(plain_password)
        test_user.is_verified = True
        
        # Login
        response = await async_client.post("/auth/login", data={
            "username": test_user.email,
            "password": plain_password
        })
        
        assert response.status_code == 200
        tokens = response.json()
        
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"
        
        # Test protected endpoint with token
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        profile_response = await async_client.get("/auth/profile", headers=headers)
        
        assert profile_response.status_code == 200
        profile = profile_response.json()
        assert profile["email"] == test_user.email
