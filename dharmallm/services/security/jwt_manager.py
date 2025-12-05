"""JWT token management for authentication - SECURE VERSION"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Set
import jwt
from jwt.exceptions import (
    ExpiredSignatureError,
    InvalidTokenError,
    InvalidSignatureError
)

logger = logging.getLogger(__name__)


class JWTManager:
    """Manages JWT token creation and validation with enhanced security"""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 60,
        refresh_token_expire_days: int = 7
    ):
        # Validate secret key
        if not secret_key:
            raise ValueError("JWT secret key cannot be empty")
        if len(secret_key) < 32:
            raise ValueError(
                f"JWT secret key too short ({len(secret_key)} chars). "
                "Must be at least 32 characters for security."
            )
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        # Token blacklist for revoked tokens
        self.blacklist: Set[str] = set()
        
        logger.info("✓ JWT Manager initialized securely")
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a new access token
        
        Args:
            data: Payload data to encode
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()
        
        # Set expiration
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )
        
        # Add standard claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),  # Issued at
            "type": "access"  # Token type
        })
        
        # Encode token
        encoded_jwt = jwt.encode(
            to_encode, self.secret_key, algorithm=self.algorithm
        )
        
        logger.debug(f"Access token created for: {data.get('sub', 'unknown')}")
        return encoded_jwt
    
    def create_refresh_token(
        self,
        data: Dict[str, Any]
    ) -> str:
        """
        Create a refresh token with longer expiration
        
        Args:
            data: Payload data to encode
            
        Returns:
            Encoded JWT refresh token string
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(
            days=self.refresh_token_expire_days
        )
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(
            to_encode, self.secret_key, algorithm=self.algorithm
        )
        
        logger.debug(
            f"Refresh token created for: {data.get('sub', 'unknown')}"
        )
        return encoded_jwt
    
    def verify_token(
        self,
        token: str,
        expected_type: str = "access"
    ) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token with enhanced security
        
        Args:
            token: JWT token string
            expected_type: Expected token type (access/refresh)
            
        Returns:
            Decoded payload if valid, None otherwise
        """
        # Check blacklist first
        if token in self.blacklist:
            logger.warning("Attempt to use blacklisted token")
            return None
        
        try:
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require_exp": True
                }
            )
            
            # Verify token type
            if payload.get("type") != expected_type:
                logger.warning(
                    f"Token type mismatch: expected {expected_type}, "
                    f"got {payload.get('type')}"
                )
                return None
            
            return payload
            
        except ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except InvalidSignatureError:
            logger.error("Invalid token signature")
            return None
        except InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected token verification error: {e}")
            return None
    
    def revoke_token(self, token: str) -> None:
        """
        Revoke a token by adding it to the blacklist
        
        Args:
            token: Token to revoke
        """
        self.blacklist.add(token)
        logger.info("Token revoked and added to blacklist")
    
    def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        return token in self.blacklist
    
    def clear_blacklist(self) -> None:
        """Clear the token blacklist (admin function)"""
        count = len(self.blacklist)
        self.blacklist.clear()
        logger.info(f"Blacklist cleared ({count} tokens removed)")
    
    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """
        Get the expiration time of a token
        
        Args:
            token: JWT token string
            
        Returns:
            Expiration datetime if valid, None otherwise
        """
        try:
            # Decode without verification to get expiry
            payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            exp_timestamp = payload.get("exp")
            if exp_timestamp:
                return datetime.fromtimestamp(exp_timestamp)
            return None
        except Exception as e:
            logger.error(f"Error getting token expiry: {e}")
            return None


_jwt_manager: Optional[JWTManager] = None


def init_jwt_manager(
    secret_key: str,
    algorithm: str = "HS256",
    access_token_expire_minutes: int = 60,
    refresh_token_expire_days: int = 7
) -> JWTManager:
    """
    Initialize the global JWT manager
    
    Args:
        secret_key: Secret key for signing tokens
        algorithm: JWT algorithm (default: HS256)
        access_token_expire_minutes: Access token expiration in minutes
        refresh_token_expire_days: Refresh token expiration in days
        
    Returns:
        Initialized JWTManager instance
    """
    global _jwt_manager
    _jwt_manager = JWTManager(
        secret_key,
        algorithm,
        access_token_expire_minutes,
        refresh_token_expire_days
    )
    return _jwt_manager


def get_jwt_manager() -> Optional[JWTManager]:
    """Get the current JWT manager instance"""
    return _jwt_manager

