"""
Secure Authentication System with Password Hashing
Implements bcrypt password hashing and secure user management
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, validator
import re

logger = logging.getLogger(__name__)

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(BaseModel):
    """User model"""
    username: str
    email: EmailStr
    hashed_password: str
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = datetime.utcnow()
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


class UserCreate(BaseModel):
    """User creation model with validation"""
    username: str
    email: EmailStr
    password: str
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format"""
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if len(v) > 50:
            raise ValueError('Username must be less than 50 characters')
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(
                'Username can only contain letters, numbers, '
                'underscores and hyphens'
            )
        return v
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if len(v) > 128:
            raise ValueError('Password must be less than 128 characters')
        
        # Check password strength
        strength_checks = {
            'lowercase': bool(re.search(r'[a-z]', v)),
            'uppercase': bool(re.search(r'[A-Z]', v)),
            'digit': bool(re.search(r'\d', v)),
            'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', v))
        }
        
        if sum(strength_checks.values()) < 3:
            raise ValueError(
                'Password must contain at least 3 of: '
                'lowercase, uppercase, digit, special character'
            )
        
        return v


class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str


class PasswordHash:
    """Password hashing utilities"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against a hash
        
        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password to compare against
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def needs_rehash(hashed_password: str) -> bool:
        """
        Check if password hash needs to be updated
        
        Args:
            hashed_password: Hashed password to check
            
        Returns:
            True if hash needs update, False otherwise
        """
        return pwd_context.needs_update(hashed_password)


class UserAuthenticator:
    """User authentication manager"""
    
    def __init__(
        self,
        max_attempts: int = 5,
        lockout_duration_minutes: int = 15
    ):
        self.max_attempts = max_attempts
        self.lockout_duration = timedelta(minutes=lockout_duration_minutes)
        self.users_db: Dict[str, User] = {}  # In-memory store (replace with real DB)
        logger.info("UserAuthenticator initialized")
    
    def create_user(self, user_data: UserCreate) -> User:
        """
        Create a new user with hashed password
        
        Args:
            user_data: User creation data
            
        Returns:
            Created user object
            
        Raises:
            ValueError: If username already exists
        """
        # Check if user exists
        if user_data.username in self.users_db:
            raise ValueError(f"Username {user_data.username} already exists")
        
        # Hash password
        hashed_password = PasswordHash.hash_password(user_data.password)
        
        # Create user
        user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            created_at=datetime.utcnow()
        )
        
        # Store user (in production, save to database)
        self.users_db[user.username] = user
        
        logger.info(f"✓ User created: {user.username}")
        return user
    
    def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[User]:
        """
        Authenticate a user
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        # Get user
        user = self.users_db.get(username)
        if not user:
            logger.warning(f"Login attempt for non-existent user: {username}")
            return None
        
        # Check if account is locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            logger.warning(f"Login attempt for locked account: {username}")
            return None
        
        # Verify password
        if not PasswordHash.verify_password(password, user.hashed_password):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failures
            if user.failed_login_attempts >= self.max_attempts:
                user.locked_until = datetime.utcnow() + self.lockout_duration
                logger.warning(
                    f"Account locked due to failed attempts: {username}"
                )
            
            logger.warning(
                f"Failed login attempt for {username} "
                f"(attempt {user.failed_login_attempts})"
            )
            return None
        
        # Check if account is active
        if not user.is_active:
            logger.warning(f"Login attempt for inactive account: {username}")
            return None
        
        # Authentication successful
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        # Check if password needs rehash (bcrypt updates)
        if PasswordHash.needs_rehash(user.hashed_password):
            user.hashed_password = PasswordHash.hash_password(password)
            logger.info(f"Password rehashed for user: {username}")
        
        logger.info(f"✓ User authenticated: {username}")
        return user
    
    def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Returns:
            True if successful, False otherwise
        """
        user = self.authenticate_user(username, old_password)
        if not user:
            return False
        
        # Validate new password
        try:
            UserCreate(
                username=username,
                email=user.email,
                password=new_password
            )
        except ValueError as e:
            logger.error(f"Invalid new password: {e}")
            return False
        
        # Hash and update password
        user.hashed_password = PasswordHash.hash_password(new_password)
        logger.info(f"✓ Password changed for user: {username}")
        return True
    
    def reset_failed_attempts(self, username: str) -> bool:
        """Reset failed login attempts for a user"""
        user = self.users_db.get(username)
        if not user:
            return False
        
        user.failed_login_attempts = 0
        user.locked_until = None
        logger.info(f"✓ Failed attempts reset for user: {username}")
        return True
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.users_db.get(username)


# Global authenticator instance
_authenticator: Optional[UserAuthenticator] = None


def init_authenticator(
    max_attempts: int = 5,
    lockout_duration_minutes: int = 15
) -> UserAuthenticator:
    """Initialize the global authenticator"""
    global _authenticator
    _authenticator = UserAuthenticator(max_attempts, lockout_duration_minutes)
    return _authenticator


def get_authenticator() -> UserAuthenticator:
    """Get the global authenticator instance"""
    if _authenticator is None:
        return init_authenticator()
    return _authenticator
