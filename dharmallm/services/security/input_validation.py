"""
Input Validation and Sanitization Utilities
Protects against XSS, SQL injection, and other injection attacks
"""

import html
import re
import logging
from typing import Any, Optional
from pydantic import validator

logger = logging.getLogger(__name__)


class InputValidator:
    """Input validation and sanitization utilities"""
    
    # Dangerous patterns to detect
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(--|\#|\/\*|\*\/)",  # SQL comments
        r"(\bEXEC\b|\bEXECUTE\b)",
        r"(\bxp_cmdshell\b)",
    ]
    
    XSS_PATTERNS = [
        r"(<script[^>]*>.*?</script>)",
        r"(javascript:)",
        r"(onerror\s*=)",
        r"(onload\s*=)",
        r"(onclick\s*=)",
        r"(onmouseover\s*=)",
        r"(<iframe[^>]*>)",
        r"(<object[^>]*>)",
        r"(<embed[^>]*>)",
        r"(data:text/html)",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",  # Directory traversal
        r"\.\.",  # Parent directory
        r"/etc/",  # System files
        r"/proc/",  # Process files
        r"~",  # Home directory
    ]
    
    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """
        Escape HTML special characters to prevent XSS
        
        Args:
            text: Text to sanitize
            
        Returns:
            HTML-escaped text
        """
        if not text:
            return text
        return html.escape(text, quote=True)
    
    @classmethod
    def detect_sql_injection(cls, text: str) -> bool:
        """
        Detect potential SQL injection attempts
        
        Args:
            text: Text to check
            
        Returns:
            True if SQL injection detected
        """
        if not text:
            return False
        
        text_upper = text.upper()
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_upper, re.IGNORECASE):
                logger.warning(f"SQL injection pattern detected: {pattern}")
                return True
        return False
    
    @classmethod
    def detect_xss(cls, text: str) -> bool:
        """
        Detect potential XSS attempts
        
        Args:
            text: Text to check
            
        Returns:
            True if XSS detected
        """
        if not text:
            return False
        
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"XSS pattern detected: {pattern}")
                return True
        return False
    
    @classmethod
    def detect_path_traversal(cls, text: str) -> bool:
        """
        Detect path traversal attempts
        
        Args:
            text: Text to check
            
        Returns:
            True if path traversal detected
        """
        if not text:
            return False
        
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Path traversal pattern detected: {pattern}")
                return True
        return False
    
    @classmethod
    def validate_safe_string(cls, text: str, max_length: int = 5000) -> str:
        """
        Validate and sanitize a string for safe use
        
        Args:
            text: Text to validate
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
            
        Raises:
            ValueError: If validation fails
        """
        if not text:
            return text
        
        # Check length
        if len(text) > max_length:
            raise ValueError(f"Text too long (max {max_length} characters)")
        
        # Check for SQL injection
        if cls.detect_sql_injection(text):
            raise ValueError("Potential SQL injection detected")
        
        # Check for XSS
        if cls.detect_xss(text):
            raise ValueError("Potential XSS attack detected")
        
        # Sanitize HTML
        sanitized = cls.sanitize_html(text)
        
        return sanitized
    
    @classmethod
    def validate_safe_filename(cls, filename: str) -> str:
        """
        Validate and sanitize a filename
        
        Args:
            filename: Filename to validate
            
        Returns:
            Sanitized filename
            
        Raises:
            ValueError: If validation fails
        """
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        # Check for path traversal
        if cls.detect_path_traversal(filename):
            raise ValueError("Invalid filename: path traversal detected")
        
        # Remove any directory separators
        filename = filename.replace("/", "").replace("\\", "")
        
        # Only allow alphanumeric, dash, underscore, and dot
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
        
        if not sanitized:
            raise ValueError("Invalid filename")
        
        # Ensure it's not too long
        if len(sanitized) > 255:
            raise ValueError("Filename too long")
        
        return sanitized
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """
        Validate email format
        
        Args:
            email: Email to validate
            
        Returns:
            True if valid
        """
        if not email:
            return False
        
        # Basic email regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @classmethod
    def validate_username(cls, username: str) -> str:
        """
        Validate and sanitize username
        
        Args:
            username: Username to validate
            
        Returns:
            Validated username
            
        Raises:
            ValueError: If validation fails
        """
        if not username:
            raise ValueError("Username cannot be empty")
        
        if len(username) < 3:
            raise ValueError("Username too short (minimum 3 characters)")
        
        if len(username) > 50:
            raise ValueError("Username too long (maximum 50 characters)")
        
        # Only alphanumeric, underscore, and dash
        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            raise ValueError(
                "Username can only contain letters, numbers, "
                "underscores and hyphens"
            )
        
        return username
    
    @classmethod
    def strip_control_characters(cls, text: str) -> str:
        """
        Remove control characters from text
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Remove control characters (except newlines and tabs)
        return ''.join(
            char for char in text
            if char.isprintable() or char in ['\n', '\t', '\r']
        )
    
    @classmethod
    def truncate_text(cls, text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to maximum length
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated text
        """
        if not text or len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix


# Pydantic validators for use in models
def validate_safe_string_field(max_length: int = 5000):
    """Create a Pydantic validator for safe strings"""
    @validator('*', pre=True)
    def validator_func(cls, v):
        if isinstance(v, str):
            return InputValidator.validate_safe_string(v, max_length)
        return v
    return validator_func


def validate_username_field():
    """Create a Pydantic validator for usernames"""
    @validator('*', pre=True)
    def validator_func(cls, v):
        if isinstance(v, str):
            return InputValidator.validate_username(v)
        return v
    return validator_func


def validate_filename_field():
    """Create a Pydantic validator for filenames"""
    @validator('*', pre=True)
    def validator_func(cls, v):
        if isinstance(v, str):
            return InputValidator.validate_safe_filename(v)
        return v
    return validator_func


# Example usage in Pydantic models:
"""
from pydantic import BaseModel, validator

class ChatMessage(BaseModel):
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        return InputValidator.validate_safe_string(v, max_length=5000)

class UserInput(BaseModel):
    username: str
    comment: str
    
    @validator('username')
    def validate_username(cls, v):
        return InputValidator.validate_username(v)
    
    @validator('comment')
    def validate_comment(cls, v):
        return InputValidator.validate_safe_string(v, max_length=10000)
"""
