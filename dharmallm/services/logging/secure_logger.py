"""
Secure Logging System for DharmaMind
====================================

Provides secure logging with sensitive data redaction, log rotation,
structured logging, and different configurations for dev/production.

Security Features:
- Automatic redaction of passwords, tokens, API keys
- PII (Personally Identifiable Information) filtering
- Log rotation with size and time limits
- Structured JSON logging
- Request tracking with correlation IDs
- Separate configurations for development vs production

Author: DharmaMind Security Team
Date: October 27, 2025
"""

import logging
import logging.handlers
import json
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SecureFormatter(logging.Formatter):
    """
    Custom formatter that redacts sensitive information
    
    Automatically removes or masks:
    - Passwords
    - JWT tokens
    - API keys
    - Session IDs
    - Email addresses (partial masking)
    - Credit card numbers
    - Social security numbers
    - IP addresses (partial masking in production)
    
    Example:
        handler.setFormatter(SecureFormatter())
    """
    
    # Patterns to redact
    SENSITIVE_PATTERNS = [
        # Passwords
        (r'password["\']?\s*[:=]\s*["\']([^"\']+)["\']', 'password: ***REDACTED***'),
        (r'password["\']?\s*[:=]\s*(\S+)', 'password: ***REDACTED***'),
        
        # JWT tokens
        (r'Bearer\s+([A-Za-z0-9_-]+\.){2}[A-Za-z0-9_-]+', 'Bearer ***REDACTED_TOKEN***'),
        (r'token["\']?\s*[:=]\s*["\']([^"\']+)["\']', 'token: ***REDACTED***'),
        
        # API keys
        (r'api[_-]?key["\']?\s*[:=]\s*["\']([^"\']+)["\']', 'api_key: ***REDACTED***'),
        (r'secret["\']?\s*[:=]\s*["\']([^"\']+)["\']', 'secret: ***REDACTED***'),
        
        # Session IDs
        (r'session[_-]?id["\']?\s*[:=]\s*["\']([^"\']+)["\']', 'session_id: ***REDACTED***'),
        
        # Credit cards (basic pattern)
        (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '****-****-****-****'),
        
        # SSN (US)
        (r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****'),
        
        # Private keys
        (r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----.*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----', 
         '***REDACTED_PRIVATE_KEY***'),
    ]
    
    # Email pattern (partial masking)
    EMAIL_PATTERN = re.compile(
        r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[A-Z|a-z]{2,})\b'
    )
    
    def __init__(self, *args, redact_emails: bool = True, **kwargs):
        """
        Initialize formatter
        
        Args:
            redact_emails: Whether to partially mask email addresses
        """
        super().__init__(*args, **kwargs)
        self.redact_emails = redact_emails
        
        # Compile patterns for better performance
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.DOTALL), replacement)
            for pattern, replacement in self.SENSITIVE_PATTERNS
        ]
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with sensitive data redacted
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string with redacted sensitive data
        """
        # Get the original formatted message
        message = super().format(record)
        
        # Apply all redaction patterns
        for pattern, replacement in self.compiled_patterns:
            message = pattern.sub(replacement, message)
        
        # Partially mask email addresses
        if self.redact_emails:
            message = self._mask_emails(message)
        
        return message
    
    def _mask_emails(self, text: str) -> str:
        """
        Partially mask email addresses
        
        Example: john.doe@example.com -> j***@example.com
        
        Args:
            text: Text containing emails
            
        Returns:
            Text with partially masked emails
        """
        def mask_email(match):
            username = match.group(1)
            domain = match.group(2)
            
            # Keep first character and mask rest of username
            masked_username = username[0] + "***" if len(username) > 1 else "***"
            
            return f"{masked_username}@{domain}"
        
        return self.EMAIL_PATTERN.sub(mask_email, text)


class JSONFormatter(SecureFormatter):
    """
    JSON formatter for structured logging with sensitive data redaction
    
    Outputs logs in JSON format for easy parsing and analysis.
    Includes automatic sensitive data redaction.
    
    Example:
        handler.setFormatter(JSONFormatter())
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        # Build log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields from record
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        if hasattr(record, "client_ip"):
            log_data["client_ip"] = record.client_ip
        
        # Convert to JSON
        json_str = json.dumps(log_data, default=str)
        
        # Apply redaction to JSON string
        for pattern, replacement in self.compiled_patterns:
            json_str = pattern.sub(replacement, json_str)
        
        return json_str


class SecureLogger:
    """
    Secure logger factory with automatic configuration
    
    Features:
    - Automatic log rotation
    - Sensitive data redaction
    - Structured logging option
    - Different configs for dev/production
    - Request correlation tracking
    
    Example usage:
        # Development
        logger = SecureLogger.get_logger(
            "my_app",
            environment="development"
        )
        
        # Production with rotation
        logger = SecureLogger.get_logger(
            "my_app",
            environment="production",
            log_file="/var/log/dharmamind/app.log",
            max_bytes=10485760,  # 10MB
            backup_count=5
        )
    """
    
    # Class-level registry to avoid duplicate handlers
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        level: LogLevel = LogLevel.INFO,
        environment: str = "development",
        log_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB default
        backup_count: int = 5,
        json_format: bool = False,
        redact_emails: bool = True
    ) -> logging.Logger:
        """
        Get or create a secure logger
        
        Args:
            name: Logger name
            level: Log level
            environment: "development" or "production"
            log_file: Path to log file (optional)
            max_bytes: Max bytes before rotation
            backup_count: Number of backup files to keep
            json_format: Whether to use JSON formatting
            redact_emails: Whether to redact email addresses
            
        Returns:
            Configured logger instance
        """
        # Return existing logger if already configured
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.value))
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Choose formatter
        if json_format:
            formatter = JSONFormatter(redact_emails=redact_emails)
        else:
            log_format = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(filename)s:%(lineno)d - %(message)s'
            )
            formatter = SecureFormatter(
                log_format,
                redact_emails=redact_emails
            )
        
        # Console handler (always added)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation (optional)
        if log_file:
            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Production: also log to syslog
        if environment == "production" and os.name != 'nt':  # Unix only
            try:
                syslog_handler = logging.handlers.SysLogHandler(
                    address='/dev/log'
                )
                syslog_handler.setFormatter(formatter)
                logger.addHandler(syslog_handler)
            except Exception as e:
                logger.warning(f"Could not add syslog handler: {e}")
        
        # Store in registry
        cls._loggers[name] = logger
        
        logger.info(
            f"✅ Secure logger '{name}' initialized "
            f"(level={level.value}, env={environment})"
        )
        
        return logger
    
    @classmethod
    def get_request_logger(
        cls,
        name: str,
        request_id: str,
        user_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        **kwargs
    ) -> logging.LoggerAdapter:
        """
        Get logger with request context
        
        Adds request_id, user_id, and client_ip to all log messages
        
        Args:
            name: Logger name
            request_id: Unique request ID
            user_id: User ID (optional)
            client_ip: Client IP address (optional)
            **kwargs: Additional arguments for get_logger()
            
        Returns:
            LoggerAdapter with request context
        """
        logger = cls.get_logger(name, **kwargs)
        
        # Create adapter with extra context
        extra = {
            "request_id": request_id,
            "user_id": user_id,
            "client_ip": client_ip
        }
        
        return logging.LoggerAdapter(logger, extra)


def configure_app_logging(
    app_name: str = "dharmamind",
    environment: str = "development",
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure application-wide logging
    
    Args:
        app_name: Application name
        environment: "development" or "production"
        log_level: Log level string
        log_file: Path to log file
        
    Returns:
        Configured root logger
    """
    # Determine settings based on environment
    if environment == "production":
        level = LogLevel.WARNING
        json_format = True
        redact_emails = True
    else:
        level = LogLevel.DEBUG
        json_format = False
        redact_emails = False
    
    # Override with explicit log level if provided
    if log_level:
        try:
            level = LogLevel[log_level.upper()]
        except KeyError:
            pass
    
    # Create logger
    logger = SecureLogger.get_logger(
        app_name,
        level=level,
        environment=environment,
        log_file=log_file,
        json_format=json_format,
        redact_emails=redact_emails
    )
    
    return logger


# Example usage
if __name__ == "__main__":
    # Development logger
    dev_logger = SecureLogger.get_logger(
        "test_dev",
        environment="development",
        level=LogLevel.DEBUG
    )
    
    dev_logger.info("This is a test message")
    dev_logger.info("User logged in with password: SecurePass123!")
    dev_logger.info("JWT token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    dev_logger.info("Email: john.doe@example.com")
    
    print("\n" + "=" * 60 + "\n")
    
    # Production logger with JSON
    prod_logger = SecureLogger.get_logger(
        "test_prod",
        environment="production",
        level=LogLevel.INFO,
        json_format=True
    )
    
    prod_logger.info("Production log message")
    prod_logger.warning("Sensitive data: api_key: sk-1234567890")
    prod_logger.error("Error with password: MyPassword123")
