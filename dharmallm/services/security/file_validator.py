"""
Secure File Operations for DharmaMind
=====================================

Provides secure file handling with:
- Path traversal prevention
- File type validation (magic number checking)
- Size limit enforcement
- Atomic write operations
- Quarantine system for suspicious files
- Safe filename sanitization

Security Features:
- Blocks directory traversal attempts (../, /etc/, etc.)
- Validates file types using magic numbers (not just extensions)
- Enforces configurable size limits
- Atomic writes prevent partial file corruption
- Quarantine system for malicious files
- Sanitizes filenames to prevent injection

Author: DharmaMind Security Team
Date: October 27, 2025
"""

import os
import re
import hashlib
import tempfile
import shutil
import magic  # python-magic library
from pathlib import Path
from typing import Optional, List, Tuple, BinaryIO
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Raised when file validation fails"""
    pass


class FileValidator:
    """
    Secure file validator with comprehensive security checks
    
    Example usage:
        validator = FileValidator(
            allowed_extensions=[".txt", ".pdf"],
            max_size_mb=10,
            base_dir="/var/uploads"
        )
        
        # Validate file
        is_safe, errors = validator.validate_file("upload.txt", file_content)
        if not is_safe:
            raise FileValidationError(f"Invalid file: {errors}")
    """
    
    # Dangerous path patterns
    DANGEROUS_PATH_PATTERNS = [
        r'\.\.',  # Parent directory
        r'^/',    # Absolute path
        r'^\\',   # Windows absolute path
        r'~',     # Home directory
        r'\$',    # Environment variables
        r'%',     # Windows environment variables
    ]
    
    # Known dangerous magic numbers (file signatures)
    DANGEROUS_SIGNATURES = {
        b'\x4D\x5A': 'Windows executable (EXE)',
        b'\x7F\x45\x4C\x46': 'Linux executable (ELF)',
        b'\x23\x21': 'Script with shebang',
        b'\xCA\xFE\xBA\xBE': 'Java class file',
        b'\x50\x4B\x03\x04': 'ZIP archive (could contain executables)',
    }
    
    # Safe file type mappings (MIME type -> extensions)
    SAFE_MIME_TYPES = {
        'text/plain': ['.txt', '.log', '.csv'],
        'application/pdf': ['.pdf'],
        'image/jpeg': ['.jpg', '.jpeg'],
        'image/png': ['.png'],
        'image/gif': ['.gif'],
        'application/json': ['.json'],
        'text/csv': ['.csv'],
        'text/html': ['.html', '.htm'],
        'application/vnd.ms-excel': ['.xls'],
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    }
    
    def __init__(
        self,
        allowed_extensions: Optional[List[str]] = None,
        allowed_mime_types: Optional[List[str]] = None,
        max_size_mb: float = 10.0,
        base_dir: Optional[str] = None,
        quarantine_dir: Optional[str] = None
    ):
        """
        Initialize file validator
        
        Args:
            allowed_extensions: List of allowed file extensions (e.g., ['.txt', '.pdf'])
            allowed_mime_types: List of allowed MIME types
            max_size_mb: Maximum file size in megabytes
            base_dir: Base directory for file operations (restricts file access)
            quarantine_dir: Directory for quarantined suspicious files
        """
        self.allowed_extensions = [
            ext.lower() if not ext.startswith('.') else ext.lower()
            for ext in (allowed_extensions or [])
        ]
        self.allowed_mime_types = allowed_mime_types or []
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.quarantine_dir = Path(quarantine_dir) if quarantine_dir else None
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create quarantine directory if specified
        if self.quarantine_dir:
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"✅ FileValidator initialized: "
            f"max_size={max_size_mb}MB, base_dir={self.base_dir}"
        )
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent injection attacks
        
        Removes or replaces dangerous characters:
        - Path separators (/, \\)
        - Null bytes
        - Control characters
        - Shell metacharacters
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
            
        Raises:
            FileValidationError: If filename is invalid
        """
        if not filename:
            raise FileValidationError("Filename cannot be empty")
        
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove null bytes and control characters
        filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)
        
        # Remove shell metacharacters
        dangerous_chars = r'[;&|`$(){}[\]<>"\']'
        filename = re.sub(dangerous_chars, '', filename)
        
        # Replace spaces and multiple dots
        filename = filename.replace(' ', '_')
        filename = re.sub(r'\.{2,}', '.', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        if not filename or filename in ['.', '..']:
            raise FileValidationError("Invalid filename after sanitization")
        
        return filename
    
    def validate_path(self, filepath: str) -> Path:
        """
        Validate file path to prevent directory traversal
        
        Args:
            filepath: File path to validate
            
        Returns:
            Resolved Path object
            
        Raises:
            FileValidationError: If path is dangerous
        """
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATH_PATTERNS:
            if re.search(pattern, filepath):
                raise FileValidationError(
                    f"Path contains dangerous pattern: {pattern}"
                )
        
        # Resolve path and check if within base directory
        try:
            resolved_path = (self.base_dir / filepath).resolve()
        except Exception as e:
            raise FileValidationError(f"Invalid path: {e}")
        
        # Ensure path is within base directory
        try:
            resolved_path.relative_to(self.base_dir)
        except ValueError:
            raise FileValidationError(
                "Path traversal detected: path is outside base directory"
            )
        
        return resolved_path
    
    def validate_extension(self, filename: str) -> Tuple[bool, str]:
        """
        Validate file extension
        
        Args:
            filename: Filename to check
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not self.allowed_extensions:
            return True, "No extension restrictions"
        
        ext = Path(filename).suffix.lower()
        
        if ext not in self.allowed_extensions:
            return False, (
                f"Extension '{ext}' not allowed. "
                f"Allowed: {', '.join(self.allowed_extensions)}"
            )
        
        return True, "Extension valid"
    
    def validate_size(self, size_bytes: int) -> Tuple[bool, str]:
        """
        Validate file size
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Tuple of (is_valid, message)
        """
        if size_bytes > self.max_size_bytes:
            max_mb = self.max_size_bytes / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            return False, (
                f"File too large: {actual_mb:.2f}MB "
                f"(max: {max_mb:.2f}MB)"
            )
        
        if size_bytes == 0:
            return False, "File is empty"
        
        return True, "Size valid"
    
    def validate_magic_number(
        self,
        content: bytes,
        filename: str
    ) -> Tuple[bool, str]:
        """
        Validate file type using magic numbers (file signatures)
        
        More secure than checking extensions, as it reads the actual
        file content to determine type.
        
        Args:
            content: File content bytes
            filename: Filename for extension checking
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not content:
            return False, "Empty file content"
        
        # Check for dangerous signatures
        for signature, description in self.DANGEROUS_SIGNATURES.items():
            if content.startswith(signature):
                return False, f"Dangerous file type detected: {description}"
        
        # If no MIME type restrictions, only check for dangerous types
        if not self.allowed_mime_types:
            return True, "No MIME type restrictions"
        
        # Detect MIME type using python-magic
        try:
            mime = magic.from_buffer(content[:2048], mime=True)
        except Exception as e:
            logger.warning(f"Could not detect MIME type: {e}")
            # Fall back to extension checking
            ext = Path(filename).suffix.lower()
            for allowed_mime, extensions in self.SAFE_MIME_TYPES.items():
                if ext in extensions and allowed_mime in self.allowed_mime_types:
                    return True, f"Extension matches allowed type: {allowed_mime}"
            return False, "Could not verify file type"
        
        # Check if MIME type is allowed
        if mime not in self.allowed_mime_types:
            return False, (
                f"MIME type '{mime}' not allowed. "
                f"Allowed: {', '.join(self.allowed_mime_types)}"
            )
        
        return True, f"MIME type valid: {mime}"
    
    def validate_file(
        self,
        filename: str,
        content: bytes
    ) -> Tuple[bool, List[str]]:
        """
        Perform comprehensive file validation
        
        Args:
            filename: Filename to validate
            content: File content bytes
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # 1. Sanitize and validate filename
        try:
            sanitized_name = self.sanitize_filename(filename)
        except FileValidationError as e:
            errors.append(f"Filename error: {e}")
            return False, errors
        
        # 2. Validate extension
        valid, msg = self.validate_extension(sanitized_name)
        if not valid:
            errors.append(msg)
        
        # 3. Validate size
        valid, msg = self.validate_size(len(content))
        if not valid:
            errors.append(msg)
        
        # 4. Validate magic number
        valid, msg = self.validate_magic_number(content, sanitized_name)
        if not valid:
            errors.append(msg)
        
        return len(errors) == 0, errors
    
    def quarantine_file(self, filename: str, content: bytes, reason: str):
        """
        Quarantine a suspicious file
        
        Args:
            filename: Original filename
            content: File content
            reason: Reason for quarantine
        """
        if not self.quarantine_dir:
            logger.warning("No quarantine directory configured")
            return
        
        # Generate unique quarantine filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.sha256(content).hexdigest()[:8]
        quarantine_name = f"{timestamp}_{file_hash}_{filename}"
        
        quarantine_path = self.quarantine_dir / quarantine_name
        
        # Write file
        quarantine_path.write_bytes(content)
        
        # Write metadata
        metadata_path = quarantine_path.with_suffix('.meta')
        metadata = {
            'original_filename': filename,
            'timestamp': timestamp,
            'reason': reason,
            'size': len(content),
            'sha256': hashlib.sha256(content).hexdigest()
        }
        
        import json
        metadata_path.write_text(json.dumps(metadata, indent=2))
        
        logger.warning(
            f"🚨 File quarantined: {filename} -> {quarantine_name} "
            f"(reason: {reason})"
        )


class SecureFileWriter:
    """
    Secure file writer with atomic operations
    
    Ensures files are written atomically (all-or-nothing) to prevent
    partial writes or corruption.
    
    Example usage:
        writer = SecureFileWriter(base_dir="/var/data")
        writer.write_file("output.txt", b"data", mode=0o600)
    """
    
    def __init__(
        self,
        base_dir: str,
        validator: Optional[FileValidator] = None
    ):
        """
        Initialize secure file writer
        
        Args:
            base_dir: Base directory for file operations
            validator: Optional FileValidator instance
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.validator = validator or FileValidator(base_dir=str(base_dir))
    
    def write_file(
        self,
        filename: str,
        content: bytes,
        mode: int = 0o644
    ) -> Path:
        """
        Write file atomically with security checks
        
        Args:
            filename: Target filename
            content: File content
            mode: File permissions (Unix)
            
        Returns:
            Path to written file
            
        Raises:
            FileValidationError: If validation fails
        """
        # Validate file
        is_valid, errors = self.validator.validate_file(filename, content)
        if not is_valid:
            # Quarantine if configured
            self.validator.quarantine_file(
                filename,
                content,
                f"Validation failed: {', '.join(errors)}"
            )
            raise FileValidationError(f"File validation failed: {errors}")
        
        # Sanitize filename
        safe_filename = self.validator.sanitize_filename(filename)
        
        # Validate path
        target_path = self.validator.validate_path(safe_filename)
        
        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write atomically using temporary file
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=target_path.parent,
            delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(content)
        
        try:
            # Set permissions
            os.chmod(tmp_path, mode)
            
            # Atomic move
            shutil.move(str(tmp_path), str(target_path))
            
            logger.info(f"✅ File written securely: {target_path}")
            
            return target_path
            
        except Exception as e:
            # Clean up temp file on error
            if tmp_path.exists():
                tmp_path.unlink()
            raise FileValidationError(f"Failed to write file: {e}")
    
    def read_file(self, filename: str) -> bytes:
        """
        Read file securely with path validation
        
        Args:
            filename: Filename to read
            
        Returns:
            File content bytes
            
        Raises:
            FileValidationError: If validation fails
        """
        # Validate path
        file_path = self.validator.validate_path(filename)
        
        if not file_path.exists():
            raise FileValidationError(f"File not found: {filename}")
        
        if not file_path.is_file():
            raise FileValidationError(f"Not a file: {filename}")
        
        # Read file
        try:
            return file_path.read_bytes()
        except Exception as e:
            raise FileValidationError(f"Failed to read file: {e}")


# Example usage
if __name__ == "__main__":
    # Create validator
    validator = FileValidator(
        allowed_extensions=['.txt', '.pdf', '.jpg'],
        max_size_mb=5.0,
        base_dir="/tmp/secure_uploads",
        quarantine_dir="/tmp/quarantine"
    )
    
    # Test safe file
    safe_content = b"This is a safe text file"
    is_valid, errors = validator.validate_file("test.txt", safe_content)
    print(f"Safe file valid: {is_valid}")
    
    # Test dangerous file (simulated executable)
    dangerous_content = b'\x4D\x5A' + b'\x00' * 100  # EXE signature
    is_valid, errors = validator.validate_file("malware.exe", dangerous_content)
    print(f"Dangerous file valid: {is_valid}, errors: {errors}")
    
    # Test path traversal
    try:
        validator.validate_path("../../etc/passwd")
    except FileValidationError as e:
        print(f"Path traversal blocked: {e}")
    
    # Test secure writer
    writer = SecureFileWriter("/tmp/secure_uploads", validator)
    try:
        path = writer.write_file("output.txt", b"Secure content")
        print(f"File written: {path}")
    except FileValidationError as e:
        print(f"Write failed: {e}")
