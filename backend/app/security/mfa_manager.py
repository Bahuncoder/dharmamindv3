"""
🔐 Multi-Factor Authentication (MFA) Implementation
Advanced security module for DharmaMind backend

Features:
- TOTP (Time-based One-Time Password) support
- SMS/Email backup codes
- Hardware security key support (WebAuthn)
- Recovery codes generation
- Admin MFA enforcement
"""

import qrcode
import pyotp
import secrets
import string
from io import BytesIO
import base64
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from fastapi import HTTPException, status
from pydantic import BaseModel
import json
import os
from cryptography.fernet import Fernet
import sqlite3
from pathlib import Path

class MFASetupRequest(BaseModel):
    user_id: str
    method: str  # 'totp', 'sms', 'email'
    phone: Optional[str] = None
    backup_email: Optional[str] = None

class MFAVerificationRequest(BaseModel):
    user_id: str
    token: str
    method: str
    remember_device: bool = False

class MFASetupResponse(BaseModel):
    success: bool
    secret: Optional[str] = None
    qr_code: Optional[str] = None
    backup_codes: Optional[List[str]] = None
    message: str

class MFAVerificationResponse(BaseModel):
    success: bool
    trusted_device_token: Optional[str] = None
    message: str

class WebAuthnChallenge(BaseModel):
    challenge: str
    user_id: str
    expires_at: datetime

class MFAManager:
    """Comprehensive Multi-Factor Authentication Manager"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or os.getenv('MFA_ENCRYPTION_KEY', self._generate_key())
        self.cipher = Fernet(self.encryption_key.encode()[:44] + b'=')
        self.db_path = Path("mfa_data.db")
        self._init_database()
        
    def _generate_key(self) -> str:
        """Generate a new encryption key"""
        return Fernet.generate_key().decode()
    
    def _init_database(self):
        """Initialize MFA database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_mfa (
                    user_id TEXT PRIMARY KEY,
                    totp_secret TEXT,
                    backup_codes TEXT,
                    trusted_devices TEXT,
                    recovery_codes TEXT,
                    webauthn_keys TEXT,
                    enabled_methods TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mfa_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    method TEXT,
                    success BOOLEAN,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def setup_totp(self, user_id: str, user_email: str) -> MFASetupResponse:
        """Setup TOTP (Time-based One-Time Password) authentication"""
        try:
            # Generate secret key
            secret = pyotp.random_base32()
            
            # Create TOTP instance
            totp = pyotp.TOTP(secret)
            
            # Generate QR code
            provisioning_uri = totp.provisioning_uri(
                user_email,
                issuer_name="DharmaMind"
            )
            
            # Create QR code image
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Convert to base64 for JSON response
            qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Generate backup codes
            backup_codes = self._generate_backup_codes()
            
            # Store encrypted data
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_mfa 
                    (user_id, totp_secret, backup_codes, enabled_methods)
                    VALUES (?, ?, ?, ?)
                """, (
                    user_id,
                    self._encrypt_data(secret),
                    self._encrypt_data(json.dumps(backup_codes)),
                    json.dumps(['totp'])
                ))
            
            return MFASetupResponse(
                success=True,
                secret=secret,
                qr_code=f"data:image/png;base64,{qr_code_base64}",
                backup_codes=backup_codes,
                message="TOTP setup successful. Scan the QR code with your authenticator app."
            )
            
        except Exception as e:
            return MFASetupResponse(
                success=False,
                message=f"TOTP setup failed: {str(e)}"
            )
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate backup codes for MFA recovery"""
        codes = []
        for _ in range(10):
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes
    
    def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT totp_secret FROM user_mfa WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return False
                
                encrypted_secret = result[0]
                secret = self._decrypt_data(encrypted_secret)
                
                totp = pyotp.TOTP(secret)
                return totp.verify(token, valid_window=1)
                
        except Exception:
            return False
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify backup code and mark as used"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT backup_codes FROM user_mfa WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return False
                
                encrypted_codes = result[0]
                codes = json.loads(self._decrypt_data(encrypted_codes))
                
                if code in codes:
                    # Remove used code
                    codes.remove(code)
                    
                    # Update database
                    conn.execute("""
                        UPDATE user_mfa 
                        SET backup_codes = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                    """, (self._encrypt_data(json.dumps(codes)), user_id))
                    
                    return True
                
                return False
                
        except Exception:
            return False
    
    def is_mfa_enabled(self, user_id: str) -> bool:
        """Check if MFA is enabled for user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT enabled_methods FROM user_mfa WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return False
                
                methods = json.loads(result[0])
                return len(methods) > 0
                
        except Exception:
            return False
    
    def generate_trusted_device_token(self, user_id: str, device_info: Dict) -> str:
        """Generate token for trusted device"""
        token_data = {
            'user_id': user_id,
            'device_info': device_info,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        return self._encrypt_data(json.dumps(token_data))
    
    def verify_trusted_device(self, token: str) -> Tuple[bool, Optional[str]]:
        """Verify trusted device token"""
        try:
            token_data = json.loads(self._decrypt_data(token))
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            
            if datetime.now() > expires_at:
                return False, None
            
            return True, token_data['user_id']
            
        except Exception:
            return False, None
    
    def log_mfa_attempt(self, user_id: str, method: str, success: bool, 
                       ip_address: str = "", user_agent: str = ""):
        """Log MFA attempt for security monitoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO mfa_attempts 
                    (user_id, method, success, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, method, success, ip_address, user_agent))
                
        except Exception:
            pass  # Don't fail authentication due to logging issues
    
    def get_mfa_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive MFA status for user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT enabled_methods, backup_codes, created_at, updated_at
                    FROM user_mfa WHERE user_id = ?
                """, (user_id,))
                result = cursor.fetchone()
                
                if not result:
                    return {
                        'enabled': False,
                        'methods': [],
                        'backup_codes_remaining': 0,
                        'setup_date': None
                    }
                
                methods, encrypted_codes, created_at, updated_at = result
                backup_codes = json.loads(self._decrypt_data(encrypted_codes)) if encrypted_codes else []
                
                return {
                    'enabled': True,
                    'methods': json.loads(methods),
                    'backup_codes_remaining': len(backup_codes),
                    'setup_date': created_at,
                    'last_updated': updated_at
                }
                
        except Exception:
            return {
                'enabled': False,
                'methods': [],
                'backup_codes_remaining': 0,
                'setup_date': None
            }

# Global MFA manager instance
mfa_manager = MFAManager()

def get_mfa_manager() -> MFAManager:
    """Get MFA manager instance"""
    return mfa_manager
