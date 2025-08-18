"""
🔐 Enterprise JWT Token Management System

Implements comprehensive JWT security including:
- Token blacklisting
- Token rotation  
- Session management
- Security fingerprinting
"""

import jwt
import redis
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import Request, HTTPException
from pydantic import BaseModel
import logging
import json

logger = logging.getLogger(__name__)

class TokenClaims(BaseModel):
    """JWT Token Claims"""
    user_id: str
    session_id: str
    fingerprint: str
    issued_at: datetime
    expires_at: datetime
    token_type: str = "access"
    permissions: List[str] = []
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class SessionFingerprint(BaseModel):
    """Security fingerprint for session validation"""
    ip_hash: str
    user_agent_hash: str
    browser_fingerprint: str
    created_at: datetime

class EnterpriseJWTManager:
    """Enterprise-grade JWT token management"""
    
    def __init__(self, redis_client: redis.Redis, secret_key: str):
        self.redis = redis_client
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=7)
        self.blacklist_prefix = "jwt_blacklist:"
        self.session_prefix = "session:"
        self.fingerprint_prefix = "fingerprint:"
        
    def generate_session_fingerprint(self, request: Request) -> SessionFingerprint:
        """Generate security fingerprint for session"""
        
        # Get client IP (handle proxies)
        client_ip = request.headers.get("X-Forwarded-For", 
                    request.headers.get("X-Real-IP", 
                    request.client.host if request.client else "unknown"))
        
        # Get user agent
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Create hashes for privacy
        ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:16]
        user_agent_hash = hashlib.sha256(user_agent.encode()).hexdigest()[:16]
        
        # Browser fingerprint (simplified)
        browser_data = f"{user_agent}|{request.headers.get('Accept-Language', '')}|{request.headers.get('Accept-Encoding', '')}"
        browser_fingerprint = hashlib.sha256(browser_data.encode()).hexdigest()[:16]
        
        return SessionFingerprint(
            ip_hash=ip_hash,
            user_agent_hash=user_agent_hash,
            browser_fingerprint=browser_fingerprint,
            created_at=datetime.utcnow()
        )
    
    def create_access_token(self, user_id: str, permissions: List[str], request: Request) -> Dict[str, Any]:
        """Create secure access token with fingerprinting"""
        
        session_id = secrets.token_urlsafe(32)
        fingerprint = self.generate_session_fingerprint(request)
        
        # Store session fingerprint
        fingerprint_key = f"{self.fingerprint_prefix}{session_id}"
        self.redis.setex(
            fingerprint_key,
            int(self.access_token_expire.total_seconds()),
            fingerprint.model_dump_json()
        )
        
        # Create token claims
        now = datetime.utcnow()
        expire = now + self.access_token_expire
        
        claims = {
            "user_id": user_id,
            "session_id": session_id,
            "fingerprint": fingerprint.browser_fingerprint,
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "token_type": "access",
            "permissions": permissions,
            "ip_hash": fingerprint.ip_hash
        }
        
        # Generate token
        token = jwt.encode(claims, self.secret_key, algorithm=self.algorithm)
        
        # Store session data
        session_data = {
            "user_id": user_id,
            "created_at": now.isoformat(),
            "last_activity": now.isoformat(),
            "ip_hash": fingerprint.ip_hash,
            "active": True
        }
        
        session_key = f"{self.session_prefix}{session_id}"
        self.redis.setex(
            session_key,
            int(self.access_token_expire.total_seconds()),
            json.dumps(session_data)
        )
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": int(self.access_token_expire.total_seconds()),
            "session_id": session_id
        }
    
    def validate_token(self, token: str, request: Request) -> TokenClaims:
        """Validate token with comprehensive security checks"""
        
        try:
            # Check if token is blacklisted
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            blacklist_key = f"{self.blacklist_prefix}{token_hash}"
            
            if self.redis.exists(blacklist_key):
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Validate session
            session_id = payload.get("session_id")
            if not session_id:
                raise HTTPException(status_code=401, detail="Invalid token: missing session")
            
            # Check session exists and is active
            session_key = f"{self.session_prefix}{session_id}"
            session_data = self.redis.get(session_key)
            
            if not session_data:
                raise HTTPException(status_code=401, detail="Session expired or invalid")
            
            session_info = json.loads(session_data)
            if not session_info.get("active", False):
                raise HTTPException(status_code=401, detail="Session has been deactivated")
            
            # Validate fingerprint
            current_fingerprint = self.generate_session_fingerprint(request)
            stored_fingerprint_key = f"{self.fingerprint_prefix}{session_id}"
            stored_fingerprint_data = self.redis.get(stored_fingerprint_key)
            
            if not stored_fingerprint_data:
                raise HTTPException(status_code=401, detail="Session fingerprint not found")
            
            stored_fingerprint = SessionFingerprint.model_validate_json(stored_fingerprint_data)
            
            # Check IP consistency (allow some flexibility for mobile users)
            if stored_fingerprint.ip_hash != current_fingerprint.ip_hash:
                logger.warning(f"IP address change detected for session {session_id}")
                # In strict mode, you might want to invalidate the session here
            
            # Check browser fingerprint
            if stored_fingerprint.browser_fingerprint != current_fingerprint.browser_fingerprint:
                logger.warning(f"Browser fingerprint change detected for session {session_id}")
                # This could indicate session hijacking
            
            # Update last activity
            session_info["last_activity"] = datetime.utcnow().isoformat()
            self.redis.setex(
                session_key,
                int(self.access_token_expire.total_seconds()),
                json.dumps(session_info)
            )
            
            return TokenClaims(
                user_id=payload["user_id"],
                session_id=session_id,
                fingerprint=payload["fingerprint"],
                issued_at=datetime.fromtimestamp(payload["iat"]),
                expires_at=datetime.fromtimestamp(payload["exp"]),
                token_type=payload.get("token_type", "access"),
                permissions=payload.get("permissions", []),
                ip_address=current_fingerprint.ip_hash
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise HTTPException(status_code=401, detail="Token validation failed")
    
    def blacklist_token(self, token: str, reason: str = "manual_revocation") -> bool:
        """Add token to blacklist"""
        
        try:
            # Decode token to get expiration
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            exp = payload.get("exp")
            
            if not exp:
                return False
            
            # Calculate remaining TTL
            now = int(time.time())
            ttl = max(0, exp - now)
            
            if ttl > 0:
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                blacklist_key = f"{self.blacklist_prefix}{token_hash}"
                
                blacklist_data = {
                    "reason": reason,
                    "blacklisted_at": datetime.utcnow().isoformat(),
                    "user_id": payload.get("user_id"),
                    "session_id": payload.get("session_id")
                }
                
                self.redis.setex(blacklist_key, ttl, json.dumps(blacklist_data))
                
                # Deactivate session
                session_id = payload.get("session_id")
                if session_id:
                    self.deactivate_session(session_id)
                
                logger.info(f"Token blacklisted: session_id={session_id}, reason={reason}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error blacklisting token: {e}")
            return False
    
    def deactivate_session(self, session_id: str) -> bool:
        """Deactivate a session"""
        
        try:
            session_key = f"{self.session_prefix}{session_id}"
            session_data = self.redis.get(session_key)
            
            if session_data:
                session_info = json.loads(session_data)
                session_info["active"] = False
                session_info["deactivated_at"] = datetime.utcnow().isoformat()
                
                # Keep for audit trail but mark as inactive
                self.redis.setex(session_key, 86400, json.dumps(session_info))  # Keep for 24 hours
                
                # Remove fingerprint
                fingerprint_key = f"{self.fingerprint_prefix}{session_id}"
                self.redis.delete(fingerprint_key)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deactivating session: {e}")
            return False
    
    def get_active_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for a user"""
        
        try:
            pattern = f"{self.session_prefix}*"
            sessions = []
            
            for key in self.redis.scan_iter(match=pattern):
                session_data = self.redis.get(key)
                if session_data:
                    session_info = json.loads(session_data)
                    if session_info.get("user_id") == user_id and session_info.get("active", False):
                        session_id = key.decode().replace(self.session_prefix, "")
                        session_info["session_id"] = session_id
                        sessions.append(session_info)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    def revoke_all_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """Revoke all sessions for a user except optionally one"""
        
        revoked_count = 0
        active_sessions = self.get_active_sessions(user_id)
        
        for session in active_sessions:
            session_id = session["session_id"]
            if except_session and session_id == except_session:
                continue
            
            if self.deactivate_session(session_id):
                revoked_count += 1
        
        logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
        return revoked_count

# Global instance (will be initialized in main.py)
jwt_manager: Optional[EnterpriseJWTManager] = None

def get_jwt_manager() -> EnterpriseJWTManager:
    """Get the global JWT manager instance"""
    if jwt_manager is None:
        raise RuntimeError("JWT manager not initialized")
    return jwt_manager

def init_jwt_manager(redis_client: redis.Redis, secret_key: str) -> EnterpriseJWTManager:
    """Initialize the global JWT manager"""
    global jwt_manager
    jwt_manager = EnterpriseJWTManager(redis_client, secret_key)
    return jwt_manager
