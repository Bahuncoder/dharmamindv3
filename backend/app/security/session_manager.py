"""
🔐 Session Management & Token Blacklisting
==========================================

Provides secure session management including:
- Active session tracking
- Token blacklisting (logout, password change)
- Concurrent session limits
- Session expiration
- Suspicious session detection
"""

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Session(BaseModel):
    """Session data model"""
    session_id: str
    user_id: str
    token_hash: str
    created_at: str
    expires_at: str
    last_activity: str
    ip_address: str
    user_agent: str
    device_fingerprint: Optional[str] = None
    is_active: bool = True


class SessionConfig:
    """Session configuration"""
    # Session limits
    MAX_CONCURRENT_SESSIONS = 5
    SESSION_DURATION_HOURS = 24
    REFRESH_TOKEN_DURATION_DAYS = 30
    
    # Inactivity timeout
    INACTIVITY_TIMEOUT_MINUTES = 60
    
    # Token blacklist cleanup
    BLACKLIST_CLEANUP_INTERVAL = 3600  # 1 hour
    
    # Security settings
    REQUIRE_RE_AUTH_FOR_SENSITIVE = True
    SESSION_BINDING = True  # Bind to IP/User-Agent


class SessionManager:
    """
    Secure session management with token blacklisting
    """
    
    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = Path(db_path)
        self.blacklisted_tokens: Set[str] = set()
        self._init_database()
        self._load_blacklist()
    
    def _init_database(self):
        """Initialize session database"""
        with sqlite3.connect(self.db_path) as conn:
            # Active sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    device_fingerprint TEXT,
                    is_active INTEGER DEFAULT 1,
                    UNIQUE(token_hash)
                )
            """)
            
            # Token blacklist table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_blacklist (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    reason TEXT,
                    blacklisted_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)
            
            # Session audit log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    ip_address TEXT,
                    timestamp TEXT NOT NULL,
                    details TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_token ON token_blacklist(token_hash)")
            
            conn.commit()
    
    def _load_blacklist(self):
        """Load blacklisted tokens into memory for fast lookup"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT token_hash FROM token_blacklist WHERE expires_at > ?",
                (datetime.now(timezone.utc).isoformat(),)
            )
            self.blacklisted_tokens = {row[0] for row in cursor.fetchall()}
        logger.info(f"Loaded {len(self.blacklisted_tokens)} blacklisted tokens")
    
    def _hash_token(self, token: str) -> str:
        """Create hash of token for storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"sess_{secrets.token_hex(16)}"
    
    def create_session(
        self,
        user_id: str,
        token: str,
        ip_address: str,
        user_agent: str,
        device_fingerprint: Optional[str] = None
    ) -> Session:
        """
        Create a new session for user
        
        Args:
            user_id: User identifier
            token: JWT access token
            ip_address: Client IP address
            user_agent: Client user agent
            device_fingerprint: Optional device fingerprint
            
        Returns:
            Session object
        """
        # Check concurrent session limit
        active_sessions = self.get_user_sessions(user_id)
        if len(active_sessions) >= SessionConfig.MAX_CONCURRENT_SESSIONS:
            # Revoke oldest session
            oldest = min(active_sessions, key=lambda s: s.created_at)
            self.revoke_session(oldest.session_id, "max_sessions_exceeded")
            logger.info(f"Revoked oldest session for user {user_id}")
        
        now = datetime.now(timezone.utc)
        session = Session(
            session_id=self._generate_session_id(),
            user_id=user_id,
            token_hash=self._hash_token(token),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=SessionConfig.SESSION_DURATION_HOURS)).isoformat(),
            last_activity=now.isoformat(),
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint,
            is_active=True
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions 
                (session_id, user_id, token_hash, created_at, expires_at, 
                 last_activity, ip_address, user_agent, device_fingerprint, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id, session.user_id, session.token_hash,
                session.created_at, session.expires_at, session.last_activity,
                session.ip_address, session.user_agent, session.device_fingerprint,
                1
            ))
            
            # Audit log
            conn.execute("""
                INSERT INTO session_audit (session_id, user_id, action, ip_address, timestamp)
                VALUES (?, ?, 'created', ?, ?)
            """, (session.session_id, user_id, ip_address, now.isoformat()))
            
            conn.commit()
        
        logger.info(f"Created session {session.session_id} for user {user_id}")
        return session
    
    def validate_session(
        self,
        token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> tuple[bool, Optional[Session], Optional[str]]:
        """
        Validate a session token
        
        Returns:
            (is_valid, session, error_message)
        """
        token_hash = self._hash_token(token)
        
        # Quick blacklist check
        if token_hash in self.blacklisted_tokens:
            return False, None, "Token has been revoked"
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if token is blacklisted in DB (backup check)
            cursor = conn.execute(
                "SELECT 1 FROM token_blacklist WHERE token_hash = ?",
                (token_hash,)
            )
            if cursor.fetchone():
                self.blacklisted_tokens.add(token_hash)
                return False, None, "Token has been revoked"
            
            # Get session
            cursor = conn.execute("""
                SELECT session_id, user_id, token_hash, created_at, expires_at,
                       last_activity, ip_address, user_agent, device_fingerprint, is_active
                FROM sessions WHERE token_hash = ?
            """, (token_hash,))
            
            row = cursor.fetchone()
            if not row:
                return False, None, "Session not found"
            
            session = Session(
                session_id=row[0],
                user_id=row[1],
                token_hash=row[2],
                created_at=row[3],
                expires_at=row[4],
                last_activity=row[5],
                ip_address=row[6],
                user_agent=row[7],
                device_fingerprint=row[8],
                is_active=bool(row[9])
            )
            
            # Check if session is active
            if not session.is_active:
                return False, None, "Session has been revoked"
            
            # Check expiration
            if datetime.now(timezone.utc) > datetime.fromisoformat(session.expires_at.replace('Z', '+00:00')):
                self.revoke_session(session.session_id, "expired")
                return False, None, "Session has expired"
            
            # Check inactivity timeout
            last_activity = datetime.fromisoformat(session.last_activity.replace('Z', '+00:00'))
            if datetime.now(timezone.utc) - last_activity > timedelta(minutes=SessionConfig.INACTIVITY_TIMEOUT_MINUTES):
                self.revoke_session(session.session_id, "inactivity")
                return False, None, "Session timed out due to inactivity"
            
            # Session binding check (optional)
            if SessionConfig.SESSION_BINDING:
                if ip_address and session.ip_address != ip_address:
                    logger.warning(f"IP mismatch for session {session.session_id}")
                    # Don't revoke, just log - could be legitimate (mobile/VPN)
            
            # Update last activity
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (now, session.session_id)
            )
            conn.commit()
            
            return True, session, None
    
    def revoke_session(self, session_id: str, reason: str = "manual"):
        """Revoke a specific session"""
        with sqlite3.connect(self.db_path) as conn:
            # Get session token hash
            cursor = conn.execute(
                "SELECT token_hash, user_id FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                token_hash, user_id = row
                
                # Add to blacklist
                expires = datetime.now(timezone.utc) + timedelta(hours=SessionConfig.SESSION_DURATION_HOURS)
                conn.execute("""
                    INSERT OR REPLACE INTO token_blacklist 
                    (token_hash, user_id, reason, blacklisted_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (token_hash, user_id, reason, datetime.now(timezone.utc).isoformat(), expires.isoformat()))
                
                # Mark session inactive
                conn.execute(
                    "UPDATE sessions SET is_active = 0 WHERE session_id = ?",
                    (session_id,)
                )
                
                # Audit log
                conn.execute("""
                    INSERT INTO session_audit (session_id, user_id, action, timestamp, details)
                    VALUES (?, ?, 'revoked', ?, ?)
                """, (session_id, user_id, datetime.now(timezone.utc).isoformat(), reason))
                
                conn.commit()
                
                # Add to memory blacklist
                self.blacklisted_tokens.add(token_hash)
                
                logger.info(f"Revoked session {session_id} for user {user_id}: {reason}")
    
    def revoke_all_user_sessions(self, user_id: str, reason: str = "logout_all"):
        """Revoke all sessions for a user"""
        sessions = self.get_user_sessions(user_id)
        for session in sessions:
            self.revoke_session(session.session_id, reason)
        logger.info(f"Revoked all {len(sessions)} sessions for user {user_id}")
    
    def blacklist_token(self, token: str, user_id: str, reason: str = "manual"):
        """Directly blacklist a token"""
        token_hash = self._hash_token(token)
        expires = datetime.now(timezone.utc) + timedelta(hours=SessionConfig.SESSION_DURATION_HOURS)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO token_blacklist 
                (token_hash, user_id, reason, blacklisted_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
            """, (token_hash, user_id, reason, datetime.now(timezone.utc).isoformat(), expires.isoformat()))
            conn.commit()
        
        self.blacklisted_tokens.add(token_hash)
        logger.info(f"Blacklisted token for user {user_id}: {reason}")
    
    def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        token_hash = self._hash_token(token)
        return token_hash in self.blacklisted_tokens
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user"""
        sessions = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT session_id, user_id, token_hash, created_at, expires_at,
                       last_activity, ip_address, user_agent, device_fingerprint, is_active
                FROM sessions WHERE user_id = ? AND is_active = 1
            """, (user_id,))
            
            for row in cursor.fetchall():
                sessions.append(Session(
                    session_id=row[0],
                    user_id=row[1],
                    token_hash=row[2],
                    created_at=row[3],
                    expires_at=row[4],
                    last_activity=row[5],
                    ip_address=row[6],
                    user_agent=row[7],
                    device_fingerprint=row[8],
                    is_active=bool(row[9])
                ))
        
        return sessions
    
    def cleanup_expired(self):
        """Clean up expired sessions and blacklist entries"""
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Delete expired sessions
            conn.execute(
                "DELETE FROM sessions WHERE expires_at < ? OR is_active = 0",
                (now,)
            )
            
            # Delete expired blacklist entries
            cursor = conn.execute(
                "DELETE FROM token_blacklist WHERE expires_at < ?",
                (now,)
            )
            deleted = cursor.rowcount
            
            conn.commit()
        
        # Reload blacklist
        self._load_blacklist()
        logger.info(f"Cleaned up expired entries, removed {deleted} from blacklist")


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# Export
__all__ = [
    "Session",
    "SessionConfig",
    "SessionManager",
    "get_session_manager",
]
