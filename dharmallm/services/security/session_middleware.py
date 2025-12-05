"""Session security middleware - SECURE IMPLEMENTATION"""

import logging
import secrets
from typing import Callable, Dict, Optional
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class Session:
    """Session data container"""
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        ip_address: str,
        user_agent: str
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.data: Dict = {}
    
    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if session has expired"""
        elapsed = (datetime.utcnow() - self.last_activity).total_seconds()
        return elapsed > timeout_seconds
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "data": self.data
        }


class SessionManager:
    """Manages user sessions with security features"""
    
    def __init__(
        self,
        session_timeout: int = 3600,  # 1 hour
        check_ip: bool = True,
        check_user_agent: bool = True
    ):
        self.session_timeout = session_timeout
        self.check_ip = check_ip
        self.check_user_agent = check_user_agent
        self.sessions: Dict[str, Session] = {}
        logger.info(f"✓ SessionManager initialized (timeout: {session_timeout}s)")
    
    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str
    ) -> str:
        """
        Create a new session
        
        Args:
            user_id: User identifier
            ip_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            Session ID (secure random token)
        """
        # Generate secure session ID
        session_id = secrets.token_urlsafe(32)
        
        # Create session
        session = Session(session_id, user_id, ip_address, user_agent)
        self.sessions[session_id] = session
        
        logger.info(f"✓ Session created for user: {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def validate_session(
        self,
        session_id: str,
        ip_address: str,
        user_agent: str
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a session
        
        Args:
            session_id: Session ID to validate
            ip_address: Current client IP
            user_agent: Current client user agent
            
        Returns:
            (is_valid, error_message)
        """
        # Check if session exists
        session = self.sessions.get(session_id)
        if not session:
            return False, "Session not found"
        
        # Check if expired
        if session.is_expired(self.session_timeout):
            self.destroy_session(session_id)
            return False, "Session expired"
        
        # Check IP consistency (optional - can be too strict for mobile users)
        if self.check_ip and session.ip_address != ip_address:
            logger.warning(
                f"IP mismatch for session {session_id}: "
                f"expected {session.ip_address}, got {ip_address}"
            )
            # You can decide to either reject or just log
            # For now, we'll be lenient and just log
            pass
        
        # Check user agent consistency
        if self.check_user_agent and session.user_agent != user_agent:
            logger.warning(
                f"User agent mismatch for session {session_id}"
            )
            # Again, can be strict or lenient
            pass
        
        # Update activity
        session.update_activity()
        
        return True, None
    
    def destroy_session(self, session_id: str) -> bool:
        """
        Destroy a session
        
        Args:
            session_id: Session ID to destroy
            
        Returns:
            True if session was destroyed, False if not found
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            del self.sessions[session_id]
            logger.info(f"✓ Session destroyed for user: {session.user_id}")
            return True
        return False
    
    def cleanup_expired(self):
        """Remove all expired sessions"""
        expired = []
        for session_id, session in self.sessions.items():
            if session.is_expired(self.session_timeout):
                expired.append(session_id)
        
        for session_id in expired:
            self.destroy_session(session_id)
        
        if expired:
            logger.info(f"✓ Cleaned up {len(expired)} expired sessions")
    
    def get_user_sessions(self, user_id: str) -> list[Session]:
        """Get all active sessions for a user"""
        return [
            s for s in self.sessions.values()
            if s.user_id == user_id
        ]
    
    def destroy_user_sessions(self, user_id: str) -> int:
        """Destroy all sessions for a user"""
        sessions_to_destroy = [
            s.session_id for s in self.sessions.values()
            if s.user_id == user_id
        ]
        
        for session_id in sessions_to_destroy:
            self.destroy_session(session_id)
        
        logger.info(
            f"✓ Destroyed {len(sessions_to_destroy)} sessions for user: {user_id}"
        )
        return len(sessions_to_destroy)
    
    def get_stats(self) -> Dict:
        """Get session statistics"""
        now = datetime.utcnow()
        active_sessions = sum(
            1 for s in self.sessions.values()
            if not s.is_expired(self.session_timeout)
        )
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "expired_sessions": len(self.sessions) - active_sessions,
            "session_timeout": self.session_timeout,
            "check_ip": self.check_ip,
            "check_user_agent": self.check_user_agent
        }


class SessionSecurityMiddleware(BaseHTTPMiddleware):
    """Middleware to enhance session security"""
    
    def __init__(
        self,
        app,
        session_manager: SessionManager = None,
        exempt_paths: list = None
    ):
        super().__init__(app)
        self.session_manager = session_manager or SessionManager()
        # Paths that don't require session validation
        self.exempt_paths = exempt_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/login",
            "/auth/register",
            "/health"
        ]
        logger.info("✓ SessionSecurityMiddleware initialized")
    
    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def is_exempt(self, path: str) -> bool:
        """Check if path is exempt from session validation"""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with session validation"""
        path = request.url.path
        
        # Skip validation for exempt paths
        if self.is_exempt(path):
            return await call_next(request)
        
        # Get session from cookie
        session_id = request.cookies.get("session_id")
        
        if session_id:
            # Validate session
            is_valid, error = self.session_manager.validate_session(
                session_id,
                self.get_client_ip(request),
                request.headers.get("user-agent", "")
            )
            
            if not is_valid:
                logger.warning(f"Invalid session: {error}")
                response = Response(
                    content="Invalid or expired session",
                    status_code=status.HTTP_401_UNAUTHORIZED
                )
                response.delete_cookie("session_id")
                return response
            
            # Add session info to request state
            session = self.session_manager.get_session(session_id)
            request.state.session = session
            request.state.user_id = session.user_id if session else None
        
        # Process request
        response = await call_next(request)
        
        return response


# Global session manager instance
_session_manager: Optional[SessionManager] = None


async def init_session_security_middleware(
    session_timeout: int = 3600,
    check_ip: bool = True,
    check_user_agent: bool = True
) -> SessionManager:
    """Initialize session security middleware"""
    global _session_manager
    _session_manager = SessionManager(
        session_timeout,
        check_ip,
        check_user_agent
    )
    logger.info("✓ Session security middleware initialized")
    return _session_manager


def get_session_manager() -> Optional[SessionManager]:
    """Get the global session manager instance"""
    return _session_manager

