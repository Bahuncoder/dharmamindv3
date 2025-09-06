"""
🔐 Enhanced Session Security Middleware

Advanced session management with hijacking protection, fingerprinting validation,
and comprehensive session security for enterprise-grade authentication.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import hashlib
import json
import redis
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from app.security.jwt_manager import get_jwt_manager, SessionFingerprint
from app.security.monitoring import get_security_monitor, SecurityEventType, ThreatLevel
import logging

logger = logging.getLogger(__name__)

class SessionSecurityError(Exception):
    """Session security related errors"""
    pass

class SessionHijackingDetected(SessionSecurityError):
    """Session hijacking attempt detected"""
    pass

class InvalidSessionFingerprint(SessionSecurityError):
    """Session fingerprint validation failed"""
    pass

class SessionSecurityMiddleware:
    """Enhanced session security middleware with hijacking protection"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.jwt_manager = get_jwt_manager()
        self.security_monitor = get_security_monitor()
        
        # Security configuration
        self.max_session_duration = 28800  # 8 hours
        self.fingerprint_validation_enabled = True
        self.session_rotation_enabled = True
        self.concurrent_session_limit = 3
        
    async def __call__(self, request: Request, call_next):
        """Process request through session security middleware"""
        
        try:
            # Skip security checks for public endpoints
            if self._is_public_endpoint(request.url.path):
                return await call_next(request)
            
            # Extract session information
            session_info = await self._extract_session_info(request)
            
            if not session_info:
                # No session required for this endpoint
                return await call_next(request)
            
            # Validate session security
            await self._validate_session_security(request, session_info)
            
            # Add session info to request state
            request.state.session_info = session_info
            request.state.user_id = session_info.get("user_id")
            
            # Process request
            response = await call_next(request)
            
            # Post-process session (rotation, etc.)
            await self._post_process_session(request, response, session_info)
            
            return response
            
        except SessionSecurityError as e:
            await self._handle_session_security_error(request, e)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "session_security_violation",
                    "message": "Session security validation failed",
                    "details": str(e)
                }
            )
        except Exception as e:
            logger.error(f"Session security middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "internal_security_error"}
            )
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (no session required)"""
        public_endpoints = [
            "/auth/login",
            "/auth/register",
            "/auth/refresh",
            "/health",
            "/docs",
            "/openapi.json",
            "/favicon.ico"
        ]
        
        return any(path.startswith(endpoint) for endpoint in public_endpoints)
    
    async def _extract_session_info(self, request: Request) -> Optional[Dict[str, Any]]:
        """Extract session information from request"""
        
        # Get JWT token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        
        try:
            # Validate and decode token
            payload = await self.jwt_manager.validate_token(token)
            
            if not payload:
                return None
            
            # Get session ID from token
            session_id = payload.get("session_id")
            if not session_id:
                return None
            
            # Get session data from Redis
            session_data = await self._get_session_data(session_id)
            
            if not session_data:
                await self.security_monitor.log_security_event(
                    SecurityEventType.INVALID_TOKEN,
                    ThreatLevel.MEDIUM,
                    request,
                    {"reason": "session_not_found", "session_id": session_id},
                    user_id=payload.get("sub")
                )
                return None
            
            return {
                "token": token,
                "payload": payload,
                "session_id": session_id,
                "session_data": session_data,
                "user_id": payload.get("sub")
            }
            
        except Exception as e:
            logger.warning(f"Session extraction error: {e}")
            await self.security_monitor.log_security_event(
                SecurityEventType.INVALID_TOKEN,
                ThreatLevel.MEDIUM,
                request,
                {"reason": "token_extraction_failed", "error": str(e)}
            )
            return None
    
    async def _get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from Redis"""
        
        try:
            session_key = f"session:{session_id}"
            session_raw = self.redis.get(session_key)
            
            if not session_raw:
                return None
            
            return json.loads(session_raw)
            
        except Exception as e:
            logger.error(f"Error getting session data: {e}")
            return None
    
    async def _validate_session_security(self, request: Request, session_info: Dict[str, Any]):
        """Validate session security against various threats"""
        
        session_data = session_info["session_data"]
        payload = session_info["payload"]
        user_id = session_info["user_id"]
        session_id = session_info["session_id"]
        
        # 1. Check session expiration
        await self._validate_session_expiration(session_data, session_id)
        
        # 2. Check session fingerprint
        if self.fingerprint_validation_enabled:
            await self._validate_session_fingerprint(request, session_data, user_id, session_id)
        
        # 3. Check for session hijacking indicators
        await self._detect_session_hijacking(request, session_data, user_id, session_id)
        
        # 4. Check concurrent session limits
        await self._validate_concurrent_sessions(user_id, session_id)
        
        # 5. Check token freshness and rotation needs
        await self._validate_token_freshness(payload, session_id)
        
        # Log successful validation
        await self.security_monitor.log_security_event(
            SecurityEventType.TOKEN_VALIDATION,
            ThreatLevel.LOW,
            request,
            {"session_id": session_id, "validation": "success"},
            user_id=user_id,
            session_id=session_id
        )
    
    async def _validate_session_expiration(self, session_data: Dict[str, Any], session_id: str):
        """Validate session has not expired"""
        
        created_at = datetime.fromisoformat(session_data["created_at"])
        last_activity = datetime.fromisoformat(session_data.get("last_activity", session_data["created_at"]))
        
        now = datetime.utcnow()
        
        # Check absolute session duration
        if (now - created_at).total_seconds() > self.max_session_duration:
            await self._invalidate_session(session_id)
            raise SessionSecurityError("Session has expired (absolute timeout)")
        
        # Check inactivity timeout (2 hours)
        if (now - last_activity).total_seconds() > 7200:
            await self._invalidate_session(session_id)
            raise SessionSecurityError("Session has expired (inactivity timeout)")
    
    async def _validate_session_fingerprint(
        self, 
        request: Request, 
        session_data: Dict[str, Any], 
        user_id: str, 
        session_id: str
    ):
        """Validate session fingerprint to detect hijacking"""
        
        # Generate current request fingerprint
        current_fingerprint = self._generate_request_fingerprint(request)
        
        # Get stored fingerprint
        stored_fingerprint = session_data.get("fingerprint")
        
        if not stored_fingerprint:
            # First time - store fingerprint
            await self._update_session_fingerprint(session_id, current_fingerprint)
            return
        
        # Compare fingerprints
        if not self._fingerprints_match(stored_fingerprint, current_fingerprint):
            # Potential session hijacking
            await self.security_monitor.log_security_event(
                SecurityEventType.SESSION_HIJACK_ATTEMPT,
                ThreatLevel.CRITICAL,
                request,
                {
                    "stored_fingerprint": stored_fingerprint,
                    "current_fingerprint": current_fingerprint,
                    "session_id": session_id
                },
                user_id=user_id,
                session_id=session_id
            )
            
            # Invalidate session
            await self._invalidate_session(session_id)
            raise SessionHijackingDetected("Session fingerprint mismatch detected")
    
    def _generate_request_fingerprint(self, request: Request) -> str:
        """Generate fingerprint for request"""
        
        # Combine various request characteristics
        fingerprint_data = {
            "user_agent": request.headers.get("User-Agent", ""),
            "accept_language": request.headers.get("Accept-Language", ""),
            "accept_encoding": request.headers.get("Accept-Encoding", ""),
            "ip": self._get_client_ip(request)
        }
        
        fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()
    
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP address"""
        
        # Check for forwarded headers (proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _fingerprints_match(self, stored: str, current: str) -> bool:
        """Check if fingerprints match (with some tolerance)"""
        
        # Exact match
        if stored == current:
            return True
        
        # For now, require exact match for maximum security
        # In production, you might want to implement fuzzy matching
        # for legitimate cases like browser updates
        
        return False
    
    async def _detect_session_hijacking(
        self, 
        request: Request, 
        session_data: Dict[str, Any], 
        user_id: str, 
        session_id: str
    ):
        """Detect various session hijacking indicators"""
        
        current_ip = self._get_client_ip(request)
        stored_ip = session_data.get("ip_address")
        
        # Check for IP address changes
        if stored_ip and stored_ip != current_ip:
            # IP changed - potential hijacking
            await self.security_monitor.log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                ThreatLevel.HIGH,
                request,
                {
                    "reason": "ip_address_changed",
                    "stored_ip": stored_ip,
                    "current_ip": current_ip,
                    "session_id": session_id
                },
                user_id=user_id,
                session_id=session_id
            )
            
            # For high security, invalidate session on IP change
            # In production, you might want to challenge the user instead
            await self._invalidate_session(session_id)
            raise SessionHijackingDetected("IP address change detected")
        
        # Check for impossible travel (rapid geographic changes)
        await self._check_impossible_travel(request, session_data, user_id, session_id)
        
        # Check for suspicious user agent changes
        current_ua = request.headers.get("User-Agent", "")
        stored_ua = session_data.get("user_agent", "")
        
        if stored_ua and self._is_suspicious_ua_change(stored_ua, current_ua):
            await self.security_monitor.log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                ThreatLevel.MEDIUM,
                request,
                {
                    "reason": "suspicious_user_agent_change",
                    "stored_ua": stored_ua,
                    "current_ua": current_ua
                },
                user_id=user_id,
                session_id=session_id
            )
    
    async def _check_impossible_travel(
        self, 
        request: Request, 
        session_data: Dict[str, Any], 
        user_id: str, 
        session_id: str
    ):
        """Check for impossible travel patterns"""
        
        # This would require geolocation data
        # Implementation depends on your geolocation service
        # For now, we'll skip this check
        pass
    
    def _is_suspicious_ua_change(self, stored_ua: str, current_ua: str) -> bool:
        """Check if user agent change is suspicious"""
        
        # Major browser changes are suspicious
        browsers = ["Chrome", "Firefox", "Safari", "Edge"]
        
        stored_browser = None
        current_browser = None
        
        for browser in browsers:
            if browser in stored_ua:
                stored_browser = browser
            if browser in current_ua:
                current_browser = browser
        
        # Different browsers = suspicious
        return stored_browser != current_browser
    
    async def _validate_concurrent_sessions(self, user_id: str, current_session_id: str):
        """Validate concurrent session limits"""
        
        # Get all active sessions for user
        session_pattern = f"session:*"
        active_sessions = []
        
        for key in self.redis.scan_iter(match=session_pattern):
            try:
                session_data = json.loads(self.redis.get(key))
                if session_data.get("user_id") == user_id:
                    active_sessions.append({
                        "session_id": key.decode().replace("session:", ""),
                        "created_at": session_data["created_at"],
                        "last_activity": session_data.get("last_activity", session_data["created_at"])
                    })
            except:
                continue
        
        # Check limit
        if len(active_sessions) > self.concurrent_session_limit:
            # Remove oldest sessions
            active_sessions.sort(key=lambda x: x["last_activity"])
            sessions_to_remove = active_sessions[:-self.concurrent_session_limit]
            
            for session in sessions_to_remove:
                if session["session_id"] != current_session_id:
                    await self._invalidate_session(session["session_id"])
    
    async def _validate_token_freshness(self, payload: Dict[str, Any], session_id: str):
        """Validate token freshness and rotation needs"""
        
        issued_at = datetime.fromtimestamp(payload.get("iat", 0))
        now = datetime.utcnow()
        
        # Check if token needs rotation (every 2 hours)
        if (now - issued_at).total_seconds() > 7200:
            # Mark for rotation
            session_key = f"session:{session_id}"
            session_data = json.loads(self.redis.get(session_key))
            session_data["needs_rotation"] = True
            self.redis.set(session_key, json.dumps(session_data))
    
    async def _update_session_fingerprint(self, session_id: str, fingerprint: str):
        """Update session fingerprint"""
        
        session_key = f"session:{session_id}"
        session_data = json.loads(self.redis.get(session_key))
        session_data["fingerprint"] = fingerprint
        session_data["last_activity"] = datetime.utcnow().isoformat()
        
        self.redis.set(session_key, json.dumps(session_data))
    
    async def _invalidate_session(self, session_id: str):
        """Invalidate a session"""
        
        # Remove from Redis
        session_key = f"session:{session_id}"
        self.redis.delete(session_key)
        
        # Add to blacklist
        await self.jwt_manager.blacklist_session(session_id)
        
        logger.info(f"Session {session_id} invalidated due to security violation")
    
    async def _post_process_session(
        self, 
        request: Request, 
        response, 
        session_info: Dict[str, Any]
    ):
        """Post-process session after request"""
        
        session_id = session_info["session_id"]
        
        # Update last activity
        await self._update_last_activity(session_id)
        
        # Check if token rotation is needed
        session_data = session_info["session_data"]
        if session_data.get("needs_rotation"):
            # Add rotation header
            response.headers["X-Token-Rotation-Required"] = "true"
    
    async def _update_last_activity(self, session_id: str):
        """Update session last activity timestamp"""
        
        session_key = f"session:{session_id}"
        session_raw = self.redis.get(session_key)
        
        if session_raw:
            session_data = json.loads(session_raw)
            session_data["last_activity"] = datetime.utcnow().isoformat()
            self.redis.set(session_key, json.dumps(session_data))
    
    async def _handle_session_security_error(self, request: Request, error: SessionSecurityError):
        """Handle session security errors"""
        
        await self.security_monitor.log_security_event(
            SecurityEventType.UNAUTHORIZED_ACCESS,
            ThreatLevel.HIGH,
            request,
            {
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
        )

# Global session security middleware instance
session_security_middleware: Optional[SessionSecurityMiddleware] = None

def get_session_security_middleware() -> SessionSecurityMiddleware:
    """Get the global session security middleware instance"""
    if session_security_middleware is None:
        raise RuntimeError("Session security middleware not initialized")
    return session_security_middleware

def init_session_security_middleware(redis_client: redis.Redis) -> SessionSecurityMiddleware:
    """Initialize the global session security middleware"""
    global session_security_middleware
    session_security_middleware = SessionSecurityMiddleware(redis_client)
    return session_security_middleware
