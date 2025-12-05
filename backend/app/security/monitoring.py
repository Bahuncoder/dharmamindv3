"""
🔍 Enterprise Security Monitoring System

Real-time security monitoring, threat detection, and incident response
for DharmaMind backend infrastructure.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import redis
from collections import defaultdict, deque
import ipaddress
from fastapi import Request
import aiofiles

logger = logging.getLogger(__name__)

class ThreatLevel(str, Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityEventType(str, Enum):
    """Types of security events"""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    TOKEN_VALIDATION = "token_validation"
    INVALID_TOKEN = "invalid_token"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SESSION_HIJACK_ATTEMPT = "session_hijack_attempt"
    BRUTE_FORCE_DETECTED = "brute_force_detected"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    INJECTION_ATTEMPT = "injection_attempt"
    XSS_ATTEMPT = "xss_attempt"

@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    user_agent: str
    endpoint: str
    details: Dict[str, Any]
    session_id: Optional[str] = None
    location: Optional[str] = None
    fingerprint: Optional[str] = None

@dataclass
class ThreatPattern:
    """Threat detection pattern"""
    name: str
    description: str
    event_types: List[SecurityEventType]
    threshold: int
    time_window: int  # seconds
    threat_level: ThreatLevel
    action: str  # block, alert, monitor

class SecurityMonitor:
    """Enterprise security monitoring system"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.event_queue = asyncio.Queue()
        self.threat_patterns = self._initialize_threat_patterns()
        self.active_threats = {}
        self.event_history = deque(maxlen=10000)  # Keep last 10k events in memory
        self.monitoring_active = False
        self.alert_callbacks: List[Callable] = []
        
        # Rate limiting tracking
        self.ip_activity = defaultdict(lambda: deque(maxlen=100))
        self.user_activity = defaultdict(lambda: deque(maxlen=100))
        
    def _initialize_threat_patterns(self) -> List[ThreatPattern]:
        """Initialize threat detection patterns"""
        return [
            ThreatPattern(
                name="Brute Force Login",
                description="Multiple failed login attempts from same IP",
                event_types=[SecurityEventType.LOGIN_FAILURE],
                threshold=5,
                time_window=300,  # 5 minutes
                threat_level=ThreatLevel.HIGH,
                action="block"
            ),
            ThreatPattern(
                name="Session Hijacking",
                description="Session used from different IP/fingerprint",
                event_types=[SecurityEventType.SESSION_HIJACK_ATTEMPT],
                threshold=1,
                time_window=60,
                threat_level=ThreatLevel.CRITICAL,
                action="block"
            ),
            ThreatPattern(
                name="Rate Limit Abuse",
                description="Consistent rate limit violations",
                event_types=[SecurityEventType.RATE_LIMIT_EXCEEDED],
                threshold=10,
                time_window=600,  # 10 minutes
                threat_level=ThreatLevel.MEDIUM,
                action="block"
            ),
            ThreatPattern(
                name="Token Manipulation",
                description="Multiple invalid token attempts",
                event_types=[SecurityEventType.INVALID_TOKEN],
                threshold=20,
                time_window=300,
                threat_level=ThreatLevel.HIGH,
                action="alert"
            ),
            ThreatPattern(
                name="Injection Attempts",
                description="SQL injection or code injection attempts",
                event_types=[SecurityEventType.INJECTION_ATTEMPT],
                threshold=3,
                time_window=300,
                threat_level=ThreatLevel.CRITICAL,
                action="block"
            )
        ]
    
    async def start_monitoring(self):
        """Start the security monitoring system"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("🔍 Security monitoring system started")
        
        # Start background tasks
        asyncio.create_task(self._process_events())
        asyncio.create_task(self._threat_detection())
        asyncio.create_task(self._cleanup_old_data())
    
    async def stop_monitoring(self):
        """Stop the security monitoring system"""
        self.monitoring_active = False
        logger.info("🔍 Security monitoring system stopped")
    
    def extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract security-relevant information from request"""
        
        # Get real client IP (handle proxies)
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
            request.headers.get("X-Real-IP") or
            (request.client.host if request.client else "unknown")
        )
        
        return {
            "ip": client_ip,
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "endpoint": str(request.url.path),
            "method": request.method,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "fingerprint": self._generate_request_fingerprint(request)
        }
    
    def _generate_request_fingerprint(self, request: Request) -> str:
        """Generate a fingerprint for the request"""
        fingerprint_data = f"{request.headers.get('User-Agent', '')}{request.headers.get('Accept-Language', '')}{request.headers.get('Accept-Encoding', '')}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    
    async def log_security_event(
        self,
        event_type: SecurityEventType,
        threat_level: ThreatLevel,
        request: Request,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log a security event"""
        
        request_info = self.extract_request_info(request)
        
        event = SecurityEvent(
            event_id=f"{int(time.time())}-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            event_type=event_type,
            threat_level=threat_level,
            timestamp=datetime.utcnow(),
            source_ip=request_info["ip"],
            user_id=user_id,
            user_agent=request_info["user_agent"],
            endpoint=request_info["endpoint"],
            details={**details, **request_info},
            session_id=session_id,
            fingerprint=request_info["fingerprint"]
        )
        
        # Add to queue for processing
        await self.event_queue.put(event)
        
        # Store in Redis for persistence
        await self._store_event(event)
        
        # Add to in-memory history
        self.event_history.append(event)
        
        logger.info(f"Security event logged: {event_type.value} from {request_info['ip']}")
    
    async def _store_event(self, event: SecurityEvent):
        """Store security event in Redis"""
        try:
            event_key = f"security_event:{event.event_id}"
            event_data = {
                **asdict(event),
                "timestamp": event.timestamp.isoformat()
            }
            
            # Store event (expire after 30 days)
            self.redis.setex(event_key, 2592000, json.dumps(event_data, default=str))
            
            # Add to time-based index
            date_key = f"security_events:{event.timestamp.date()}"
            self.redis.sadd(date_key, event.event_id)
            self.redis.expire(date_key, 2592000)
            
            # Add to IP-based index
            ip_key = f"security_events_ip:{event.source_ip}"
            self.redis.lpush(ip_key, event.event_id)
            self.redis.ltrim(ip_key, 0, 999)  # Keep last 1000 events per IP
            self.redis.expire(ip_key, 604800)  # 7 days
            
        except Exception as e:
            logger.error(f"Error storing security event: {e}")
    
    async def _process_events(self):
        """Process security events from the queue"""
        while self.monitoring_active:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Update activity tracking
                self._update_activity_tracking(event)
                
                # Check for immediate threats
                await self._check_immediate_threats(event)
                
                # Save to persistent storage if needed
                await self._save_to_audit_log(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing security event: {e}")
    
    def _update_activity_tracking(self, event: SecurityEvent):
        """Update activity tracking for rate limiting and pattern detection"""
        
        # Track IP activity
        self.ip_activity[event.source_ip].append({
            "timestamp": event.timestamp,
            "event_type": event.event_type.value,
            "endpoint": event.endpoint
        })
        
        # Track user activity
        if event.user_id:
            self.user_activity[event.user_id].append({
                "timestamp": event.timestamp,
                "event_type": event.event_type.value,
                "ip": event.source_ip,
                "endpoint": event.endpoint
            })
    
    async def _check_immediate_threats(self, event: SecurityEvent):
        """Check for immediate threats requiring instant action"""
        
        # Check for critical events that need immediate response
        if event.threat_level == ThreatLevel.CRITICAL:
            await self._handle_critical_threat(event)
        
        # Check for known malicious patterns
        if await self._is_malicious_pattern(event):
            await self._handle_malicious_activity(event)
    
    async def _handle_critical_threat(self, event: SecurityEvent):
        """Handle critical security threats"""
        
        logger.critical(f"CRITICAL THREAT DETECTED: {event.event_type.value} from {event.source_ip}")
        
        # Immediate actions for critical threats
        if event.event_type == SecurityEventType.SESSION_HIJACK_ATTEMPT:
            # Invalidate all sessions for the user
            if event.user_id:
                await self._invalidate_user_sessions(event.user_id)
        
        # Block IP temporarily
        await self._temporary_ip_block(event.source_ip, duration=3600)  # 1 hour
        
        # Send immediate alert
        await self._send_security_alert(event)
    
    async def _is_malicious_pattern(self, event: SecurityEvent) -> bool:
        """Check if event matches known malicious patterns"""
        
        # Check for SQL injection patterns
        if event.event_type == SecurityEventType.INJECTION_ATTEMPT:
            return True
        
        # Check for XSS patterns
        if event.event_type == SecurityEventType.XSS_ATTEMPT:
            return True
        
        # Check for suspicious user agents
        suspicious_agents = ["sqlmap", "nikto", "dirb", "gobuster", "nmap"]
        if any(agent in event.user_agent.lower() for agent in suspicious_agents):
            return True
        
        return False
    
    async def _handle_malicious_activity(self, event: SecurityEvent):
        """Handle detected malicious activity"""
        
        logger.warning(f"MALICIOUS ACTIVITY: {event.event_type.value} from {event.source_ip}")
        
        # Block IP for longer period
        await self._temporary_ip_block(event.source_ip, duration=86400)  # 24 hours
        
        # Log to security incidents
        await self._log_security_incident(event)
    
    async def _threat_detection(self):
        """Background threat detection based on patterns"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                for pattern in self.threat_patterns:
                    await self._check_threat_pattern(pattern)
                
            except Exception as e:
                logger.error(f"Error in threat detection: {e}")
    
    async def _check_threat_pattern(self, pattern: ThreatPattern):
        """Check if a threat pattern is triggered"""
        
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=pattern.time_window)
        
        # Group events by source IP
        ip_events = defaultdict(list)
        
        for event in self.event_history:
            if (event.timestamp >= window_start and 
                event.event_type in pattern.event_types):
                ip_events[event.source_ip].append(event)
        
        # Check each IP against the pattern
        for ip, events in ip_events.items():
            if len(events) >= pattern.threshold:
                await self._trigger_threat_response(pattern, ip, events)
    
    async def _trigger_threat_response(self, pattern: ThreatPattern, ip: str, events: List[SecurityEvent]):
        """Trigger response to detected threat pattern"""
        
        logger.warning(f"THREAT PATTERN DETECTED: {pattern.name} from {ip} ({len(events)} events)")
        
        if pattern.action == "block":
            await self._temporary_ip_block(ip, duration=3600)
        elif pattern.action == "alert":
            await self._send_threat_alert(pattern, ip, events)
        
        # Log the threat detection
        threat_event = SecurityEvent(
            event_id=f"threat-{int(time.time())}-{hashlib.sha256(ip.encode()).hexdigest()[:8]}",
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            threat_level=pattern.threat_level,
            timestamp=datetime.utcnow(),
            source_ip=ip,
            user_id=None,
            user_agent="threat_detection_system",
            endpoint="/threat_detection",
            details={
                "pattern_name": pattern.name,
                "pattern_description": pattern.description,
                "event_count": len(events),
                "time_window": pattern.time_window,
                "action_taken": pattern.action
            }
        )
        
        await self.event_queue.put(threat_event)
    
    async def _temporary_ip_block(self, ip: str, duration: int):
        """Temporarily block an IP address"""
        
        block_key = f"ip_blocked:{ip}"
        self.redis.setex(block_key, duration, json.dumps({
            "blocked_at": datetime.utcnow().isoformat(),
            "duration": duration,
            "reason": "security_threat_detected"
        }))
        
        logger.info(f"IP {ip} blocked for {duration} seconds")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP is currently blocked"""
        block_key = f"ip_blocked:{ip}"
        return self.redis.exists(block_key)
    
    async def _send_security_alert(self, event: SecurityEvent):
        """Send security alert to registered callbacks"""
        
        for callback in self.alert_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in security alert callback: {e}")
    
    async def _send_threat_alert(self, pattern: ThreatPattern, ip: str, events: List[SecurityEvent]):
        """Send threat pattern alert"""
        
        alert_data = {
            "type": "threat_pattern_detected",
            "pattern": pattern.name,
            "ip": ip,
            "event_count": len(events),
            "threat_level": pattern.threat_level.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.warning(f"Threat alert: {json.dumps(alert_data)}")
    
    async def _save_to_audit_log(self, event: SecurityEvent):
        """Save security event to audit log file"""
        
        try:
            log_entry = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "endpoint": event.endpoint,
                "details": event.details
            }
            
            # Async file writing
            async with aiofiles.open("logs/security_audit.log", "a") as f:
                await f.write(f"{json.dumps(log_entry)}\\n")
                
        except Exception as e:
            logger.error(f"Error saving to audit log: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old IP activity data
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(hours=24)
                
                for ip in list(self.ip_activity.keys()):
                    activity = self.ip_activity[ip]
                    # Keep only recent activity
                    recent_activity = deque([
                        item for item in activity 
                        if item["timestamp"] > cutoff_time
                    ], maxlen=100)
                    self.ip_activity[ip] = recent_activity
                    
                    if not recent_activity:
                        del self.ip_activity[ip]
                
                # Similar cleanup for user activity
                for user_id in list(self.user_activity.keys()):
                    activity = self.user_activity[user_id]
                    recent_activity = deque([
                        item for item in activity 
                        if item["timestamp"] > cutoff_time
                    ], maxlen=100)
                    self.user_activity[user_id] = recent_activity
                    
                    if not recent_activity:
                        del self.user_activity[user_id]
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback for security alerts"""
        self.alert_callbacks.append(callback)
    
    async def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get data for security monitoring dashboard"""
        
        current_time = datetime.utcnow()
        last_24h = current_time - timedelta(hours=24)
        last_1h = current_time - timedelta(hours=1)
        
        # Count events in different time windows
        events_24h = [e for e in self.event_history if e.timestamp >= last_24h]
        events_1h = [e for e in self.event_history if e.timestamp >= last_1h]
        
        # Group by threat level
        threat_counts = defaultdict(int)
        for event in events_24h:
            threat_counts[event.threat_level.value] += 1
        
        # Top source IPs
        ip_counts = defaultdict(int)
        for event in events_24h:
            ip_counts[event.source_ip] += 1
        
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Event type distribution
        event_type_counts = defaultdict(int)
        for event in events_24h:
            event_type_counts[event.event_type.value] += 1
        
        return {
            "total_events_24h": len(events_24h),
            "total_events_1h": len(events_1h),
            "threat_level_distribution": dict(threat_counts),
            "top_source_ips": top_ips,
            "event_type_distribution": dict(event_type_counts),
            "active_blocks": self._get_active_blocks(),
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "last_updated": current_time.isoformat()
        }
    
    def _get_active_blocks(self) -> List[str]:
        """Get list of currently blocked IPs"""
        blocked_ips = []
        pattern = "ip_blocked:*"
        
        for key in self.redis.scan_iter(match=pattern):
            ip = key.decode().replace("ip_blocked:", "")
            blocked_ips.append(ip)
        
        return blocked_ips

# Global security monitor instance
security_monitor: Optional[SecurityMonitor] = None

def get_security_monitor() -> SecurityMonitor:
    """Get the global security monitor instance"""
    if security_monitor is None:
        raise RuntimeError("Security monitor not initialized")
    return security_monitor

def init_security_monitor(redis_client: redis.Redis) -> SecurityMonitor:
    """Initialize the global security monitor"""
    global security_monitor
    security_monitor = SecurityMonitor(redis_client)
    return security_monitor
