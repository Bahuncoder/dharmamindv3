"""
🔍 Security Dashboard API Routes

Real-time security monitoring dashboard endpoints for enterprise security management.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.security.monitoring import get_security_monitor, SecurityEventType, ThreatLevel
from app.security.jwt_manager import get_jwt_manager
from app.routes.admin_auth import verify_admin_token
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/security", tags=["security-dashboard"])
security = HTTPBearer()

@router.get("/dashboard", dependencies=[Depends(verify_admin_token)])
async def get_security_dashboard() -> Dict[str, Any]:
    """
    Get comprehensive security dashboard data
    
    Requires admin authentication
    Returns real-time security metrics and monitoring data
    """
    try:
        security_monitor = get_security_monitor()
        dashboard_data = await security_monitor.get_security_dashboard_data()
        
        # Add additional system security metrics
        jwt_manager = get_jwt_manager()
        
        # Get blacklisted tokens count
        blacklisted_count = await jwt_manager.get_blacklisted_tokens_count()
        
        # Get active sessions count
        active_sessions = await jwt_manager.get_active_sessions_count()
        
        dashboard_data.update({
            "authentication": {
                "blacklisted_tokens": blacklisted_count,
                "active_sessions": active_sessions,
                "jwt_status": "operational"
            },
            "security_framework": {
                "status": "enterprise-grade",
                "monitoring": "active",
                "features": [
                    "jwt_token_blacklisting",
                    "session_fingerprinting", 
                    "real_time_monitoring",
                    "threat_detection",
                    "automatic_blocking",
                    "audit_logging"
                ]
            }
        })
        
        return {
            "status": "success",
            "data": dashboard_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Security dashboard error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security dashboard data"
        )

@router.get("/events", dependencies=[Depends(verify_admin_token)])
async def get_recent_security_events(
    limit: int = 100,
    threat_level: Optional[ThreatLevel] = None,
    event_type: Optional[SecurityEventType] = None,
    hours: int = 24
) -> Dict[str, Any]:
    """
    Get recent security events with filtering
    
    Args:
        limit: Maximum number of events to return
        threat_level: Filter by threat level
        event_type: Filter by event type
        hours: Time window in hours
    """
    try:
        security_monitor = get_security_monitor()
        
        # Filter events from memory history
        events = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        for event in security_monitor.event_history:
            if event.timestamp < cutoff_time:
                continue
                
            if threat_level and event.threat_level != threat_level:
                continue
                
            if event_type and event.event_type != event_type:
                continue
                
            events.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "timestamp": event.timestamp.isoformat(),
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "endpoint": event.endpoint,
                "user_agent": event.user_agent[:100] + "..." if len(event.user_agent) > 100 else event.user_agent,
                "details": {k: v for k, v in event.details.items() if k not in ['headers', 'query_params']}
            })
            
            if len(events) >= limit:
                break
        
        return {
            "status": "success",
            "events": events,
            "total": len(events),
            "filters": {
                "threat_level": threat_level.value if threat_level else None,
                "event_type": event_type.value if event_type else None,
                "hours": hours,
                "limit": limit
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Security events retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security events"
        )

@router.get("/threats/active", dependencies=[Depends(verify_admin_token)])
async def get_active_threats() -> Dict[str, Any]:
    """Get currently active security threats and blocked IPs"""
    try:
        security_monitor = get_security_monitor()
        
        # Get blocked IPs
        blocked_ips = security_monitor._get_active_blocks()
        
        # Get recent critical events (last hour)
        critical_events = []
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for event in security_monitor.event_history:
            if (event.timestamp >= cutoff_time and 
                event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]):
                critical_events.append({
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "threat_level": event.threat_level.value,
                    "timestamp": event.timestamp.isoformat(),
                    "source_ip": event.source_ip,
                    "user_id": event.user_id,
                    "endpoint": event.endpoint
                })
        
        return {
            "status": "success",
            "active_threats": {
                "blocked_ips": blocked_ips,
                "critical_events_1h": critical_events,
                "total_blocked": len(blocked_ips),
                "total_critical_events": len(critical_events)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Active threats retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active threats"
        )

@router.post("/block-ip", dependencies=[Depends(verify_admin_token)])
async def manual_ip_block(
    ip_address: str,
    duration: int = 3600,
    reason: str = "manual_admin_block"
) -> Dict[str, Any]:
    """
    Manually block an IP address
    
    Args:
        ip_address: IP address to block
        duration: Block duration in seconds (default 1 hour)
        reason: Reason for blocking
    """
    try:
        security_monitor = get_security_monitor()
        
        # Validate IP address format
        import ipaddress
        try:
            ipaddress.ip_address(ip_address)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid IP address format"
            )
        
        # Block the IP
        await security_monitor._temporary_ip_block(ip_address, duration)
        
        logger.info(f"Admin manually blocked IP {ip_address} for {duration} seconds")
        
        return {
            "status": "success",
            "message": f"IP {ip_address} blocked successfully",
            "ip_address": ip_address,
            "duration": duration,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual IP block error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to block IP address"
        )

@router.delete("/unblock-ip", dependencies=[Depends(verify_admin_token)])
async def manual_ip_unblock(ip_address: str) -> Dict[str, Any]:
    """
    Manually unblock an IP address
    
    Args:
        ip_address: IP address to unblock
    """
    try:
        security_monitor = get_security_monitor()
        
        # Remove block
        block_key = f"ip_blocked:{ip_address}"
        result = security_monitor.redis.delete(block_key)
        
        if result:
            logger.info(f"Admin manually unblocked IP {ip_address}")
            return {
                "status": "success",
                "message": f"IP {ip_address} unblocked successfully",
                "ip_address": ip_address,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "info",
                "message": f"IP {ip_address} was not blocked",
                "ip_address": ip_address,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Manual IP unblock error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unblock IP address"
        )

@router.get("/audit-log", dependencies=[Depends(verify_admin_token)])
async def get_audit_log(
    lines: int = 1000,
    search: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get security audit log entries
    
    Args:
        lines: Number of recent lines to retrieve
        search: Search term to filter log entries
    """
    try:
        import aiofiles
        import json
        
        log_entries = []
        
        try:
            async with aiofiles.open("logs/security_audit.log", "r") as f:
                # Read all lines
                all_lines = await f.readlines()
                
                # Get recent lines
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                for line in recent_lines:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Apply search filter if provided
                        if search:
                            line_str = line.lower()
                            if search.lower() not in line_str:
                                continue
                        
                        log_entries.append(entry)
                    except json.JSONDecodeError:
                        continue
                        
        except FileNotFoundError:
            logger.warning("Security audit log file not found")
            log_entries = []
        
        return {
            "status": "success",
            "audit_log": log_entries,
            "total_entries": len(log_entries),
            "search_term": search,
            "lines_requested": lines,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Audit log retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit log"
        )

@router.get("/config", dependencies=[Depends(verify_admin_token)])
async def get_security_config() -> Dict[str, Any]:
    """Get current security configuration"""
    try:
        security_monitor = get_security_monitor()
        jwt_manager = get_jwt_manager()
        
        config = {
            "monitoring": {
                "active": security_monitor.monitoring_active,
                "threat_patterns": len(security_monitor.threat_patterns),
                "event_history_size": len(security_monitor.event_history),
                "max_event_history": security_monitor.event_history.maxlen
            },
            "jwt": {
                "token_expiry": jwt_manager.token_expiry,
                "refresh_token_expiry": jwt_manager.refresh_token_expiry,
                "issuer": jwt_manager.issuer,
                "audience": jwt_manager.audience
            },
            "session_security": {
                "fingerprint_validation": True,
                "session_rotation": True,
                "max_session_duration": 28800,  # 8 hours
                "concurrent_session_limit": 3
            },
            "features": {
                "real_time_monitoring": True,
                "automatic_threat_response": True,
                "session_hijacking_protection": True,
                "token_blacklisting": True,
                "rate_limiting": True,
                "audit_logging": True
            }
        }
        
        return {
            "status": "success",
            "security_config": config,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Security config retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security configuration"
        )
