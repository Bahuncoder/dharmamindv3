"""
📊 Enhanced Performance Dashboard API Routes

Comprehensive performance monitoring, metrics visualization, and system analytics
for enterprise-grade performance management.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.security import HTTPBearer
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.monitoring.performance_monitor import get_metrics_collector, get_performance_profiler
from app.services.advanced_llm_router import get_advanced_llm_router
from app.services.intelligent_cache import get_intelligent_cache
from app.db.advanced_pool import get_db_pool_manager
from app.middleware.security import verify_admin_token
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/performance", tags=["performance-dashboard"])
security = HTTPBearer()

@router.get("/dashboard", dependencies=[Depends(verify_admin_token)])
async def get_performance_dashboard(
    time_range: int = Query(3600, description="Time range in seconds")
) -> Dict[str, Any]:
    """
    Get comprehensive performance dashboard data
    
    Args:
        time_range: Time range for metrics in seconds (default: 1 hour)
    
    Returns comprehensive performance metrics and system health data
    """
    try:
        metrics_collector = get_metrics_collector()
        
        # Get metrics summary
        metrics_summary = await metrics_collector.get_metrics_summary(time_range)
        
        # Get LLM router statistics
        llm_stats = {}
        try:
            llm_router = get_advanced_llm_router()
            llm_stats = await llm_router.get_router_stats()
        except RuntimeError:
            llm_stats = {"status": "not_initialized"}
        
        # Get cache statistics
        cache_stats = {}
        try:
            cache = get_intelligent_cache()
            cache_stats = await cache.get_cache_stats()
        except RuntimeError:
            cache_stats = {"status": "not_initialized"}
        
        # Get database pool statistics
        db_stats = {}
        try:
            db_pool_mgr = get_db_pool_manager()
            db_stats = await db_pool_mgr.get_all_pool_stats()
        except RuntimeError:
            db_stats = {"status": "not_initialized"}
        
        dashboard_data = {
            "system_metrics": metrics_summary,
            "llm_router": llm_stats,
            "caching": cache_stats,
            "database": db_stats,
            "performance_score": await _calculate_performance_score(metrics_summary),
            "recommendations": await _generate_performance_recommendations(
                metrics_summary, llm_stats, cache_stats, db_stats
            ),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return {
            "status": "success",
            "data": dashboard_data,
            "time_range": time_range
        }
        
    except Exception as e:
        logger.error(f"Performance dashboard error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance dashboard data"
        )

@router.get("/metrics", dependencies=[Depends(verify_admin_token)])
async def get_performance_metrics(
    metric_names: Optional[str] = Query(None, description="Comma-separated metric names"),
    time_range: int = Query(3600, description="Time range in seconds"),
    aggregation: str = Query("avg", description="Aggregation method: avg, min, max, sum")
) -> Dict[str, Any]:
    """
    Get specific performance metrics with filtering and aggregation
    
    Args:
        metric_names: Comma-separated list of metric names to retrieve
        time_range: Time range for metrics in seconds
        aggregation: Aggregation method for metric values
    """
    try:
        metrics_collector = get_metrics_collector()
        
        # Get all metrics
        all_metrics = await metrics_collector.get_metrics_summary(time_range)
        
        # Filter by metric names if specified
        if metric_names:
            requested_metrics = [name.strip() for name in metric_names.split(',')]
            filtered_metrics = {
                name: data for name, data in all_metrics.get("metrics", {}).items()
                if name in requested_metrics
            }
            all_metrics["metrics"] = filtered_metrics
        
        # Apply aggregation
        aggregated_metrics = {}
        for metric_name, metric_data in all_metrics.get("metrics", {}).items():
            if aggregation == "avg":
                aggregated_metrics[metric_name] = metric_data.get("avg", 0)
            elif aggregation == "min":
                aggregated_metrics[metric_name] = metric_data.get("min", 0)
            elif aggregation == "max":
                aggregated_metrics[metric_name] = metric_data.get("max", 0)
            elif aggregation == "sum":
                aggregated_metrics[metric_name] = metric_data.get("total", 0)
            else:
                aggregated_metrics[metric_name] = metric_data
        
        return {
            "status": "success",
            "metrics": aggregated_metrics,
            "aggregation": aggregation,
            "time_range": time_range,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )

@router.get("/alerts", dependencies=[Depends(verify_admin_token)])
async def get_performance_alerts(
    limit: int = Query(100, description="Maximum number of alerts to return"),
    level: Optional[str] = Query(None, description="Filter by alert level")
) -> Dict[str, Any]:
    """
    Get recent performance alerts
    
    Args:
        limit: Maximum number of alerts to return
        level: Filter by alert level (info, warning, error, critical)
    """
    try:
        metrics_collector = get_metrics_collector()
        
        # Get recent alerts (last 24 hours)
        alerts = await metrics_collector._get_recent_alerts(86400)
        
        # Filter by level if specified
        if level:
            alerts = [alert for alert in alerts if alert.get("level") == level.lower()]
        
        # Limit results
        alerts = alerts[:limit]
        
        return {
            "status": "success",
            "alerts": alerts,
            "total": len(alerts),
            "filters": {"level": level, "limit": limit},
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance alerts error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance alerts"
        )

@router.get("/system-health", dependencies=[Depends(verify_admin_token)])
async def get_system_health() -> Dict[str, Any]:
    """Get real-time system health metrics"""
    try:
        metrics_collector = get_metrics_collector()
        
        system_health = await metrics_collector._get_current_system_health()
        
        # Calculate health score
        health_score = 100.0
        
        # Deduct points for high resource usage
        if system_health.get("cpu_percent", 0) > 80:
            health_score -= 20
        elif system_health.get("cpu_percent", 0) > 60:
            health_score -= 10
        
        if system_health.get("memory_percent", 0) > 85:
            health_score -= 20
        elif system_health.get("memory_percent", 0) > 70:
            health_score -= 10
        
        if system_health.get("disk_percent", 0) > 90:
            health_score -= 25
        elif system_health.get("disk_percent", 0) > 80:
            health_score -= 15
        
        # Add health classification
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 75:
            health_status = "good"
        elif health_score >= 60:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            "status": "success",
            "health": {
                **system_health,
                "health_score": max(0, health_score),
                "health_status": health_status
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System health error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system health"
        )

@router.get("/llm-performance", dependencies=[Depends(verify_admin_token)])
async def get_llm_performance() -> Dict[str, Any]:
    """Get LLM router performance statistics"""
    try:
        llm_router = get_advanced_llm_router()
        stats = await llm_router.get_router_stats()
        
        # Calculate performance insights
        insights = []
        
        for endpoint in stats.get("endpoints", []):
            if endpoint["requests"] > 0:
                if endpoint["success_rate"] < 0.95:
                    insights.append({
                        "type": "warning",
                        "message": f"Low success rate for {endpoint['provider']}: {endpoint['success_rate']:.2%}"
                    })
                
                if endpoint["average_latency"] > 5.0:
                    insights.append({
                        "type": "warning", 
                        "message": f"High latency for {endpoint['provider']}: {endpoint['average_latency']:.2f}s"
                    })
        
        return {
            "status": "success",
            "llm_stats": stats,
            "insights": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except RuntimeError:
        return {
            "status": "error",
            "message": "LLM router not initialized",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"LLM performance error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve LLM performance data"
        )

@router.get("/cache-performance", dependencies=[Depends(verify_admin_token)])
async def get_cache_performance() -> Dict[str, Any]:
    """Get cache performance statistics"""
    try:
        cache = get_intelligent_cache()
        stats = await cache.get_cache_stats()
        
        # Calculate cache insights
        insights = []
        metrics = stats.get("metrics", {})
        
        hit_rate = metrics.get("hit_rate", 0)
        if hit_rate < 0.6:
            insights.append({
                "type": "warning",
                "message": f"Low cache hit rate: {hit_rate:.2%} (target: >60%)"
            })
        elif hit_rate > 0.8:
            insights.append({
                "type": "success",
                "message": f"Excellent cache hit rate: {hit_rate:.2%}"
            })
        
        evictions = metrics.get("evictions", 0)
        if evictions > 100:
            insights.append({
                "type": "warning",
                "message": f"High cache evictions: {evictions} (consider increasing cache size)"
            })
        
        return {
            "status": "success",
            "cache_stats": stats,
            "insights": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except RuntimeError:
        return {
            "status": "error",
            "message": "Intelligent cache not initialized",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache performance error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache performance data"
        )

@router.get("/database-performance", dependencies=[Depends(verify_admin_token)])
async def get_database_performance() -> Dict[str, Any]:
    """Get database performance statistics"""
    try:
        db_pool_mgr = get_db_pool_manager()
        stats = await db_pool_mgr.get_all_pool_stats()
        
        # Calculate database insights
        insights = []
        
        for pool_name, pool_stats in stats.get("pools", {}).items():
            utilization = pool_stats.get("checked_out", 0) / max(pool_stats.get("pool_size", 1), 1)
            
            if utilization > 0.9:
                insights.append({
                    "type": "warning",
                    "message": f"High connection utilization in pool {pool_name}: {utilization:.2%}"
                })
            
            if pool_stats.get("health_status") != "healthy":
                insights.append({
                    "type": "error",
                    "message": f"Pool {pool_name} health issue: {pool_stats.get('health_status')}"
                })
        
        return {
            "status": "success",
            "database_stats": stats,
            "insights": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except RuntimeError:
        return {
            "status": "error",
            "message": "Database pool manager not initialized",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Database performance error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve database performance data"
        )

async def _calculate_performance_score(metrics_summary: Dict[str, Any]) -> float:
    """Calculate overall performance score"""
    
    score = 100.0
    system_health = metrics_summary.get("system_health", {})
    
    # System resource penalties
    cpu_percent = system_health.get("cpu_percent", 0)
    memory_percent = system_health.get("memory_percent", 0)
    
    if cpu_percent > 80:
        score -= 15
    elif cpu_percent > 60:
        score -= 8
    
    if memory_percent > 85:
        score -= 15
    elif memory_percent > 70:
        score -= 8
    
    # Response time penalties
    avg_response_time = 0
    metrics = metrics_summary.get("metrics", {})
    for metric_name, metric_data in metrics.items():
        if "request.duration" in metric_name:
            avg_response_time = metric_data.get("avg", 0)
            break
    
    if avg_response_time > 2.0:
        score -= 20
    elif avg_response_time > 1.0:
        score -= 10
    elif avg_response_time > 0.5:
        score -= 5
    
    return max(0, min(100, score))

async def _generate_performance_recommendations(
    metrics_summary: Dict[str, Any],
    llm_stats: Dict[str, Any],
    cache_stats: Dict[str, Any],
    db_stats: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Generate performance improvement recommendations"""
    
    recommendations = []
    
    # System resource recommendations
    system_health = metrics_summary.get("system_health", {})
    
    if system_health.get("cpu_percent", 0) > 80:
        recommendations.append({
            "type": "critical",
            "area": "system",
            "message": "High CPU usage detected. Consider scaling horizontally or optimizing CPU-intensive operations."
        })
    
    if system_health.get("memory_percent", 0) > 85:
        recommendations.append({
            "type": "critical",
            "area": "system",
            "message": "High memory usage detected. Consider increasing available memory or optimizing memory-intensive operations."
        })
    
    # Cache recommendations
    cache_metrics = cache_stats.get("metrics", {})
    hit_rate = cache_metrics.get("hit_rate", 0)
    
    if hit_rate < 0.6:
        recommendations.append({
            "type": "optimization",
            "area": "caching",
            "message": f"Low cache hit rate ({hit_rate:.2%}). Consider reviewing cache TTL settings and cache key strategies."
        })
    
    # Database recommendations
    for pool_name, pool_stats in db_stats.get("pools", {}).items():
        utilization = pool_stats.get("checked_out", 0) / max(pool_stats.get("pool_size", 1), 1)
        
        if utilization > 0.8:
            recommendations.append({
                "type": "scaling",
                "area": "database",
                "message": f"High database connection utilization in pool {pool_name}. Consider increasing pool size."
            })
    
    # LLM recommendations
    for endpoint in llm_stats.get("endpoints", []):
        if endpoint.get("success_rate", 1.0) < 0.95:
            recommendations.append({
                "type": "reliability",
                "area": "llm",
                "message": f"Low success rate for LLM provider {endpoint.get('provider')}. Check provider status and failover configuration."
            })
    
    return recommendations
