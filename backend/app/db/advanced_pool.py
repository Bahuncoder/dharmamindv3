"""
🗄️ High-Performance Database Connection Pool

Advanced database connection management with pooling, health monitoring,
and performance optimization for enterprise scalability.
"""

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import aiosqlite
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy import text, event
from sqlalchemy.engine import Engine
import redis

logger = logging.getLogger(__name__)

class ConnectionStatus(str, Enum):
    """Connection status types"""
    ACTIVE = "active"
    IDLE = "idle"
    TESTING = "testing"
    FAILED = "failed"
    TERMINATED = "terminated"

@dataclass
class ConnectionMetrics:
    """Connection performance metrics"""
    connection_id: str
    created_at: datetime
    last_used: datetime
    total_queries: int = 0
    total_query_time: float = 0.0
    errors: int = 0
    status: ConnectionStatus = ConnectionStatus.IDLE
    current_transaction: Optional[str] = None

@dataclass
class PoolMetrics:
    """Database pool performance metrics"""
    pool_size: int
    active_connections: int
    idle_connections: int
    failed_connections: int
    total_connections_created: int
    total_queries_executed: int
    average_query_time: float
    peak_active_connections: int
    last_health_check: datetime
    health_status: str = "healthy"

class DatabaseHealthMonitor:
    """Database health monitoring and alerting"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.health_check_interval = 60  # seconds
        self.metrics_retention = 86400  # 24 hours
        self.is_monitoring = False
        
    async def start_monitoring(self, pool_manager):
        """Start health monitoring background task"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        asyncio.create_task(self._monitor_loop(pool_manager))
        logger.info("Database health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        logger.info("Database health monitoring stopped")
    
    async def _monitor_loop(self, pool_manager):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Perform health checks
                health_status = await self._perform_health_check(pool_manager)
                
                # Store metrics
                await self._store_health_metrics(health_status)
                
                # Check for alerts
                await self._check_health_alerts(health_status)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def _perform_health_check(self, pool_manager) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        
        health_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "pools": {},
            "alerts": []
        }
        
        for pool_name, pool_info in pool_manager.pools.items():
            try:
                pool_health = await self._check_pool_health(pool_info)
                health_data["pools"][pool_name] = pool_health
                
                # Check for issues
                if pool_health["error_rate"] > 0.1:  # 10% error rate
                    health_data["alerts"].append({
                        "level": "warning",
                        "message": f"High error rate in pool {pool_name}: {pool_health['error_rate']:.2%}"
                    })
                
                if pool_health["active_connections"] / pool_health["pool_size"] > 0.9:  # 90% utilization
                    health_data["alerts"].append({
                        "level": "warning", 
                        "message": f"High connection utilization in pool {pool_name}: {pool_health['utilization']:.2%}"
                    })
                    
            except Exception as e:
                health_data["pools"][pool_name] = {"status": "failed", "error": str(e)}
                health_data["alerts"].append({
                    "level": "error",
                    "message": f"Health check failed for pool {pool_name}: {e}"
                })
        
        # Set overall status
        if any(alert["level"] == "error" for alert in health_data["alerts"]):
            health_data["overall_status"] = "unhealthy"
        elif health_data["alerts"]:
            health_data["overall_status"] = "degraded"
        
        return health_data
    
    async def _check_pool_health(self, pool_info) -> Dict[str, Any]:
        """Check individual pool health"""
        
        pool = pool_info["engine"].pool
        
        # Test connection
        start_time = time.time()
        try:
            async with pool_info["session_factory"]() as session:
                await session.execute(text("SELECT 1"))
            
            connection_test_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "active_connections": pool.checkedout(),
                "utilization": pool.checkedout() / pool.size() if pool.size() > 0 else 0,
                "connection_test_time": connection_test_time,
                "error_rate": 0.0  # Would be calculated from metrics
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "connection_test_time": time.time() - start_time
            }
    
    async def _store_health_metrics(self, health_data: Dict[str, Any]):
        """Store health metrics in Redis"""
        try:
            metrics_key = f"db_health:{int(time.time())}"
            self.redis.setex(metrics_key, self.metrics_retention, json.dumps(health_data))
            
            # Keep only recent metrics
            current_time = int(time.time())
            cutoff_time = current_time - self.metrics_retention
            
            # Clean up old metrics
            pattern = "db_health:*"
            for key in self.redis.scan_iter(match=pattern):
                try:
                    timestamp = int(key.decode().split(':')[1])
                    if timestamp < cutoff_time:
                        self.redis.delete(key)
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Error storing health metrics: {e}")
    
    async def _check_health_alerts(self, health_data: Dict[str, Any]):
        """Check for health alerts and notifications"""
        
        alerts = health_data.get("alerts", [])
        if not alerts:
            return
        
        # Log alerts
        for alert in alerts:
            if alert["level"] == "error":
                logger.error(f"Database Health Alert: {alert['message']}")
            elif alert["level"] == "warning":
                logger.warning(f"Database Health Warning: {alert['message']}")
        
        # Store alerts for dashboard
        alert_key = f"db_alerts:{int(time.time())}"
        self.redis.setex(alert_key, 3600, json.dumps(alerts))  # Keep alerts for 1 hour

class AdvancedConnectionPool:
    """Advanced database connection pool with monitoring"""
    
    def __init__(
        self,
        database_url: str,
        pool_name: str = "default",
        min_size: int = 5,
        max_size: int = 20,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True
    ):
        self.database_url = database_url
        self.pool_name = pool_name
        self.min_size = min_size
        self.max_size = max_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        
        self.engine = None
        self.session_factory = None
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        self.pool_metrics = PoolMetrics(
            pool_size=max_size,
            active_connections=0,
            idle_connections=0,
            failed_connections=0,
            total_connections_created=0,
            total_queries_executed=0,
            average_query_time=0.0,
            peak_active_connections=0,
            last_health_check=datetime.utcnow()
        )
    
    async def initialize(self):
        """Initialize the connection pool"""
        
        # Create async engine with advanced pooling
        self.engine = create_async_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=self.max_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            pool_pre_ping=self.pool_pre_ping,
            echo=False,  # Set to True for SQL query logging
            future=True
        )
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Set up event listeners for monitoring
        self._setup_event_listeners()
        
        logger.info(f"Database pool '{self.pool_name}' initialized with {self.max_size} max connections")
    
    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for monitoring"""
        
        @event.listens_for(self.engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle new connection creation"""
            connection_id = f"{self.pool_name}_{id(dbapi_connection)}"
            self.connection_metrics[connection_id] = ConnectionMetrics(
                connection_id=connection_id,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                status=ConnectionStatus.ACTIVE
            )
            self.pool_metrics.total_connections_created += 1
            logger.debug(f"New database connection created: {connection_id}")
        
        @event.listens_for(self.engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool"""
            connection_id = f"{self.pool_name}_{id(dbapi_connection)}"
            if connection_id in self.connection_metrics:
                self.connection_metrics[connection_id].status = ConnectionStatus.ACTIVE
                self.connection_metrics[connection_id].last_used = datetime.utcnow()
            
            self.pool_metrics.active_connections += 1
            self.pool_metrics.peak_active_connections = max(
                self.pool_metrics.peak_active_connections,
                self.pool_metrics.active_connections
            )
        
        @event.listens_for(self.engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Handle connection checkin to pool"""
            connection_id = f"{self.pool_name}_{id(dbapi_connection)}"
            if connection_id in self.connection_metrics:
                self.connection_metrics[connection_id].status = ConnectionStatus.IDLE
            
            self.pool_metrics.active_connections = max(0, self.pool_metrics.active_connections - 1)
        
        @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
        def on_before_execute(conn, cursor, statement, parameters, context, executemany):
            """Handle query execution start"""
            context._query_start_time = time.time()
        
        @event.listens_for(self.engine.sync_engine, "after_cursor_execute") 
        def on_after_execute(conn, cursor, statement, parameters, context, executemany):
            """Handle query execution completion"""
            if hasattr(context, '_query_start_time'):
                query_time = time.time() - context._query_start_time
                
                # Update pool metrics
                self.pool_metrics.total_queries_executed += 1
                total_time = (self.pool_metrics.average_query_time * 
                            (self.pool_metrics.total_queries_executed - 1) + query_time)
                self.pool_metrics.average_query_time = total_time / self.pool_metrics.total_queries_executed
                
                # Update connection metrics
                connection_id = f"{self.pool_name}_{id(conn.connection)}"
                if connection_id in self.connection_metrics:
                    metrics = self.connection_metrics[connection_id]
                    metrics.total_queries += 1
                    metrics.total_query_time += query_time
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        
        if not self.session_factory:
            raise RuntimeError(f"Pool '{self.pool_name}' not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error in pool '{self.pool_name}': {e}")
                raise
            finally:
                await session.close()
    
    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> Any:
        """Execute a query with connection pooling"""
        
        async with self.get_session() as session:
            result = await session.execute(text(query), parameters or {})
            await session.commit()
            return result
    
    async def health_check(self) -> bool:
        """Perform health check on the pool"""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            
            self.pool_metrics.last_health_check = datetime.utcnow()
            self.pool_metrics.health_status = "healthy"
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for pool '{self.pool_name}': {e}")
            self.pool_metrics.health_status = f"unhealthy: {e}"
            return False
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool performance statistics"""
        
        pool = self.engine.pool if self.engine else None
        
        stats = {
            "pool_name": self.pool_name,
            "pool_size": pool.size() if pool else 0,
            "checked_out": pool.checkedout() if pool else 0,
            "checked_in": pool.checkedin() if pool else 0,
            "overflow": pool.overflow() if pool else 0,
            "total_connections_created": self.pool_metrics.total_connections_created,
            "total_queries_executed": self.pool_metrics.total_queries_executed,
            "average_query_time": self.pool_metrics.average_query_time,
            "peak_active_connections": self.pool_metrics.peak_active_connections,
            "health_status": self.pool_metrics.health_status,
            "last_health_check": self.pool_metrics.last_health_check.isoformat(),
            "active_connection_details": []
        }
        
        # Add active connection details
        for conn_id, metrics in self.connection_metrics.items():
            if metrics.status == ConnectionStatus.ACTIVE:
                stats["active_connection_details"].append({
                    "connection_id": conn_id,
                    "created_at": metrics.created_at.isoformat(),
                    "last_used": metrics.last_used.isoformat(),
                    "total_queries": metrics.total_queries,
                    "average_query_time": metrics.total_query_time / max(metrics.total_queries, 1)
                })
        
        return stats
    
    async def cleanup(self):
        """Clean up pool resources"""
        if self.engine:
            await self.engine.dispose()
            logger.info(f"Database pool '{self.pool_name}' cleaned up")

class DatabasePoolManager:
    """Manages multiple database connection pools"""
    
    def __init__(self, redis_client: redis.Redis):
        self.pools: Dict[str, Dict[str, Any]] = {}
        self.health_monitor = DatabaseHealthMonitor(redis_client)
        self.default_pool_name = "default"
    
    async def add_pool(
        self,
        name: str,
        database_url: str,
        min_size: int = 5,
        max_size: int = 20,
        **kwargs
    ) -> AdvancedConnectionPool:
        """Add a new connection pool"""
        
        if name in self.pools:
            raise ValueError(f"Pool '{name}' already exists")
        
        pool = AdvancedConnectionPool(
            database_url=database_url,
            pool_name=name,
            min_size=min_size,
            max_size=max_size,
            **kwargs
        )
        
        await pool.initialize()
        
        # Create session factory
        session_factory = pool.session_factory
        
        self.pools[name] = {
            "pool": pool,
            "engine": pool.engine,
            "session_factory": session_factory
        }
        
        logger.info(f"Added database pool: {name}")
        return pool
    
    async def get_pool(self, name: str = None) -> AdvancedConnectionPool:
        """Get a connection pool by name"""
        pool_name = name or self.default_pool_name
        
        if pool_name not in self.pools:
            raise ValueError(f"Pool '{pool_name}' not found")
        
        return self.pools[pool_name]["pool"]
    
    @asynccontextmanager
    async def get_session(self, pool_name: str = None) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session from specified pool"""
        pool = await self.get_pool(pool_name)
        async with pool.get_session() as session:
            yield session
    
    async def start_monitoring(self):
        """Start health monitoring for all pools"""
        await self.health_monitor.start_monitoring(self)
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        await self.health_monitor.stop_monitoring()
    
    async def get_all_pool_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        
        all_stats = {
            "pools": {},
            "summary": {
                "total_pools": len(self.pools),
                "total_connections": 0,
                "total_queries": 0,
                "healthy_pools": 0
            }
        }
        
        for name, pool_info in self.pools.items():
            pool_stats = await pool_info["pool"].get_pool_stats()
            all_stats["pools"][name] = pool_stats
            
            # Update summary
            all_stats["summary"]["total_connections"] += pool_stats["checked_out"]
            all_stats["summary"]["total_queries"] += pool_stats["total_queries_executed"]
            
            if pool_stats["health_status"] == "healthy":
                all_stats["summary"]["healthy_pools"] += 1
        
        return all_stats
    
    async def cleanup_all(self):
        """Clean up all pools"""
        await self.health_monitor.stop_monitoring()
        
        for name, pool_info in self.pools.items():
            await pool_info["pool"].cleanup()
        
        self.pools.clear()
        logger.info("All database pools cleaned up")

# Global pool manager instance
db_pool_manager: Optional[DatabasePoolManager] = None

def get_db_pool_manager() -> DatabasePoolManager:
    """Get the global database pool manager"""
    if db_pool_manager is None:
        raise RuntimeError("Database pool manager not initialized")
    return db_pool_manager

def init_db_pool_manager(redis_client: redis.Redis) -> DatabasePoolManager:
    """Initialize the global database pool manager"""
    global db_pool_manager
    db_pool_manager = DatabasePoolManager(redis_client)
    return db_pool_manager
