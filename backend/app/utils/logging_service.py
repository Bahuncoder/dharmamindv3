"""
🕉️ DharmaMind Production Logging System

Structured logging system for production debugging and analytics:
- Multi-level logging with dharmic context
- Performance tracing and profiling
- Security event logging
- User interaction analytics
- Error tracking and debugging
- Dharmic wisdom analytics

May our logs guide us toward digital enlightenment 📝✨
"""

import logging
import json
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import sys
import os

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    logging.warning("structlog not installed - using standard logging")

from ..config import settings
from ..db.database import DatabaseManager

# Context variables for request tracking
request_id_context: ContextVar[str] = ContextVar('request_id', default='')
user_id_context: ContextVar[str] = ContextVar('user_id', default='')
session_id_context: ContextVar[str] = ContextVar('session_id', default='')


class LogLevel(str, Enum):
    """Enhanced log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    DHARMIC = "dharmic"  # Special level for dharmic insights
    PERFORMANCE = "performance"
    ANALYTICS = "analytics"


class EventType(str, Enum):
    """Types of events to log"""
    USER_ACTION = "user_action"
    API_REQUEST = "api_request"
    DATABASE_QUERY = "database_query"
    CACHE_OPERATION = "cache_operation"
    AI_INFERENCE = "ai_inference"
    DHARMIC_INSIGHT = "dharmic_insight"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"
    SYSTEM_EVENT = "system_event"


@dataclass
class LogContext:
    """Log context information"""
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    function_name: Optional[str] = None
    dharmic_context: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    start_time: float
    end_time: float
    duration: float
    memory_start: int
    memory_end: int
    memory_delta: int
    cpu_percent: Optional[float] = None


class DharmicLogger:
    """🚀 Enhanced Production Logging System"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize dharmic logging system"""
        
        self.db = db_manager
        self.performance_trackers: Dict[str, PerformanceMetrics] = {}
        
        # Logging configuration
        self.config = {
            'max_log_size': 100 * 1024 * 1024,  # 100MB
            'backup_count': 10,
            'log_retention_days': 30,
            'async_logging': True,
            'include_stack_info': True,
            'performance_threshold': 1.0,  # Log slow operations > 1s
            'dharmic_insights_enabled': True,
            'analytics_sampling_rate': 0.1,  # 10% sampling for analytics
        }
        
        # Setup logging infrastructure
        self._setup_logging()
        self._setup_log_database()
        
        # Event handlers
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        
        # Performance tracking
        self.slow_operations: List[Dict[str, Any]] = []
        
        self.logger = self._get_logger("dharmamind")
        self.logger.info("🔱 Dharmic Logging System initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(logs_dir / "dharmamind.log")
            ]
        )
        
        # Setup structured logging if available
        if HAS_STRUCTLOG:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    self._add_dharmic_context,
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        
        # Setup specialized loggers
        self._setup_specialized_loggers(logs_dir)
    
    def _setup_specialized_loggers(self, logs_dir: Path):
        """Setup specialized loggers for different purposes"""
        
        # Security logger
        security_logger = logging.getLogger("dharmamind.security")
        security_handler = logging.FileHandler(logs_dir / "security.log")
        security_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
        )
        security_logger.addHandler(security_handler)
        security_logger.setLevel(logging.WARNING)
        
        # Performance logger
        performance_logger = logging.getLogger("dharmamind.performance")
        performance_handler = logging.FileHandler(logs_dir / "performance.log")
        performance_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - PERFORMANCE - %(message)s'
            )
        )
        performance_logger.addHandler(performance_handler)
        performance_logger.setLevel(logging.INFO)
        
        # Dharmic insights logger
        dharmic_logger = logging.getLogger("dharmamind.dharmic")
        dharmic_handler = logging.FileHandler(logs_dir / "dharmic_insights.log")
        dharmic_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - DHARMIC - %(message)s'
            )
        )
        dharmic_logger.addHandler(dharmic_handler)
        dharmic_logger.setLevel(logging.INFO)
        
        # Analytics logger
        analytics_logger = logging.getLogger("dharmamind.analytics")
        analytics_handler = logging.FileHandler(logs_dir / "analytics.log")
        analytics_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - ANALYTICS - %(message)s'
            )
        )
        analytics_logger.addHandler(analytics_handler)
        analytics_logger.setLevel(logging.INFO)
        
        # API access logger
        api_logger = logging.getLogger("dharmamind.api")
        api_handler = logging.FileHandler(logs_dir / "api_access.log")
        api_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - API - %(message)s'
            )
        )
        api_logger.addHandler(api_handler)
        api_logger.setLevel(logging.INFO)
    
    async def _setup_log_database(self):
        """Setup database tables for log storage"""
        
        if not self.db:
            return
        
        try:
            # Application logs table
            app_logs_table = """
                CREATE TABLE IF NOT EXISTS application_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    level VARCHAR(20) NOT NULL,
                    logger_name VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    request_id VARCHAR(255),
                    user_id VARCHAR(255),
                    session_id VARCHAR(255),
                    component VARCHAR(100),
                    function_name VARCHAR(255),
                    event_type VARCHAR(50),
                    context_data JSONB DEFAULT '{}',
                    stack_trace TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            
            # Performance logs table
            performance_logs_table = """
                CREATE TABLE IF NOT EXISTS performance_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    operation_name VARCHAR(255) NOT NULL,
                    duration_ms FLOAT NOT NULL,
                    memory_delta_mb FLOAT,
                    cpu_percent FLOAT,
                    component VARCHAR(100),
                    request_id VARCHAR(255),
                    user_id VARCHAR(255),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            
            # Security events table
            security_events_table = """
                CREATE TABLE IF NOT EXISTS security_events (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    event_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    user_id VARCHAR(255),
                    ip_address INET,
                    user_agent TEXT,
                    endpoint VARCHAR(255),
                    details JSONB DEFAULT '{}',
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            
            # Dharmic insights table
            dharmic_insights_table = """
                CREATE TABLE IF NOT EXISTS dharmic_insights (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    insight_type VARCHAR(100) NOT NULL,
                    wisdom_level FLOAT,
                    compassion_score FLOAT,
                    user_id VARCHAR(255),
                    session_id VARCHAR(255),
                    query_text TEXT,
                    response_text TEXT,
                    dharmic_principles JSONB DEFAULT '{}',
                    user_impact JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            
            # Analytics events table
            analytics_events_table = """
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    event_name VARCHAR(255) NOT NULL,
                    event_category VARCHAR(100),
                    user_id VARCHAR(255),
                    session_id VARCHAR(255),
                    properties JSONB DEFAULT '{}',
                    metrics JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            
            # Execute table creation
            await self.db.execute_query(app_logs_table)
            await self.db.execute_query(performance_logs_table)
            await self.db.execute_query(security_events_table)
            await self.db.execute_query(dharmic_insights_table)
            await self.db.execute_query(analytics_events_table)
            
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_app_logs_timestamp ON application_logs(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_app_logs_level ON application_logs(level)",
                "CREATE INDEX IF NOT EXISTS idx_app_logs_request_id ON application_logs(request_id)",
                "CREATE INDEX IF NOT EXISTS idx_app_logs_user_id ON application_logs(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_performance_duration ON performance_logs(duration_ms)",
                "CREATE INDEX IF NOT EXISTS idx_performance_operation ON performance_logs(operation_name)",
                "CREATE INDEX IF NOT EXISTS idx_security_timestamp ON security_events(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_security_event_type ON security_events(event_type)",
                "CREATE INDEX IF NOT EXISTS idx_dharmic_wisdom_level ON dharmic_insights(wisdom_level)",
                "CREATE INDEX IF NOT EXISTS idx_dharmic_user_id ON dharmic_insights(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_analytics_event_name ON analytics_events(event_name)",
                "CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics_events(timestamp)"
            ]
            
            for index_query in indexes:
                await self.db.execute_query(index_query)
            
            logging.info("✅ Logging database tables created")
            
        except Exception as e:
            logging.error(f"❌ Failed to setup logging database: {e}")
    
    def _get_logger(self, name: str):
        """Get logger instance"""
        if HAS_STRUCTLOG:
            return structlog.get_logger(name)
        else:
            return logging.getLogger(name)
    
    def _add_dharmic_context(self, _, __, event_dict):
        """Add dharmic context to log events"""
        
        # Add request context
        event_dict["request_id"] = request_id_context.get()
        event_dict["user_id"] = user_id_context.get()
        event_dict["session_id"] = session_id_context.get()
        
        # Add timestamp if not present
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return event_dict
    
    def set_context(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Set logging context variables"""
        
        if request_id:
            request_id_context.set(request_id)
        if user_id:
            user_id_context.set(user_id)
        if session_id:
            session_id_context.set(session_id)
    
    def generate_request_id(self) -> str:
        """Generate unique request ID"""
        return str(uuid.uuid4())
    
    async def log_event(
        self,
        level: LogLevel,
        message: str,
        event_type: EventType,
        component: Optional[str] = None,
        function_name: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
        performance_data: Optional[Dict[str, Any]] = None,
        dharmic_context: Optional[Dict[str, Any]] = None,
        store_in_db: bool = True
    ):
        """Log event with full context"""
        
        try:
            # Build log context
            log_context = LogContext(
                request_id=request_id_context.get(),
                user_id=user_id_context.get(),
                session_id=session_id_context.get(),
                component=component,
                function_name=function_name,
                dharmic_context=dharmic_context,
                performance_data=performance_data
            )
            
            # Build log entry
            log_entry = {
                "message": message,
                "level": level.value,
                "event_type": event_type.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": asdict(log_context),
                "data": context_data or {}
            }
            
            # Log to appropriate logger
            logger_name = f"dharmamind.{event_type.value}"
            logger = self._get_logger(logger_name)
            
            # Log with appropriate level
            log_func = getattr(logger, level.value, logger.info)
            log_func(message, **log_entry)
            
            # Store in database if enabled and available
            if store_in_db and self.db:
                await self._store_log_in_database(log_entry)
            
            # Trigger event handlers
            await self._trigger_event_handlers(event_type, log_entry)
            
        except Exception as e:
            # Fallback logging - don't let logging failures break the application
            logging.error(f"Logging system error: {e}")
    
    async def _store_log_in_database(self, log_entry: Dict[str, Any]):
        """Store log entry in database"""
        
        try:
            query = """
                INSERT INTO application_logs (
                    level, logger_name, message, request_id, user_id, session_id,
                    component, function_name, event_type, context_data
                ) VALUES (
                    %(level)s, %(logger_name)s, %(message)s, %(request_id)s,
                    %(user_id)s, %(session_id)s, %(component)s, %(function_name)s,
                    %(event_type)s, %(context_data)s
                )
            """
            
            context = log_entry.get("context", {})
            
            await self.db.execute_query(query, {
                'level': log_entry["level"],
                'logger_name': f"dharmamind.{log_entry['event_type']}",
                'message': log_entry["message"],
                'request_id': context.get("request_id"),
                'user_id': context.get("user_id"),
                'session_id': context.get("session_id"),
                'component': context.get("component"),
                'function_name': context.get("function_name"),
                'event_type': log_entry["event_type"],
                'context_data': {**context, **log_entry.get("data", {})}
            })
            
        except Exception as e:
            logging.error(f"Failed to store log in database: {e}")
    
    async def log_performance(
        self,
        operation_name: str,
        duration: float,
        memory_delta: Optional[int] = None,
        cpu_percent: Optional[float] = None,
        component: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log performance metrics"""
        
        try:
            performance_data = {
                'operation_name': operation_name,
                'duration_ms': duration * 1000,  # Convert to milliseconds
                'memory_delta_mb': memory_delta / (1024 * 1024) if memory_delta else None,
                'cpu_percent': cpu_percent,
                'component': component,
                'metadata': metadata or {}
            }
            
            # Log performance event
            await self.log_event(
                level=LogLevel.PERFORMANCE,
                message=f"Performance: {operation_name} took {duration:.3f}s",
                event_type=EventType.PERFORMANCE_METRIC,
                component=component,
                performance_data=performance_data
            )
            
            # Store in performance logs table
            if self.db:
                await self._store_performance_log(performance_data)
            
            # Check if this is a slow operation
            if duration > self.config['performance_threshold']:
                self.slow_operations.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'operation_name': operation_name,
                    'duration': duration,
                    'component': component,
                    'metadata': metadata
                })
                
                # Keep only recent slow operations
                if len(self.slow_operations) > 100:
                    self.slow_operations = self.slow_operations[-100:]
                
                # Log as warning for slow operations
                await self.log_event(
                    level=LogLevel.WARNING,
                    message=f"Slow operation detected: {operation_name} took {duration:.3f}s",
                    event_type=EventType.PERFORMANCE_METRIC,
                    component=component,
                    context_data={'slow_operation': True, **performance_data}
                )
            
        except Exception as e:
            logging.error(f"Performance logging error: {e}")
    
    async def _store_performance_log(self, performance_data: Dict[str, Any]):
        """Store performance log in database"""
        
        try:
            query = """
                INSERT INTO performance_logs (
                    operation_name, duration_ms, memory_delta_mb, cpu_percent,
                    component, request_id, user_id, metadata
                ) VALUES (
                    %(operation_name)s, %(duration_ms)s, %(memory_delta_mb)s,
                    %(cpu_percent)s, %(component)s, %(request_id)s, %(user_id)s,
                    %(metadata)s
                )
            """
            
            await self.db.execute_query(query, {
                'operation_name': performance_data['operation_name'],
                'duration_ms': performance_data['duration_ms'],
                'memory_delta_mb': performance_data['memory_delta_mb'],
                'cpu_percent': performance_data['cpu_percent'],
                'component': performance_data['component'],
                'request_id': request_id_context.get(),
                'user_id': user_id_context.get(),
                'metadata': performance_data['metadata']
            })
            
        except Exception as e:
            logging.error(f"Failed to store performance log: {e}")
    
    async def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        endpoint: Optional[str] = None
    ):
        """Log security event"""
        
        try:
            security_data = {
                'event_type': event_type,
                'severity': severity,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'endpoint': endpoint,
                'details': details,
                'user_id': user_id_context.get()
            }
            
            # Log security event
            await self.log_event(
                level=LogLevel.SECURITY,
                message=f"Security event: {event_type} - {severity}",
                event_type=EventType.SECURITY_EVENT,
                context_data=security_data
            )
            
            # Store in security events table
            if self.db:
                await self._store_security_event(security_data)
            
        except Exception as e:
            logging.error(f"Security logging error: {e}")
    
    async def _store_security_event(self, security_data: Dict[str, Any]):
        """Store security event in database"""
        
        try:
            query = """
                INSERT INTO security_events (
                    event_type, severity, user_id, ip_address, user_agent,
                    endpoint, details
                ) VALUES (
                    %(event_type)s, %(severity)s, %(user_id)s, %(ip_address)s,
                    %(user_agent)s, %(endpoint)s, %(details)s
                )
            """
            
            await self.db.execute_query(query, security_data)
            
        except Exception as e:
            logging.error(f"Failed to store security event: {e}")
    
    async def log_dharmic_insight(
        self,
        insight_type: str,
        wisdom_level: float,
        compassion_score: float,
        query_text: Optional[str] = None,
        response_text: Optional[str] = None,
        dharmic_principles: Optional[Dict[str, Any]] = None,
        user_impact: Optional[Dict[str, Any]] = None
    ):
        """Log dharmic insights and wisdom analytics"""
        
        if not self.config['dharmic_insights_enabled']:
            return
        
        try:
            dharmic_data = {
                'insight_type': insight_type,
                'wisdom_level': wisdom_level,
                'compassion_score': compassion_score,
                'query_text': query_text,
                'response_text': response_text,
                'dharmic_principles': dharmic_principles or {},
                'user_impact': user_impact or {}
            }
            
            # Log dharmic insight
            await self.log_event(
                level=LogLevel.DHARMIC,
                message=f"Dharmic insight: {insight_type} (wisdom: {wisdom_level:.2f}, compassion: {compassion_score:.2f})",
                event_type=EventType.DHARMIC_INSIGHT,
                dharmic_context=dharmic_data
            )
            
            # Store in dharmic insights table
            if self.db:
                await self._store_dharmic_insight(dharmic_data)
            
        except Exception as e:
            logging.error(f"Dharmic insight logging error: {e}")
    
    async def _store_dharmic_insight(self, dharmic_data: Dict[str, Any]):
        """Store dharmic insight in database"""
        
        try:
            query = """
                INSERT INTO dharmic_insights (
                    insight_type, wisdom_level, compassion_score, user_id,
                    session_id, query_text, response_text, dharmic_principles,
                    user_impact
                ) VALUES (
                    %(insight_type)s, %(wisdom_level)s, %(compassion_score)s,
                    %(user_id)s, %(session_id)s, %(query_text)s, %(response_text)s,
                    %(dharmic_principles)s, %(user_impact)s
                )
            """
            
            await self.db.execute_query(query, {
                **dharmic_data,
                'user_id': user_id_context.get(),
                'session_id': session_id_context.get()
            })
            
        except Exception as e:
            logging.error(f"Failed to store dharmic insight: {e}")
    
    async def log_analytics_event(
        self,
        event_name: str,
        event_category: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Log analytics event with sampling"""
        
        # Apply sampling rate
        import random
        if random.random() > self.config['analytics_sampling_rate']:
            return
        
        try:
            analytics_data = {
                'event_name': event_name,
                'event_category': event_category,
                'properties': properties or {},
                'metrics': metrics or {}
            }
            
            # Log analytics event
            await self.log_event(
                level=LogLevel.ANALYTICS,
                message=f"Analytics: {event_name}",
                event_type=EventType.USER_ACTION,
                context_data=analytics_data
            )
            
            # Store in analytics events table
            if self.db:
                await self._store_analytics_event(analytics_data)
            
        except Exception as e:
            logging.error(f"Analytics logging error: {e}")
    
    async def _store_analytics_event(self, analytics_data: Dict[str, Any]):
        """Store analytics event in database"""
        
        try:
            query = """
                INSERT INTO analytics_events (
                    event_name, event_category, user_id, session_id,
                    properties, metrics
                ) VALUES (
                    %(event_name)s, %(event_category)s, %(user_id)s,
                    %(session_id)s, %(properties)s, %(metrics)s
                )
            """
            
            await self.db.execute_query(query, {
                **analytics_data,
                'user_id': user_id_context.get(),
                'session_id': session_id_context.get()
            })
            
        except Exception as e:
            logging.error(f"Failed to store analytics event: {e}")
    
    def start_performance_tracking(self, operation_name: str) -> str:
        """Start performance tracking for an operation"""
        
        tracking_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        
        try:
            import psutil
            process = psutil.Process()
            memory_start = process.memory_info().rss
        except:
            memory_start = 0
        
        self.performance_trackers[tracking_id] = PerformanceMetrics(
            start_time=time.time(),
            end_time=0,
            duration=0,
            memory_start=memory_start,
            memory_end=0,
            memory_delta=0
        )
        
        return tracking_id
    
    async def end_performance_tracking(
        self,
        tracking_id: str,
        component: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """End performance tracking and log results"""
        
        if tracking_id not in self.performance_trackers:
            logging.warning(f"Performance tracking ID not found: {tracking_id}")
            return
        
        try:
            tracker = self.performance_trackers[tracking_id]
            tracker.end_time = time.time()
            tracker.duration = tracker.end_time - tracker.start_time
            
            try:
                import psutil
                process = psutil.Process()
                tracker.memory_end = process.memory_info().rss
                tracker.memory_delta = tracker.memory_end - tracker.memory_start
                tracker.cpu_percent = process.cpu_percent()
            except:
                tracker.memory_end = tracker.memory_start
                tracker.memory_delta = 0
                tracker.cpu_percent = None
            
            # Extract operation name from tracking ID
            operation_name = tracking_id.split('_')[0]
            
            # Log performance
            await self.log_performance(
                operation_name=operation_name,
                duration=tracker.duration,
                memory_delta=tracker.memory_delta,
                cpu_percent=tracker.cpu_percent,
                component=component,
                metadata=metadata
            )
            
            # Cleanup
            del self.performance_trackers[tracking_id]
            
        except Exception as e:
            logging.error(f"Performance tracking error: {e}")
    
    def performance_monitor(self, operation_name: str, component: Optional[str] = None):
        """Decorator for performance monitoring"""
        
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    tracking_id = self.start_performance_tracking(operation_name)
                    try:
                        result = await func(*args, **kwargs)
                        await self.end_performance_tracking(tracking_id, component)
                        return result
                    except Exception as e:
                        await self.end_performance_tracking(tracking_id, component, {'error': str(e)})
                        raise
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    tracking_id = self.start_performance_tracking(operation_name)
                    try:
                        result = func(*args, **kwargs)
                        # Note: Can't await in sync function, so performance tracking won't work perfectly
                        # For sync functions, recommend using async versions where possible
                        return result
                    except Exception as e:
                        raise
                return sync_wrapper
        
        return decorator
    
    def add_event_handler(self, event_type: EventType, handler: Callable):
        """Add event handler for specific event types"""
        
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    async def _trigger_event_handlers(self, event_type: EventType, log_entry: Dict[str, Any]):
        """Trigger registered event handlers"""
        
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(log_entry)
                    else:
                        handler(log_entry)
                except Exception as e:
                    logging.error(f"Event handler error: {e}")
    
    async def get_log_analytics(
        self,
        hours: int = 24,
        component: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get log analytics and insights"""
        
        if not self.db:
            return {"error": "Database not available"}
        
        try:
            # Get log statistics
            stats_query = """
                SELECT 
                    level,
                    event_type,
                    component,
                    COUNT(*) as count,
                    AVG(EXTRACT(EPOCH FROM (created_at - timestamp))) as avg_processing_time
                FROM application_logs 
                WHERE created_at >= NOW() - INTERVAL '%s hours'
                %s
                GROUP BY level, event_type, component
                ORDER BY count DESC
            """
            
            component_filter = "AND component = %(component)s" if component else ""
            query = stats_query % (hours, component_filter)
            
            params = {'component': component} if component else {}
            
            stats_result = await self.db.fetch_all(query, params)
            
            # Get performance analytics
            perf_query = """
                SELECT 
                    operation_name,
                    component,
                    COUNT(*) as executions,
                    AVG(duration_ms) as avg_duration_ms,
                    MAX(duration_ms) as max_duration_ms,
                    MIN(duration_ms) as min_duration_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration_ms
                FROM performance_logs 
                WHERE created_at >= NOW() - INTERVAL '%s hours'
                %s
                GROUP BY operation_name, component
                ORDER BY avg_duration_ms DESC
                LIMIT 20
            """
            
            perf_result = await self.db.fetch_all(perf_query % (hours, component_filter), params)
            
            # Get error analytics
            error_query = """
                SELECT 
                    event_type,
                    component,
                    COUNT(*) as error_count,
                    COUNT(DISTINCT user_id) as affected_users
                FROM application_logs 
                WHERE level IN ('error', 'critical') 
                AND created_at >= NOW() - INTERVAL '%s hours'
                %s
                GROUP BY event_type, component
                ORDER BY error_count DESC
            """
            
            error_result = await self.db.fetch_all(error_query % (hours, component_filter), params)
            
            # Get dharmic insights analytics
            dharmic_query = """
                SELECT 
                    insight_type,
                    COUNT(*) as count,
                    AVG(wisdom_level) as avg_wisdom,
                    AVG(compassion_score) as avg_compassion,
                    COUNT(DISTINCT user_id) as unique_users
                FROM dharmic_insights 
                WHERE created_at >= NOW() - INTERVAL '%s hours'
                GROUP BY insight_type
                ORDER BY avg_wisdom DESC
            """
            
            dharmic_result = await self.db.fetch_all(dharmic_query % hours)
            
            return {
                'time_range_hours': hours,
                'component_filter': component,
                'log_statistics': [dict(row) for row in stats_result],
                'performance_metrics': [dict(row) for row in perf_result],
                'error_analytics': [dict(row) for row in error_result],
                'dharmic_insights': [dict(row) for row in dharmic_result],
                'slow_operations_recent': self.slow_operations[-10:],  # Last 10 slow operations
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Log analytics error: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_logs(self, retention_days: Optional[int] = None):
        """Clean up old log data"""
        
        retention_days = retention_days or self.config['log_retention_days']
        
        if not self.db:
            return
        
        try:
            tables_to_clean = [
                'application_logs',
                'performance_logs',
                'security_events',
                'dharmic_insights',
                'analytics_events'
            ]
            
            for table in tables_to_clean:
                cleanup_query = f"""
                    DELETE FROM {table} 
                    WHERE created_at < NOW() - INTERVAL '{retention_days} days'
                """
                
                result = await self.db.execute_query(cleanup_query)
                logging.info(f"Cleaned {table}: {result} rows removed")
            
            logging.info(f"Log cleanup completed for {retention_days} days retention")
            
        except Exception as e:
            logging.error(f"Log cleanup error: {e}")


# Global logger instance
_dharmic_logger: Optional[DharmicLogger] = None


def get_dharmic_logger(db_manager: Optional[DatabaseManager] = None) -> DharmicLogger:
    """Get global dharmic logger instance"""
    global _dharmic_logger
    
    if _dharmic_logger is None:
        _dharmic_logger = DharmicLogger(db_manager)
    
    return _dharmic_logger


# Convenience functions
async def log_info(message: str, **kwargs):
    """Log info message"""
    logger = get_dharmic_logger()
    await logger.log_event(LogLevel.INFO, message, EventType.SYSTEM_EVENT, **kwargs)


async def log_error(message: str, error: Optional[Exception] = None, **kwargs):
    """Log error message"""
    logger = get_dharmic_logger()
    context_data = kwargs.get('context_data', {})
    
    if error:
        context_data['error'] = str(error)
        context_data['traceback'] = traceback.format_exc()
    
    kwargs['context_data'] = context_data
    
    await logger.log_event(LogLevel.ERROR, message, EventType.ERROR_EVENT, **kwargs)


async def log_performance(operation_name: str, duration: float, **kwargs):
    """Log performance metric"""
    logger = get_dharmic_logger()
    await logger.log_performance(operation_name, duration, **kwargs)


async def log_security(event_type: str, severity: str, **kwargs):
    """Log security event"""
    logger = get_dharmic_logger()
    await logger.log_security_event(event_type, severity, **kwargs)


async def log_dharmic_insight(insight_type: str, wisdom_level: float, compassion_score: float, **kwargs):
    """Log dharmic insight"""
    logger = get_dharmic_logger()
    await logger.log_dharmic_insight(insight_type, wisdom_level, compassion_score, **kwargs)


# Export logging components
__all__ = [
    "DharmicLogger",
    "LogLevel",
    "EventType",
    "LogContext",
    "PerformanceMetrics",
    "get_dharmic_logger",
    "log_info",
    "log_error",
    "log_performance",
    "log_security",
    "log_dharmic_insight"
]
