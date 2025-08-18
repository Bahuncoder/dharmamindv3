"""
🔍 DharmaMind Distributed Tracing & APM System

Enterprise-grade observability with OpenTelemetry integration:

Core Features:
- Distributed request tracing across all services
- Performance monitoring and bottleneck detection
- Custom span attributes and metrics
- Jaeger and Prometheus integration
- Automatic instrumentation for FastAPI, SQLAlchemy, Redis
- Business logic tracing with dharmic context
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from contextvars import ContextVar
from functools import wraps
from dataclasses import dataclass
import json

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

# FastAPI and other imports
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Set up logging
logger = logging.getLogger("dharmamind.tracing")

# Context variables for trace correlation
current_trace_id: ContextVar[Optional[str]] = ContextVar('current_trace_id', default=None)
current_span_id: ContextVar[Optional[str]] = ContextVar('current_span_id', default=None)
current_user_id: ContextVar[Optional[str]] = ContextVar('current_user_id', default=None)


@dataclass
class TracingConfig:
    """Configuration for distributed tracing"""
    service_name: str = "dharmamind-backend"
    service_version: str = "2.0.0"
    environment: str = "production"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    prometheus_port: int = 8000
    sample_rate: float = 1.0
    enable_db_tracing: bool = True
    enable_redis_tracing: bool = True
    enable_custom_metrics: bool = True


class DharmaTracer:
    """
    Advanced distributed tracing system for DharmaMind
    """
    
    def __init__(self, config: TracingConfig):
        self.config = config
        self.tracer = None
        self.meter = None
        self.resource = None
        self._setup_telemetry()
        
        # Custom metrics
        self.request_counter = None
        self.request_duration = None
        self.dharma_wisdom_score = None
        self.llm_performance = None
        self.cache_performance = None
        
        self._setup_custom_metrics()
    
    def _setup_telemetry(self):
        """Initialize OpenTelemetry providers and exporters"""
        
        # Create resource with service information
        self.resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": self.config.service_version,
            "deployment.environment": self.config.environment,
            "telemetry.sdk.language": "python",
            "service.type": "dharma_backend"
        })
        
        # Setup trace provider
        trace_provider = TracerProvider(resource=self.resource)
        trace.set_tracer_provider(trace_provider)
        
        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            endpoint=self.config.jaeger_endpoint,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace_provider.add_span_processor(span_processor)
        
        # Setup metrics provider with Prometheus
        prometheus_reader = PrometheusMetricReader()
        metrics_provider = MeterProvider(
            resource=self.resource,
            metric_readers=[prometheus_reader]
        )
        metrics.set_meter_provider(metrics_provider)
        
        # Get tracer and meter
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        logger.info(f"🔍 Distributed tracing initialized for {self.config.service_name}")
    
    def _setup_custom_metrics(self):
        """Setup custom business metrics"""
        
        # Request metrics
        self.request_counter = self.meter.create_counter(
            name="dharma_requests_total",
            description="Total number of requests processed",
            unit="1"
        )
        
        self.request_duration = self.meter.create_histogram(
            name="dharma_request_duration_seconds",
            description="Request processing duration",
            unit="s"
        )
        
        # Business metrics
        self.dharma_wisdom_score = self.meter.create_histogram(
            name="dharma_wisdom_score",
            description="Wisdom score of responses",
            unit="1"
        )
        
        self.llm_performance = self.meter.create_histogram(
            name="llm_response_time",
            description="LLM response time",
            unit="s"
        )
        
        self.cache_performance = self.meter.create_counter(
            name="cache_operations_total",
            description="Cache operations",
            unit="1"
        )
    
    def instrument_app(self, app):
        """Automatically instrument FastAPI application"""
        
        # FastAPI instrumentation
        FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
        
        # SQLAlchemy instrumentation
        if self.config.enable_db_tracing:
            SQLAlchemyInstrumentor().instrument()
        
        # Redis instrumentation
        if self.config.enable_redis_tracing:
            RedisInstrumentor().instrument()
        
        logger.info("🔧 Automatic instrumentation enabled")
    
    def create_span(self, name: str, kind: trace.SpanKind = trace.SpanKind.INTERNAL):
        """Create a new span for tracing"""
        return self.tracer.start_span(name, kind=kind)
    
    def trace_function(self, span_name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
        """Decorator for tracing functions"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__name__}"
                
                with self.tracer.start_span(name) as span:
                    # Add function attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    
                    # Add context attributes
                    if current_user_id.get():
                        span.set_attribute("user.id", current_user_id.get())
                    
                    try:
                        start_time = time.time()
                        
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)
                        
                        # Record timing
                        duration = time.time() - start_time
                        span.set_attribute("function.duration", duration)
                        
                        # Mark as successful
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        
                        return result
                        
                    except Exception as e:
                        # Record error
                        span.set_status(trace.Status(
                            trace.StatusCode.ERROR,
                            str(e)
                        ))
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__name__}"
                
                with self.tracer.start_span(name) as span:
                    # Add function attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        
                        # Record timing
                        duration = time.time() - start_time
                        span.set_attribute("function.duration", duration)
                        
                        # Mark as successful
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        
                        return result
                        
                    except Exception as e:
                        # Record error
                        span.set_status(trace.Status(
                            trace.StatusCode.ERROR,
                            str(e)
                        ))
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def record_dharma_metrics(self, wisdom_score: float, processing_time: float, 
                            user_satisfaction: Optional[float] = None):
        """Record dharmic business metrics"""
        
        # Wisdom score
        self.dharma_wisdom_score.record(wisdom_score, {
            "service": self.config.service_name,
            "environment": self.config.environment
        })
        
        # Processing time
        self.request_duration.record(processing_time, {
            "operation": "dharma_processing",
            "service": self.config.service_name
        })
        
        # User satisfaction if available
        if user_satisfaction is not None:
            if hasattr(self, 'user_satisfaction'):
                self.user_satisfaction.record(user_satisfaction)
    
    def record_llm_metrics(self, provider: str, model: str, response_time: float, 
                          tokens_used: int, success: bool):
        """Record LLM performance metrics"""
        
        self.llm_performance.record(response_time, {
            "llm_provider": provider,
            "llm_model": model,
            "success": str(success),
            "service": self.config.service_name
        })
        
        # Token usage metric
        if hasattr(self, 'token_usage'):
            self.token_usage.add(tokens_used, {
                "llm_provider": provider,
                "llm_model": model
            })
    
    def record_cache_metrics(self, operation: str, hit: bool, size_bytes: Optional[int] = None):
        """Record cache performance metrics"""
        
        self.cache_performance.add(1, {
            "cache_operation": operation,
            "cache_hit": str(hit),
            "service": self.config.service_name
        })
        
        if size_bytes and hasattr(self, 'cache_size'):
            self.cache_size.record(size_bytes, {
                "cache_operation": operation
            })


class TracingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request tracing and context correlation
    """
    
    def __init__(self, app, tracer: DharmaTracer):
        super().__init__(app)
        self.tracer = tracer
    
    async def dispatch(self, request: Request, call_next):
        """Process request with distributed tracing"""
        
        start_time = time.time()
        
        # Extract trace context from headers
        trace_id = request.headers.get("X-Trace-ID")
        user_id = request.headers.get("X-User-ID")
        
        # Set context variables
        if trace_id:
            current_trace_id.set(trace_id)
        if user_id:
            current_user_id.set(user_id)
        
        # Create main request span
        with self.tracer.create_span(
            f"{request.method} {request.url.path}",
            kind=trace.SpanKind.SERVER
        ) as span:
            
            # Add request attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.scheme", request.url.scheme)
            span.set_attribute("http.host", request.url.hostname or "")
            span.set_attribute("http.user_agent", request.headers.get("user-agent", ""))
            
            if user_id:
                span.set_attribute("user.id", user_id)
            
            if trace_id:
                span.set_attribute("trace.id", trace_id)
            
            try:
                # Process request
                response = await call_next(request)
                
                # Add response attributes
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.response_size", 
                                 len(response.body) if hasattr(response, 'body') else 0)
                
                # Record metrics
                duration = time.time() - start_time
                
                self.tracer.request_counter.add(1, {
                    "method": request.method,
                    "status_code": str(response.status_code),
                    "endpoint": request.url.path
                })
                
                self.tracer.request_duration.record(duration, {
                    "method": request.method,
                    "status_code": str(response.status_code),
                    "endpoint": request.url.path
                })
                
                # Set span status
                if response.status_code >= 400:
                    span.set_status(trace.Status(
                        trace.StatusCode.ERROR,
                        f"HTTP {response.status_code}"
                    ))
                else:
                    span.set_status(trace.Status(trace.StatusCode.OK))
                
                return response
                
            except Exception as e:
                # Record error
                span.set_status(trace.Status(
                    trace.StatusCode.ERROR,
                    str(e)
                ))
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                
                # Record error metrics
                duration = time.time() - start_time
                self.tracer.request_counter.add(1, {
                    "method": request.method,
                    "status_code": "500",
                    "endpoint": request.url.path,
                    "error": "true"
                })
                
                raise


# Global tracer instance
_global_tracer: Optional[DharmaTracer] = None


def get_tracer() -> Optional[DharmaTracer]:
    """Get the global tracer instance"""
    return _global_tracer


def initialize_tracing(config: TracingConfig) -> DharmaTracer:
    """Initialize distributed tracing system"""
    global _global_tracer
    
    _global_tracer = DharmaTracer(config)
    
    logger.info(f"🔍 Distributed tracing system initialized")
    logger.info(f"📊 Service: {config.service_name} v{config.service_version}")
    logger.info(f"🌍 Environment: {config.environment}")
    logger.info(f"📈 Jaeger endpoint: {config.jaeger_endpoint}")
    
    return _global_tracer


# Convenience decorators using global tracer
def trace(span_name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Convenience decorator for tracing using global tracer"""
    def decorator(func):
        if _global_tracer:
            return _global_tracer.trace_function(span_name, attributes)(func)
        return func
    return decorator


def trace_dharma_operation(operation_type: str):
    """Specialized decorator for dharmic operations"""
    return trace(
        span_name=f"dharma.{operation_type}",
        attributes={
            "dharma.operation_type": operation_type,
            "dharma.service": "dharmamind"
        }
    )


def trace_llm_operation(provider: str, model: str):
    """Specialized decorator for LLM operations"""
    return trace(
        span_name=f"llm.{provider}.{model}",
        attributes={
            "llm.provider": provider,
            "llm.model": model,
            "operation.type": "llm_inference"
        }
    )


# Context managers for manual span creation
class DharmaSpan:
    """Context manager for creating dharmic spans"""
    
    def __init__(self, name: str, operation_type: str, **attributes):
        self.name = name
        self.operation_type = operation_type
        self.attributes = attributes
        self.span = None
    
    def __enter__(self):
        if _global_tracer:
            self.span = _global_tracer.create_span(f"dharma.{self.name}")
            self.span.set_attribute("dharma.operation_type", self.operation_type)
            
            for key, value in self.attributes.items():
                self.span.set_attribute(f"dharma.{key}", value)
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.set_status(trace.Status(
                    trace.StatusCode.ERROR,
                    str(exc_val)
                ))
            else:
                self.span.set_status(trace.Status(trace.StatusCode.OK))
            
            self.span.end()


# Export all main components
__all__ = [
    "DharmaTracer",
    "TracingConfig", 
    "TracingMiddleware",
    "initialize_tracing",
    "get_tracer",
    "trace",
    "trace_dharma_operation",
    "trace_llm_operation",
    "DharmaSpan",
    "current_trace_id",
    "current_span_id",
    "current_user_id"
]
