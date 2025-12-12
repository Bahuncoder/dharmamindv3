"""
System metrics models for DharmaMind platform

Defines system performance metrics, analytics, and health monitoring data structures.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class MetricType(str, Enum):
    """Types of system metrics"""
    PERFORMANCE = "performance"
    USAGE = "usage"
    ERROR = "error"
    SECURITY = "security"
    BUSINESS = "business"

class MetricUnit(str, Enum):
    """Metric units"""
    COUNT = "count"
    PERCENTAGE = "percentage"
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    BYTES = "bytes"
    REQUESTS_PER_SECOND = "requests_per_second"

class HealthStatus(str, Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"

class SystemMetrics(BaseModel):
    """System performance and usage metrics"""
    metric_id: str = Field(..., description="Unique metric identifier")
    name: str = Field(..., description="Metric name")
    type: MetricType = Field(..., description="Metric type")
    value: Union[int, float, str] = Field(..., description="Metric value")
    unit: MetricUnit = Field(..., description="Metric unit")
    
    # Metadata
    description: Optional[str] = Field(default=None, description="Metric description")
    tags: Dict[str, str] = Field(default_factory=dict, description="Metric tags")
    dimensions: Dict[str, Any] = Field(default_factory=dict, description="Metric dimensions")
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.now, description="Metric timestamp")
    period_start: Optional[datetime] = Field(default=None, description="Period start time")
    period_end: Optional[datetime] = Field(default=None, description="Period end time")
    
    # Statistical data
    min_value: Optional[float] = Field(default=None, description="Minimum value in period")
    max_value: Optional[float] = Field(default=None, description="Maximum value in period")
    avg_value: Optional[float] = Field(default=None, description="Average value in period")
    std_deviation: Optional[float] = Field(default=None, description="Standard deviation")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class DharmicAnalytics(BaseModel):
    """Analytics specific to dharmic and spiritual metrics"""
    analytics_id: str = Field(..., description="Analytics identifier")
    user_id: Optional[str] = Field(default=None, description="Associated user ID")
    
    # Spiritual journey metrics
    wisdom_requests_count: int = Field(default=0, description="Number of wisdom requests")
    meditation_sessions: int = Field(default=0, description="Meditation sessions completed")
    spiritual_growth_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Spiritual growth score")
    dharmic_alignment_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Dharmic alignment score")
    
    # Engagement metrics
    session_duration_avg: float = Field(default=0.0, description="Average session duration in minutes")
    questions_asked: int = Field(default=0, description="Total questions asked")
    wisdom_applied: int = Field(default=0, description="Wisdom applications reported")
    community_interactions: int = Field(default=0, description="Community interactions")
    
    # Content metrics
    favorite_topics: List[str] = Field(default_factory=list, description="Most engaged topics")
    spiritual_traditions_explored: List[str] = Field(default_factory=list, description="Traditions explored")
    practices_learned: List[str] = Field(default_factory=list, description="Practices learned")
    
    # Behavioral insights
    peak_engagement_times: List[str] = Field(default_factory=list, description="Peak engagement time periods")
    learning_patterns: Dict[str, Any] = Field(default_factory=dict, description="Learning behavior patterns")
    progress_milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Achievement milestones")
    
    # Period information
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")
    generated_at: datetime = Field(default_factory=datetime.now, description="Analytics generation time")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class SystemHealth(BaseModel):
    """Overall system health status"""
    health_id: str = Field(..., description="Health check identifier")
    overall_status: HealthStatus = Field(..., description="Overall system health status")
    
    # Component health
    api_status: HealthStatus = Field(..., description="API health status")
    database_status: HealthStatus = Field(..., description="Database health status")
    cache_status: HealthStatus = Field(..., description="Cache system health")
    llm_status: HealthStatus = Field(..., description="LLM services health")
    
    # Performance metrics
    response_time_avg: float = Field(..., description="Average response time in milliseconds")
    cpu_usage: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0.0, le=100.0, description="Memory usage percentage")
    disk_usage: float = Field(..., ge=0.0, le=100.0, description="Disk usage percentage")
    
    # Service availability
    uptime_percentage: float = Field(..., ge=0.0, le=100.0, description="System uptime percentage")
    error_rate: float = Field(..., ge=0.0, le=100.0, description="Error rate percentage")
    active_users: int = Field(default=0, description="Currently active users")
    
    # Alerts and issues
    active_alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Active system alerts")
    recent_issues: List[Dict[str, Any]] = Field(default_factory=list, description="Recent system issues")
    maintenance_scheduled: Optional[datetime] = Field(default=None, description="Next scheduled maintenance")
    
    # Timestamp
    checked_at: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class ModelConfiguration(BaseModel):
    """AI model configuration settings"""
    model_config = {'protected_namespaces': ()}  # Allow 'model_' prefix in fields
    
    config_id: str = Field(..., description="Configuration identifier")
    model_name: str = Field(..., description="AI model name")
    
    # Model parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=2000, ge=1, le=8000, description="Maximum tokens")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Dharmic settings
    dharmic_guidance_enabled: bool = Field(default=True, description="Enable dharmic guidance")
    spiritual_context_weight: float = Field(default=0.8, ge=0.0, le=1.0, description="Spiritual context importance")
    wisdom_integration_level: str = Field(default="medium", description="Wisdom integration level")
    
    # Safety and filtering
    content_filter_enabled: bool = Field(default=True, description="Enable content filtering")
    safety_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Safety threshold")
    cultural_sensitivity: float = Field(default=0.9, ge=0.0, le=1.0, description="Cultural sensitivity level")
    
    # Performance settings
    response_timeout: int = Field(default=30, ge=1, le=120, description="Response timeout in seconds")
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Retry attempts on failure")
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    
    # Model metadata
    version: str = Field(..., description="Model version")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    limitations: List[str] = Field(default_factory=list, description="Known limitations")
    
    # Configuration metadata
    created_by: str = Field(..., description="Configuration creator")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    is_active: bool = Field(default=True, description="Whether configuration is active")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }