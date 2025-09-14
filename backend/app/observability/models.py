"""
📊 Observability Models
======================

Pydantic models for observability and monitoring endpoints.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class ResponseStatus(str, Enum):
    """Response status types"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"

class FeedbackType(str, Enum):
    """Feedback type categories"""
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    GENERAL_FEEDBACK = "general_feedback"
    SPIRITUAL_GUIDANCE = "spiritual_guidance"
    TECHNICAL_ISSUE = "technical_issue"
    CONTENT_SUGGESTION = "content_suggestion"
    USER_EXPERIENCE = "user_experience"

class FeedbackSentiment(str, Enum):
    """Feedback sentiment analysis"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class FeedbackRequest(BaseModel):
    """User feedback submission"""
    user_id: Optional[str] = Field(default=None, description="User ID (optional for anonymous feedback)")
    conversation_id: Optional[str] = Field(default=None, description="Related conversation ID")
    message_id: Optional[str] = Field(default=None, description="Related message ID")
    
    # Feedback content
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    title: str = Field(..., min_length=3, max_length=200, description="Feedback title")
    content: str = Field(..., min_length=10, max_length=4000, description="Feedback content")
    
    # Ratings (1-5 score)
    overall_rating: Optional[int] = Field(default=None, ge=1, le=5, description="Overall rating (1-5)")
    response_quality: Optional[int] = Field(default=None, ge=1, le=5, description="Response quality rating")
    helpfulness: Optional[int] = Field(default=None, ge=1, le=5, description="Helpfulness rating")
    spiritual_value: Optional[int] = Field(default=None, ge=1, le=5, description="Spiritual value rating")
    
    # Additional context
    user_email: Optional[str] = Field(default=None, description="Contact email (optional)")
    browser_info: Optional[str] = Field(default=None, description="Browser information")
    device_info: Optional[str] = Field(default=None, description="Device information")
    
    # Privacy settings
    allow_contact: bool = Field(default=False, description="Allow contact for follow-up")
    share_anonymously: bool = Field(default=True, description="Share feedback anonymously for improvements")
    
    @validator("content")
    def validate_feedback_content(cls, v):
        """Validate feedback content"""
        if not v.strip():
            raise ValueError("Feedback content cannot be empty")
        return v.strip()

class FeedbackResponse(BaseModel):
    """Feedback submission response"""
    feedback_id: str = Field(..., description="Unique feedback ID")
    status: str = Field(..., description="Submission status")
    message: str = Field(..., description="Response message")
    created_at: datetime = Field(default_factory=datetime.now, description="Submission timestamp")
    
    # Analytics info (optional)
    sentiment_score: Optional[float] = Field(default=None, ge=-1, le=1, description="AI-analyzed sentiment score")
    priority_score: Optional[float] = Field(default=None, ge=0, le=1, description="Priority score")
    category_confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Category classification confidence")

class FeedbackAnalytics(BaseModel):
    """Feedback analytics and insights"""
    id: str = Field(..., description="Feedback record ID")
    user_id: Optional[str] = Field(default=None, description="User ID")
    conversation_id: Optional[str] = Field(default=None, description="Related conversation")
    message_id: Optional[str] = Field(default=None, description="Related message")
    
    # Basic feedback info
    feedback_type: FeedbackType = Field(..., description="Feedback type")
    title: str = Field(..., description="Feedback title")
    content: str = Field(..., description="Feedback content")
    
    # Ratings
    overall_rating: Optional[int] = Field(default=None, description="Overall rating")
    response_quality: Optional[int] = Field(default=None, description="Response quality rating")
    helpfulness: Optional[int] = Field(default=None, description="Helpfulness rating")
    spiritual_value: Optional[int] = Field(default=None, description="Spiritual value rating")
    
    # AI Analysis
    sentiment: FeedbackSentiment = Field(..., description="AI-analyzed sentiment")
    
class FeedbackMetrics(BaseModel):
    """Feedback metrics summary"""
    total_feedback: int = Field(..., description="Total feedback count")
    by_type: Dict[str, int] = Field(default_factory=dict, description="Feedback count by type")
    by_sentiment: Dict[str, int] = Field(default_factory=dict, description="Feedback count by sentiment")
    average_ratings: Dict[str, float] = Field(default_factory=dict, description="Average ratings")
    recent_feedback: List[FeedbackAnalytics] = Field(default_factory=list, description="Recent feedback items")

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    rishi_name: Optional[str] = None
    emotional_analysis: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()

class AuthRequest(BaseModel):
    """Authentication request model"""
    email: str
    password: str

class AuthResponse(BaseModel):
    """Authentication response model"""
    access_token: str
    token_type: str = "bearer"
    user_id: str

class UserProfile(BaseModel):
    """User profile model"""
    user_id: str
    email: str
    username: str
    created_at: datetime

class HealthStatus(BaseModel):
    """Health check status"""
    status: str = "healthy"
    timestamp: datetime = datetime.now()
    components: Dict[str, str] = {}

class UserPreferences(BaseModel):
    """User preferences model"""
    user_id: str
    spiritual_practices: List[str] = []
    preferred_rishi: Optional[str] = None
    meditation_style: Optional[str] = None
    notification_settings: Dict[str, bool] = {}

class SystemMetrics(BaseModel):
    """System metrics model"""
    cpu_usage: float
    memory_usage: float
    active_users: int
    response_time_ms: float
    timestamp: datetime = datetime.now()

class DharmicAnalytics(BaseModel):
    """Dharmic analytics model"""
    total_interactions: int
    emotional_insights: Dict[str, int]
    rishi_usage: Dict[str, int]
    spiritual_growth_metrics: Dict[str, Any]

class SystemHealth(BaseModel):
    """System health model"""
    overall_status: str
    services: Dict[str, str]
    last_checked: datetime = datetime.now()

class ModelConfiguration(BaseModel):
    """Model configuration model"""
    model_name: str
    model_type: str
    parameters: Dict[str, Any]
    active: bool = True

# Export all models
__all__ = [
    'ResponseStatus',
    'FeedbackType',
    'FeedbackSentiment', 
    'FeedbackRequest',
    'FeedbackResponse',
    'FeedbackAnalytics',
    'FeedbackMetrics',
    'ChatRequest', 
    'ChatResponse',
    'AuthRequest',
    'AuthResponse', 
    'UserProfile',
    'HealthStatus',
    'UserPreferences',
    'SystemMetrics',
    'DharmicAnalytics',
    'SystemHealth',
    'ModelConfiguration'
]
