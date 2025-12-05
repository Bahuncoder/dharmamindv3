"""
Feedback models for DharmaMind platform

Defines feedback requests, responses, and analytics for continuous improvement.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class FeedbackSentiment(str, Enum):
    """Sentiment analysis of feedback"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"

class FeedbackType(str, Enum):
    """Types of feedback"""
    GENERAL = "general"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    SPIRITUAL_GUIDANCE = "spiritual_guidance"
    USER_EXPERIENCE = "user_experience"
    CONTENT_QUALITY = "content_quality"
    PERFORMANCE = "performance"

class FeedbackPriority(str, Enum):
    """Feedback priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FeedbackStatus(str, Enum):
    """Feedback processing status"""
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    IN_REVIEW = "in_review"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"

class FeedbackRequest(BaseModel):
    """User feedback request"""
    user_id: Optional[str] = Field(default=None, description="User ID (optional for anonymous feedback)")
    type: FeedbackType = Field(..., description="Type of feedback")
    title: str = Field(..., description="Feedback title", min_length=5, max_length=200)
    description: str = Field(..., description="Detailed feedback description", min_length=10, max_length=5000)
    
    # Context information
    page_url: Optional[str] = Field(default=None, description="Page where feedback was submitted")
    user_agent: Optional[str] = Field(default=None, description="User browser/device info")
    session_id: Optional[str] = Field(default=None, description="User session ID")
    
    # Rating and priority
    rating: Optional[int] = Field(default=None, ge=1, le=5, description="User rating (1-5 stars)")
    priority: FeedbackPriority = Field(default=FeedbackPriority.MEDIUM, description="Feedback priority")
    
    # Additional context
    expected_behavior: Optional[str] = Field(default=None, description="What was expected")
    actual_behavior: Optional[str] = Field(default=None, description="What actually happened")
    steps_to_reproduce: Optional[List[str]] = Field(default=None, description="Steps to reproduce issue")
    
    # Contact info (for follow-up)
    contact_email: Optional[str] = Field(default=None, description="Contact email for follow-up")
    allow_contact: bool = Field(default=True, description="Allow team to contact for clarification")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True

class FeedbackResponse(BaseModel):
    """Feedback response from the system"""
    feedback_id: str = Field(..., description="Unique feedback identifier")
    status: FeedbackStatus = Field(..., description="Current feedback status")
    message: str = Field(..., description="Response message to user")
    
    # Processing information
    assigned_to: Optional[str] = Field(default=None, description="Team member assigned")
    estimated_resolution: Optional[datetime] = Field(default=None, description="Estimated resolution date")
    
    # Response metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Response creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class FeedbackItem(BaseModel):
    """Complete feedback item with all details"""
    feedback_id: str = Field(..., description="Unique feedback identifier")
    user_id: Optional[str] = Field(default=None, description="User ID")
    type: FeedbackType = Field(..., description="Feedback type")
    title: str = Field(..., description="Feedback title")
    description: str = Field(..., description="Feedback description")
    
    # Status and priority
    status: FeedbackStatus = Field(default=FeedbackStatus.SUBMITTED, description="Current status")
    priority: FeedbackPriority = Field(default=FeedbackPriority.MEDIUM, description="Priority level")
    
    # Context
    page_url: Optional[str] = Field(default=None, description="Source page URL")
    user_agent: Optional[str] = Field(default=None, description="User agent string")
    session_context: Optional[Dict[str, Any]] = Field(default=None, description="Session context")
    
    # Ratings and metrics
    user_rating: Optional[int] = Field(default=None, ge=1, le=5, description="User rating")
    internal_rating: Optional[int] = Field(default=None, ge=1, le=5, description="Internal team rating")
    
    # Processing details
    assigned_to: Optional[str] = Field(default=None, description="Assigned team member")
    tags: List[str] = Field(default_factory=list, description="Feedback tags")
    resolution_notes: Optional[str] = Field(default=None, description="Resolution notes")
    
    # Timestamps
    submitted_at: datetime = Field(default_factory=datetime.now, description="Submission timestamp")
    acknowledged_at: Optional[datetime] = Field(default=None, description="Acknowledgment timestamp")
    resolved_at: Optional[datetime] = Field(default=None, description="Resolution timestamp")
    
    # Communication
    responses: List[FeedbackResponse] = Field(default_factory=list, description="Team responses")
    contact_email: Optional[str] = Field(default=None, description="Contact email")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class FeedbackAnalytics(BaseModel):
    """Feedback analytics and insights"""
    total_feedback: int = Field(default=0, description="Total feedback items")
    feedback_by_type: Dict[str, int] = Field(default_factory=dict, description="Feedback count by type")
    feedback_by_status: Dict[str, int] = Field(default_factory=dict, description="Feedback count by status")
    feedback_by_priority: Dict[str, int] = Field(default_factory=dict, description="Feedback count by priority")
    
    # Ratings analysis
    average_rating: Optional[float] = Field(default=None, description="Average user rating")
    rating_distribution: Dict[str, int] = Field(default_factory=dict, description="Rating distribution")
    
    # Time-based metrics
    average_resolution_time: Optional[float] = Field(default=None, description="Average resolution time in hours")
    response_time_avg: Optional[float] = Field(default=None, description="Average response time in hours")
    
    # Trends
    monthly_trends: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Monthly feedback trends")
    top_issues: List[str] = Field(default_factory=list, description="Most common issues")
    improvement_areas: List[str] = Field(default_factory=list, description="Areas needing improvement")
    
    # Generated metadata
    period_start: datetime = Field(..., description="Analysis period start")
    period_end: datetime = Field(..., description="Analysis period end")
    generated_at: datetime = Field(default_factory=datetime.now, description="Analytics generation time")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class FeedbackUpdate(BaseModel):
    """Model for updating feedback status"""
    status: Optional[FeedbackStatus] = Field(default=None, description="New status")
    priority: Optional[FeedbackPriority] = Field(default=None, description="New priority")
    assigned_to: Optional[str] = Field(default=None, description="Assign to team member")
    tags: Optional[List[str]] = Field(default=None, description="Add/update tags")
    resolution_notes: Optional[str] = Field(default=None, description="Resolution notes")
    internal_notes: Optional[str] = Field(default=None, description="Internal notes")

class FeedbackSummary(BaseModel):
    """Summary view of feedback for dashboards"""
    feedback_id: str = Field(..., description="Feedback ID")
    title: str = Field(..., description="Feedback title")
    type: FeedbackType = Field(..., description="Feedback type")
    status: FeedbackStatus = Field(..., description="Current status")
    priority: FeedbackPriority = Field(..., description="Priority level")
    user_rating: Optional[int] = Field(default=None, description="User rating")
    submitted_at: datetime = Field(..., description="Submission date")
    days_open: int = Field(..., description="Days since submission")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }