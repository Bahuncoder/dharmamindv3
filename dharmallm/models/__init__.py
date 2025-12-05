"""
🕉️ DharmaMind Advanced Data Models - Complete System

Comprehensive Pydantic models for all DharmaMind services and operations:

Core Models:
- Chat and conversation management
- User profiles and preferences  
- Analytics and monitoring
- Dharmic compliance tracking
- AI/ML model configurations
- System health and performance

Advanced Features:
- Validation with custom dharmic rules
- Automatic serialization/deserialization
- Type safety and documentation
- Integration with all backend services

May these models serve all beings with precision and compassion 🙏
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid


# ===============================
# CORE ENUMERATIONS
# ===============================

class MessageRole(str, Enum):
    """Message role in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DHARMA_GUIDE = "dharma_guide"


class ResponseType(str, Enum):
    """Type of AI response"""
    DIRECT_ANSWER = "direct_answer"
    GUIDED_INQUIRY = "guided_inquiry"
    WISDOM_TEACHING = "wisdom_teaching"
    COMPASSIONATE_SUPPORT = "compassionate_support"
    DHARMIC_GUIDANCE = "dharmic_guidance"


class DharmicPrinciple(str, Enum):
    """Core dharmic principles"""
    AHIMSA = "ahimsa"  # Non-violence
    SATYA = "satya"    # Truthfulness
    ASTEYA = "asteya"  # Non-stealing
    BRAHMACHARYA = "brahmacharya"  # Moderation
    APARIGRAHA = "aparigraha"  # Non-possessiveness
    DHARMA = "dharma"  # Righteousness
    KARMA = "karma"    # Action-consequence
    MOKSHA = "moksha"  # Liberation


class Priority(str, Enum):
    """Priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """System health status"""
    EXCELLENT = "excellent"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"


# ===============================
# CHAT & CONVERSATION MODELS
# ===============================

class ChatRequest(BaseModel):
    """Enhanced chat request model"""
    message: str = Field(..., min_length=1, max_length=4096, description="User message")
    context: Optional[str] = Field(default=None, max_length=2048, description="Additional context")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    user_id: Optional[str] = Field(default=None, description="User ID")
    language: str = Field(default="en", pattern="^[a-z]{2}$", description="Response language (ISO 639-1)")
    
    # Advanced options
    response_type: Optional[ResponseType] = Field(default=None, description="Preferred response type")
    dharmic_focus: Optional[List[DharmicPrinciple]] = Field(default=None, description="Dharmic principles to emphasize")
    max_response_length: Optional[int] = Field(default=2048, ge=100, le=4096, description="Maximum response length")
    include_sources: bool = Field(default=True, description="Include source references")
    include_suggestions: bool = Field(default=True, description="Include follow-up suggestions")
    
    # Privacy and preferences
    anonymous: bool = Field(default=False, description="Anonymous chat session")
    save_conversation: bool = Field(default=True, description="Save conversation history")
    
    @validator("message")
    def validate_message_content(cls, v):
        """Validate message content for dharmic compliance"""
        # Basic content validation
        if not v.strip():
            raise ValueError("Message cannot be empty")
        
        # Check for obviously harmful content
        harmful_keywords = ["violence", "hate", "harm", "illegal"]
        if any(keyword in v.lower() for keyword in harmful_keywords):
            # In production, this would be more sophisticated
            pass  # Allow for now, let the dharmic validation handle it
        
        return v.strip()


class ChatMessage(BaseModel):
    """Enhanced chat message model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Message ID")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")
    tokens_used: Optional[int] = Field(default=None, description="Token count")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    
    # Dharmic analysis
    dharmic_score: Optional[float] = Field(default=None, ge=0, le=1, description="Dharmic alignment score")
    principle_scores: Dict[str, float] = Field(default_factory=dict, description="Individual principle scores")
    
    # Source tracking
    model_used: Optional[str] = Field(default=None, description="AI model used")
    modules_engaged: List[str] = Field(default_factory=list, description="Spiritual modules engaged")
    sources: List[str] = Field(default_factory=list, description="Source references")


class ChatResponse(BaseModel):
    """Comprehensive chat response model"""
    response: str = Field(..., description="AI response content")
    conversation_id: str = Field(..., description="Conversation ID")
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Response message ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    # Performance metrics
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    tokens_used: int = Field(..., ge=0, description="Total tokens used")
    model_used: str = Field(..., description="Primary LLM model used")
    
    # Quality scores
    confidence_score: float = Field(..., ge=0, le=1, description="Response confidence")
    dharmic_alignment: float = Field(..., ge=0, le=1, description="Dharmic alignment score")
    relevance_score: float = Field(..., ge=0, le=1, description="Response relevance")
    wisdom_score: float = Field(..., ge=0, le=1, description="Wisdom content score")
    
    # Content analysis
    response_type: ResponseType = Field(..., description="Type of response")
    dharmic_principles: List[DharmicPrinciple] = Field(default_factory=list, description="Dharmic principles addressed")
    modules_used: List[str] = Field(default_factory=list, description="Spiritual modules engaged")
    
    # Supporting information
    sources: List[str] = Field(default_factory=list, description="Source references")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    related_topics: List[str] = Field(default_factory=list, description="Related spiritual topics")
    
    # Warnings and flags
    content_warnings: List[str] = Field(default_factory=list, description="Content warnings if any")
    dharmic_concerns: List[str] = Field(default_factory=list, description="Dharmic concerns identified")
    
    # User guidance
    practice_suggestions: List[str] = Field(default_factory=list, description="Suggested spiritual practices")
    meditation_guidance: Optional[str] = Field(default=None, description="Meditation guidance")
    scripture_references: List[str] = Field(default_factory=list, description="Relevant scripture passages")


class ConversationSummary(BaseModel):
    """Conversation summary and analytics"""
    conversation_id: str = Field(..., description="Conversation ID")
    user_id: Optional[str] = Field(default=None, description="User ID")
    
    # Basic metrics
    total_messages: int = Field(..., ge=0, description="Total messages in conversation")
    user_messages: int = Field(..., ge=0, description="User messages count")
    assistant_messages: int = Field(..., ge=0, description="Assistant messages count")
    
    # Time tracking
    started_at: datetime = Field(..., description="Conversation start time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    duration: timedelta = Field(..., description="Conversation duration")
    
    # Content analysis
    topics_discussed: List[str] = Field(default_factory=list, description="Topics covered")
    dharmic_themes: List[DharmicPrinciple] = Field(default_factory=list, description="Dharmic themes explored")
    spiritual_level: str = Field(..., description="Assessed spiritual level")
    
    # Quality metrics
    avg_dharmic_score: float = Field(..., ge=0, le=1, description="Average dharmic alignment")
    avg_wisdom_score: float = Field(..., ge=0, le=1, description="Average wisdom content")
    user_satisfaction: Optional[float] = Field(default=None, ge=0, le=1, description="User satisfaction score")
    
    # Recommendations
    suggested_practices: List[str] = Field(default_factory=list, description="Recommended practices")
    recommended_reading: List[str] = Field(default_factory=list, description="Recommended scriptures/texts")
    next_steps: List[str] = Field(default_factory=list, description="Suggested next steps")


# ===============================
# FEEDBACK & RATING MODELS
# ===============================

class FeedbackType(str, Enum):
    """Type of feedback"""
    GENERAL = "general"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    CONTENT_QUALITY = "content_quality"
    PERFORMANCE = "performance"
    DHARMIC_CONCERN = "dharmic_concern"
    SPIRITUAL_GUIDANCE = "spiritual_guidance"
    USER_EXPERIENCE = "user_experience"


class FeedbackSentiment(str, Enum):
    """Sentiment of feedback"""
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
    sentiment_score: float = Field(..., ge=-1, le=1, description="Sentiment confidence score")
    priority_score: float = Field(..., ge=0, le=1, description="Priority score")
    
    # Content analysis
    key_topics: List[str] = Field(default_factory=list, description="Extracted key topics")
    mentioned_features: List[str] = Field(default_factory=list, description="Mentioned features")
    suggestions: List[str] = Field(default_factory=list, description="Extracted suggestions")
    issues_identified: List[str] = Field(default_factory=list, description="Identified issues")
    
    # Dharmic analysis
    dharmic_concerns: List[str] = Field(default_factory=list, description="Dharmic concerns mentioned")
    spiritual_insights: List[str] = Field(default_factory=list, description="Spiritual insights")
    
    # Status and workflow
    status: str = Field(default="new", description="Feedback status")
    assigned_to: Optional[str] = Field(default=None, description="Assigned team member")
    resolution: Optional[str] = Field(default=None, description="Resolution notes")
    
    # Timestamps
    created_at: datetime = Field(..., description="Feedback creation time")
    analyzed_at: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    resolved_at: Optional[datetime] = Field(default=None, description="Resolution timestamp")
    
    # Contact info (if provided)
    user_email: Optional[str] = Field(default=None, description="User contact email")
    allow_contact: bool = Field(default=False, description="Allow follow-up contact")
    
    # Technical context
    browser_info: Optional[str] = Field(default=None, description="Browser information")
    device_info: Optional[str] = Field(default=None, description="Device information")


# ===============================
# USER & PROFILE MODELS
# ===============================

class UserPreferences(BaseModel):
    """User preferences and settings"""
    user_id: str = Field(..., description="User ID")
    
    # Language and communication
    preferred_language: str = Field(default="en", description="Preferred language")
    response_style: str = Field(default="balanced", description="Response style preference")
    max_response_length: int = Field(default=2048, ge=100, le=4096, description="Preferred response length")
    
    # Spiritual preferences
    dharmic_focus: List[DharmicPrinciple] = Field(default_factory=list, description="Preferred dharmic principles")
    spiritual_tradition: Optional[str] = Field(default=None, description="Primary spiritual tradition")
    meditation_experience: str = Field(default="beginner", description="Meditation experience level")
    scripture_familiarity: List[str] = Field(default_factory=list, description="Familiar scriptures")
    
    # Privacy settings
    save_conversations: bool = Field(default=True, description="Save conversation history")
    share_analytics: bool = Field(default=True, description="Share anonymous analytics")
    personalized_responses: bool = Field(default=True, description="Enable personalized responses")
    
    # Notification preferences
    daily_wisdom: bool = Field(default=False, description="Receive daily wisdom")
    practice_reminders: bool = Field(default=False, description="Receive practice reminders")
    progress_updates: bool = Field(default=False, description="Receive progress updates")
    
    # Content filters
    content_maturity: str = Field(default="general", description="Content maturity level")
    trigger_warnings: bool = Field(default=False, description="Show trigger warnings")
    
    updated_at: datetime = Field(default_factory=datetime.now, description="Last updated")


class UserProfile(BaseModel):
    """Comprehensive user profile"""
    user_id: str = Field(..., description="Unique user ID")
    created_at: datetime = Field(default_factory=datetime.now, description="Profile creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last profile update")
    
    # Basic information
    display_name: Optional[str] = Field(default=None, description="Display name")
    email: Optional[str] = Field(default=None, description="Email address")
    timezone: str = Field(default="UTC", description="User timezone")
    
    # Spiritual journey tracking
    spiritual_level: str = Field(default="seeker", description="Current spiritual level")
    dharmic_progress: Dict[str, float] = Field(default_factory=dict, description="Progress in dharmic principles")
    meditation_streak: int = Field(default=0, ge=0, description="Current meditation streak")
    wisdom_points: int = Field(default=0, ge=0, description="Accumulated wisdom points")
    
    # Learning preferences
    learning_style: str = Field(default="balanced", description="Preferred learning style")
    difficulty_preference: str = Field(default="moderate", description="Content difficulty preference")
    pace_preference: str = Field(default="steady", description="Learning pace preference")
    
    # Goals and intentions
    spiritual_goals: List[str] = Field(default_factory=list, description="Spiritual goals")
    current_practices: List[str] = Field(default_factory=list, description="Current spiritual practices")
    areas_of_interest: List[str] = Field(default_factory=list, description="Areas of spiritual interest")
    
    # Progress tracking
    total_conversations: int = Field(default=0, ge=0, description="Total conversations")
    total_messages: int = Field(default=0, ge=0, description="Total messages sent")
    avg_session_duration: float = Field(default=0, ge=0, description="Average session duration")
    
    # Preferences
    preferences: UserPreferences = Field(default_factory=lambda: UserPreferences(user_id=""), description="User preferences")


# ===============================
# ANALYTICS & MONITORING MODELS
# ===============================

class SystemMetrics(BaseModel):
    """System performance metrics"""
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
    # Performance metrics
    response_time_avg: float = Field(..., ge=0, description="Average response time")
    response_time_p95: float = Field(..., ge=0, description="95th percentile response time")
    throughput_per_minute: float = Field(..., ge=0, description="Requests per minute")
    
    # Resource utilization
    cpu_usage_percent: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage_percent: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_usage_percent: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    
    # Database metrics
    db_connections_active: int = Field(..., ge=0, description="Active database connections")
    db_query_time_avg: float = Field(..., ge=0, description="Average database query time")
    cache_hit_rate: float = Field(..., ge=0, le=1, description="Cache hit rate")
    
    # AI/ML metrics
    model_inference_time: float = Field(..., ge=0, description="Model inference time")
    embedding_generation_time: float = Field(..., ge=0, description="Embedding generation time")
    vector_search_time: float = Field(..., ge=0, description="Vector search time")
    
    # Quality metrics
    avg_confidence_score: float = Field(..., ge=0, le=1, description="Average confidence score")
    avg_dharmic_alignment: float = Field(..., ge=0, le=1, description="Average dharmic alignment")
    user_satisfaction_score: float = Field(..., ge=0, le=1, description="User satisfaction score")
    
    # Error tracking
    error_rate: float = Field(..., ge=0, le=1, description="Error rate")
    timeout_rate: float = Field(..., ge=0, le=1, description="Timeout rate")
    retry_rate: float = Field(..., ge=0, le=1, description="Retry rate")


class DharmicAnalytics(BaseModel):
    """Dharmic compliance and wisdom analytics"""
    timestamp: datetime = Field(default_factory=datetime.now, description="Analytics timestamp")
    period: str = Field(..., description="Analytics period (hour/day/week/month)")
    
    # Content analysis
    total_interactions: int = Field(..., ge=0, description="Total interactions analyzed")
    dharmic_compliance_rate: float = Field(..., ge=0, le=1, description="Dharmic compliance rate")
    wisdom_content_rate: float = Field(..., ge=0, le=1, description="Wisdom content rate")
    
    # Principle breakdown
    principle_scores: Dict[str, float] = Field(..., description="Average scores per principle")
    principle_violations: Dict[str, int] = Field(..., description="Violations per principle")
    
    # Content categories
    guidance_requests: int = Field(..., ge=0, description="Spiritual guidance requests")
    wisdom_teachings: int = Field(..., ge=0, description="Wisdom teaching interactions")
    meditation_guidance: int = Field(..., ge=0, description="Meditation guidance requests")
    scripture_discussions: int = Field(..., ge=0, description="Scripture-related discussions")
    
    # User impact
    user_spiritual_growth: float = Field(..., ge=0, le=1, description="Average user spiritual growth")
    practice_adoption_rate: float = Field(..., ge=0, le=1, description="Practice adoption rate")
    wisdom_application_rate: float = Field(..., ge=0, le=1, description="Wisdom application rate")
    
    # Quality indicators
    harmful_content_blocked: int = Field(..., ge=0, description="Harmful content interactions blocked")
    compassionate_responses: int = Field(..., ge=0, description="Compassionate responses generated")
    wisdom_insights_shared: int = Field(..., ge=0, description="Wisdom insights shared")


class AlertDefinition(BaseModel):
    """System alert configuration"""
    id: str = Field(..., description="Alert ID")
    name: str = Field(..., description="Alert name")
    description: str = Field(..., description="Alert description")
    
    # Trigger conditions
    metric_name: str = Field(..., description="Metric to monitor")
    threshold_value: float = Field(..., description="Threshold value")
    comparison_operator: Literal["gt", "lt", "eq", "gte", "lte"] = Field(..., description="Comparison operator")
    
    # Alert settings
    priority: Priority = Field(..., description="Alert priority")
    enabled: bool = Field(default=True, description="Alert enabled status")
    cooldown_minutes: int = Field(default=5, ge=1, description="Cooldown period in minutes")
    
    # Notification settings
    notify_channels: List[str] = Field(default_factory=list, description="Notification channels")
    escalation_minutes: int = Field(default=15, ge=1, description="Escalation time in minutes")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Alert creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")


# ===============================
# CONFIGURATION MODELS
# ===============================

class ModelConfiguration(BaseModel):
    """AI model configuration"""
    model_id: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Model provider")
    model_name: str = Field(..., description="Model name")
    
    # Model parameters
    max_tokens: int = Field(default=2048, ge=1, description="Maximum tokens")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Temperature setting")
    top_p: float = Field(default=0.9, ge=0, le=1, description="Top-p sampling")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling")
    
    # Performance settings
    timeout_seconds: int = Field(default=30, ge=1, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, description="Maximum retries")
    
    # Dharmic settings
    dharmic_filter_enabled: bool = Field(default=True, description="Enable dharmic filtering")
    wisdom_enhancement: bool = Field(default=True, description="Enable wisdom enhancement")
    compassion_bias: float = Field(default=0.1, ge=0, le=1, description="Compassion bias factor")
    
    # Usage limits
    requests_per_minute: int = Field(default=60, ge=1, description="Requests per minute limit")
    tokens_per_day: int = Field(default=100000, ge=1, description="Tokens per day limit")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Configuration creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")


# ===============================
# HEALTH & STATUS MODELS  
# ===============================

class ComponentHealth(BaseModel):
    """Individual component health status"""
    component_name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Health status")
    
    # Health metrics
    response_time: float = Field(..., ge=0, description="Response time in seconds")
    uptime_percentage: float = Field(..., ge=0, le=100, description="Uptime percentage")
    error_rate: float = Field(..., ge=0, le=1, description="Error rate")
    
    # Last check
    last_check: datetime = Field(..., description="Last health check time")
    next_check: datetime = Field(..., description="Next scheduled check")
    
    # Additional details
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health details")
    issues: List[str] = Field(default_factory=list, description="Current issues")
    recommendations: List[str] = Field(default_factory=list, description="Health recommendations")


class SystemHealth(BaseModel):
    """Overall system health status"""
    overall_status: HealthStatus = Field(..., description="Overall system status")
    health_score: float = Field(..., ge=0, le=1, description="Overall health score")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    
    # Component health
    components: List[ComponentHealth] = Field(..., description="Individual component health")
    
    # Summary metrics
    healthy_components: int = Field(..., ge=0, description="Number of healthy components")
    total_components: int = Field(..., ge=0, description="Total number of components")
    
    # System-wide metrics
    avg_response_time: float = Field(..., ge=0, description="Average response time")
    system_uptime: float = Field(..., ge=0, le=100, description="System uptime percentage")
    
    # Active issues
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues")
    warnings: List[str] = Field(default_factory=list, description="System warnings")
    
    # Recommendations
    immediate_actions: List[str] = Field(default_factory=list, description="Immediate actions needed")
    preventive_measures: List[str] = Field(default_factory=list, description="Preventive measures")


# ===============================
# MODEL EXPORTS
# ===============================

__all__ = [
    # Enums
    "MessageRole",
    "ResponseType", 
    "DharmicPrinciple",
    "Priority",
    "HealthStatus",
    "FeedbackType",
    "FeedbackSentiment",
    
    # Chat models
    "ChatRequest",
    "ChatMessage", 
    "ChatResponse",
    "ConversationSummary",
    
    # Feedback models
    "FeedbackRequest",
    "FeedbackResponse",
    "FeedbackAnalytics",
    
    # User models
    "UserPreferences",
    "UserProfile",
    
    # Analytics models
    "SystemMetrics",
    "DharmicAnalytics",
    "AlertDefinition",
    
    # Configuration models
    "ModelConfiguration",
    
    # Health models
    "ComponentHealth",
    "SystemHealth"
]
