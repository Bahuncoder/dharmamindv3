<<<<<<< HEAD
"""
Chat models for DharmaMind API

These models define the structure of chat-related data exchanges
between the frontend and backend systems.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    """Message role in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MessageType(str, Enum):
    """Type of message content"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

class ChatMessage(BaseModel):
    """Individual chat message in a conversation"""
    id: Optional[str] = Field(default=None, description="Message unique identifier")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class ChatRequest(BaseModel):
    """Request for chat completion"""
    message: str = Field(..., description="User message", min_length=1, max_length=10000)
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    context: Optional[str] = Field(default=None, description="Additional context")
    language: str = Field(default="en", description="Response language preference")
    message_type: MessageType = Field(default=MessageType.TEXT, description="Message content type")
    stream: bool = Field(default=False, description="Whether to stream the response")
    
    # Advanced options
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity")
    max_tokens: Optional[int] = Field(default=2000, ge=1, le=8000, description="Maximum response length")
    dharmic_guidance: bool = Field(default=True, description="Apply dharmic principles")
    spiritual_context: bool = Field(default=True, description="Include spiritual wisdom")

class ChatResponse(BaseModel):
    """Enhanced response from chat completion with dharmic features"""
    message: str = Field(..., description="Assistant response")
    conversation_id: str = Field(..., description="Conversation identifier")
    message_id: str = Field(..., description="Message unique identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    # Metadata
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    model_used: Optional[str] = Field(default=None, description="AI model used for response")
    modules_used: Optional[List[str]] = Field(default=None, description="Chakra modules used")
    
    # Standard metrics
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Response confidence")
    dharmic_alignment: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Dharmic alignment score")
    relevance: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Response relevance")
    
    # Dharmic Features
    dharmic_insights: Optional[List[str]] = Field(default=None, description="Dharmic insights provided")
    growth_suggestions: Optional[List[str]] = Field(default=None, description="Personal growth suggestions")
    spiritual_context: Optional[str] = Field(default=None, description="Spiritual context")
    ethical_guidance: Optional[str] = Field(default=None, description="Ethical guidance")
    response_style: Optional[str] = Field(default=None, description="Response style used")
    
    # Enhanced Enterprise Features
    cultural_sensitivity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Cultural sensitivity score")
    compassion_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Compassion score")
    wisdom_assessment: Optional[Dict[str, float]] = Field(default=None, description="Wisdom assessment metrics")
    safety_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Safety score")
    tradition_alignment: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Tradition alignment score")
    
    # Quality indicators
    quality_gates_passed: Optional[bool] = Field(default=True, description="Whether quality gates passed")
    evaluation_details: Optional[Dict[str, Any]] = Field(default=None, description="Detailed evaluation metrics")
    
    # Backward compatibility
    spiritual_insights: Optional[List[str]] = Field(default=None, description="Spiritual insights (legacy)")
    wisdom_level: Optional[str] = Field(default=None, description="Wisdom level category")

class ConversationSummary(BaseModel):
    """Summary of a conversation"""
    conversation_id: str = Field(..., description="Conversation identifier")
    title: Optional[str] = Field(default=None, description="Conversation title")
    summary: Optional[str] = Field(default=None, description="Conversation summary")
    key_topics: Optional[List[str]] = Field(default=None, description="Key topics discussed")
    spiritual_themes: Optional[List[str]] = Field(default=None, description="Spiritual themes explored")
    start_time: Optional[datetime] = Field(default=None, description="Conversation start time")
    end_time: Optional[datetime] = Field(default=None, description="Conversation end time")
    message_count: int = Field(default=0, description="Total messages in conversation")

class ConversationHistory(BaseModel):
    """Complete conversation history"""
    conversation_id: str = Field(..., description="Conversation identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="All messages in conversation")
    summary: Optional[ConversationSummary] = Field(default=None, description="Conversation summary")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Conversation metadata
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used")
    spiritual_journey_progress: Optional[Dict[str, Any]] = Field(default=None, description="Spiritual progress")
    wisdom_gained: Optional[List[str]] = Field(default=None, description="Wisdom insights gained")

class StreamingChunk(BaseModel):
    """Chunk of streaming response"""
    chunk_id: str = Field(..., description="Chunk identifier")
    conversation_id: str = Field(..., description="Conversation identifier")
    content: str = Field(..., description="Chunk content")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Chunk metadata")

class WisdomRequest(BaseModel):
    """Request for spiritual wisdom"""
    question: str = Field(..., description="Wisdom question", min_length=1, max_length=5000)
    category: Optional[str] = Field(default=None, description="Wisdom category")
    tradition: Optional[str] = Field(default=None, description="Spiritual tradition preference")
    depth: Optional[str] = Field(default="medium", description="Depth of wisdom (basic, medium, deep)")
    context: Optional[str] = Field(default=None, description="Personal context for wisdom")

class WisdomResponse(BaseModel):
    """Response containing spiritual wisdom"""
    wisdom: str = Field(..., description="Wisdom content")
    source: Optional[str] = Field(default=None, description="Source of wisdom")
    tradition: Optional[str] = Field(default=None, description="Spiritual tradition")
    category: Optional[str] = Field(default=None, description="Wisdom category")
    related_concepts: Optional[List[str]] = Field(default=None, description="Related spiritual concepts")
    practical_application: Optional[str] = Field(default=None, description="How to apply this wisdom")
    depth_level: Optional[str] = Field(default=None, description="Depth level of wisdom")

class ModuleInfo(BaseModel):
    """Information about system modules"""
    module_name: str = Field(..., description="Module name")
    module_type: str = Field(..., description="Module type")
    version: str = Field(default="1.0.0", description="Module version")
    status: str = Field(default="active", description="Module status")
    capabilities: List[str] = Field(default_factory=list, description="Module capabilities")
    description: Optional[str] = Field(default=None, description="Module description")
    health_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Module health score")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class EvaluationResult(BaseModel):
    """Result of response evaluation"""
    evaluation_id: str = Field(..., description="Evaluation identifier")
    response_quality: float = Field(..., ge=0.0, le=1.0, description="Response quality score")
    dharmic_alignment: float = Field(..., ge=0.0, le=1.0, description="Dharmic alignment score")
    spiritual_depth: float = Field(..., ge=0.0, le=1.0, description="Spiritual depth score")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy score")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    compassion_level: float = Field(..., ge=0.0, le=1.0, description="Compassion level score")
    
    # Qualitative assessments
    strengths: List[str] = Field(default_factory=list, description="Response strengths")
    improvements: List[str] = Field(default_factory=list, description="Areas for improvement")
    spiritual_insights: List[str] = Field(default_factory=list, description="Spiritual insights")
    
    # Overall evaluation
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall evaluation score")
    recommendation: str = Field(..., description="Evaluation recommendation")
    evaluation_timestamp: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class DharmicChatRequest(BaseModel):
    """Request for dharmic chat with enhanced spiritual processing"""
    message: str = Field(..., description="User message", min_length=1, max_length=10000)
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")
    user_id: Optional[str] = Field(default="seeker", description="User identifier")
    
    # Dharmic enhancement options
    include_personal_growth: bool = Field(default=True, description="Include personal growth insights")
    include_spiritual_guidance: bool = Field(default=True, description="Include spiritual guidance")
    include_ethical_guidance: bool = Field(default=True, description="Include ethical guidance")
    response_style: str = Field(default="conversational", description="Response style: conversational, wise, practical")

class DharmicChatResponse(BaseModel):
    """Response from dharmic chat with spiritual enhancements"""
    response: str = Field(..., description="Generated response")
    conversation_id: str = Field(..., description="Conversation identifier")
    
    # Dharmic enhancements
    dharmic_insights: List[str] = Field(default_factory=list, description="Dharmic insights")
    growth_suggestions: List[str] = Field(default_factory=list, description="Personal growth suggestions")
    spiritual_context: str = Field(default="", description="Spiritual context")
    ethical_guidance: str = Field(default="", description="Ethical guidance")
    conversation_style: str = Field(default="conversational", description="Applied conversation style")
    
    # Processing information
    processing_info: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

# ===============================
# RISHI PERSONA MODELS
# ===============================

class RishiType(str, Enum):
    """Available Rishi personas"""
    VALMIKI = "valmiki"
    VYASA = "vyasa" 
    NARADA = "narada"
    VASISHTA = "vasishta"
    PATANJALI = "patanjali"

class ConversationType(str, Enum):
    """Types of conversations"""
    RISHI_GUIDANCE = "rishi_guidance"
    GENERAL_SPIRITUAL = "general_spiritual"
    MEDITATION_SUPPORT = "meditation_support"
    SCRIPTURAL_STUDY = "scriptural_study"

class RishiChatRequest(BaseModel):
    """Chat request with specific Rishi"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    rishi_id: RishiType = Field(..., description="Selected Rishi persona")
    conversation_id: str = Field(..., description="Conversation identifier")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    guidance_level: Optional[str] = Field(default="balanced", description="Guidance depth: light, balanced, deep")

class RishiSelectionRequest(BaseModel):
    """Request to select a Rishi for conversation"""
    rishi_id: RishiType = Field(..., description="Selected Rishi persona")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Initial context")
    conversation_goal: Optional[str] = Field(default=None, description="Goal for the conversation")

class RishiInfo(BaseModel):
    """Information about a Rishi persona"""
    id: RishiType
    name: str
    specialty: str
    description: str
    style: str
    context: str
    avatar_url: Optional[str] = None
    premium_only: bool = False
    
class UserChatPreferences(BaseModel):
    """User's chat preferences"""
    preferred_rishis: List[RishiType] = Field(default_factory=list, description="Preferred Rishi personas")
    conversation_style: str = Field(default="balanced", description="Conversation style: casual, balanced, formal")
    spiritual_background: str = Field(default="general", description="Spiritual background")
    language_preference: str = Field(default="english", description="Language preference")
    guidance_depth: str = Field(default="balanced", description="Guidance depth preference")
    topics_of_interest: List[str] = Field(default_factory=list, description="Topics of interest")

class ChatHistoryResponse(BaseModel):
    """Conversation history response"""
    conversation_id: str
    conversation_type: ConversationType
    rishi_name: Optional[str] = None
    started_at: datetime
    last_message_at: datetime
    message_count: int
    messages: List[ChatMessage]
    summary: Optional[str] = None

class ChatUsageStats(BaseModel):
    """User's chat usage statistics"""
    user_id: str
    current_period_usage: Dict[str, int] = Field(default_factory=dict, description="Current period usage by feature")
    total_usage: Dict[str, int] = Field(default_factory=dict, description="Total usage by feature")
    favorite_rishi: Optional[str] = Field(default=None, description="Most used Rishi")
    most_discussed_topics: List[str] = Field(default_factory=list, description="Most discussed topics")
    average_session_length: Optional[float] = Field(default=None, description="Average session length in minutes")
=======
"""
Chat Data Models

Pydantic models for chat-related API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    """Message role enumeration"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message")
    context: Optional[str] = Field(default=None, description="Additional context")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    user_id: Optional[str] = Field(default=None, description="User ID")
    language: str = Field(default="en", description="Response language")

class ChatMessage(BaseModel):
    """Individual chat message in conversation"""
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI response")
    conversation_id: str = Field(..., description="Conversation ID")
    modules_used: List[str] = Field(default=[], description="Dharma modules used")
    confidence_score: float = Field(..., description="Response confidence (0-1)")
    dharmic_alignment: float = Field(..., description="Dharmic alignment score (0-1)")
    sources: List[str] = Field(default=[], description="Source references")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")
    timestamp: datetime = Field(..., description="Response timestamp")
    model_used: str = Field(..., description="LLM model used")
    processing_time: float = Field(..., description="Processing time in seconds")

class ConversationHistory(BaseModel):
    """Conversation history model"""
    conversation_id: str
    messages: List[ChatMessage]
    total_messages: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class EvaluationResult(BaseModel):
    """Response evaluation result"""
    confidence_score: float = Field(..., ge=0, le=1)
    dharmic_alignment: float = Field(..., ge=0, le=1)
    relevance_score: float = Field(..., ge=0, le=1)
    sources: List[str] = Field(default=[])
    suggestions: List[str] = Field(default=[])
    explanation: Optional[str] = None
    
class ModuleInfo(BaseModel):
    """Dharma module information"""
    name: str
    description: str
    category: str
    expertise_areas: List[str]
    confidence: float = Field(..., ge=0, le=1)
    yaml_path: Optional[str] = None
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
