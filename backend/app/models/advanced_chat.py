"""
🕉️ Advanced Chat Models - Enterprise Grade
==========================================

Comprehensive models for advanced chat functionality including:
- Advanced chat requests with complex context
- Streaming responses and real-time features  
- Personalization and user preferences
- Analytics and performance metrics
- Enterprise features and workflows
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

# Extend existing chat models with advanced features
from .chat import ChatRequest, ChatResponse, RishiType, ConversationType

class ComplexityLevel(str, Enum):
    """Request complexity levels"""
    SIMPLE = "simple"           # Basic questions
    MODERATE = "moderate"       # Multi-part questions  
    COMPLEX = "complex"         # Deep philosophical queries
    RESEARCH = "research"       # Extensive research needed

class PersonalizationLevel(str, Enum):
    """Personalization depth levels"""
    MINIMAL = "minimal"         # Basic personalization
    BALANCED = "balanced"       # Standard personalization
    DEEP = "deep"              # Comprehensive personalization
    CUSTOM = "custom"          # Fully customized responses

class ResponseStyle(str, Enum):
    """AI response styles"""
    CONVERSATIONAL = "conversational"  # Natural conversation
    SCHOLARLY = "scholarly"            # Academic/formal
    PRACTICAL = "practical"            # Action-oriented  
    INSPIRATIONAL = "inspirational"    # Motivational
    MEDITATIVE = "meditative"          # Contemplative

class CreativityLevel(str, Enum):
    """Response creativity levels"""
    CONSERVATIVE = "conservative"      # Traditional wisdom
    BALANCED = "balanced"              # Mix of traditional and creative
    CREATIVE = "creative"              # Innovative interpretations
    EXPERIMENTAL = "experimental"     # Cutting-edge insights

class AdvancedChatRequest(BaseModel):
    """Advanced chat request with comprehensive options"""
    
    # Core message data
    message: str = Field(..., min_length=1, max_length=5000)
    conversation_id: Optional[str] = None
    rishi_id: Optional[RishiType] = None
    conversation_type: ConversationType = ConversationType.GENERAL_SPIRITUAL
    
    # Context and personalization
    context: Optional[Dict[str, Any]] = None
    session_context: Optional[Dict[str, Any]] = None
    mood_context: Optional[str] = None
    personalization_level: PersonalizationLevel = PersonalizationLevel.BALANCED
    
    # AI processing options
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, ge=100, le=4000)
    creativity_level: CreativityLevel = CreativityLevel.BALANCED
    response_style: ResponseStyle = ResponseStyle.CONVERSATIONAL
    complexity_score: Optional[float] = Field(default=1.0, ge=0.1, le=10.0)
    
    # Advanced features
    include_insights: bool = True
    include_suggestions: bool = True
    include_related_topics: bool = False
    include_practical_steps: bool = False
    include_meditation_guidance: bool = False
    
    # Performance options
    use_cache: bool = True
    context_window_size: Optional[int] = Field(default=10, ge=1, le=50)
    priority_processing: bool = False
    
    # Multi-modal support
    attachments: Optional[List[Dict[str, Any]]] = None
    image_context: Optional[str] = None
    audio_context: Optional[str] = None

class StreamingChatResponse(BaseModel):
    """Streaming chat response chunk"""
    chunk_id: str
    conversation_id: str
    content: str
    is_final: bool = False
    chunk_type: str = "text"  # text, insight, suggestion, completion
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ContextualChatRequest(BaseModel):
    """Chat request with rich contextual information"""
    message: str
    conversation_id: str
    
    # Rich context
    user_intent: Optional[str] = None
    emotional_state: Optional[str] = None
    spiritual_goal: Optional[str] = None
    life_situation: Optional[str] = None
    
    # Conversation flow
    previous_topic: Optional[str] = None
    desired_depth: Optional[str] = None
    time_available: Optional[int] = None  # minutes
    
    # Preferences
    teaching_style: Optional[str] = None
    examples_preferred: bool = True
    scriptural_references: bool = True

class PersonalizedResponse(BaseModel):
    """Personalized AI response with user-specific adaptations"""
    base_response: str
    personalized_response: str
    personalization_applied: List[str]
    user_context_used: Dict[str, Any]
    adaptation_confidence: float = Field(ge=0.0, le=1.0)

class ChatTemplate(BaseModel):
    """Pre-defined conversation templates"""
    template_id: str
    name: str
    description: str
    category: str
    initial_prompts: List[str]
    expected_flow: List[str]
    required_context: List[str]
    estimated_duration: int  # minutes
    difficulty_level: str
    prerequisites: Optional[List[str]] = None

class ChatWorkflow(BaseModel):
    """Structured conversation workflows"""
    workflow_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    decision_points: List[Dict[str, Any]]
    completion_criteria: Dict[str, Any]
    adaptive_branching: bool = True
    personalization_points: List[str]

class ConversationInsight(BaseModel):
    """Insights generated from conversation analysis"""
    insight_id: str
    conversation_id: str
    insight_type: str  # pattern, growth_area, strength, recommendation
    insight_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    supporting_evidence: List[str]
    actionable_steps: Optional[List[str]] = None
    relevant_teachings: Optional[List[str]] = None

class UserGrowthProfile(BaseModel):
    """User's spiritual growth profile based on conversations"""
    user_id: str
    current_level: str
    growth_areas: List[str]
    strengths: List[str]
    recommended_practices: List[str]
    preferred_rishis: List[RishiType]
    conversation_patterns: Dict[str, Any]
    progress_metrics: Dict[str, float]
    next_milestones: List[str]
    
class AdvancedAnalytics(BaseModel):
    """Advanced analytics for chat interactions"""
    session_id: str
    user_id: str
    
    # Engagement metrics
    message_count: int
    session_duration: float  # minutes
    avg_response_time: float  # seconds
    user_satisfaction_score: Optional[float] = None
    
    # Quality metrics  
    avg_dharmic_alignment: float
    avg_confidence_score: float
    conversation_coherence: float
    spiritual_depth_achieved: float
    
    # Learning indicators
    concepts_explored: List[str]
    insights_generated: List[str]
    growth_indicators: List[str]
    knowledge_gaps_identified: List[str]
    
    # Personalization effectiveness
    personalization_success_rate: float
    context_utilization_score: float
    user_preference_alignment: float

class ChatGamification(BaseModel):
    """Gamification elements for chat engagement"""
    user_id: str
    
    # Achievement tracking
    conversations_completed: int
    rishis_consulted: List[RishiType]
    wisdom_points_earned: int
    streak_days: int
    milestones_reached: List[str]
    
    # Skill development
    spiritual_skills: Dict[str, int]  # skill -> level
    knowledge_areas: Dict[str, float]  # area -> mastery
    practice_consistency: Dict[str, float]  # practice -> consistency
    
    # Social features
    community_contributions: int
    teachings_shared: int
    discussions_participated: int
    mentorship_given: int

class MultiModalChatRequest(BaseModel):
    """Multi-modal chat supporting text, images, audio"""
    text_message: Optional[str] = None
    image_data: Optional[str] = None  # base64 encoded
    audio_data: Optional[str] = None  # base64 encoded
    document_data: Optional[str] = None
    
    # Modal-specific options
    image_analysis_type: Optional[str] = None
    audio_transcription_needed: bool = True
    document_summary_needed: bool = True
    
    # Integration preferences
    cross_modal_insights: bool = True
    unified_response: bool = True