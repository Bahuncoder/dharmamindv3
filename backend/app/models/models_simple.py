"""
Simplified models for DharmaMind backend
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import uuid

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ResponseType(str, Enum):
    """Response type preferences"""
    CONTEMPLATIVE = "contemplative"
    PRACTICAL = "practical"
    SCHOLARLY = "scholarly"
    DEVOTIONAL = "devotional"
    ADAPTIVE = "adaptive"


class DharmicPrinciple(str, Enum):
    """Core dharmic principles"""
    DHARMA = "dharma"
    ARTHA = "artha"
    KAMA = "kama"
    MOKSHA = "moksha"
    AHIMSA = "ahimsa"
    SATYA = "satya"
    ASTEYA = "asteya"
    BRAHMACHARYA = "brahmacharya"
    APARIGRAHA = "aparigraha"


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=4096, description="User message")
    context: Optional[str] = Field(default=None, max_length=2048, description="Additional context")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    user_id: Optional[str] = Field(default=None, description="User ID")
    language: str = Field(default="en", pattern="^[a-z]{2}$", description="Response language (ISO 639-1)")
    response_type: Optional[ResponseType] = Field(default=None, description="Preferred response type")
    dharmic_focus: Optional[List[DharmicPrinciple]] = Field(default=None, description="Dharmic principles to emphasize")
    max_response_length: Optional[int] = Field(default=2048, ge=100, le=4096, description="Maximum response length")
    include_sources: bool = Field(default=True, description="Include source references")
    include_suggestions: bool = Field(default=True, description="Include follow-up suggestions")
    save_conversation: bool = Field(default=True, description="Save conversation history")


class ChatMessage(BaseModel):
    """Chat message model"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Message ID")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Dict = Field(default_factory=dict, description="Message metadata")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI response content")
    conversation_id: str = Field(..., description="Conversation ID")
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Response message ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    metadata: Dict = Field(default_factory=dict, description="Response metadata")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Response confidence")
    dharmic_score: Optional[float] = Field(default=None, ge=0, le=1, description="Dharmic alignment score")
    relevance: Optional[float] = Field(default=None, ge=0, le=1, description="Response relevance")
    wisdom_score: Optional[float] = Field(default=None, ge=0, le=1, description="Wisdom content score")
    dharmic_principles: List[DharmicPrinciple] = Field(default_factory=list, description="Dharmic principles addressed")
    spiritual_modules: List[str] = Field(default_factory=list, description="Spiritual modules engaged")
    sources: List[str] = Field(default_factory=list, description="Source references")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    related_topics: List[str] = Field(default_factory=list, description="Related spiritual topics")


class UserPreferences(BaseModel):
    """User preferences model"""
    user_id: str = Field(..., description="User ID")
    language: str = Field(default="en", description="Preferred language")
    response_type: ResponseType = Field(default=ResponseType.ADAPTIVE, description="Preferred response type")
    dharmic_focus: List[DharmicPrinciple] = Field(default_factory=list, description="Preferred dharmic principles")
    max_response_length: int = Field(default=2048, ge=100, le=4096, description="Maximum response length")
    include_sources: bool = Field(default=True, description="Include source references")
    include_suggestions: bool = Field(default=True, description="Include follow-up suggestions")
    spiritual_level: str = Field(default="beginner", description="Spiritual understanding level")


class UserProfile(BaseModel):
    """User profile model"""
    user_id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    created_at: datetime = Field(default_factory=datetime.now, description="Account creation time")
    last_active: datetime = Field(default_factory=datetime.now, description="Last activity time")
    preferences: UserPreferences = Field(..., description="User preferences")
    conversation_count: int = Field(default=0, description="Total conversations")
    dharmic_progress: Dict[str, float] = Field(default_factory=dict, description="Dharmic progress tracking")


# Export all models
__all__ = [
    "MessageRole",
    "ResponseType", 
    "DharmicPrinciple",
    "ChatRequest",
    "ChatMessage",
    "ChatResponse",
    "UserPreferences",
    "UserProfile"
]
