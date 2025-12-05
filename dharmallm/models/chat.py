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
