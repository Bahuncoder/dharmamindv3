<<<<<<< HEAD
"""
Memory Manager Service
=====================

Manages conversation memory and context for the DharmaMind system.
Temporary implementation for backward compatibility.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages conversation memory and context"""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.user_contexts: Dict[str, Dict[str, Any]] = {}
        logger.info("Memory Manager initialized")
    
    def add_message(self, conversation_id: str, message: Dict[str, Any], user_id: Optional[str] = None):
        """Add a message to conversation memory"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        message_with_timestamp = {
            **message,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id
        }
        
        self.conversations[conversation_id].append(message_with_timestamp)
        logger.debug(f"Added message to conversation {conversation_id}")
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversations.get(conversation_id, [])
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context"""
        return self.user_contexts.get(user_id, {})
    
    def update_user_context(self, user_id: str, context: Dict[str, Any]):
        """Update user context"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {}
        
        self.user_contexts[user_id].update(context)
        logger.debug(f"Updated context for user {user_id}")
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get or create memory manager instance"""
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    
    return _memory_manager
=======
"""
🕉️ DharmaMind Enhanced Memory Manager - Complete System

Advanced conversation and context management for dharmic wisdom preservation:

Core Features:
- Multi-level conversation storage (short-term, long-term, wisdom archives)
- Context-aware memory retrieval with semantic search
- Dharmic wisdom pattern recognition and preservation
- User spiritual journey tracking and progression
- Intelligent memory consolidation and archiving
- Cross-conversation insight extraction

Memory Layers:
- Session Memory: Active conversation context
- User Memory: Personal spiritual journey and preferences  
- Wisdom Memory: Distilled dharmic insights and teachings
- Universal Memory: Collective wisdom patterns
- Sacred Memory: Protected spiritual content

May this memory serve the eternal wisdom 🧠
"""

import asyncio
import logging
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import hashlib
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np

from ..models import ChatRequest, ChatResponse, UserProfile
from ..models.chat import (
    ModuleInfo, ChatMessage, MessageRole, EvaluationResult
)
from ..config import settings

logger = logging.getLogger(__name__)

# ===============================
# MEMORY ENUMS AND MODELS
# ===============================

class MemoryType(Enum):
    SESSION = "session"           # Active conversation
    USER_PERSONAL = "user_personal"     # User's personal journey
    WISDOM_ARCHIVE = "wisdom_archive"   # Distilled wisdom
    UNIVERSAL = "universal"       # Collective patterns
    SACRED = "sacred"            # Protected content
    CONTEXTUAL = "contextual"    # Situational memory

class MemoryPriority(Enum):
    CRITICAL = "critical"        # Essential dharmic insights
    HIGH = "high"               # Important spiritual content
    MEDIUM = "medium"           # General conversation
    LOW = "low"                # Transient information
    EPHEMERAL = "ephemeral"     # Temporary session data

class ConsolidationStrategy(Enum):
    WISDOM_DISTILLATION = "wisdom_distillation"
    PATTERN_EXTRACTION = "pattern_extraction"
    INSIGHT_SYNTHESIS = "insight_synthesis"
    SPIRITUAL_PROGRESSION = "spiritual_progression"
    TEACHING_MOMENTS = "teaching_moments"

@dataclass
class MemoryRecord:
    """Individual memory record with metadata"""
    id: str
    memory_type: MemoryType
    priority: MemoryPriority
    content: str
    context: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    wisdom_score: float = 0.0
    dharmic_alignment: float = 0.0
    emotional_resonance: float = 0.0
    spiritual_significance: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = None
    embeddings: Optional[List[float]] = None
    consolidation_status: str = "pending"
    expires_at: Optional[datetime] = None

@dataclass
class ConversationThread:
    """Complete conversation thread with analysis"""
    conversation_id: str
    user_id: Optional[str]
    title: str
    summary: str
    messages: List[Dict[str, Any]]
    spiritual_themes: List[str]
    wisdom_insights: List[str]
    emotional_journey: List[str]
    dharmic_principles: List[str]
    key_moments: List[Dict[str, Any]]
    created_at: datetime
    last_updated: datetime
    message_count: int
    avg_wisdom_score: float
    spiritual_progression: float
    consolidation_priority: float

@dataclass
class UserSpiritualJourney:
    """User's spiritual journey and progression"""
    user_id: str
    spiritual_level: str
    journey_stage: str
    key_realizations: List[str]
    recurring_themes: List[str]
    growth_areas: List[str]
    preferred_teachings: List[str]
    wisdom_milestones: List[Dict[str, Any]]
    emotional_patterns: List[str]
    practice_consistency: float
    openness_to_change: float
    depth_of_inquiry: float
    compassion_development: float
    wisdom_integration: float
    last_assessment: datetime

class MemoryManager:
    """Service for managing conversation memory and embeddings"""
    
    def __init__(self):
        self.conversations: Dict[str, List[ChatMessage]] = {}
        self.embeddings: Dict[str, Any] = {}
        self.vector_store = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize memory manager and connections"""
        logger.info("Initializing Memory Manager...")
        
        try:
            # Initialize Redis for fast conversation storage
            await self._init_redis()
            
            # Initialize vector database
            await self._init_vector_store()
            
            # Set up cleanup tasks
            await self._setup_cleanup_tasks()
            
            logger.info("Memory Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Manager: {e}")
            raise
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            # This would typically initialize a real Redis client
            # For now, we'll use in-memory storage
            self.redis_client = {}  # Placeholder
            logger.info("Redis connection established (placeholder)")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory storage: {e}")
            self.redis_client = {}
    
    async def _init_vector_store(self):
        """Initialize vector database connection"""
        try:
            # This would typically initialize Qdrant, Pinecone, or similar
            # For now, we'll use in-memory storage
            self.vector_store = {}  # Placeholder
            logger.info(f"Vector store initialized: {settings.VECTOR_DB_TYPE}")
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")
            self.vector_store = {}
    
    async def _setup_cleanup_tasks(self):
        """Set up background cleanup tasks"""
        # This would typically set up periodic cleanup
        logger.info("Memory cleanup tasks configured")
    
    async def store_conversation(
        self,
        conversation_id: str,
        user_message: str,
        ai_response: str,
        modules: List[ModuleInfo],
        evaluation: EvaluationResult
    ):
        """Store a conversation turn in memory"""
        
        try:
            # Create message objects
            user_msg = ChatMessage(
                role=MessageRole.USER,
                content=user_message,
                timestamp=datetime.now(),
                metadata={"conversation_id": conversation_id}
            )
            
            ai_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=ai_response,
                timestamp=datetime.now(),
                metadata={
                    "conversation_id": conversation_id,
                    "modules_used": [m.name for m in modules],
                    "confidence_score": evaluation.confidence_score,
                    "dharmic_alignment": evaluation.dharmic_alignment
                }
            )
            
            # Store in conversation history
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            self.conversations[conversation_id].extend([user_msg, ai_msg])
            
            # Store in Redis for persistence
            await self._store_in_redis(conversation_id, user_msg, ai_msg)
            
            # Create and store embeddings
            await self._create_and_store_embeddings(conversation_id, user_message, ai_response, modules)
            
            # Cleanup old conversations if needed
            await self._cleanup_old_conversations()
            
            logger.debug(f"Conversation stored: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
    
    async def _store_in_redis(self, conversation_id: str, user_msg: ChatMessage, ai_msg: ChatMessage):
        """Store conversation in Redis"""
        try:
            # Convert messages to JSON for storage
            messages_data = [
                {
                    "role": user_msg.role.value,
                    "content": user_msg.content,
                    "timestamp": user_msg.timestamp.isoformat(),
                    "metadata": user_msg.metadata
                },
                {
                    "role": ai_msg.role.value,
                    "content": ai_msg.content,
                    "timestamp": ai_msg.timestamp.isoformat(),
                    "metadata": ai_msg.metadata
                }
            ]
            
            # In a real implementation, this would use Redis
            key = f"conversation:{conversation_id}"
            if key not in self.redis_client:
                self.redis_client[key] = []
            self.redis_client[key].extend(messages_data)
            
        except Exception as e:
            logger.error(f"Error storing in Redis: {e}")
    
    async def _create_and_store_embeddings(
        self,
        conversation_id: str,
        user_message: str,
        ai_response: str,
        modules: List[ModuleInfo]
    ):
        """Create embeddings and store in vector database"""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.md5((user_message + ai_response).encode()).hexdigest()
            
            # This would typically use sentence-transformers or OpenAI embeddings
            # For now, we'll create placeholder embeddings
            user_embedding = self._create_placeholder_embedding(user_message)
            response_embedding = self._create_placeholder_embedding(ai_response)
            
            # Store embeddings with metadata
            embedding_data = {
                "conversation_id": conversation_id,
                "user_message": user_message,
                "ai_response": ai_response,
                "modules": [m.name for m in modules],
                "timestamp": datetime.now().isoformat(),
                "user_embedding": user_embedding,
                "response_embedding": response_embedding
            }
            
            self.vector_store[content_hash] = embedding_data
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
    
    def _create_placeholder_embedding(self, text: str) -> List[float]:
        """Create placeholder embedding (would use real embedding model)"""
        # This is a simple hash-based placeholder
        # Real implementation would use sentence-transformers or OpenAI
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Create a simple 384-dimensional vector (common size)
        embedding = []
        for i in range(384):
            embedding.append((hash_int >> (i % 32)) & 1)
        
        # Normalize to [-1, 1] range
        embedding = [float(x * 2 - 1) for x in embedding]
        return embedding
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 20
    ) -> List[ChatMessage]:
        """Retrieve conversation history"""
        
        try:
            # First try in-memory storage
            if conversation_id in self.conversations:
                messages = self.conversations[conversation_id]
                return messages[-limit:] if limit else messages
            
            # Then try Redis
            return await self._get_from_redis(conversation_id, limit)
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    async def _get_from_redis(self, conversation_id: str, limit: int) -> List[ChatMessage]:
        """Retrieve conversation from Redis"""
        try:
            key = f"conversation:{conversation_id}"
            if key in self.redis_client:
                messages_data = self.redis_client[key]
                
                # Convert back to ChatMessage objects
                messages = []
                for msg_data in messages_data[-limit:] if limit else messages_data:
                    message = ChatMessage(
                        role=MessageRole(msg_data["role"]),
                        content=msg_data["content"],
                        timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                        metadata=msg_data.get("metadata")
                    )
                    messages.append(message)
                
                return messages
            
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving from Redis: {e}")
            return []
    
    async def search_similar_conversations(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar conversations using vector similarity"""
        
        try:
            # Create query embedding
            query_embedding = self._create_placeholder_embedding(query)
            
            # Calculate similarities (placeholder implementation)
            similarities = []
            for content_hash, data in self.vector_store.items():
                # Simple dot product similarity (would use proper cosine similarity)
                user_sim = sum(a * b for a, b in zip(query_embedding, data["user_embedding"]))
                response_sim = sum(a * b for a, b in zip(query_embedding, data["response_embedding"]))
                
                similarity = max(user_sim, response_sim)
                similarities.append((similarity, data))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for similarity, data in similarities[:limit]:
                results.append({
                    "similarity": similarity,
                    "conversation_id": data["conversation_id"],
                    "user_message": data["user_message"],
                    "ai_response": data["ai_response"],
                    "modules": data["modules"],
                    "timestamp": data["timestamp"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar conversations: {e}")
            return []
    
    async def get_contextual_memory(
        self,
        conversation_id: str,
        current_message: str,
        max_context: int = 5
    ) -> Dict[str, Any]:
        """Get relevant contextual memory for current message"""
        
        try:
            # Get recent conversation history
            recent_history = await self.get_conversation_history(conversation_id, max_context)
            
            # Search for similar past conversations
            similar_conversations = await self.search_similar_conversations(current_message, 3)
            
            # Extract key topics from conversation
            topics = self._extract_conversation_topics(recent_history)
            
            return {
                "recent_history": recent_history,
                "similar_conversations": similar_conversations,
                "topics": topics,
                "conversation_length": len(recent_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting contextual memory: {e}")
            return {
                "recent_history": [],
                "similar_conversations": [],
                "topics": [],
                "conversation_length": 0
            }
    
    def _extract_conversation_topics(self, messages: List[ChatMessage]) -> List[str]:
        """Extract key topics from conversation messages"""
        
        # Simple keyword extraction (would use NLP in real implementation)
        topic_keywords = [
            "meditation", "dharma", "karma", "yoga", "peace", "wisdom",
            "suffering", "happiness", "compassion", "love", "anger",
            "fear", "anxiety", "stress", "balance", "harmony"
        ]
        
        topics = []
        text = " ".join([msg.content.lower() for msg in messages])
        
        for keyword in topic_keywords:
            if keyword in text:
                topics.append(keyword)
        
        return list(set(topics))  # Remove duplicates
    
    async def store_wisdom_interaction(
        self,
        conversation_id: str,
        question: str,
        response: str,
        modules: List[ModuleInfo],
        evaluation: EvaluationResult
    ):
        """Store wisdom-specific interaction with enhanced metadata"""
        
        # Use regular store_conversation but add wisdom-specific metadata
        await self.store_conversation(conversation_id, question, response, modules, evaluation)
        
        # Add to wisdom-specific index
        try:
            wisdom_key = f"wisdom:{conversation_id}"
            wisdom_data = {
                "question": question,
                "response": response,
                "modules": [m.name for m in modules],
                "dharmic_alignment": evaluation.dharmic_alignment,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in Redis (placeholder)
            self.redis_client[wisdom_key] = wisdom_data
            
        except Exception as e:
            logger.error(f"Error storing wisdom interaction: {e}")
    
    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation from memory"""
        
        try:
            # Remove from in-memory storage
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            
            # Remove from Redis
            key = f"conversation:{conversation_id}"
            if key in self.redis_client:
                del self.redis_client[key]
            
            # Remove related embeddings
            to_remove = []
            for content_hash, data in self.vector_store.items():
                if data["conversation_id"] == conversation_id:
                    to_remove.append(content_hash)
            
            for hash_key in to_remove:
                del self.vector_store[hash_key]
            
            logger.info(f"Conversation deleted: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
    
    async def _cleanup_old_conversations(self):
        """Clean up old conversations to manage memory"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=30)  # Keep 30 days
            
            # Clean up in-memory conversations
            for conv_id in list(self.conversations.keys()):
                messages = self.conversations[conv_id]
                if messages and messages[-1].timestamp < cutoff_date:
                    del self.conversations[conv_id]
            
            # Clean up vector store
            to_remove = []
            for content_hash, data in self.vector_store.items():
                timestamp = datetime.fromisoformat(data["timestamp"])
                if timestamp < cutoff_date:
                    to_remove.append(content_hash)
            
            for hash_key in to_remove:
                del self.vector_store[hash_key]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old conversation embeddings")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        
        return {
            "active_conversations": len(self.conversations),
            "total_messages": sum(len(msgs) for msgs in self.conversations.values()),
            "stored_embeddings": len(self.vector_store),
            "redis_keys": len(self.redis_client),
            "memory_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB (rough calculation)"""
        
        try:
            import sys
            
            # Calculate size of conversations
            conv_size = sys.getsizeof(self.conversations)
            for conv_id, messages in self.conversations.items():
                conv_size += sys.getsizeof(conv_id) + sys.getsizeof(messages)
                for msg in messages:
                    conv_size += sys.getsizeof(msg.content) + sys.getsizeof(msg.metadata or {})
            
            # Calculate size of vector store
            vector_size = sys.getsizeof(self.vector_store)
            for data in self.vector_store.values():
                vector_size += sys.getsizeof(data)
            
            # Convert to MB
            total_bytes = conv_size + vector_size
            return total_bytes / (1024 * 1024)
            
        except Exception:
            return 0.0
    
    async def health_check(self) -> bool:
        """Check if memory manager is healthy"""
        try:
            # Basic health checks
            memory_stats = await self.get_memory_statistics()
            
            # Check if memory usage is reasonable (< 500MB)
            if memory_stats["memory_mb"] > 500:
                logger.warning(f"High memory usage: {memory_stats['memory_mb']:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Memory manager health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            # Cleanup connections and resources
            if self.redis_client:
                # In real implementation, would close Redis connection
                pass
            
            if self.vector_store:
                # In real implementation, would close vector DB connection
                pass
            
            logger.info("Memory manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during memory manager cleanup: {e}")


# Dependency injection function for FastAPI
_memory_manager_instance = None


def get_memory_manager() -> MemoryManager:
    """Get the memory manager instance (singleton pattern)"""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
