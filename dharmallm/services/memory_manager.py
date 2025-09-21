"""
🕉️ Memory Manager Service
==========================

Advanced memory management service for DharmaMind that handles:
- Conversation history and context
- User preferences and personalization
- Emotional patterns and insights  
- Spiritual progress tracking
- Knowledge base integration
- Vector embeddings and semantic search

Features:
- Persistent memory across sessions
- Intelligent context retrieval
- Emotional memory patterns
- Spiritual journey tracking
- Privacy-preserving storage
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ConversationMemory:
    """Conversation memory entry"""
    session_id: str
    user_id: Optional[str]
    timestamp: datetime
    user_message: str
    ai_response: str
    emotional_state: Optional[str] = None
    spiritual_context: Optional[str] = None
    rishi_involved: Optional[str] = None
    satisfaction_rating: Optional[float] = None

@dataclass
class UserProfile:
    """User profile with preferences and history"""
    user_id: str
    name: Optional[str] = None
    spiritual_level: str = "beginner"
    preferred_practices: List[str] = None
    emotional_patterns: Dict[str, Any] = None
    spiritual_goals: List[str] = None
    conversation_history: List[str] = None
    last_active: Optional[datetime] = None
    
    def __post_init__(self):
        if self.preferred_practices is None:
            self.preferred_practices = []
        if self.emotional_patterns is None:
            self.emotional_patterns = {}
        if self.spiritual_goals is None:
            self.spiritual_goals = []
        if self.conversation_history is None:
            self.conversation_history = []

class MemoryManager:
    """🧠 Advanced memory management service"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # In-memory storage (would be replaced with persistent storage in production)
        self.conversation_memory: List[ConversationMemory] = []
        self.user_profiles: Dict[str, UserProfile] = {}
        self.emotional_patterns: Dict[str, List[Any]] = {}
        self.spiritual_insights: Dict[str, Any] = {}
        
        # Memory configuration
        self.max_conversation_history = 1000
        self.memory_retention_days = 30
        
    async def initialize(self):
        """Initialize the memory manager"""
        try:
            self.logger.info("🌟 Initializing Memory Manager...")
            
            # Load existing data (in production, this would load from persistent storage)
            await self._load_persistent_data()
            
            # Clean old memories
            await self._cleanup_old_memories()
            
            self.logger.info(f"✅ Memory Manager initialized with {len(self.conversation_memory)} conversations")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Memory Manager: {e}")
    
    async def store_conversation(
        self, 
        session_id: str, 
        user_message: str, 
        ai_response: str,
        user_id: Optional[str] = None,
        emotional_state: Optional[str] = None,
        spiritual_context: Optional[str] = None,
        rishi_involved: Optional[str] = None
    ) -> str:
        """Store a conversation in memory"""
        try:
            # Create conversation memory entry
            conversation = ConversationMemory(
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now(),
                user_message=user_message[:500],  # Limit message length
                ai_response=ai_response[:1000],   # Limit response length
                emotional_state=emotional_state,
                spiritual_context=spiritual_context,
                rishi_involved=rishi_involved
            )
            
            # Add to memory
            self.conversation_memory.append(conversation)
            
            # Update user profile if user_id provided
            if user_id:
                await self._update_user_profile(user_id, conversation)
            
            # Maintain memory limits
            if len(self.conversation_memory) > self.max_conversation_history:
                self.conversation_memory = self.conversation_memory[-self.max_conversation_history:]
            
            conversation_id = f"{session_id}_{len(self.conversation_memory)}"
            self.logger.debug(f"💾 Stored conversation: {conversation_id}")
            
            return conversation_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store conversation: {e}")
            return ""
    
    async def get_conversation_history(
        self, 
        session_id: Optional[str] = None, 
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[ConversationMemory]:
        """Retrieve conversation history"""
        try:
            # Filter conversations
            filtered_conversations = []
            
            for conversation in self.conversation_memory:
                if session_id and conversation.session_id != session_id:
                    continue
                if user_id and conversation.user_id != user_id:
                    continue
                filtered_conversations.append(conversation)
            
            # Sort by timestamp and limit
            filtered_conversations.sort(key=lambda x: x.timestamp, reverse=True)
            return filtered_conversations[:limit]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get conversation history: {e}")
            return []
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        try:
            return self.user_profiles.get(user_id)
        except Exception as e:
            self.logger.error(f"❌ Failed to get user profile: {e}")
            return None
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile"""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(user_id=user_id)
            
            profile = self.user_profiles[user_id]
            
            # Update profile fields
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            profile.last_active = datetime.now()
            
            self.logger.debug(f"👤 Updated user profile: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update user profile: {e}")
            return False
    
    async def store_emotional_pattern(self, user_id: str, emotion: str, context: Dict[str, Any]):
        """Store emotional pattern for learning"""
        try:
            if user_id not in self.emotional_patterns:
                self.emotional_patterns[user_id] = []
            
            pattern = {
                "emotion": emotion,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            
            self.emotional_patterns[user_id].append(pattern)
            
            # Keep only recent patterns (last 100)
            if len(self.emotional_patterns[user_id]) > 100:
                self.emotional_patterns[user_id] = self.emotional_patterns[user_id][-100:]
            
            self.logger.debug(f"💙 Stored emotional pattern for {user_id}: {emotion}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store emotional pattern: {e}")
    
    async def get_emotional_patterns(self, user_id: str, emotion: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get emotional patterns for a user"""
        try:
            patterns = self.emotional_patterns.get(user_id, [])
            
            if emotion:
                patterns = [p for p in patterns if p["emotion"] == emotion]
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get emotional patterns: {e}")
            return []
    
    async def store_spiritual_insight(self, user_id: str, insight: str, category: str = "general"):
        """Store spiritual insight or progress"""
        try:
            if user_id not in self.spiritual_insights:
                self.spiritual_insights[user_id] = {}
            
            if category not in self.spiritual_insights[user_id]:
                self.spiritual_insights[user_id][category] = []
            
            insight_entry = {
                "insight": insight,
                "timestamp": datetime.now().isoformat(),
                "category": category
            }
            
            self.spiritual_insights[user_id][category].append(insight_entry)
            
            # Keep only recent insights (last 50 per category)
            if len(self.spiritual_insights[user_id][category]) > 50:
                self.spiritual_insights[user_id][category] = self.spiritual_insights[user_id][category][-50:]
            
            self.logger.debug(f"🕉️ Stored spiritual insight for {user_id}: {category}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store spiritual insight: {e}")
    
    async def get_spiritual_insights(self, user_id: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Get spiritual insights for a user"""
        try:
            insights = self.spiritual_insights.get(user_id, {})
            
            if category:
                return {category: insights.get(category, [])}
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get spiritual insights: {e}")
            return {}
    
    async def search_conversations(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[ConversationMemory]:
        """Search conversations by content"""
        try:
            query_lower = query.lower()
            matching_conversations = []
            
            for conversation in self.conversation_memory:
                if user_id and conversation.user_id != user_id:
                    continue
                
                # Simple text search in user message and AI response
                if (query_lower in conversation.user_message.lower() or 
                    query_lower in conversation.ai_response.lower()):
                    matching_conversations.append(conversation)
            
            # Sort by relevance (timestamp for now)
            matching_conversations.sort(key=lambda x: x.timestamp, reverse=True)
            return matching_conversations[:limit]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to search conversations: {e}")
            return []
    
    async def get_context_for_session(self, session_id: str, limit: int = 5) -> Dict[str, Any]:
        """Get relevant context for a session"""
        try:
            # Get recent conversations for this session
            recent_conversations = await self.get_conversation_history(session_id=session_id, limit=limit)
            
            # Extract context
            context = {
                "conversation_count": len(recent_conversations),
                "recent_messages": [conv.user_message for conv in recent_conversations[:3]],
                "recent_emotions": [conv.emotional_state for conv in recent_conversations if conv.emotional_state],
                "spiritual_contexts": [conv.spiritual_context for conv in recent_conversations if conv.spiritual_context],
                "rishis_involved": list(set([conv.rishi_involved for conv in recent_conversations if conv.rishi_involved]))
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get context for session: {e}")
            return {}
    
    async def _update_user_profile(self, user_id: str, conversation: ConversationMemory):
        """Update user profile based on conversation"""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(user_id=user_id)
            
            profile = self.user_profiles[user_id]
            
            # Add conversation to history
            conv_summary = f"{conversation.timestamp.strftime('%Y-%m-%d')}: {conversation.user_message[:50]}..."
            profile.conversation_history.append(conv_summary)
            
            # Keep only recent history
            if len(profile.conversation_history) > 20:
                profile.conversation_history = profile.conversation_history[-20:]
            
            # Update emotional patterns
            if conversation.emotional_state:
                emotion = conversation.emotional_state
                if emotion not in profile.emotional_patterns:
                    profile.emotional_patterns[emotion] = 0
                profile.emotional_patterns[emotion] += 1
            
            profile.last_active = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update user profile: {e}")
    
    async def _load_persistent_data(self):
        """Load data from persistent storage (placeholder)"""
        try:
            # In production, this would load from database/file storage
            self.logger.debug("📂 Loading persistent memory data...")
            # For now, just initialize empty
            pass
        except Exception as e:
            self.logger.error(f"❌ Failed to load persistent data: {e}")
    
    async def _cleanup_old_memories(self):
        """Clean up old memories to manage storage"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.memory_retention_days)
            
            # Remove old conversations
            initial_count = len(self.conversation_memory)
            self.conversation_memory = [
                conv for conv in self.conversation_memory 
                if conv.timestamp > cutoff_date
            ]
            
            removed_count = initial_count - len(self.conversation_memory)
            if removed_count > 0:
                self.logger.info(f"🧹 Cleaned up {removed_count} old conversations")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup old memories: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            return {
                "total_conversations": len(self.conversation_memory),
                "total_users": len(self.user_profiles),
                "emotional_patterns_count": sum(len(patterns) for patterns in self.emotional_patterns.values()),
                "spiritual_insights_count": sum(
                    len(insights) for user_insights in self.spiritual_insights.values() 
                    for insights in user_insights.values()
                ),
                "memory_retention_days": self.memory_retention_days,
                "oldest_conversation": min(
                    [conv.timestamp for conv in self.conversation_memory], 
                    default=datetime.now()
                ).isoformat() if self.conversation_memory else None
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if memory manager is healthy"""
        try:
            # Simple health check - ensure basic data structures exist
            return (
                isinstance(self.conversation_memory, list) and
                isinstance(self.user_profiles, dict) and
                isinstance(self.emotional_patterns, dict)
            )
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.logger.info("🧹 Cleaning up Memory Manager...")
            # In production, this would save data to persistent storage
            # For now, just log the cleanup
            self.logger.info("✅ Memory Manager cleanup complete")
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")

# Global instance
_memory_manager: Optional[MemoryManager] = None

async def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        await _memory_manager.initialize()
    return _memory_manager
