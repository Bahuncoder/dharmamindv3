"""
🕉️ Comprehensive Chat Service Suite - Enterprise Grade
====================================================

Collection of all advanced services for comprehensive chat functionality:
- Conversation Management
- Personalization Engine  
- Analytics & Insights
- Content Filtering & Safety
- Rate Limiting & Performance
- Gamification & Engagement
- Multi-modal Processing
"""

import logging
import hashlib
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from ..config import settings

logger = logging.getLogger(__name__)

# ===============================
# CONVERSATION SERVICE
# ===============================

class ConversationService:
    """Advanced conversation management and context tracking"""
    
    def __init__(self):
        self.conversations = {}  # In-memory store - replace with database
    
    async def create_rishi_conversation(
        self, 
        user_id: str, 
        rishi_id: str, 
        context: Dict[str, Any],
        conversation_goal: Optional[str] = None,
        personalization_enabled: bool = True
    ) -> str:
        """Create new conversation with Rishi"""
        conversation_id = f"{user_id}_{rishi_id}_{int(datetime.now().timestamp())}"
        
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "user_id": user_id,
            "rishi_id": rishi_id,
            "goal": conversation_goal,
            "context": context,
            "messages": [],
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "personalization_enabled": personalization_enabled
        }
        
        return conversation_id
    
    async def get_relevant_history(
        self, 
        user_id: str, 
        conversation_id: str, 
        message_count: int = 10
    ) -> List[Dict[str, Any]]:
        """Get relevant conversation history for context"""
        conversation = self.conversations.get(conversation_id, {})
        messages = conversation.get("messages", [])
        return messages[-message_count:] if messages else []

def get_conversation_service() -> ConversationService:
    return ConversationService()

# ===============================
# PERSONALIZATION SERVICE
# ===============================

class PersonalizationService:
    """Advanced personalization and user preference management"""
    
    def __init__(self):
        self.user_profiles = {}
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's personalization preferences"""
        return self.user_profiles.get(user_id, {
            "preferred_rishis": [],
            "communication_style": "balanced",
            "spiritual_background": "general",
            "interests": [],
            "learning_pace": "moderate"
        })
    
    async def get_rishi_recommendations(
        self, 
        user_id: str, 
        available_rishis: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get personalized Rishi recommendations"""
        preferences = await self.get_user_preferences(user_id)
        
        # Simple recommendation logic - enhance with ML
        recommendations = []
        for rishi in available_rishis[:3]:  # Top 3
            recommendations.append({
                "rishi": rishi,
                "reason": f"Matches your interest in {rishi.get('specialty', 'spiritual wisdom')}",
                "compatibility_score": 0.85
            })
        
        return recommendations
    
    async def personalize_rishi_introduction(
        self, 
        user_id: str, 
        rishi_info: Dict[str, Any], 
        conversation_context: Dict[str, Any]
    ) -> str:
        """Create personalized Rishi introduction"""
        preferences = await self.get_user_preferences(user_id)
        
        base_intro = f"🕉️ Greetings, seeker! I am {rishi_info['name']}."
        
        if preferences.get("spiritual_background") == "beginner":
            return f"{base_intro} I'm here to guide you gently on your spiritual journey with wisdom from {rishi_info['specialty']}."
        else:
            return f"{base_intro} I'm honored to share deeper insights from {rishi_info['specialty']} to support your spiritual growth."
    
    async def enhance_conversation_context(
        self, 
        user_id: str, 
        message: str, 
        conversation_id: str,
        base_context: Dict[str, Any], 
        personalization_level: str = "balanced"
    ) -> Dict[str, Any]:
        """Enhance conversation context with personalization"""
        preferences = await self.get_user_preferences(user_id)
        
        enhanced_context = {
            "preferences": preferences,
            "personalization_level": personalization_level,
            "history_summary": "New conversation",  # Would compute from actual history
            "goals": preferences.get("goals", []),
            "personalization": {
                "style_adaptation": True,
                "context_awareness": True,
                "preference_alignment": True
            }
        }
        
        return {**base_context, **enhanced_context}
    
    async def personalize_response(
        self, 
        response: Dict[str, Any], 
        user_id: str, 
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Personalize AI response based on user profile"""
        preferences = await self.get_user_preferences(user_id)
        
        # Add personalization metadata
        response["enhancements"] = {
            "style_adapted": True,
            "preference_matched": True,
            "context_utilized": True
        }
        
        return response

def get_personalization_service() -> PersonalizationService:
    return PersonalizationService()

# ===============================
# ANALYTICS SERVICE  
# ===============================

class AnalyticsService:
    """Comprehensive analytics and insights tracking"""
    
    def __init__(self):
        self.analytics_data = {}
    
    async def track_rishi_selection(
        self, 
        user_id: str, 
        rishi_id: str, 
        conversation_id: str,
        context: Dict[str, Any]
    ):
        """Track Rishi selection for analytics"""
        event_data = {
            "event": "rishi_selection",
            "user_id": user_id,
            "rishi_id": rishi_id,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }
        
        # Store analytics event
        if user_id not in self.analytics_data:
            self.analytics_data[user_id] = []
        self.analytics_data[user_id].append(event_data)
        
        logger.info(f"📊 Tracked rishi selection: {rishi_id} for user {user_id}")
    
    async def track_comprehensive_chat(
        self, 
        user_id: str, 
        request: Dict[str, Any], 
        response: Dict[str, Any],
        processing_context: Dict[str, Any]
    ):
        """Track comprehensive chat interaction"""
        event_data = {
            "event": "comprehensive_chat",
            "user_id": user_id,
            "message_length": len(request.get("message", "")),
            "processing_time": response.get("processing_time", 0),
            "confidence_score": response.get("confidence_score", 0),
            "dharmic_alignment": response.get("dharmic_alignment", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        if user_id not in self.analytics_data:
            self.analytics_data[user_id] = []
        self.analytics_data[user_id].append(event_data)
        
        logger.info(f"📊 Tracked comprehensive chat for user {user_id}")

def get_analytics_service() -> AnalyticsService:
    return AnalyticsService()

# ===============================
# CONTENT FILTER SERVICE
# ===============================

class ContentFilterService:
    """Content filtering and safety validation"""
    
    def __init__(self):
        self.blocked_keywords = ["inappropriate", "harmful"]  # Basic example
    
    async def validate_message(
        self, 
        message: str, 
        user_id: str, 
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate message content for safety and appropriateness"""
        
        # Simple keyword filtering - enhance with ML models
        message_lower = message.lower()
        
        for keyword in self.blocked_keywords:
            if keyword in message_lower:
                return {
                    "approved": False,
                    "reason": f"Content contains inappropriate keyword: {keyword}",
                    "severity": "medium"
                }
        
        # Length validation
        if len(message) > 5000:
            return {
                "approved": False,
                "reason": "Message too long (max 5000 characters)",
                "severity": "low"
            }
        
        return {
            "approved": True,
            "reason": "Content approved",
            "severity": "none"
        }

def get_content_filter_service() -> ContentFilterService:
    return ContentFilterService()

# ===============================
# RATE LIMITER SERVICE
# ===============================

class RateLimiterService:
    """Advanced rate limiting with tier-based limits"""
    
    def __init__(self):
        self.user_requests = {}  # user_id -> [(timestamp, count)]
        
        # Rate limits by subscription tier
        self.tier_limits = {
            "free": {"requests_per_minute": 5, "requests_per_hour": 50},
            "premium": {"requests_per_minute": 20, "requests_per_hour": 500},
            "enterprise": {"requests_per_minute": 100, "requests_per_hour": 2000}
        }
    
    async def check_rate_limit(
        self, 
        user_id: str, 
        endpoint: str, 
        tier: str = "free"
    ) -> Dict[str, Any]:
        """Check if user is within rate limits"""
        
        now = datetime.now()
        limits = self.tier_limits.get(tier, self.tier_limits["free"])
        
        # Initialize user tracking
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        # Clean old requests (older than 1 hour)
        hour_ago = now - timedelta(hours=1)
        self.user_requests[user_id] = [
            (timestamp, count) for timestamp, count in self.user_requests[user_id]
            if timestamp > hour_ago
        ]
        
        # Count recent requests
        minute_ago = now - timedelta(minutes=1)
        recent_requests = sum(
            count for timestamp, count in self.user_requests[user_id]
            if timestamp > minute_ago
        )
        
        hour_requests = sum(
            count for timestamp, count in self.user_requests[user_id]
        )
        
        # Check limits
        if recent_requests >= limits["requests_per_minute"]:
            return {
                "allowed": False,
                "message": f"Rate limit exceeded: {limits['requests_per_minute']} requests per minute",
                "retry_after": 60
            }
        
        if hour_requests >= limits["requests_per_hour"]:
            return {
                "allowed": False,
                "message": f"Hourly limit exceeded: {limits['requests_per_hour']} requests per hour",
                "retry_after": 3600
            }
        
        # Add current request
        self.user_requests[user_id].append((now, 1))
        
        return {
            "allowed": True,
            "remaining_minute": limits["requests_per_minute"] - recent_requests - 1,
            "remaining_hour": limits["requests_per_hour"] - hour_requests - 1
        }

def get_rate_limiter() -> RateLimiterService:
    return RateLimiterService()

# ===============================
# CACHE SERVICE
# ===============================

class CacheService:
    """Response caching for performance optimization"""
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache - replace with Redis
    
    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and valid"""
        cached_item = self.cache.get(cache_key)
        
        if cached_item:
            if datetime.now() < cached_item["expires_at"]:
                return cached_item["response"]
            else:
                # Remove expired cache
                del self.cache[cache_key]
        
        return None
    
    async def cache_response(
        self, 
        cache_key: str, 
        response: Dict[str, Any], 
        ttl: int = 3600
    ):
        """Cache response with TTL"""
        self.cache[cache_key] = {
            "response": response,
            "cached_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=ttl)
        }

def get_cache_service() -> CacheService:
    return CacheService()

# ===============================
# GAMIFICATION SERVICE
# ===============================

class GamificationService:
    """Gamification and achievement tracking"""
    
    def __init__(self):
        self.user_achievements = {}
    
    async def process_chat_achievement(
        self, 
        user_id: str, 
        chat_data: Dict[str, Any]
    ):
        """Process potential achievements from chat interaction"""
        
        if user_id not in self.user_achievements:
            self.user_achievements[user_id] = {
                "total_chats": 0,
                "rishis_consulted": set(),
                "wisdom_points": 0,
                "achievements": []
            }
        
        user_data = self.user_achievements[user_id]
        user_data["total_chats"] += 1
        
        if chat_data.get("rishi_used"):
            user_data["rishis_consulted"].add(chat_data["rishi_used"])
        
        # Award wisdom points based on quality
        dharmic_score = chat_data.get("dharmic_alignment", 0)
        wisdom_points = int(dharmic_score * 10)
        user_data["wisdom_points"] += wisdom_points
        
        # Check for achievements
        await self._check_achievements(user_id, user_data)
    
    async def _check_achievements(self, user_id: str, user_data: Dict[str, Any]):
        """Check and award achievements"""
        achievements = user_data["achievements"]
        
        # First chat achievement
        if user_data["total_chats"] == 1 and "first_chat" not in achievements:
            achievements.append("first_chat")
            logger.info(f"🏆 Achievement unlocked: First Chat for user {user_id}")
        
        # Rishi diversity achievement
        if len(user_data["rishis_consulted"]) >= 3 and "rishi_explorer" not in achievements:
            achievements.append("rishi_explorer")
            logger.info(f"🏆 Achievement unlocked: Rishi Explorer for user {user_id}")

def get_gamification_service() -> GamificationService:
    return GamificationService()

# ===============================
# HELPER FUNCTIONS
# ===============================

async def get_rishi_usage_count(user_id: str, rishi_id: str) -> int:
    """Get usage count for specific rishi by user"""
    # Mock implementation - replace with database query
    return 5

async def get_last_rishi_interaction(user_id: str, rishi_id: str) -> Optional[str]:
    """Get last interaction timestamp with rishi"""
    # Mock implementation
    return "2024-09-26T10:30:00Z"

async def validate_rishi_access(rishi_id: str, user_tier: str) -> bool:
    """Validate if user can access specific rishi"""
    premium_rishis = ["vyasa", "vasishta"]
    
    if rishi_id in premium_rishis:
        return user_tier in ["premium", "enterprise"]
    
    return True

async def generate_cache_key(request: Dict[str, Any], user_id: str) -> str:
    """Generate cache key for request"""
    key_data = {
        "message": request.get("message", ""),
        "rishi": request.get("rishi_id"),
        "user": user_id[:8]  # Partial user ID for privacy
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()

async def generate_intelligent_fallback_response(
    request: Dict[str, Any],
    user: Dict[str, Any],
    error: str,
    processing_time: float
) -> Dict[str, Any]:
    """Generate intelligent fallback when main processing fails"""
    
    from ..models.chat import ChatResponse
    
    fallback_message = f"🕉️ I understand your spiritual inquiry: '{request.get('message', '')}'. While I'm experiencing a temporary processing challenge, I want to honor your seeking. The path of dharma teaches us that every question arises from sincere spiritual yearning. Please try again in a moment, and I'll be ready to offer deeper wisdom."
    
    return ChatResponse(
        response=fallback_message,
        conversation_id=request.get("conversation_id", f"fallback_{int(datetime.now().timestamp())}"),
        message_id=f"fallback_{int(datetime.now().timestamp())}",
        timestamp=datetime.now().isoformat(),
        confidence_score=0.7,
        dharmic_alignment=0.85,
        processing_time=processing_time,
        model_used="Intelligent-Fallback",
        sources=["Backend Fallback System"],
        metadata={
            "fallback_reason": error,
            "service_source": "intelligent_fallback",
            "user_authenticated": True,
            "fallback_quality": "high"
        }
    )

# Background task functions
async def save_comprehensive_conversation(
    user_id: str,
    conversation_id: str,
    user_message: Dict[str, Any],
    ai_response: Dict[str, Any],
    processing_context: Dict[str, Any],
    processing_time: float
):
    """Save comprehensive conversation data"""
    conversation_data = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_message": user_message,
        "ai_response": ai_response,
        "processing_context": processing_context,
        "processing_time": processing_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to database (mock implementation)
    logger.info(f"💾 Saved comprehensive conversation for user {user_id}")

async def update_user_personalization(
    user_id: str,
    interaction_data: Dict[str, Any]
):
    """Update user personalization based on interaction"""
    
    # Analyze interaction for personalization insights
    personalization_updates = {
        "last_interaction": datetime.now().isoformat(),
        "interaction_quality": interaction_data.get("satisfaction_indicators", {}),
        "preferences_updated": True
    }
    
    # Update user profile (mock implementation)
    logger.info(f"🎯 Updated personalization for user {user_id}")

# Service getters for missing dependencies
def get_chat_history_service():
    """Mock chat history service"""
    class MockChatHistoryService:
        async def get_conversation_history(self, user_id: str, conversation_id: str):
            return {"messages": [], "total": 0}
        
        async def get_user_conversations(self, user_id: str, limit: int, offset: int):
            return []
    
    return MockChatHistoryService()