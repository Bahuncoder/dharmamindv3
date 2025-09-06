"""
🕉️ DharmaMind Personalization Integration
========================================

Integration layer that connects personalization and practice recommendation
engines with the main chat system and consciousness core.

This integration:
- Enhances chat responses with personalized guidance
- Suggests practices based on user context
- Tracks user progress and adapts recommendations
- Maintains spiritual alignment with dharmic principles

May this integration serve each user's unique spiritual journey 🙏
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..models.user_profile import UserProfile, PersonalizationResponse
from .personalization_engine import get_personalization_engine
from .practice_recommendation_engine import get_practice_recommendation_engine
from ..chakra_modules.consciousness_core import ConsciousnessCore

logger = logging.getLogger(__name__)


class PersonalizationIntegration:
    """🕉️ Integration layer for personalized spiritual guidance
    
    This class orchestrates the interaction between:
    - User profiles and preferences
    - Personalization engine for custom responses
    - Practice recommendation engine
    - Consciousness core for spiritual processing
    - Chat system for contextual responses
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.personalization_engine = get_personalization_engine()
        self.practice_engine = get_practice_recommendation_engine()
        self.consciousness_core = ConsciousnessCore()
        
        self.logger.info("🌟 Personalization Integration initialized")
    
    async def enhance_chat_response(
        self,
        user_profile: UserProfile,
        base_response: str,
        message_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance a chat response with personalized elements"""
        
        try:
            # Get personalized response
            personalized = await self.personalization_engine.personalize_response(
                user_profile, base_response, message_context
            )
            
            # Generate practice recommendations if appropriate
            practices = []
            if self._should_include_practices(message_context, personalized):
                practices = await self.practice_engine.generate_recommendations(
                    user_profile, message_context
                )
            
            # Add consciousness insights if relevant
            consciousness_insights = await self._get_consciousness_insights(
                user_profile, message_context
            )
            
            # Compile enhanced response
            enhanced_response = {
                "response": personalized.response,
                "personalization_factors": personalized.personalization_factors,
                "recommended_practices": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "duration": p.duration,
                        "description": p.description,
                        "priority": p.priority,
                        "personalized_notes": p.personalized_notes
                    }
                    for p in practices[:3]  # Limit to top 3
                ],
                "suggested_topics": personalized.suggested_topics,
                "chakra_guidance": personalized.chakra_guidance,
                "path_specific_advice": personalized.path_specific_advice,
                "progress_encouragement": personalized.progress_encouragement,
                "consciousness_insights": consciousness_insights,
                "user_context": {
                    "spiritual_level": user_profile.spiritual_level.value,
                    "primary_path": user_profile.primary_path.value,
                    "meditation_experience": user_profile.meditation_experience.value,
                    "current_focus": user_profile.learning_progress.current_focus
                }
            }
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Error enhancing chat response: {e}")
            # Return minimal enhanced response
            return {
                "response": base_response,
                "personalization_factors": ["error_fallback"],
                "recommended_practices": [],
                "suggested_topics": [],
                "user_context": {
                    "spiritual_level": "seeker",
                    "primary_path": "mixed"
                }
            }
    
    async def process_user_message(
        self,
        user_profile: UserProfile,
        message: str,
        conversation_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user message with full personalization context"""
        
        try:
            # Analyze message for spiritual context
            message_analysis = await self._analyze_message_context(
                message, conversation_history or []
            )
            
            # Check if user is asking for practices
            if message_analysis.get("requesting_practices", False):
                practices = await self.practice_engine.generate_recommendations(
                    user_profile, message_analysis
                )
                
                return {
                    "type": "practice_recommendations",
                    "practices": practices,
                    "context": message_analysis,
                    "personalized_intro": await self._generate_practice_intro(
                        user_profile, message_analysis
                    )
                }
            
            # Check if user is sharing progress
            if message_analysis.get("sharing_progress", False):
                updated_profile = await self._process_progress_update(
                    user_profile, message_analysis
                )
                
                return {
                    "type": "progress_acknowledgment",
                    "updated_profile": updated_profile,
                    "encouragement": await self._generate_progress_response(
                        updated_profile, message_analysis
                    )
                }
            
            # Check if user needs guidance on specific topic
            if message_analysis.get("seeking_guidance", False):
                guidance_context = await self._prepare_guidance_context(
                    user_profile, message_analysis
                )
                
                return {
                    "type": "personalized_guidance",
                    "context": guidance_context,
                    "ready_for_consciousness_processing": True
                }
            
            # Default: prepare for regular chat enhancement
            return {
                "type": "regular_chat",
                "context": message_analysis,
                "ready_for_enhancement": True
            }
            
        except Exception as e:
            self.logger.error(f"Error processing user message: {e}")
            return {
                "type": "regular_chat",
                "context": {"error": str(e)},
                "ready_for_enhancement": False
            }
    
    async def track_user_interaction(
        self,
        user_profile: UserProfile,
        interaction_data: Dict[str, Any]
    ) -> UserProfile:
        """Track and learn from user interactions"""
        
        try:
            # Update personalization engine
            updated_profile = await self.personalization_engine.update_user_interaction(
                user_profile, interaction_data
            )
            
            # Track practice completion if relevant
            if interaction_data.get("completed_practice"):
                practice_name = interaction_data["completed_practice"]
                duration = interaction_data.get("practice_duration", 10)
                rating = interaction_data.get("experience_rating", 3)
                
                updated_profile = await self.practice_engine.track_practice_completion(
                    updated_profile, practice_name, duration, rating
                )
            
            # Update learning progress
            if interaction_data.get("learned_something"):
                topic = interaction_data.get("topic", "general")
                updated_profile.add_learning_progress(topic)
            
            return updated_profile
            
        except Exception as e:
            self.logger.error(f"Error tracking user interaction: {e}")
            return user_profile
    
    async def _should_include_practices(
        self,
        message_context: Dict[str, Any],
        personalized_response: PersonalizationResponse
    ) -> bool:
        """Determine if practice recommendations should be included"""
        
        # Include practices if:
        # 1. User specifically asked for practices
        if message_context.get("requesting_practices", False):
            return True
        
        # 2. Response mentions challenges that practices could help
        if any(challenge in personalized_response.response.lower() 
               for challenge in ["stress", "anxiety", "difficulty", "struggle"]):
            return True
        
        # 3. User is new and might benefit from guidance
        if message_context.get("user_level") == "seeker":
            # Include practices occasionally for new users
            return message_context.get("message_count", 0) % 3 == 0
        
        # 4. User hasn't received practices recently
        if message_context.get("last_practice_suggestion", 0) > 5:
            return True
        
        return False
    
    async def _get_consciousness_insights(
        self,
        user_profile: UserProfile,
        message_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get consciousness-level insights for the user"""
        
        try:
            # Only provide consciousness insights for advanced users
            if user_profile.spiritual_level.value in ["practitioner", "devotee", "teacher", "sage"]:
                # Create consciousness context
                consciousness_context = {
                    "user_path": user_profile.primary_path.value,
                    "current_challenges": user_profile.recent_challenges,
                    "chakra_state": user_profile.chakra_profile.get_weak_chakras(),
                    "message_theme": message_context.get("theme", "general")
                }
                
                # Get consciousness processing insights
                insights = await self.consciousness_core.process_consciousness_layer(
                    message_context.get("user_message", ""),
                    consciousness_context
                )
                
                return {
                    "consciousness_level": insights.get("consciousness_level"),
                    "dharmic_alignment": insights.get("dharmic_alignment"),
                    "spiritual_guidance": insights.get("spiritual_guidance"),
                    "awakening_potential": insights.get("awakening_potential")
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting consciousness insights: {e}")
            return None
    
    async def _analyze_message_context(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze message for spiritual and emotional context"""
        
        context = {
            "message_length": len(message),
            "message_count": len(conversation_history),
            "timestamp": datetime.now().isoformat()
        }
        
        # Detect if user is requesting practices
        practice_keywords = [
            "practice", "meditation", "technique", "exercise", 
            "how to", "what should I do", "recommend", "suggest"
        ]
        context["requesting_practices"] = any(
            keyword in message.lower() for keyword in practice_keywords
        )
        
        # Detect if user is sharing progress
        progress_keywords = [
            "I practiced", "I meditated", "I tried", "I've been",
            "my practice", "feeling better", "progress", "improved"
        ]
        context["sharing_progress"] = any(
            keyword in message.lower() for keyword in progress_keywords
        )
        
        # Detect if user is seeking guidance
        guidance_keywords = [
            "help", "guidance", "lost", "confused", "don't know",
            "struggling", "difficult", "challenge", "problem"
        ]
        context["seeking_guidance"] = any(
            keyword in message.lower() for keyword in guidance_keywords
        )
        
        # Detect emotional tone
        if any(word in message.lower() for word in ["stress", "anxious", "worried"]):
            context["emotional_tone"] = "stressed"
        elif any(word in message.lower() for word in ["sad", "depressed", "down"]):
            context["emotional_tone"] = "low"
        elif any(word in message.lower() for word in ["happy", "grateful", "joy"]):
            context["emotional_tone"] = "positive"
        else:
            context["emotional_tone"] = "neutral"
        
        # Detect spiritual themes
        spiritual_themes = {
            "meditation": ["meditat", "mindful", "awareness", "breath"],
            "devotion": ["prayer", "surrender", "divine", "god", "devotion"],
            "wisdom": ["understand", "knowledge", "wisdom", "truth", "insight"],
            "service": ["service", "help others", "karma", "action", "work"],
            "ethics": ["right", "wrong", "dharma", "moral", "ethical"]
        }
        
        detected_themes = []
        for theme, keywords in spiritual_themes.items():
            if any(keyword in message.lower() for keyword in keywords):
                detected_themes.append(theme)
        
        context["themes"] = detected_themes
        context["primary_theme"] = detected_themes[0] if detected_themes else "general"
        
        return context
    
    async def _generate_practice_intro(
        self,
        user_profile: UserProfile,
        message_context: Dict[str, Any]
    ) -> str:
        """Generate personalized introduction for practice recommendations"""
        
        level = user_profile.spiritual_level.value
        path = user_profile.primary_path.value
        
        intros = {
            "seeker": f"As you begin your spiritual journey, here are some gentle practices to support your growth:",
            "student": f"Based on your dedication to learning, these practices will deepen your understanding:",
            "practitioner": f"For your advanced practice on the {path.replace('_', ' ')} path, consider these techniques:",
            "devotee": f"To nurture your devotional heart, these loving practices are recommended:",
            "teacher": f"As one who guides others, these practices will support your own continued growth:",
            "sage": f"In your wisdom, these subtle practices may offer new dimensions of realization:"
        }
        
        return intros.get(level, "Here are some practices that may support your spiritual journey:")
    
    async def _process_progress_update(
        self,
        user_profile: UserProfile,
        message_analysis: Dict[str, Any]
    ) -> UserProfile:
        """Process when user shares their spiritual progress"""
        
        # Update interaction data
        interaction_data = {
            "topic": "progress_sharing",
            "response_quality": 4,  # Assume positive when sharing progress
            "learned_something": True,
            "mood": "positive" if message_analysis.get("emotional_tone") == "positive" else "neutral"
        }
        
        return await self.personalization_engine.update_user_interaction(
            user_profile, interaction_data
        )
    
    async def _generate_progress_response(
        self,
        user_profile: UserProfile,
        message_analysis: Dict[str, Any]
    ) -> str:
        """Generate encouraging response for progress sharing"""
        
        responses = [
            "How wonderful to hear about your spiritual growth! Every step on the path is sacred.",
            "Your dedication to practice is inspiring. The divine works through consistent effort.",
            "What beautiful progress you're making! Trust in your journey and continue with faith.",
            "Your spiritual development brings joy to witness. Keep nurturing this inner flame.",
            "This growth you describe is the fruit of sincere practice. Continue with devotion."
        ]
        
        # Select response based on user's path preference
        if user_profile.primary_path.value == "bhakti_yoga":
            return "Your heart's devotion is blossoming beautifully! The divine surely feels your love and dedication."
        elif user_profile.primary_path.value == "raja_yoga":
            return "Your disciplined practice is bearing fruit! The mind's mastery you're developing is truly commendable."
        elif user_profile.primary_path.value == "karma_yoga":
            return "Your selfless efforts are transforming you wonderfully! Continue offering your actions with pure intention."
        elif user_profile.primary_path.value == "jnana_yoga":
            return "Your growing understanding reflects the light of wisdom within! Keep inquiring into truth."
        
        return responses[0]  # Default encouraging response
    
    async def _prepare_guidance_context(
        self,
        user_profile: UserProfile,
        message_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare context for personalized spiritual guidance"""
        
        return {
            "user_level": user_profile.spiritual_level.value,
            "user_path": user_profile.primary_path.value,
            "current_challenges": user_profile.recent_challenges,
            "weak_chakras": user_profile.chakra_profile.get_weak_chakras(),
            "learning_focus": user_profile.learning_progress.current_focus,
            "message_themes": message_analysis.get("themes", []),
            "emotional_tone": message_analysis.get("emotional_tone", "neutral"),
            "guidance_style": user_profile.personalization_settings.guidance_style,
            "include_sanskrit": user_profile.practice_preferences.include_sanskrit
        }


# Global integration instance
_personalization_integration = None


def get_personalization_integration() -> PersonalizationIntegration:
    """Get global personalization integration instance"""
    global _personalization_integration
    if _personalization_integration is None:
        _personalization_integration = PersonalizationIntegration()
    return _personalization_integration


# Export main class
__all__ = ["PersonalizationIntegration", "get_personalization_integration"]
