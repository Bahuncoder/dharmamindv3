"""
🌟 DharmaMind Universal Personalization Engine
============================================

Advanced personalization system that adapts spiritual guidance to each user's
unique journey, drawing from eternal wisdom principles while remaining
accessible and inclusive for seekers from all backgrounds.

This engine provides:
- Universal spiritual guidance adapted to user's path and level
- Secular presentation of ancient wisdom principles  
- Inclusive practices suitable for all backgrounds
- Respectful, non-sectarian approach to inner growth

May this wisdom serve all souls seeking truth and inner peace 🙏
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

from .models.user_profile import (
    UserProfile, SpiritualLevel, SpiritualPath,
    MeditationExperience, PersonalizationResponse
)

logger = logging.getLogger(__name__)


class PersonalizationEngine:
    """🕉️ Advanced Personalization Engine for DharmaMind
    
    Provides highly personalized spiritual guidance by:
    - Analyzing user's spiritual profile and journey
    - Adapting response style and content
    - Suggesting personalized practices
    - Tracking progress and growth
    - Providing path-specific wisdom
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Personalization templates and patterns
        self.guidance_styles = {
            "gentle": {
                "tone": "compassionate and nurturing",
                "approach": "gradual and supportive",
                "language": "soft and encouraging"
            },
            "direct": {
                "tone": "clear and straightforward",
                "approach": "practical and focused",
                "language": "precise and actionable"
            },
            "scholarly": {
                "tone": "informed and detailed",
                "approach": "educational and thorough",
                "language": "traditional and reverent"
            },
            "modern": {
                "tone": "contemporary and relatable",
                "approach": "practical for modern life",
                "language": "accessible and current"
            }
        }
        
        # Path-specific wisdom templates (Universal approach)
        self.path_wisdom = {
            SpiritualPath.SERVICE_PATH: {
                "focus": "selfless service and contribution to others",
                "practices": ["volunteer work", "acts of kindness", "helping others"],
                "teachings": "Find fulfillment through serving others",
                "guidance": "Transform your daily actions into opportunities to help"
            },
            SpiritualPath.DEVOTION_PATH: {
                "focus": "love, gratitude, and surrender to life's flow",
                "practices": ["gratitude practice", "loving-kindness", "heart opening"],
                "teachings": "Love and gratitude open the path to inner peace",
                "guidance": "Cultivate an open heart and trust in life's wisdom"
            },
            SpiritualPath.MEDITATION_PATH: {
                "focus": "inner stillness and mindful awareness",
                "practices": ["meditation", "mindfulness", "breath awareness"],
                "teachings": "Inner peace comes through cultivating stillness",
                "guidance": "Develop discipline in your inner practices"
            },
            SpiritualPath.WISDOM_PATH: {
                "focus": "understanding truth through inquiry and study",
                "practices": ["self-inquiry", "contemplation", "study"],
                "teachings": "Know yourself to understand life's deeper truths",
                "guidance": "Question deeply and seek understanding"
            }
        }
        
        # Level-appropriate content
        self.level_content = {
            SpiritualLevel.SEEKER: {
                "complexity": "basic",
                "concepts": ["dharma", "karma", "basic meditation"],
                "avoid": ["advanced tantric practices", "complex philosophy"]
            },
            SpiritualLevel.STUDENT: {
                "complexity": "intermediate",
                "concepts": [
                    "yogic practices", "scriptural study", "ethical living"
                ],
                "avoid": ["extreme practices", "sectarian debates"]
            },
            SpiritualLevel.PRACTITIONER: {
                "complexity": "advanced",
                "concepts": [
                    "energy work", "advanced meditation", 
                    "spiritual psychology"
                ],
                "avoid": ["beginner explanations"]
            },
            SpiritualLevel.DEVOTEE: {
                "complexity": "devotional",
                "concepts": [
                    "bhakti practices", "surrender", "divine relationship"
                ],
                "avoid": ["dry intellectual discussions"]
            }
        }
        
        self.logger.info("🌟 Personalization Engine initialized")
    
    async def personalize_response(
        self,
        user_profile: UserProfile,
        base_response: str,
        context: Dict[str, Any]
    ) -> PersonalizationResponse:
        """Personalize a response based on user profile"""
        
        try:
            # Get personalization context
            context = user_profile.get_personalization_context()
            
            # Analyze response needs
            factors = await self._analyze_personalization_needs(
                user_profile, context
            )
            
            # Adapt response style and content
            personalized_response = await self._adapt_response_content(
                base_response, user_profile, context
            )
            
            # Generate practice recommendations
            practices = await self._generate_practice_recommendations(
                user_profile, context
            )
            
            # Suggest learning topics
            suggested_topics = user_profile.suggest_next_learning_topics()
            
            # Generate chakra guidance
            chakra_guide = await self._generate_chakra_guidance(user_profile)
            
            # Generate path-specific advice
            path_advice = await self._generate_path_specific_advice(
                user_profile
            )
            
            # Add progress encouragement
            encouragement = await self._generate_progress_encouragement(
                user_profile
            )
            
            return PersonalizationResponse(
                response=personalized_response,
                personalization_factors=factors,
                recommended_practices=practices,
                suggested_topics=suggested_topics,
                chakra_guidance=chakra_guide,
                path_specific_advice=path_advice,
                progress_encouragement=encouragement
            )
            
        except Exception as e:
            self.logger.error(f"Error personalizing response: {e}")
            # Return base response with minimal personalization
            return PersonalizationResponse(
                response=base_response,
                personalization_factors=["error_fallback"],
                recommended_practices=[],
                suggested_topics=[]
            )
    
    async def _analyze_personalization_needs(
        self,
        user_profile: UserProfile,
        context: Dict[str, Any]
    ) -> List[str]:
        """Analyze what personalization factors to apply"""
        
        factors = []
        
        # Spiritual level factors
        factors.append(f"spiritual_level:{user_profile.spiritual_level.value}")
        
        # Path-specific factors
        factors.append(f"primary_path:{user_profile.primary_path.value}")
        
        # Experience level
        exp_val = user_profile.meditation_experience.value
        factors.append(f"meditation_experience:{exp_val}")
        
        # Life circumstances
        life_stage = user_profile.life_circumstances.life_stage.value
        stress_level = user_profile.life_circumstances.stress_level
        factors.append(f"life_stage:{life_stage}")
        factors.append(f"stress_level:{stress_level}")
        
        # Personality type
        factors.append(f"personality:{user_profile.personality_type.value}")
        
        # Current focus areas
        if user_profile.learning_progress.current_focus:
            factors.append("current_focus")
        
        # Recent challenges
        if user_profile.recent_challenges:
            factors.append("addressing_challenges")
        
        # Chakra energy state
        weak_chakras = user_profile.chakra_profile.get_weak_chakras()
        if weak_chakras:
            factors.append("chakra_balancing")
        
        return factors
    
    async def _adapt_response_content(
        self,
        base_response: str,
        user_profile: UserProfile,
        context: Dict[str, Any]
    ) -> str:
        """Adapt response content based on user profile"""
        
        try:
            # Get guidance style (not using the config yet, but it's available)
            style = user_profile.personalization_settings.guidance_style
            
            # Start with base response
            adapted_response = base_response
            
            # Add personalized greeting based on spiritual level
            greeting = await self._generate_personalized_greeting(user_profile)
            if greeting:
                adapted_response = f"{greeting}\n\n{adapted_response}"
            
            # Adapt language complexity
            if user_profile.spiritual_level == SpiritualLevel.SEEKER:
                adapted_response = await self._simplify_language(
                    adapted_response
                )
            elif user_profile.spiritual_level in [
                SpiritualLevel.TEACHER, SpiritualLevel.SAGE
            ]:
                adapted_response = await self._enrich_language(
                    adapted_response
                )
            
            # Add path-specific context
            if user_profile.primary_path != SpiritualPath.MIXED:
                path_context = await self._add_path_context(
                    adapted_response, user_profile.primary_path
                )
                adapted_response = path_context
            
            # Include Sanskrit if preferred
            if user_profile.practice_preferences.include_sanskrit:
                adapted_response = await self._add_sanskrit_terms(
                    adapted_response
                )
            
            # Adapt to current mood if available
            if user_profile.current_mood:
                adapted_response = await self._adapt_to_mood(
                    adapted_response, user_profile.current_mood
                )
            
            # Add personal encouragement
            encouragement = await self._generate_personal_encouragement(
                user_profile
            )
            if encouragement:
                adapted_response = f"{adapted_response}\n\n{encouragement}"
            
            return adapted_response
            
        except Exception as e:
            self.logger.error(f"Error adapting response content: {e}")
            return base_response
    
    async def _generate_practice_recommendations(
        self,
        user_profile: UserProfile,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate personalized practice recommendations"""
        
        recommendations = []
        
        try:
            # Based on spiritual level
            if user_profile.spiritual_level == SpiritualLevel.SEEKER:
                recommendations.extend([
                    "Start with 5-10 minutes of daily meditation",
                    "Practice simple breathing exercises (pranayama)",
                    "Read introductory texts like the Bhagavad Gita"
                ])
            
            elif user_profile.spiritual_level == SpiritualLevel.STUDENT:
                recommendations.extend([
                    "Establish a regular meditation practice",
                    "Study sacred texts with contemplation",
                    "Practice karma yoga through selfless service"
                ])
            
            elif user_profile.spiritual_level == SpiritualLevel.PRACTITIONER:
                recommendations.extend([
                    "Deepen meditation with advanced techniques",
                    "Explore energy practices and chakra work",
                    "Engage in spiritual community (satsang)"
                ])
            
            # Based on primary path
            path_info = self.path_wisdom.get(user_profile.primary_path, {})
            path_practices = path_info.get("practices", [])
            recommendations.extend(path_practices)
            
            # Based on weak chakras
            weak_chakras = user_profile.chakra_profile.get_weak_chakras()
            for chakra in weak_chakras[:2]:  # Top 2 weak chakras
                chakra_practices = await self._get_chakra_practices(chakra)
                recommendations.extend(chakra_practices)
            
            # Based on current challenges
            if user_profile.recent_challenges:
                for challenge in user_profile.recent_challenges[:2]:
                    practices = await self._get_practices_for_challenge(
                        challenge
                    )
                    recommendations.extend(practices)
            
            # Based on stress level
            if user_profile.life_circumstances.stress_level > 7:
                recommendations.extend([
                    "Practice stress-reducing meditation techniques",
                    "Try gentle yoga or walking meditation",
                    "Use calming mantras like 'Om Shanti'"
                ])
            
            # Based on time availability
            time_avail = user_profile.life_circumstances.time_availability
            if time_avail == "limited":
                recommendations.extend([
                    "Practice micro-meditations (2-3 minutes)",
                    "Use mindful breathing during daily activities",
                    "Listen to spiritual teachings during commute"
                ])
            
            # Remove duplicates and limit to 8 recommendations
            unique_recommendations = list(set(recommendations))
            return unique_recommendations[:8]
            
        except Exception as e:
            self.logger.error(
                f"Error generating practice recommendations: {e}"
            )
            return [
                "Practice daily meditation",
                "Study sacred texts",
                "Cultivate compassion and kindness"
            ]
    
    async def _generate_chakra_guidance(
        self, user_profile: UserProfile
    ) -> Optional[str]:
        """Generate chakra-specific guidance"""
        
        try:
            weak_chakras = user_profile.chakra_profile.get_weak_chakras()
            if not weak_chakras:
                return None
            
            primary_weak = weak_chakras[0]
            
            chakra_guidance = {
                "root": (
                    "Focus on grounding practices. Spend time in nature, "
                    "practice standing meditation, and work on feeling "
                    "secure and stable in your life."
                ),
                "sacral": (
                    "Nurture your creativity and emotional flow. Practice "
                    "hip-opening yoga, creative expression, and healthy "
                    "relationship with emotions."
                ),
                "solar_plexus": (
                    "Strengthen your personal power and confidence. Practice "
                    "core-strengthening exercises, assertiveness, and "
                    "decision-making skills."
                ),
                "heart": (
                    "Cultivate love and compassion. Practice loving-kindness "
                    "meditation, heart-opening yoga, and service to others."
                ),
                "throat": (
                    "Express your authentic truth. Practice chanting, singing, "
                    "truthful communication, and creative expression."
                ),
                "third_eye": (
                    "Develop intuition and inner wisdom. Practice meditation, "
                    "visualization, and trust in your inner guidance."
                ),
                "crown": (
                    "Connect with divine consciousness. Practice spiritual "
                    "meditation, study sacred texts, and cultivate surrender."
                )
            }
            
            return chakra_guidance.get(primary_weak)
            
        except Exception as e:
            self.logger.error(f"Error generating chakra guidance: {e}")
            return None
    
    async def _generate_path_specific_advice(self, user_profile: UserProfile) -> Optional[str]:
        """Generate advice specific to user's spiritual path"""
        
        try:
            path = user_profile.primary_path
            if path == SpiritualPath.MIXED:
                return "Continue exploring different paths to find what resonates most deeply with your soul."
            
            path_info = self.path_wisdom.get(path)
            if not path_info:
                return None
            
            return f"As a practitioner of {path.value.replace('_', ' ').title()}, {path_info['guidance']}. {path_info['teachings']}."
            
        except Exception as e:
            self.logger.error(f"Error generating path-specific advice: {e}")
            return None
    
    async def _generate_progress_encouragement(self, user_profile: UserProfile) -> Optional[str]:
        """Generate personalized progress encouragement"""
        
        try:
            encouragements = []
            
            # Based on spiritual level
            if user_profile.spiritual_level == SpiritualLevel.SEEKER:
                encouragements.append("Every step on the spiritual path is sacred. Trust your journey.")
            elif user_profile.spiritual_level == SpiritualLevel.STUDENT:
                encouragements.append("Your dedication to learning shows the wisdom of your heart.")
            elif user_profile.spiritual_level == SpiritualLevel.PRACTITIONER:
                encouragements.append("Your consistent practice is transforming you in beautiful ways.")
            
            # Based on progress
            if len(user_profile.learning_progress.topics_learned) > 10:
                encouragements.append("Your growing wisdom illuminates the path for others.")
            
            if len(user_profile.learning_progress.practices_tried) > 5:
                encouragements.append("Your willingness to explore different practices shows spiritual maturity.")
            
            # Based on challenges
            if user_profile.recent_challenges:
                encouragements.append("Challenges are opportunities for spiritual growth. You're becoming stronger.")
            
            # Based on meditation experience
            if user_profile.meditation_experience in [MeditationExperience.ADVANCED, MeditationExperience.EXPERT]:
                encouragements.append("Your meditation practice is a gift to the world's consciousness.")
            
            return random.choice(encouragements) if encouragements else None
            
        except Exception as e:
            self.logger.error(f"Error generating progress encouragement: {e}")
            return None
    
    async def _generate_personalized_greeting(self, user_profile: UserProfile) -> Optional[str]:
        """Generate a personalized greeting"""
        
        try:
            greetings = []
            
            # Based on spiritual level
            if user_profile.spiritual_level == SpiritualLevel.SEEKER:
                greetings.extend([
                    "Namaste, fellow seeker of truth.",
                    "Welcome to your spiritual journey, dear soul."
                ])
            elif user_profile.spiritual_level == SpiritualLevel.DEVOTEE:
                greetings.extend([
                    "Namaste, beloved devotee.",
                    "Blessings on your path of love and surrender."
                ])
            elif user_profile.spiritual_level == SpiritualLevel.TEACHER:
                greetings.extend([
                    "Namaste, respected teacher.",
                    "Honor to connect with a guide of wisdom."
                ])
            
            # Based on path
            if user_profile.primary_path == SpiritualPath.BHAKTI_YOGA:
                greetings.extend([
                    "May divine love guide our conversation.",
                    "In the spirit of devotion, I greet you."
                ])
            
            # Based on time of interaction
            if user_profile.last_interaction:
                days_since = (datetime.now() - user_profile.last_interaction).days
                if days_since > 7:
                    greetings.append("Welcome back to your spiritual practice.")
                elif days_since > 30:
                    greetings.append("How wonderful to reconnect on the path.")
            
            return random.choice(greetings) if greetings else None
            
        except Exception as e:
            self.logger.error(f"Error generating personalized greeting: {e}")
            return None
    
    async def _get_chakra_practices(self, chakra: str) -> List[str]:
        """Get practices for specific chakra"""
        
        chakra_practices = {
            "root": [
                "Practice grounding meditation",
                "Try standing or walking meditation",
                "Connect with nature daily"
            ],
            "sacral": [
                "Practice hip-opening yoga poses",
                "Engage in creative activities",
                "Practice emotional acceptance meditation"
            ],
            "solar_plexus": [
                "Practice core-strengthening exercises",
                "Work on personal boundaries",
                "Practice confidence-building affirmations"
            ],
            "heart": [
                "Practice loving-kindness meditation",
                "Perform acts of service",
                "Practice heart-opening yoga poses"
            ],
            "throat": [
                "Practice chanting or singing",
                "Work on truthful communication",
                "Practice neck and shoulder releases"
            ],
            "third_eye": [
                "Practice concentration meditation",
                "Work with visualization techniques",
                "Practice intuition-building exercises"
            ],
            "crown": [
                "Practice spiritual meditation",
                "Study sacred texts",
                "Practice surrender and letting go"
            ]
        }
        
        return chakra_practices.get(chakra, [])
    
    async def _get_practices_for_challenge(self, challenge: str) -> List[str]:
        """Get practices to address specific challenges"""
        
        challenge_practices = {
            "stress": [
                "Practice stress-reducing breathing techniques",
                "Try gentle yoga or tai chi",
                "Use calming mantras"
            ],
            "anxiety": [
                "Practice grounding meditation",
                "Try progressive muscle relaxation",
                "Focus on present-moment awareness"
            ],
            "depression": [
                "Practice gratitude meditation",
                "Engage in uplifting spiritual practices",
                "Connect with spiritual community"
            ],
            "anger": [
                "Practice cooling breathing techniques",
                "Try compassion meditation",
                "Work with patience and tolerance"
            ],
            "fear": [
                "Practice courage-building affirmations",
                "Try protective visualization",
                "Cultivate faith and trust"
            ]
        }
        
        # Simple keyword matching for now
        for key, practices in challenge_practices.items():
            if key.lower() in challenge.lower():
                return practices
        
        return ["Practice mindfulness meditation", "Cultivate inner peace"]
    
    async def update_user_interaction(
        self,
        user_profile: UserProfile,
        interaction_data: Dict[str, Any]
    ) -> UserProfile:
        """Update user profile based on interaction"""
        
        try:
            # Update interaction history
            topic = interaction_data.get("topic", "")
            response_quality = interaction_data.get("response_quality", 3)
            
            user_profile.update_interaction(topic, response_quality)
            
            # Update learning progress
            if "learned_topic" in interaction_data:
                user_profile.add_learning_progress(
                    interaction_data["learned_topic"],
                    interaction_data.get("practiced_technique")
                )
            
            # Update recent insights
            if "insight" in interaction_data:
                user_profile.recent_insights.append(interaction_data["insight"])
                # Keep only last 10 insights
                user_profile.recent_insights = user_profile.recent_insights[-10:]
            
            # Update current mood if provided
            if "mood" in interaction_data:
                user_profile.current_mood = interaction_data["mood"]
            
            # Update challenges if mentioned
            if "challenge" in interaction_data:
                challenge = interaction_data["challenge"]
                if challenge not in user_profile.recent_challenges:
                    user_profile.recent_challenges.append(challenge)
                    # Keep only last 5 challenges
                    user_profile.recent_challenges = user_profile.recent_challenges[-5:]
            
            return user_profile
            
        except Exception as e:
            self.logger.error(f"Error updating user interaction: {e}")
            return user_profile


# Global personalization engine instance
_personalization_engine = None


def get_personalization_engine() -> PersonalizationEngine:
    """Get global personalization engine instance"""
    global _personalization_engine
    if _personalization_engine is None:
        _personalization_engine = PersonalizationEngine()
    return _personalization_engine


# Export main class
__all__ = ["PersonalizationEngine", "get_personalization_engine"]
