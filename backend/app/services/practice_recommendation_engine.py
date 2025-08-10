"""
🕉️ DharmaMind Practice Recommendation Engine
===========================================

Advanced system for generating personalized spiritual practice recommendations
based on user profiles, spiritual paths, current challenges, and progress.

This engine provides:
- Path-specific practice suggestions
- Level-appropriate techniques
- Challenge-based interventions
- Progress-tracking practices
- Chakra balancing recommendations

May these practices guide each soul towards liberation 🙏
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models.user_profile import (
    UserProfile, SpiritualLevel, SpiritualPath,
    PracticeRecommendation
)

logger = logging.getLogger(__name__)


class PracticeRecommendationEngine:
    """🕉️ Advanced Practice Recommendation Engine
    
    Generates personalized spiritual practice recommendations by:
    - Analyzing user's spiritual level and path
    - Considering current challenges and goals
    - Adapting to time availability and preferences
    - Tracking practice effectiveness
    - Suggesting progressive skill development
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Practice database organized by categories
        self.meditation_practices = {
            SpiritualLevel.SEEKER: [
                {
                    "name": "Basic Breath Awareness",
                    "duration": "5-10 minutes",
                    "description": "Simple breathing meditation for beginners",
                    "instructions": "Focus on natural breath without controlling",
                    "benefits": ["Reduces stress", "Improves focus"],
                    "difficulty": 1
                },
                {
                    "name": "Body Scan Meditation",
                    "duration": "10-15 minutes",
                    "description": "Progressive relaxation technique",
                    "instructions": "Systematically relax each body part",
                    "benefits": ["Physical relaxation", "Mind-body awareness"],
                    "difficulty": 1
                }
            ],
            SpiritualLevel.STUDENT: [
                {
                    "name": "Mindfulness of Breathing (Anapanasati)",
                    "duration": "15-20 minutes",
                    "description": "Traditional Buddhist breathing meditation",
                    "instructions": "Four-step breath awareness practice",
                    "benefits": ["Deep concentration", "Mental clarity"],
                    "difficulty": 2
                },
                {
                    "name": "Loving-Kindness Meditation (Metta)",
                    "duration": "15-25 minutes",
                    "description": "Heart-opening compassion practice",
                    "instructions": "Send love to self, loved ones, all beings",
                    "benefits": ["Emotional healing", "Compassion development"],
                    "difficulty": 2
                }
            ],
            SpiritualLevel.PRACTITIONER: [
                {
                    "name": "Vipassana Insight Meditation",
                    "duration": "20-45 minutes",
                    "description": "Advanced awareness practice",
                    "instructions": "Observe thoughts without attachment",
                    "benefits": ["Deep insight", "Liberation from patterns"],
                    "difficulty": 3
                },
                {
                    "name": "Chakra Meditation",
                    "duration": "25-40 minutes",
                    "description": "Energy center activation practice",
                    "instructions": "Systematically activate each chakra",
                    "benefits": ["Energy balance", "Spiritual awakening"],
                    "difficulty": 3
                }
            ]
        }
        
        # Path-specific practices
        self.path_practices = {
            SpiritualPath.KARMA_YOGA: [
                {
                    "name": "Selfless Service (Seva)",
                    "category": "Action",
                    "description": "Volunteer work with no expectation",
                    "frequency": "Weekly",
                    "guidance": "Find opportunities to serve others"
                },
                {
                    "name": "Mindful Work Practice",
                    "category": "Daily Life",
                    "description": "Transform work into spiritual practice",
                    "frequency": "Daily",
                    "guidance": "Offer all actions to the divine"
                }
            ],
            SpiritualPath.BHAKTI_YOGA: [
                {
                    "name": "Devotional Chanting (Kirtan)",
                    "category": "Devotion",
                    "description": "Sacred sound and music practice",
                    "frequency": "Daily",
                    "guidance": "Sing divine names with pure love"
                },
                {
                    "name": "Prayer and Surrender",
                    "category": "Devotion",
                    "description": "Heartfelt communication with divine",
                    "frequency": "Multiple times daily",
                    "guidance": "Surrender all concerns to divine will"
                }
            ],
            SpiritualPath.RAJA_YOGA: [
                {
                    "name": "Pranayama (Breath Control)",
                    "category": "Energy",
                    "description": "Advanced breathing techniques",
                    "frequency": "Daily",
                    "guidance": "Master the breath to master the mind"
                },
                {
                    "name": "Dharana (Concentration)",
                    "category": "Mental",
                    "description": "One-pointed focus practice",
                    "frequency": "Daily",
                    "guidance": "Choose one object and maintain focus"
                }
            ],
            SpiritualPath.JNANA_YOGA: [
                {
                    "name": "Self-Inquiry (Atma Vichara)",
                    "category": "Inquiry",
                    "description": "Who am I? investigation",
                    "frequency": "Daily",
                    "guidance": "Question the nature of the self"
                },
                {
                    "name": "Scriptural Study (Svadhyaya)",
                    "category": "Study",
                    "description": "Sacred text contemplation",
                    "frequency": "Daily",
                    "guidance": "Study with contemplation and reflection"
                }
            ]
        }
        
        # Challenge-specific practices
        self.challenge_practices = {
            "stress": [
                {
                    "name": "Cooling Breath (Sheetali)",
                    "type": "Pranayama",
                    "immediate": True,
                    "description": "Cooling breathing technique for stress relief"
                },
                {
                    "name": "Progressive Muscle Relaxation",
                    "type": "Relaxation",
                    "immediate": True,
                    "description": "Systematic tension release"
                }
            ],
            "anxiety": [
                {
                    "name": "Grounding Meditation",
                    "type": "Stabilization",
                    "immediate": True,
                    "description": "Root chakra stabilization practice"
                },
                {
                    "name": "4-7-8 Breathing",
                    "type": "Pranayama",
                    "immediate": True,
                    "description": "Calming breath pattern"
                }
            ],
            "anger": [
                {
                    "name": "Patience Meditation",
                    "type": "Emotional",
                    "immediate": False,
                    "description": "Cultivate patience and tolerance"
                },
                {
                    "name": "Forgiveness Practice",
                    "type": "Heart",
                    "immediate": False,
                    "description": "Release resentment and anger"
                }
            ]
        }
        
        self.logger.info("🌟 Practice Recommendation Engine initialized")
    
    async def generate_recommendations(
        self,
        user_profile: UserProfile,
        context: Dict[str, Any] = None
    ) -> List[PracticeRecommendation]:
        """Generate personalized practice recommendations"""
        
        try:
            recommendations = []
            
            # Core meditation practice based on level
            meditation_rec = await self._get_meditation_recommendation(
                user_profile
            )
            if meditation_rec:
                recommendations.append(meditation_rec)
            
            # Path-specific practices
            path_recs = await self._get_path_specific_practices(user_profile)
            recommendations.extend(path_recs)
            
            # Challenge-based practices
            if user_profile.recent_challenges:
                challenge_recs = await self._get_challenge_practices(
                    user_profile
                )
                recommendations.extend(challenge_recs)
            
            # Chakra balancing if needed
            chakra_recs = await self._get_chakra_practices(user_profile)
            recommendations.extend(chakra_recs)
            
            # Progress-based practices
            progress_recs = await self._get_progress_practices(user_profile)
            recommendations.extend(progress_recs)
            
            # Adapt to time availability
            adapted_recs = await self._adapt_to_time_constraints(
                recommendations, user_profile
            )
            
            # Limit to top 6 recommendations
            return adapted_recs[:6]
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return await self._get_fallback_recommendations()
    
    async def _get_meditation_recommendation(
        self, user_profile: UserProfile
    ) -> Optional[PracticeRecommendation]:
        """Get primary meditation practice recommendation"""
        
        try:
            level_practices = self.meditation_practices.get(
                user_profile.spiritual_level, []
            )
            
            if not level_practices:
                return None
            
            # Choose practice based on experience and preferences
            suitable_practices = []
            
            for practice in level_practices:
                # Check if user has tried this practice
                if practice["name"] not in user_profile.learning_progress.practices_tried:
                    suitable_practices.append(practice)
            
            # If all practices tried, recommend advancing difficulty
            if not suitable_practices:
                suitable_practices = level_practices
            
            # Select best practice
            selected = suitable_practices[0]  # For now, take first suitable
            
            return PracticeRecommendation(
                name=selected["name"],
                type="Meditation",
                duration=selected["duration"],
                description=selected["description"],
                instructions=selected["instructions"],
                benefits=selected["benefits"],
                difficulty=selected["difficulty"],
                frequency="Daily",
                priority="High",
                personalized_notes=f"Recommended for {user_profile.spiritual_level.value} level"
            )
            
        except Exception as e:
            self.logger.error(f"Error getting meditation recommendation: {e}")
            return None
    
    async def _get_path_specific_practices(
        self, user_profile: UserProfile
    ) -> List[PracticeRecommendation]:
        """Get practices specific to user's spiritual path"""
        
        try:
            if user_profile.primary_path == SpiritualPath.MIXED:
                return []
            
            path_practices = self.path_practices.get(
                user_profile.primary_path, []
            )
            
            recommendations = []
            
            for practice in path_practices[:2]:  # Top 2 path practices
                rec = PracticeRecommendation(
                    name=practice["name"],
                    type=practice["category"],
                    description=practice["description"],
                    frequency=practice["frequency"],
                    instructions=practice["guidance"],
                    priority="Medium",
                    personalized_notes=f"Essential for {user_profile.primary_path.value.replace('_', ' ').title()} path"
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting path practices: {e}")
            return []
    
    async def _get_challenge_practices(
        self, user_profile: UserProfile
    ) -> List[PracticeRecommendation]:
        """Get practices to address current challenges"""
        
        try:
            recommendations = []
            
            for challenge in user_profile.recent_challenges[:2]:
                # Simple keyword matching for challenges
                challenge_key = None
                for key in self.challenge_practices.keys():
                    if key.lower() in challenge.lower():
                        challenge_key = key
                        break
                
                if challenge_key:
                    practices = self.challenge_practices[challenge_key]
                    for practice in practices[:1]:  # One per challenge
                        rec = PracticeRecommendation(
                            name=practice["name"],
                            type=practice["type"],
                            description=practice["description"],
                            priority="High" if practice["immediate"] else "Medium",
                            personalized_notes=f"Helpful for addressing {challenge}"
                        )
                        recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting challenge practices: {e}")
            return []
    
    async def _get_chakra_practices(
        self, user_profile: UserProfile
    ) -> List[PracticeRecommendation]:
        """Get practices for chakra balancing"""
        
        try:
            weak_chakras = user_profile.chakra_profile.get_weak_chakras()
            if not weak_chakras:
                return []
            
            recommendations = []
            primary_weak = weak_chakras[0]
            
            # Chakra-specific practices
            chakra_practices = {
                "root": {
                    "name": "Root Chakra Grounding",
                    "description": "Grounding meditation for stability",
                    "instructions": "Visualize roots growing from your base"
                },
                "heart": {
                    "name": "Heart Opening Practice",
                    "description": "Love and compassion cultivation",
                    "instructions": "Send love to yourself and others"
                },
                "throat": {
                    "name": "Truth Expression Practice",
                    "description": "Authentic communication development",
                    "instructions": "Practice speaking your truth with kindness"
                }
            }
            
            if primary_weak in chakra_practices:
                practice = chakra_practices[primary_weak]
                rec = PracticeRecommendation(
                    name=practice["name"],
                    type="Chakra Work",
                    description=practice["description"],
                    instructions=practice["instructions"],
                    priority="Medium",
                    personalized_notes=f"Balances your {primary_weak} chakra"
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting chakra practices: {e}")
            return []
    
    async def _get_progress_practices(
        self, user_profile: UserProfile
    ) -> List[PracticeRecommendation]:
        """Get practices based on user's progress and goals"""
        
        try:
            recommendations = []
            
            # If user has been consistent, suggest advancement
            if len(user_profile.learning_progress.practices_tried) > 5:
                rec = PracticeRecommendation(
                    name="Advanced Integration Practice",
                    type="Integration",
                    description="Combine learned practices into unified session",
                    instructions="Create your own practice sequence",
                    priority="Low",
                    personalized_notes="You're ready for self-directed practice"
                )
                recommendations.append(rec)
            
            # If user needs variety
            elif len(user_profile.learning_progress.practices_tried) < 3:
                rec = PracticeRecommendation(
                    name="Exploration Practice",
                    type="Discovery",
                    description="Try different meditation styles",
                    instructions="Experiment with various techniques",
                    priority="Medium",
                    personalized_notes="Discover what resonates with you"
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting progress practices: {e}")
            return []
    
    async def _adapt_to_time_constraints(
        self,
        recommendations: List[PracticeRecommendation],
        user_profile: UserProfile
    ) -> List[PracticeRecommendation]:
        """Adapt recommendations to user's time availability"""
        
        try:
            time_available = user_profile.life_circumstances.time_availability
            
            if time_available == "limited":
                # Shorten durations and suggest micro-practices
                for rec in recommendations:
                    if rec.duration and "minutes" in rec.duration:
                        # Reduce duration by half
                        original = rec.duration
                        rec.duration = "5-10 minutes"
                        rec.personalized_notes = f"Shortened from {original} due to time constraints"
                    
                    # Add micro-practice options
                    if rec.type == "Meditation":
                        rec.instructions = f"Start with just 2-3 minutes. {rec.instructions}"
            
            elif time_available == "abundant":
                # Suggest longer, deeper practices
                for rec in recommendations:
                    if rec.type == "Meditation":
                        rec.personalized_notes = "You have time for deeper practice - extend as you feel called"
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error adapting to time constraints: {e}")
            return recommendations
    
    async def _get_fallback_recommendations(
        self
    ) -> List[PracticeRecommendation]:
        """Get basic fallback recommendations"""
        
        return [
            PracticeRecommendation(
                name="Basic Meditation",
                type="Meditation",
                duration="10 minutes",
                description="Simple breathing meditation",
                instructions="Focus on your natural breath",
                benefits=["Relaxation", "Mental clarity"],
                frequency="Daily",
                priority="High"
            ),
            PracticeRecommendation(
                name="Gratitude Practice",
                type="Reflection",
                duration="5 minutes",
                description="Count your blessings",
                instructions="List three things you're grateful for",
                benefits=["Positive mindset", "Emotional wellbeing"],
                frequency="Daily",
                priority="Medium"
            )
        ]
    
    async def track_practice_completion(
        self,
        user_profile: UserProfile,
        practice_name: str,
        duration: int,
        experience_rating: int
    ) -> UserProfile:
        """Track when user completes a practice"""
        
        try:
            # Add to practiced techniques
            if practice_name not in user_profile.learning_progress.practices_tried:
                user_profile.learning_progress.practices_tried.append(practice_name)
            
            # Update meditation minutes
            if "meditation" in practice_name.lower():
                user_profile.meditation_minutes += duration
            
            # Record feedback for future recommendations
            feedback_entry = {
                "practice": practice_name,
                "duration": duration,
                "rating": experience_rating,
                "date": datetime.now().isoformat()
            }
            
            # Add to interaction history
            user_profile.update_interaction(
                f"Completed {practice_name}",
                experience_rating
            )
            
            return user_profile
            
        except Exception as e:
            self.logger.error(f"Error tracking practice completion: {e}")
            return user_profile


# Global practice engine instance
_practice_engine = None


def get_practice_recommendation_engine() -> PracticeRecommendationEngine:
    """Get global practice recommendation engine instance"""
    global _practice_engine
    if _practice_engine is None:
        _practice_engine = PracticeRecommendationEngine()
    return _practice_engine


# Export main class
__all__ = ["PracticeRecommendationEngine", "get_practice_recommendation_engine"]
