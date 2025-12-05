"""
🌟 DharmaMind Universal Spiritual Profile Models
==============================================

Advanced user profiling for personalized spiritual guidance.
These models capture the user's spiritual journey, preferences, and progress
to provide universal wisdom and practice recommendations for all seekers.

Drawing from eternal principles while serving humanity across all backgrounds 🙏
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SpiritualLevel(str, Enum):
    """User's spiritual development level"""
    SEEKER = "seeker"           # Just beginning the journey
    STUDENT = "student"         # Learning and practicing
    PRACTITIONER = "practitioner"   # Regular practice established
    DEVOTEE = "devotee"         # Deep devotion and commitment
    TEACHER = "teacher"         # Sharing wisdom with others
    SAGE = "sage"              # Advanced spiritual understanding

class SpiritualPath(str, Enum):
    """Universal spiritual approaches accessible to all seekers"""
    SERVICE_PATH = "service_path"           # Path of selfless service
    DEVOTION_PATH = "devotion_path"         # Path of love and surrender
    MEDITATION_PATH = "meditation_path"     # Path of inner stillness
    WISDOM_PATH = "wisdom_path"             # Path of knowledge and inquiry
    ENERGY_PATH = "energy_path"             # Path of subtle energy work
    MINDFUL_LIVING = "mindful_living"       # Path of conscious daily life
    MIXED_PATH = "mixed_path"               # Combination of approaches

class LifeStage(str, Enum):
    """Hindu life stages (Ashramas)"""
    BRAHMACHARYA = "brahmacharya"   # Student stage
    GRIHASTHA = "grihastha"         # Householder stage
    VANAPRASTHA = "vanaprastha"     # Forest dweller stage
    SANNYASA = "sannyasa"           # Renunciate stage

class PersonalityType(str, Enum):
    """Spiritual personality types based on Gunas"""
    SATTVIC = "sattvic"         # Pure, balanced, wise
    RAJASIC = "rajasic"         # Active, passionate, dynamic
    TAMASIC = "tamasic"         # Inert, lazy, confused
    SATTVA_RAJAS = "sattva_rajas"   # Balanced with activity
    RAJAS_TAMAS = "rajas_tamas"     # Active but confused
    MIXED_GUNA = "mixed_guna"       # All three qualities

class MeditationExperience(str, Enum):
    """User's meditation experience level"""
    NONE = "none"
    BEGINNER = "beginner"       # < 6 months
    INTERMEDIATE = "intermediate"   # 6 months - 2 years
    ADVANCED = "advanced"       # 2+ years
    EXPERT = "expert"           # 5+ years, teaching others

class PracticeFrequency(str, Enum):
    """How often user practices"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    OCCASIONALLY = "occasionally"
    RARELY = "rarely"

class SpiritualInterest(BaseModel):
    """Specific spiritual interests and topics"""
    name: str = Field(..., description="Interest name")
    level: int = Field(..., ge=1, le=10, description="Interest level (1-10)")
    learned_about: bool = Field(default=False, description="Has user learned about this")
    wants_to_learn: bool = Field(default=True, description="Wants to learn more")
    
class ChakraProfile(BaseModel):
    """User's chakra energy profile"""
    root_chakra: int = Field(default=5, ge=1, le=10, description="Muladhara - survival, grounding")
    sacral_chakra: int = Field(default=5, ge=1, le=10, description="Svadhisthana - creativity, sexuality")
    solar_plexus: int = Field(default=5, ge=1, le=10, description="Manipura - personal power")
    heart_chakra: int = Field(default=5, ge=1, le=10, description="Anahata - love, compassion")
    throat_chakra: int = Field(default=5, ge=1, le=10, description="Vishuddha - communication")
    third_eye: int = Field(default=5, ge=1, le=10, description="Ajna - intuition, wisdom")
    crown_chakra: int = Field(default=5, ge=1, le=10, description="Sahasrara - spiritual connection")
    
    def get_dominant_chakras(self) -> List[str]:
        """Get the most active chakras"""
        chakras = {
            "root": self.root_chakra,
            "sacral": self.sacral_chakra,
            "solar_plexus": self.solar_plexus,
            "heart": self.heart_chakra,
            "throat": self.throat_chakra,
            "third_eye": self.third_eye,
            "crown": self.crown_chakra
        }
        sorted_chakras = sorted(chakras.items(), key=lambda x: x[1], reverse=True)
        return [chakra[0] for chakra in sorted_chakras[:3]]
    
    def get_weak_chakras(self) -> List[str]:
        """Get chakras that need attention"""
        chakras = {
            "root": self.root_chakra,
            "sacral": self.sacral_chakra,
            "solar_plexus": self.solar_plexus,
            "heart": self.heart_chakra,
            "throat": self.throat_chakra,
            "third_eye": self.third_eye,
            "crown": self.crown_chakra
        }
        return [name for name, level in chakras.items() if level <= 4]

class PracticePreferences(BaseModel):
    """User's practice preferences and schedule"""
    preferred_time: str = Field(default="morning", description="Preferred practice time")
    session_duration: int = Field(default=20, ge=5, le=120, description="Preferred session length in minutes")
    practice_frequency: PracticeFrequency = Field(default=PracticeFrequency.WEEKLY)
    preferred_practices: List[str] = Field(default=[], description="Preferred spiritual practices")
    avoid_practices: List[str] = Field(default=[], description="Practices to avoid")
    meditation_style: List[str] = Field(default=[], description="Preferred meditation styles")
    mantra_preferences: List[str] = Field(default=[], description="Preferred mantras")
    deity_connection: List[str] = Field(default=[], description="Connected deities")
    language_preference: str = Field(default="english", description="Preferred language for guidance")
    include_sanskrit: bool = Field(default=True, description="Include Sanskrit terms and mantras")

class LifeCircumstances(BaseModel):
    """User's current life circumstances affecting practice"""
    life_stage: LifeStage = Field(default=LifeStage.GRIHASTHA)
    occupation_type: str = Field(default="working", description="Type of occupation")
    stress_level: int = Field(default=5, ge=1, le=10, description="Current stress level")
    family_situation: str = Field(default="", description="Family circumstances")
    health_concerns: List[str] = Field(default=[], description="Health issues affecting practice")
    time_availability: str = Field(default="moderate", description="Available time for practice")
    living_situation: str = Field(default="", description="Living environment")
    spiritual_community: bool = Field(default=False, description="Part of spiritual community")

class LearningProgress(BaseModel):
    """Track user's learning and spiritual progress"""
    topics_learned: List[str] = Field(default=[], description="Topics user has learned about")
    practices_tried: List[str] = Field(default=[], description="Practices user has tried")
    milestones_achieved: List[str] = Field(default=[], description="Spiritual milestones")
    challenges_faced: List[str] = Field(default=[], description="Challenges in practice")
    breakthrough_moments: List[str] = Field(default=[], description="Spiritual breakthroughs")
    current_focus: List[str] = Field(default=[], description="Current learning focus areas")
    goals: List[str] = Field(default=[], description="Spiritual goals")
    
class InteractionHistory(BaseModel):
    """Track user's interaction patterns"""
    total_conversations: int = Field(default=0)
    favorite_topics: List[str] = Field(default=[])
    question_patterns: List[str] = Field(default=[])
    response_preferences: Dict[str, Any] = Field(default={})
    feedback_given: List[str] = Field(default=[])
    most_helpful_responses: List[str] = Field(default=[])
    spiritual_insights_gained: List[str] = Field(default=[])

class PersonalizationSettings(BaseModel):
    """User's personalization preferences"""
    guidance_style: str = Field(default="gentle", description="Preferred guidance style")
    communication_tone: str = Field(default="compassionate", description="Preferred communication tone")
    depth_level: str = Field(default="moderate", description="Preferred depth of teachings")
    cultural_context: str = Field(default="universal", description="Cultural context preference")
    modernize_teachings: bool = Field(default=True, description="Adapt ancient teachings to modern life")
    personal_examples: bool = Field(default=True, description="Include personal examples")
    practice_reminders: bool = Field(default=True, description="Send practice reminders")
    progress_tracking: bool = Field(default=True, description="Track spiritual progress")

class UserProfile(BaseModel):
    """Complete user spiritual profile for personalization"""
    
    # Basic Information
    user_id: str = Field(..., description="Unique user identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Spiritual Profile
    spiritual_level: SpiritualLevel = Field(default=SpiritualLevel.SEEKER)
    primary_path: SpiritualPath = Field(default=SpiritualPath.MIXED_PATH)
    secondary_paths: List[SpiritualPath] = Field(default=[])
    personality_type: PersonalityType = Field(default=PersonalityType.MIXED_GUNA)
    meditation_experience: MeditationExperience = Field(default=MeditationExperience.BEGINNER)
    
    # Interests and Preferences
    spiritual_interests: List[SpiritualInterest] = Field(default=[])
    chakra_profile: ChakraProfile = Field(default_factory=ChakraProfile)
    practice_preferences: PracticePreferences = Field(default_factory=PracticePreferences)
    life_circumstances: LifeCircumstances = Field(default_factory=LifeCircumstances)
    personalization_settings: PersonalizationSettings = Field(default_factory=PersonalizationSettings)
    
    # Progress Tracking
    learning_progress: LearningProgress = Field(default_factory=LearningProgress)
    interaction_history: InteractionHistory = Field(default_factory=InteractionHistory)
    
    # Dynamic Data
    last_interaction: Optional[datetime] = None
    current_mood: Optional[str] = None
    recent_challenges: List[str] = Field(default=[])
    recent_insights: List[str] = Field(default=[])
    
    def update_interaction(self, topic: str, response_quality: int):
        """Update interaction history"""
        self.interaction_history.total_conversations += 1
        if topic not in self.interaction_history.favorite_topics:
            self.interaction_history.favorite_topics.append(topic)
        self.last_interaction = datetime.now()
        self.updated_at = datetime.now()
    
    def add_learning_progress(self, topic: str, practice: str = None):
        """Add to learning progress"""
        if topic not in self.learning_progress.topics_learned:
            self.learning_progress.topics_learned.append(topic)
        
        if practice and practice not in self.learning_progress.practices_tried:
            self.learning_progress.practices_tried.append(practice)
        
        self.updated_at = datetime.now()
    
    def get_personalization_context(self) -> Dict[str, Any]:
        """Get context for personalizing responses"""
        return {
            "spiritual_level": self.spiritual_level.value,
            "primary_path": self.primary_path.value,
            "personality_type": self.personality_type.value,
            "meditation_experience": self.meditation_experience.value,
            "dominant_chakras": self.chakra_profile.get_dominant_chakras(),
            "weak_chakras": self.chakra_profile.get_weak_chakras(),
            "life_stage": self.life_circumstances.life_stage.value,
            "stress_level": self.life_circumstances.stress_level,
            "preferred_practices": self.practice_preferences.preferred_practices,
            "guidance_style": self.personalization_settings.guidance_style,
            "recent_topics": self.interaction_history.favorite_topics[-5:],
            "current_focus": self.learning_progress.current_focus,
            "goals": self.learning_progress.goals
        }
    
    def suggest_next_learning_topics(self) -> List[str]:
        """Suggest next topics to learn based on profile"""
        suggestions = []
        
        # Based on spiritual level
        if self.spiritual_level == SpiritualLevel.SEEKER:
            suggestions.extend(["dharma basics", "meditation introduction", "yoga principles"])
        elif self.spiritual_level == SpiritualLevel.STUDENT:
            suggestions.extend(["karma yoga", "bhakti practices", "scriptural study"])
        elif self.spiritual_level == SpiritualLevel.PRACTITIONER:
            suggestions.extend(["advanced meditation", "energy work", "devotional practices"])
        
        # Based on primary path
        if self.primary_path == SpiritualPath.SERVICE_PATH:
            suggestions.extend(["selfless service", "compassionate action", "helping others"])
        elif self.primary_path == SpiritualPath.DEVOTION_PATH:
            suggestions.extend(["loving practices", "gratitude cultivation", "surrender techniques"])
        elif self.primary_path == SpiritualPath.MEDITATION_PATH:
            suggestions.extend(["advanced meditation", "mindfulness", "inner stillness"])
        elif self.primary_path == SpiritualPath.WISDOM_PATH:
            suggestions.extend(["self-inquiry", "philosophical study", "truth seeking"])
        
        # Based on weak chakras
        weak_chakras = self.chakra_profile.get_weak_chakras()
        for chakra in weak_chakras:
            if chakra == "root":
                suggestions.extend(["grounding practices", "stability meditation"])
            elif chakra == "heart":
                suggestions.extend(["loving-kindness meditation", "compassion practices"])
            elif chakra == "third_eye":
                suggestions.extend(["intuition development", "inner vision practices"])
        
        # Remove duplicates and limit
        return list(set(suggestions))[:8]

class ProfileUpdateRequest(BaseModel):
    """Request to update user profile"""
    spiritual_level: Optional[SpiritualLevel] = None
    primary_path: Optional[SpiritualPath] = None
    meditation_experience: Optional[MeditationExperience] = None
    spiritual_interests: Optional[List[SpiritualInterest]] = None
    chakra_profile: Optional[ChakraProfile] = None
    practice_preferences: Optional[PracticePreferences] = None
    life_circumstances: Optional[LifeCircumstances] = None
    personalization_settings: Optional[PersonalizationSettings] = None
    current_mood: Optional[str] = None
    recent_challenges: Optional[List[str]] = None

class PersonalizationResponse(BaseModel):
    """Personalized response with recommendations"""
    response: str = Field(..., description="Personalized response")
    personalization_factors: List[str] = Field(..., description="Factors used for personalization")
    recommended_practices: List[str] = Field(default=[], description="Recommended practices")
    suggested_topics: List[str] = Field(default=[], description="Suggested learning topics")
    chakra_guidance: Optional[str] = None
    path_specific_advice: Optional[str] = None
    progress_encouragement: Optional[str] = None

class PracticeRecommendation(BaseModel):
    """Recommendation for spiritual practice"""
    practice_type: str = Field(..., description="Type of practice")
    title: str = Field(..., description="Practice title")
    description: str = Field(..., description="Practice description")
    duration: str = Field(..., description="Recommended duration")
    difficulty: str = Field(..., description="Difficulty level")
    benefits: List[str] = Field(default=[], description="Expected benefits")
    instructions: Optional[str] = None
    prerequisites: Optional[List[str]] = None
    ideal_time: Optional[str] = None
    frequency: Optional[str] = None

# Export all models
__all__ = [
    "SpiritualLevel", "SpiritualPath", "LifeStage", "PersonalityType",
    "MeditationExperience", "PracticeFrequency", "SpiritualInterest",
    "ChakraProfile", "PracticePreferences", "LifeCircumstances",
    "LearningProgress", "InteractionHistory", "PersonalizationSettings",
    "UserProfile", "ProfileUpdateRequest", "PersonalizationResponse",
    "PracticeRecommendation"
]
