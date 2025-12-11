<<<<<<< HEAD
"""
User Profile model for DharmaMind platform

Extended user profile information including spiritual journey,
preferences, and detailed personal information.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from enum import Enum

class SpiritualLevel(str, Enum):
    """Spiritual development levels"""
    SEEKER = "seeker"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PRACTITIONER = "practitioner"
    TEACHER = "teacher"
    MASTER = "master"

class MeditationExperience(str, Enum):
    """Meditation experience levels"""
    NONE = "none"
    BEGINNER = "beginner"
    SOME_EXPERIENCE = "some_experience"
    REGULAR_PRACTICE = "regular_practice"
    EXPERIENCED = "experienced"
    ADVANCED = "advanced"

class SpiritualTradition(str, Enum):
    """Spiritual traditions and practices"""
    HINDUISM = "hinduism"
    BUDDHISM = "buddhism"
    JAINISM = "jainism"
    SIKHISM = "sikhism"
    YOGA = "yoga"
    VEDANTA = "vedanta"
    TANTRA = "tantra"
    SUFISM = "sufism"
    CHRISTIANITY = "christianity"
    JUDAISM = "judaism"
    ISLAM = "islam"
    NON_DENOMINATIONAL = "non_denominational"
    SECULAR = "secular"

class UserProfile(BaseModel):
    """Extended user profile with spiritual journey information"""
    user_id: str = Field(..., description="Associated user ID")
    
    # Personal Information
    date_of_birth: Optional[date] = Field(default=None, description="Date of birth")
    gender: Optional[str] = Field(default=None, description="Gender identity")
    phone_number: Optional[str] = Field(default=None, description="Phone number")
    address: Optional[Dict[str, str]] = Field(default=None, description="Address information")
    
    # Spiritual Journey
    spiritual_level: SpiritualLevel = Field(default=SpiritualLevel.SEEKER, description="Current spiritual development level")
    meditation_experience: MeditationExperience = Field(default=MeditationExperience.NONE, description="Meditation experience")
    primary_tradition: Optional[SpiritualTradition] = Field(default=None, description="Primary spiritual tradition")
    secondary_traditions: List[SpiritualTradition] = Field(default_factory=list, description="Additional spiritual traditions")
    
    # Spiritual Practices
    daily_practices: List[str] = Field(default_factory=list, description="Daily spiritual practices")
    meditation_duration: Optional[int] = Field(default=None, description="Daily meditation duration in minutes")
    preferred_meditation_style: Optional[str] = Field(default=None, description="Preferred meditation style")
    spiritual_goals: List[str] = Field(default_factory=list, description="Spiritual goals and aspirations")
    
    # Learning and Growth
    learning_preferences: Dict[str, Any] = Field(default_factory=dict, description="Learning style preferences")
    areas_of_interest: List[str] = Field(default_factory=list, description="Areas of spiritual interest")
    books_read: List[str] = Field(default_factory=list, description="Spiritual books read")
    teachers_followed: List[str] = Field(default_factory=list, description="Spiritual teachers followed")
    
    # Wellness and Lifestyle
    health_conditions: Optional[List[str]] = Field(default=None, description="Relevant health conditions")
    dietary_preferences: Optional[str] = Field(default=None, description="Dietary preferences")
    exercise_routine: Optional[str] = Field(default=None, description="Exercise routine")
    sleep_schedule: Optional[Dict[str, str]] = Field(default=None, description="Sleep schedule")
    
    # Personality and Preferences
    personality_type: Optional[str] = Field(default=None, description="Personality type (MBTI, etc.)")
    communication_style: Optional[str] = Field(default=None, description="Preferred communication style")
    feedback_preferences: Dict[str, Any] = Field(default_factory=dict, description="How they prefer to receive feedback")
    
    # Progress Tracking
    milestones_achieved: List[Dict[str, Any]] = Field(default_factory=list, description="Spiritual milestones achieved")
    challenges_faced: List[str] = Field(default_factory=list, description="Spiritual challenges encountered")
    breakthroughs: List[Dict[str, Any]] = Field(default_factory=list, description="Spiritual breakthroughs")
    
    # Session and Interaction History
    total_sessions: int = Field(default=0, description="Total spiritual guidance sessions")
    favorite_topics: List[str] = Field(default_factory=list, description="Most discussed topics")
    session_feedback_scores: List[float] = Field(default_factory=list, description="Session feedback scores")
    
    # Privacy and Sharing
    profile_visibility: str = Field(default="private", description="Profile visibility setting")
    data_sharing_consent: Dict[str, bool] = Field(default_factory=dict, description="Data sharing consents")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Profile creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    last_spiritual_assessment: Optional[datetime] = Field(default=None, description="Last spiritual assessment date")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            date: lambda d: d.isoformat()
        }

class UserProfileUpdate(BaseModel):
    """Model for updating user profile"""
    date_of_birth: Optional[date] = Field(default=None, description="Date of birth")
    gender: Optional[str] = Field(default=None, description="Gender identity")
    phone_number: Optional[str] = Field(default=None, description="Phone number")
    spiritual_level: Optional[SpiritualLevel] = Field(default=None, description="Spiritual level")
    meditation_experience: Optional[MeditationExperience] = Field(default=None, description="Meditation experience")
    primary_tradition: Optional[SpiritualTradition] = Field(default=None, description="Primary tradition")
    daily_practices: Optional[List[str]] = Field(default=None, description="Daily practices")
    meditation_duration: Optional[int] = Field(default=None, description="Meditation duration")
    spiritual_goals: Optional[List[str]] = Field(default=None, description="Spiritual goals")
    areas_of_interest: Optional[List[str]] = Field(default=None, description="Areas of interest")

class SpiritualAssessment(BaseModel):
    """Spiritual development assessment"""
    user_id: str = Field(..., description="User ID")
    assessment_date: datetime = Field(default_factory=datetime.now, description="Assessment date")
    
    # Assessment scores
    consciousness_level: float = Field(..., ge=0.0, le=10.0, description="Consciousness development score")
    wisdom_integration: float = Field(..., ge=0.0, le=10.0, description="Wisdom integration score")
    compassion_development: float = Field(..., ge=0.0, le=10.0, description="Compassion development score")
    mindfulness_practice: float = Field(..., ge=0.0, le=10.0, description="Mindfulness practice score")
    dharmic_alignment: float = Field(..., ge=0.0, le=10.0, description="Dharmic alignment score")
    
    # Qualitative assessments
    strengths: List[str] = Field(default_factory=list, description="Spiritual strengths")
    growth_areas: List[str] = Field(default_factory=list, description="Areas for growth")
    recommendations: List[str] = Field(default_factory=list, description="Spiritual practice recommendations")
    
    # Overall assessment
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Overall spiritual development score")
    next_assessment_date: Optional[datetime] = Field(default=None, description="Recommended next assessment date")

class PersonalizationResponse(BaseModel):
    """Response for personalization recommendations"""
    user_id: str = Field(..., description="User ID")
    
    # Personalized recommendations
    recommended_practices: List[str] = Field(default_factory=list, description="Recommended spiritual practices")
    suggested_content: List[Dict[str, Any]] = Field(default_factory=list, description="Suggested content")
    personalized_guidance: str = Field(default="", description="Personalized spiritual guidance")
    
    # Learning path
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    learning_path: List[Dict[str, Any]] = Field(default_factory=list, description="Personalized learning path")
    
    # Practice adjustments
    meditation_adjustments: Optional[Dict[str, Any]] = Field(default=None, description="Meditation practice adjustments")
    practice_schedule: Optional[Dict[str, Any]] = Field(default=None, description="Recommended practice schedule")
    
    # Progress insights
    growth_insights: List[str] = Field(default_factory=list, description="Insights about spiritual growth")
    celebration_points: List[str] = Field(default_factory=list, description="Achievements to celebrate")
    
    # Personalization metadata
    personalization_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Personalization relevance score")
    generated_at: datetime = Field(default_factory=datetime.now, description="Response generation timestamp")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class SpiritualPath(BaseModel):
    """Spiritual development path and progression"""
    path_id: str = Field(..., description="Unique path identifier")
    user_id: str = Field(..., description="Associated user ID")
    
    # Path definition
    path_name: str = Field(..., description="Name of the spiritual path")
    tradition: SpiritualTradition = Field(..., description="Associated spiritual tradition")
    difficulty_level: str = Field(default="beginner", description="Path difficulty level")
    estimated_duration: Optional[int] = Field(default=None, description="Estimated duration in days")
    
    # Path structure
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="Path stages")
    current_stage: int = Field(default=0, description="Current stage index")
    practices: List[str] = Field(default_factory=list, description="Required practices")
    milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Path milestones")
    
    # Progress tracking
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Completion percentage")
    days_active: int = Field(default=0, description="Days actively following path")
    streak_count: int = Field(default=0, description="Current practice streak")
    
    # Personalization
    adapted_for_user: bool = Field(default=False, description="Whether path is adapted for user")
    adaptations: List[str] = Field(default_factory=list, description="User-specific adaptations")
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now, description="Path start timestamp")
    last_activity: Optional[datetime] = Field(default=None, description="Last activity timestamp")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class PracticeRecommendation(BaseModel):
    """Spiritual practice recommendation"""
    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    user_id: str = Field(..., description="Target user ID")
    
    # Recommendation details
    practice_name: str = Field(..., description="Name of recommended practice")
    practice_type: str = Field(..., description="Type of practice (meditation, yoga, etc.)")
    difficulty_level: str = Field(default="beginner", description="Difficulty level")
    duration_minutes: Optional[int] = Field(default=None, description="Recommended duration")
    
    # Practice description
    description: str = Field(..., description="Practice description")
    instructions: List[str] = Field(default_factory=list, description="Step-by-step instructions")
    benefits: List[str] = Field(default_factory=list, description="Expected benefits")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites")
    
    # Personalization
    personalization_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Personalization relevance")
    adaptation_notes: List[str] = Field(default_factory=list, description="User-specific adaptations")
    
    # Scheduling
    recommended_frequency: Optional[str] = Field(default=None, description="Recommended frequency")
    best_time_of_day: Optional[str] = Field(default=None, description="Optimal time of day")
    
    # Tracking
    is_accepted: Optional[bool] = Field(default=None, description="Whether user accepted recommendation")
    is_completed: bool = Field(default=False, description="Whether practice was completed")
    completion_date: Optional[datetime] = Field(default=None, description="Practice completion date")
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now, description="Recommendation timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Recommendation expiry")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
=======
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

# Export all models
__all__ = [
    "SpiritualLevel", "SpiritualPath", "LifeStage", "PersonalityType",
    "MeditationExperience", "PracticeFrequency", "SpiritualInterest",
    "ChakraProfile", "PracticePreferences", "LifeCircumstances",
    "LearningProgress", "InteractionHistory", "PersonalizationSettings",
    "UserProfile", "ProfileUpdateRequest", "PersonalizationResponse"
]
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
