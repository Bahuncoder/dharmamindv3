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