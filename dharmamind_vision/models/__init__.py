"""
📊 Data Models for DharmaMind Vision

Core data structures and models for the revolutionary vision system:
- Pose and landmark data models
- Session and user data structures  
- Analysis result models
- Traditional yoga concept models
- Performance metrics models

These models ensure consistency across all 6 revolutionary subsystems.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import numpy as np

# Pose and Movement Models

@dataclass
class Landmark3D:
    """3D landmark with confidence score."""
    x: float
    y: float
    z: float
    confidence: float
    visibility: float

@dataclass
class PoseFrame:
    """Complete pose data for a single frame."""
    timestamp: datetime
    frame_id: str
    landmarks: List[Landmark3D]
    quality_score: float
    detection_confidence: float
    
    # Analysis results
    pose_classification: Optional[str] = None
    alignment_analysis: Dict = field(default_factory=dict)
    balance_assessment: Dict = field(default_factory=dict)
    
    # Traditional yoga analysis
    chakra_analysis: Dict = field(default_factory=dict)
    traditional_assessment: Dict = field(default_factory=dict)

# Session Models

class SessionType(Enum):
    """Types of practice sessions."""
    HATHA_YOGA = "hatha_yoga"
    VINYASA_FLOW = "vinyasa_flow"
    MEDITATION = "meditation"
    PRANAYAMA = "pranayama"
    MIXED_PRACTICE = "mixed_practice"

class ExperienceLevel(Enum):
    """User experience levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    TEACHER = "teacher"

@dataclass
class SessionGoals:
    """Session-specific goals and intentions."""
    primary_focus: str  # e.g., "flexibility", "strength", "relaxation"
    specific_poses: List[str]  # Target poses for the session
    duration_minutes: int
    intensity_level: str  # "gentle", "moderate", "intense"
    traditional_focus: Optional[str] = None  # e.g., "chakra_balancing", "pranayama"

@dataclass
class SessionMetrics:
    """Comprehensive session performance metrics."""
    
    # Duration and timing
    total_duration_minutes: float
    active_practice_minutes: float
    meditation_minutes: float
    rest_minutes: float
    
    # Quality scores
    overall_quality_score: float  # 0-10
    pose_accuracy_average: float
    alignment_consistency: float
    breathing_quality: float
    mindfulness_score: float
    
    # Progress metrics
    poses_attempted: int
    poses_completed_well: int
    corrections_applied: int
    improvements_made: int
    
    # Traditional assessments
    chakra_balance_score: float
    energy_flow_quality: float
    traditional_principles_embodied: List[str]

# Analysis Result Models

@dataclass
class PostureCorrection:
    """Real-time posture correction data."""
    target_body_part: str
    current_position: Landmark3D
    ideal_position: Landmark3D
    correction_vector: Tuple[float, float, float]
    priority_level: str  # "low", "medium", "high", "critical"
    correction_instruction: str
    traditional_guidance: str
    biomechanical_reasoning: str

@dataclass
class MeditationAnalysis:
    """Meditation state analysis results."""
    current_state: str  # "settling", "focused", "absorbed", "transcendent"
    stillness_score: float  # 0-1
    micro_movement_level: float
    meditation_depth: float
    state_duration_seconds: float
    traditional_stage: str  # "pratyahara", "dharana", "dhyana", "samadhi"
    
    # Breathing during meditation
    breathing_rhythm_score: float
    breathing_depth_score: float
    pranayama_classification: Optional[str]

@dataclass
class BreathingAnalysis:
    """Breathing pattern analysis results."""
    current_phase: str  # "inhale", "exhale", "pause"
    breathing_rate_per_minute: float
    rhythm_consistency: float
    depth_adequacy: float
    pattern_classification: str  # "natural", "ujjayi", "kapalabhati", etc.
    traditional_assessment: str
    quality_score: float  # 0-1

# Learning and Progress Models

class CompetencyArea(Enum):
    """Core competency areas for yoga practice."""
    ALIGNMENT_PRECISION = "alignment_precision"
    BREATH_AWARENESS = "breath_awareness"
    STRENGTH_BUILDING = "strength_building"
    FLEXIBILITY_DEVELOPMENT = "flexibility_development"
    BALANCE_STABILITY = "balance_stability"
    MINDFUL_MOVEMENT = "mindful_movement"
    CONCENTRATION_FOCUS = "concentration_focus"
    TRADITIONAL_UNDERSTANDING = "traditional_understanding"
    ENERGY_AWARENESS = "energy_awareness"

@dataclass
class SkillAssessment:
    """Individual skill assessment in a competency area."""
    competency: CompetencyArea
    current_level: float  # 0-10 skill level
    confidence_score: float  # 0-1 confidence in assessment
    improvement_rate: float  # Rate of improvement
    strengths: List[str]
    areas_for_growth: List[str]
    recommended_practices: List[str]
    traditional_milestones: List[str]

@dataclass
class LearningPath:
    """Personalized learning path for user development."""
    user_id: str
    current_assessments: Dict[CompetencyArea, SkillAssessment]
    short_term_objectives: List[str]  # Next 2-4 weeks
    medium_term_goals: List[str]     # Next 2-3 months
    long_term_aspirations: List[str] # 6 months - 1 year
    
    # Personalization factors
    learning_style: str  # "visual", "kinesthetic", "analytical", "intuitive"
    pace_preference: str  # "gradual", "steady", "intensive"
    cultural_integration_level: str  # "basic", "moderate", "deep"

# Traditional Yoga Models

@dataclass
class AsanaClassification:
    """Traditional yoga pose classification."""
    sanskrit_name: str
    english_name: str
    pose_family: str  # "standing", "seated", "backbend", "forward_fold", etc.
    difficulty_level: str
    traditional_benefits: List[str]
    contraindications: List[str]
    preparatory_poses: List[str]
    counter_poses: List[str]
    
    # Traditional context
    classical_text_reference: Optional[str]  # e.g., "Hatha Yoga Pradipika"
    philosophical_significance: str
    energetic_qualities: str
    chakra_associations: List[str]

@dataclass
class ChakraAnalysis:
    """Chakra energy analysis from pose and movement."""
    chakra_name: str
    sanskrit_name: str
    associated_body_region: str
    current_energy_level: float  # 0-1
    balance_score: float  # 0-1
    blockage_indicators: List[str]
    enhancement_suggestions: List[str]
    traditional_practices: List[str]

# Feedback and Communication Models

class CommunicationStyle(Enum):
    """Different communication styles for feedback."""
    GENTLE_NURTURING = "gentle_nurturing"
    WISE_TEACHER = "wise_teacher"
    ENCOURAGING_COACH = "encouraging_coach"
    SCIENTIFIC_PRECISE = "scientific_precise"
    POETIC_METAPHORICAL = "poetic_metaphorical"
    PRACTICAL_DIRECT = "practical_direct"
    CONTEMPLATIVE_PHILOSOPHICAL = "contemplative_philosophical"

@dataclass
class FeedbackContent:
    """Structured feedback content."""
    message_type: str  # "correction", "encouragement", "guidance", "wisdom"
    primary_message: str
    traditional_context: Optional[str]
    scientific_rationale: Optional[str]
    actionable_steps: List[str]
    
    # Delivery preferences
    communication_style: CommunicationStyle
    urgency_level: str  # "low", "medium", "high"
    timing_preference: str  # "immediate", "end_of_pose", "session_break"

# Analytics and Insights Models

@dataclass
class PracticeInsight:
    """Intelligent insight derived from practice analysis."""
    insight_type: str  # "pattern", "improvement", "recommendation", "correlation"
    title: str
    description: str
    confidence_level: float  # 0-1
    supporting_data: Dict[str, Any]
    actionable_recommendations: List[str]
    traditional_wisdom_connection: Optional[str]
    predicted_impact: str  # "low", "medium", "high"

@dataclass
class LifeCorrelation:
    """Correlation between practice and life outcomes."""
    practice_element: str  # e.g., "morning_meditation_consistency"
    life_outcome: str      # e.g., "stress_reduction", "sleep_quality"
    correlation_strength: float  # 0-1
    confidence_level: float      # 0-1
    time_lag_days: int          # Days between practice and outcome
    supporting_evidence: List[str]
    traditional_explanation: Optional[str]

@dataclass
class PredictiveInsight:
    """Predictive insight about future outcomes."""
    prediction_type: str  # "skill_development", "health_outcome", "practice_evolution"
    timeframe: str       # "1_week", "1_month", "3_months", "6_months"
    predicted_outcome: str
    probability: float   # 0-1
    contributing_factors: List[str]
    required_conditions: List[str]
    traditional_validation: Optional[str]

# System Performance Models

@dataclass
class PerformanceMetrics:
    """System performance monitoring."""
    processing_fps: float
    latency_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_utilization_percent: Optional[float]
    
    # Analysis accuracy
    pose_detection_accuracy: float
    landmark_precision: float
    classification_confidence: float
    
    # System health
    error_rate: float
    uptime_hours: float
    last_error_timestamp: Optional[datetime]

@dataclass
class QualityAssurance:
    """Quality assurance metrics for analysis results."""
    data_completeness: float      # 0-1
    consistency_score: float      # 0-1
    reliability_index: float      # 0-1
    traditional_alignment: float  # 0-1 (alignment with traditional principles)
    user_satisfaction: float     # 0-1 (based on user feedback)
    
    # Validation results
    cross_validation_score: float
    expert_review_score: Optional[float]
    user_feedback_score: Optional[float]

# Integration Models

@dataclass
class SystemIntegration:
    """Integration status between all 6 subsystems."""
    posture_correction_status: str
    meditation_analysis_status: str
    learning_system_status: str
    life_integration_status: str
    session_management_status: str
    feedback_engine_status: str
    
    # Cross-system communication
    data_flow_health: float      # 0-1
    synchronization_score: float # 0-1
    integration_errors: List[str]
    last_sync_timestamp: datetime

# Export all models for easy importing
__all__ = [
    # Pose and Movement Models
    'Landmark3D', 'PoseFrame',
    
    # Session Models
    'SessionType', 'ExperienceLevel', 'SessionGoals', 'SessionMetrics',
    
    # Analysis Result Models
    'PostureCorrection', 'MeditationAnalysis', 'BreathingAnalysis',
    
    # Learning and Progress Models
    'CompetencyArea', 'SkillAssessment', 'LearningPath',
    
    # Traditional Yoga Models
    'AsanaClassification', 'ChakraAnalysis',
    
    # Feedback and Communication Models
    'CommunicationStyle', 'FeedbackContent',
    
    # Analytics and Insights Models
    'PracticeInsight', 'LifeCorrelation', 'PredictiveInsight',
    
    # System Performance Models
    'PerformanceMetrics', 'QualityAssurance',
    
    # Integration Models
    'SystemIntegration'
]