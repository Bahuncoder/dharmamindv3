"""
🧘 Dhyana Module - Sacred Meditation and Contemplative Practice
Complete system for spiritual meditation and inner contemplation
Based on Patanjali's Yoga Sutras and traditional dhyana practices
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DhyanaLevel(Enum):
    """Levels of meditation practice"""
    BEGINNER = "beginner"               # Starting meditation practice
    DEVELOPING = "developing"           # Building consistency
    FOCUSED = "focused"                 # Achieving sustained concentration
    ABSORBED = "absorbed"               # Experiencing absorption states
    ESTABLISHED = "established"         # Stable meditation practice
    MASTERY = "mastery"                 # Advanced meditation mastery


class MeditationType(Enum):
    """Types of meditation practice"""
    MINDFULNESS = "mindfulness"         # Vipassana awareness meditation
    CONCENTRATION = "concentration"     # Shamatha focused meditation
    MANTRA = "mantra"                   # Sacred sound meditation
    BREATH = "breath"                   # Pranayama breathing meditation
    MOVEMENT = "movement"               # Walking or moving meditation
    CONTEMPLATION = "contemplation"     # Philosophical inquiry
    DEVOTIONAL = "devotional"           # Bhakti loving meditation
    EMPTINESS = "emptiness"             # Void or formless meditation


class MeditationObject(Enum):
    """Objects of meditation focus"""
    BREATH = "breath"                   # Natural breathing
    MANTRA = "mantra"                   # Sacred syllables
    YANTRA = "yantra"                   # Sacred geometry
    DEITY = "deity"                     # Divine form
    CHAKRA = "chakra"                   # Energy centers
    VOID = "void"                       # Formless awareness
    NATURE = "nature"                   # Natural phenomena
    LIGHT = "light"                     # Inner or outer light


class DhyanaObstacle(Enum):
    """Common meditation obstacles"""
    RESTLESSNESS = "restlessness"       # Agitated mind
    DULLNESS = "dullness"              # Mental sluggishness
    DOUBT = "doubt"                    # Questioning practice
    ATTACHMENT = "attachment"          # Clinging to experiences
    AVERSION = "aversion"              # Resistance to discomfort
    COMPARISON = "comparison"          # Judging progress
    EXPECTATION = "expectation"        # Demanding results
    DISTRACTION = "distraction"        # External interruptions


class Samadhi(Enum):
    """States of absorption in meditation"""
    SAVITARKA = "savitarka"            # With gross thought
    NIRVITARKA = "nirvitarka"          # Without gross thought
    SAVICHARA = "savichara"            # With subtle thought
    NIRVICHARA = "nirvichara"          # Without subtle thought
    SANANDA = "sananda"                # Blissful absorption
    SASMITA = "sasmita"                # With sense of I-ness
    NIRBIJA = "nirbija"                # Seedless absorption


@dataclass
class DhyanaGuidance:
    """Comprehensive meditation guidance"""
    level: DhyanaLevel
    primary_teaching: str
    meditation_practices: List[str]
    concentration_methods: List[str]
    daily_integration: List[str]
    obstacle_solutions: Dict[str, str]
    progress_indicators: List[str]
    recommended_duration: str
    preparation_methods: List[str]
    deepening_techniques: List[str]


@dataclass
class MeditationSession:
    """Structure for meditation session"""
    meditation_type: MeditationType
    focus_object: MeditationObject
    duration_minutes: int
    preparation_steps: List[str]
    main_practice: str
    integration_steps: List[str]
    common_experiences: List[str]


class DhyanaResponse(BaseModel):
    """Response from Dhyana module"""
    dhyana_level: str = Field(description="Current meditation practice level")
    meditation_guidance: str = Field(description="Core meditation teaching")
    practice_methods: List[str] = Field(description="Specific meditation practices")
    concentration_training: List[str] = Field(description="Concentration development")
    daily_integration: List[str] = Field(description="Daily meditation integration")
    obstacle_solutions: Dict[str, str] = Field(description="Solutions for obstacles")
    progress_indicators: List[str] = Field(description="Signs of meditation progress")
    session_structure: Dict[str, Any] = Field(description="Meditation session format")
    deepening_practices: List[str] = Field(description="Advanced practices")
    scriptural_wisdom: str = Field(description="Traditional meditation teachings")


class DhyanaModule:
    """
    🧘 Dhyana Module - Sacred Meditation and Contemplative Practice
    
    Based on traditional meditation teachings:
    - Patanjali's Yoga Sutras on Dhyana and Samadhi
    - Vedantic meditation instructions
    - Kashmir Shaivism contemplative practices
    - Advaita Vedanta inquiry meditation
    
    Dhyana is the seventh limb of Ashtanga Yoga, representing sustained
    meditation that leads to Samadhi (absorption/enlightenment).
    """
    
    def __init__(self):
        self.name = "Dhyana"
        self.color = "🧘"
        self.element = "Consciousness"
        self.principles = ["Sustained Focus", "Present Awareness", "Inner Stillness", "Unity"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.meditation_sessions = self._initialize_meditation_sessions()
        self.obstacle_solutions = self._initialize_obstacle_solutions()
        
    def _initialize_guidance_levels(self) -> Dict[DhyanaLevel, DhyanaGuidance]:
        """Initialize guidance for different levels of meditation practice"""
        return {
            DhyanaLevel.BEGINNER: DhyanaGuidance(
                level=DhyanaLevel.BEGINNER,
                primary_teaching="Begin simply with awareness of breath. Meditation is not about stopping thoughts but developing a peaceful relationship with them. Consistency matters more than duration.",
                meditation_practices=[
                    "5-10 minute daily breath awareness",
                    "Simple mindfulness of body sensations",
                    "Basic mantra repetition (Om or So Hum)",
                    "Walking meditation in nature",
                    "Loving-kindness meditation for 5 minutes"
                ],
                concentration_methods=[
                    "Count breaths from 1 to 10, then repeat",
                    "Focus on the sensation of breath at nostrils",
                    "Use a simple word or phrase as anchor",
                    "Practice single-pointed attention for short periods"
                ],
                daily_integration=[
                    "Set specific daily meditation time",
                    "Create quiet, dedicated meditation space",
                    "Practice mindful breathing during daily activities",
                    "Use meditation apps or guided sessions initially",
                    "Keep meditation journal to track experience"
                ],
                obstacle_solutions={
                    "monkey_mind": "Gently return to breath each time mind wanders - this is the practice",
                    "physical_discomfort": "Adjust posture as needed, comfort supports concentration",
                    "impatience": "Start with very short sessions and gradually increase duration"
                },
                progress_indicators=[
                    "Sitting still for chosen duration becomes easier",
                    "Noticing mind wandering more quickly",
                    "Experiencing moments of calm and stillness",
                    "Feeling more centered throughout the day"
                ],
                recommended_duration="5-15 minutes daily",
                preparation_methods=[
                    "Choose consistent time and place",
                    "Ensure comfortable sitting position",
                    "Set gentle timer to avoid clock-watching",
                    "Take few deep breaths to settle in"
                ],
                deepening_techniques=[
                    "Gradually extend sitting time",
                    "Experiment with different meditation objects",
                    "Join meditation group or find teacher",
                    "Read authentic meditation instructions"
                ]
            ),
            
            DhyanaLevel.DEVELOPING: DhyanaGuidance(
                level=DhyanaLevel.DEVELOPING,
                primary_teaching="Develop consistency and patience. Like training a puppy, gently but firmly guide attention back to chosen object. Obstacles are part of the path, not failures.",
                meditation_practices=[
                    "15-25 minute daily sessions",
                    "Alternate between concentration and mindfulness",
                    "Practice with different meditation objects",
                    "Include body scanning meditation",
                    "Experiment with mantra and breath combination"
                ],
                concentration_methods=[
                    "Develop sustained attention on single object",
                    "Practice one-pointed focus for longer periods",
                    "Use visualization techniques",
                    "Work with subtle breath sensations"
                ],
                daily_integration=[
                    "Maintain consistent daily practice schedule",
                    "Apply mindfulness to routine activities",
                    "Practice mini-meditations throughout day",
                    "Study meditation texts for deeper understanding",
                    "Connect with meditation community or teacher"
                ],
                obstacle_solutions={
                    "inconsistency": "Lower the bar - better to sit for 5 minutes daily than 30 minutes occasionally",
                    "drowsiness": "Sit more upright, ensure adequate sleep, meditate when naturally alert",
                    "emotional_turbulence": "Allow emotions to be present without fighting or feeding them"
                },
                progress_indicators=[
                    "Natural settling into meditation posture",
                    "Periods of sustained concentration",
                    "Greater equanimity with difficult emotions",
                    "Increased awareness of mental patterns"
                ],
                recommended_duration="15-30 minutes daily",
                preparation_methods=[
                    "Prepare mind with preliminary practices",
                    "Use pranayama to settle nervous system",
                    "Set clear intention for session",
                    "Minimize external disturbances"
                ],
                deepening_techniques=[
                    "Work with specific meditation lineage",
                    "Attend meditation retreats",
                    "Practice in early morning hours",
                    "Combine study with practical experience"
                ]
            ),
            
            DhyanaLevel.FOCUSED: DhyanaGuidance(
                level=DhyanaLevel.FOCUSED,
                primary_teaching="Sustained concentration brings natural joy and peace. Let meditation deepen naturally without forcing. The mind becomes like a calm lake reflecting the sky.",
                meditation_practices=[
                    "30-45 minute daily sessions",
                    "Deep concentration on chosen object",
                    "Contemplative inquiry practices",
                    "Advanced pranayama with meditation",
                    "Silent sitting in natural absorption"
                ],
                concentration_methods=[
                    "Single-pointed absorption (ekagrata)",
                    "Subtle object meditation",
                    "Energy center (chakra) concentration",
                    "Light meditation and visualization"
                ],
                daily_integration=[
                    "Live with meditative awareness throughout day",
                    "Practice continuous mindfulness",
                    "Integrate wisdom insights into daily life",
                    "Share practice with others appropriately",
                    "Study advanced meditation texts"
                ],
                obstacle_solutions={
                    "spiritual_pride": "Remember meditation is about dissolving ego, not enhancing it",
                    "attachment_to_states": "Let experiences come and go without clinging",
                    "comparison_with_others": "Your path is unique, honor your individual process"
                },
                progress_indicators=[
                    "Natural absorption states arise",
                    "Effortless sustained concentration",
                    "Deep peace and inner joy",
                    "Spontaneous wisdom insights"
                ],
                recommended_duration="30-60 minutes daily",
                preparation_methods=[
                    "Purification practices before sitting",
                    "Invoke blessings of lineage and teachers",
                    "Create sacred atmosphere",
                    "Align with natural rhythms"
                ],
                deepening_techniques=[
                    "Practice advanced samadhi techniques",
                    "Work with formless meditation",
                    "Undertake intensive retreat practice",
                    "Receive guidance from realized teacher"
                ]
            ),
            
            DhyanaLevel.ABSORBED: DhyanaGuidance(
                level=DhyanaLevel.ABSORBED,
                primary_teaching="Meditation happens by itself. The meditator, process of meditation, and object merge into one unified awareness. Rest in natural state without manipulation.",
                meditation_practices=[
                    "Extended periods of natural sitting",
                    "Formless awareness meditation",
                    "Contemplation of ultimate nature",
                    "Spontaneous samadhi states",
                    "Integration of realization with activity"
                ],
                concentration_methods=[
                    "Effortless awareness",
                    "Recognition of natural state",
                    "Absorption in pure consciousness",
                    "Unity of subject and object"
                ],
                daily_integration=[
                    "Continuous recognition of true nature",
                    "Seamless integration of realization",
                    "Spontaneous compassionate activity",
                    "Living from place of inner freedom",
                    "Serving others' awakening naturally"
                ],
                obstacle_solutions={
                    "subtle_ego": "Even spiritual attainment can become new form of identity - transcend this too",
                    "isolation": "Balance deep inner work with appropriate engagement with world",
                    "responsibility": "Use attainments for benefit of all beings, not personal gain"
                },
                progress_indicators=[
                    "Natural spontaneous meditation",
                    "Effortless presence and awareness",
                    "Deep compassion and wisdom",
                    "Freedom from meditation techniques"
                ],
                recommended_duration="As naturally arises",
                preparation_methods=[
                    "Recognition that preparation is meditation",
                    "Resting in natural awareness",
                    "Trusting innate wisdom",
                    "Opening to what is present"
                ],
                deepening_techniques=[
                    "Post-meditation integration",
                    "Service as spiritual practice",
                    "Teaching and sharing wisdom",
                    "Continuous self-inquiry"
                ]
            ),
            
            DhyanaLevel.ESTABLISHED: DhyanaGuidance(
                level=DhyanaLevel.ESTABLISHED,
                primary_teaching="Meditation and life become one seamless flow. Awareness remains undisturbed whether in sitting practice or worldly activity. This is true dhyana.",
                meditation_practices=[
                    "Natural meditation in all circumstances",
                    "Teaching and guiding others",
                    "Silent transmission of presence",
                    "Integration with all life activities",
                    "Spontaneous sacred moments"
                ],
                concentration_methods=[
                    "Concentration without effort",
                    "Natural samadhi in activity",
                    "Unwavering awareness",
                    "Unity consciousness"
                ],
                daily_integration=[
                    "Every moment is meditation",
                    "Continuous presence and awareness",
                    "Serving spiritual evolution of humanity",
                    "Living example of realized being",
                    "Spontaneous wisdom and compassion"
                ],
                obstacle_solutions={
                    "complacency": "Continue deepening even in established state",
                    "responsibility": "Use realization to serve collective awakening",
                    "humility": "Remain student even while teaching others"
                },
                progress_indicators=[
                    "No difference between meditation and non-meditation",
                    "Effortless presence in all circumstances",
                    "Natural wisdom and compassion",
                    "Serving others' spiritual growth"
                ],
                recommended_duration="Continuous awareness",
                preparation_methods=[
                    "No special preparation needed",
                    "Presence is always available",
                    "Resting as awareness itself",
                    "Natural recognition of true nature"
                ],
                deepening_techniques=[
                    "Selfless service as highest practice",
                    "Transmission through presence",
                    "Continuous surrender and humility",
                    "Living as embodiment of truth"
                ]
            ),
            
            DhyanaLevel.MASTERY: DhyanaGuidance(
                level=DhyanaLevel.MASTERY,
                primary_teaching="The master has become meditation itself. There is no one who meditates, no act of meditation, only pure awareness expressing through form. This is the ultimate dhyana.",
                meditation_practices=[
                    "Meditation without meditator",
                    "Spontaneous emanation of wisdom",
                    "Silent blessing of all beings",
                    "Natural transmission of awakening",
                    "Being as meditation"
                ],
                concentration_methods=[
                    "Concentration is natural state",
                    "Effortless focus",
                    "Unity awareness",
                    "Beyond technique"
                ],
                daily_integration=[
                    "Life as continuous meditation",
                    "Spontaneous wisdom activity",
                    "Natural blessing and healing presence",
                    "Embodiment of divine consciousness",
                    "Serving universal awakening"
                ],
                obstacle_solutions={
                    "final_transcendence": "Even mastery must be transcended to serve perfectly",
                    "ultimate_service": "Complete surrender to divine will and cosmic purpose",
                    "beyond_personal": "No personal achievement, only universal awakening"
                },
                progress_indicators=[
                    "Meditation happens through you",
                    "Natural emanation of peace and wisdom",
                    "Effortless service to all beings",
                    "Being itself as meditation"
                ],
                recommended_duration="Timeless presence",
                preparation_methods=[
                    "Continuous recognition of what is",
                    "Natural presence without doing",
                    "Being as preparation",
                    "Always already here"
                ],
                deepening_techniques=[
                    "Serving cosmic evolution",
                    "Being instrument of universal wisdom",
                    "Continuous surrender to greater purpose",
                    "Living as pure consciousness"
                ]
            )
        }
    
    def _initialize_meditation_sessions(self) -> Dict[MeditationType, MeditationSession]:
        """Initialize structured meditation sessions"""
        return {
            MeditationType.MINDFULNESS: MeditationSession(
                meditation_type=MeditationType.MINDFULNESS,
                focus_object=MeditationObject.BREATH,
                duration_minutes=20,
                preparation_steps=[
                    "Sit comfortably with spine naturally erect",
                    "Take three deep breaths to settle",
                    "Set intention for mindful awareness",
                    "Notice body sensations and mental state"
                ],
                main_practice="Observe breath naturally without controlling. When mind wanders, gently note 'thinking' and return to breath. Maintain open awareness of present moment.",
                integration_steps=[
                    "Sit quietly for a moment before opening eyes",
                    "Notice the quality of mind after practice",
                    "Set intention to carry mindfulness into day",
                    "Dedicate merit of practice to all beings"
                ],
                common_experiences=[
                    "Mind wandering and returning to breath",
                    "Periods of calm and clarity",
                    "Awareness of mental patterns",
                    "Growing stability of attention"
                ]
            ),
            
            MeditationType.CONCENTRATION: MeditationSession(
                meditation_type=MeditationType.CONCENTRATION,
                focus_object=MeditationObject.BREATH,
                duration_minutes=25,
                preparation_steps=[
                    "Create stable, comfortable meditation posture",
                    "Choose specific object of concentration",
                    "Set clear intention for single-pointed focus",
                    "Begin with few minutes of natural breathing"
                ],
                main_practice="Maintain unwavering attention on chosen object. When mind moves away, immediately return focus. Develop sustained concentration without strain.",
                integration_steps=[
                    "Maintain focus for final few moments",
                    "Slowly expand awareness to include environment",
                    "Appreciate the calm and clarity developed",
                    "Carry concentrated awareness into activities"
                ],
                common_experiences=[
                    "Development of sustained attention",
                    "Natural joy and calm arising",
                    "Decreased mental chatter",
                    "Periods of absorption"
                ]
            ),
            
            MeditationType.MANTRA: MeditationSession(
                meditation_type=MeditationType.MANTRA,
                focus_object=MeditationObject.MANTRA,
                duration_minutes=30,
                preparation_steps=[
                    "Sit facing east if possible",
                    "Choose sacred mantra with meaning",
                    "Use mala beads if helpful",
                    "Connect with tradition and lineage"
                ],
                main_practice="Repeat chosen mantra with devotion and concentration. Let sound vibration purify mind and heart. Merge with sacred syllables completely.",
                integration_steps=[
                    "Complete the mantra repetition mindfully",
                    "Sit in silence absorbing the vibrations",
                    "Offer gratitude to the tradition",
                    "Carry mantra awareness throughout day"
                ],
                common_experiences=[
                    "Calming effect of sacred sound",
                    "Heart opening and devotion",
                    "Mental purification",
                    "Connection with divine aspect"
                ]
            )
        }
    
    def _initialize_obstacle_solutions(self) -> Dict[DhyanaObstacle, Dict[str, Any]]:
        """Initialize solutions for common meditation obstacles"""
        return {
            DhyanaObstacle.RESTLESSNESS: {
                "description": "Agitated, scattered mental state during meditation",
                "solutions": [
                    "Begin with movement meditation or walking",
                    "Use longer exhales to activate parasympathetic system",
                    "Practice body scanning to ground awareness",
                    "Ensure adequate physical exercise during day",
                    "Address underlying stress or anxiety"
                ],
                "practices": ["Progressive muscle relaxation", "Slower breathing"],
                "wisdom": "Restlessness is wind that eventually settles into natural stillness"
            },
            
            DhyanaObstacle.DULLNESS: {
                "description": "Mental sluggishness, drowsiness, or lack of clarity",
                "solutions": [
                    "Sit more upright with stronger posture",
                    "Ensure adequate sleep and proper nutrition",
                    "Meditate during naturally alert times",
                    "Use brief periods of visualization",
                    "Practice in cooler environment with fresh air"
                ],
                "practices": ["Energizing breath", "Brief standing meditation"],
                "wisdom": "Dullness cleared by gentle alertness, like dawn dispelling darkness"
            },
            
            DhyanaObstacle.DOUBT: {
                "description": "Questioning practice, teacher, or own ability",
                "solutions": [
                    "Study authentic meditation texts",
                    "Connect with experienced practitioners",
                    "Start with very simple practices",
                    "Focus on immediate benefits rather than ultimate goals",
                    "Understand doubt as normal part of spiritual path"
                ],
                "practices": ["Faith-building contemplation", "Gradual practice"],
                "wisdom": "Doubt resolved through direct experience, not intellectual understanding"
            }
        }
    
    def assess_dhyana_level(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> DhyanaLevel:
        """Assess user's current meditation practice level"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for mastery level indicators
        if any(word in query_lower for word in ["master", "teaching meditation", "transmission", "no technique"]):
            return DhyanaLevel.MASTERY
        
        # Check for established level indicators
        if any(word in query_lower for word in ["established practice", "continuous awareness", "daily life meditation"]):
            return DhyanaLevel.ESTABLISHED
        
        # Check for absorbed level indicators
        if any(word in query_lower for word in ["samadhi", "absorption", "effortless", "formless"]):
            return DhyanaLevel.ABSORBED
        
        # Check for focused level indicators
        if any(word in query_lower for word in ["concentration", "sustained focus", "deep meditation"]):
            return DhyanaLevel.FOCUSED
        
        # Check for developing level indicators
        if any(word in query_lower for word in ["consistent practice", "obstacles", "deepening"]):
            return DhyanaLevel.DEVELOPING
        
        # Default to beginner
        return DhyanaLevel.BEGINNER
    
    def identify_meditation_obstacles(self, query: str, context: Dict[str, Any]) -> List[DhyanaObstacle]:
        """Identify meditation obstacles mentioned in query"""
        obstacles = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["restless", "agitated", "scattered", "hyperactive"]):
            obstacles.append(DhyanaObstacle.RESTLESSNESS)
        
        if any(word in query_lower for word in ["sleepy", "dull", "sluggish", "tired", "drowsy"]):
            obstacles.append(DhyanaObstacle.DULLNESS)
        
        if any(word in query_lower for word in ["doubt", "questioning", "not working", "unsure"]):
            obstacles.append(DhyanaObstacle.DOUBT)
        
        if any(word in query_lower for word in ["attachment", "clinging", "want to repeat", "chasing"]):
            obstacles.append(DhyanaObstacle.ATTACHMENT)
        
        if any(word in query_lower for word in ["aversion", "resistance", "don't like", "avoiding"]):
            obstacles.append(DhyanaObstacle.AVERSION)
        
        if any(word in query_lower for word in ["distracted", "noise", "interruption", "can't focus"]):
            obstacles.append(DhyanaObstacle.DISTRACTION)
        
        return obstacles if obstacles else [DhyanaObstacle.RESTLESSNESS]
    
    def get_obstacle_solutions(self, obstacles: List[DhyanaObstacle]) -> Dict[str, str]:
        """Get solutions for identified obstacles"""
        solutions = {}
        
        for obstacle in obstacles:
            obstacle_data = self.obstacle_solutions.get(obstacle, {})
            solutions[obstacle.value] = "; ".join(obstacle_data.get("solutions", ["Practice patience and gentle persistence"])[:2])
        
        return solutions
    
    def get_session_structure(self, meditation_type: MeditationType) -> Dict[str, Any]:
        """Get structured meditation session format"""
        session = self.meditation_sessions.get(meditation_type)
        if not session:
            return {}
        
        return {
            "type": session.meditation_type.value,
            "duration": f"{session.duration_minutes} minutes",
            "preparation": session.preparation_steps,
            "main_practice": session.main_practice,
            "integration": session.integration_steps,
            "common_experiences": session.common_experiences
        }
    
    def get_scriptural_wisdom(self, level: DhyanaLevel) -> str:
        """Get scriptural wisdom appropriate to meditation level"""
        wisdom_map = {
            DhyanaLevel.BEGINNER: "Bhagavad Gita 6.19: 'As a lamp in a windless place does not flicker, so the disciplined mind of a yogi remains steady in meditation.'",
            DhyanaLevel.DEVELOPING: "Yoga Sutras 1.14: 'Practice is firmly grounded when it is cultivated continuously for an extended period with dedication.'",
            DhyanaLevel.FOCUSED: "Yoga Sutras 3.2: 'Dhyana is the uninterrupted flow of consciousness toward the object of meditation.'",
            DhyanaLevel.ABSORBED: "Yoga Sutras 3.3: 'When only the essence of the object shines forth, as if devoid of form, that is samadhi.'",
            DhyanaLevel.ESTABLISHED: "Bhagavad Gita 2.48: 'Established in yoga, perform actions, abandoning attachment and remaining balanced in success and failure.'",
            DhyanaLevel.MASTERY: "Mandukya Upanishad: 'Turiya is not that which is conscious of the inner world, nor that which is conscious of the outer world... It is pure consciousness itself.'"
        }
        return wisdom_map.get(level, "Katha Upanishad: 'When the five senses are stilled, when the mind is at rest, when the intellect wavers not - that is called the highest state.'")
    
    async def process_dhyana_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> DhyanaResponse:
        """Process meditation-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess meditation aspects
            level = self.assess_dhyana_level(query, context)
            obstacles = self.identify_meditation_obstacles(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            obstacle_solutions = self.get_obstacle_solutions(obstacles)
            session_structure = self.get_session_structure(MeditationType.MINDFULNESS)
            scriptural_wisdom = self.get_scriptural_wisdom(level)
            
            return DhyanaResponse(
                dhyana_level=level.value,
                meditation_guidance=guidance.primary_teaching,
                practice_methods=guidance.meditation_practices,
                concentration_training=guidance.concentration_methods,
                daily_integration=guidance.daily_integration,
                obstacle_solutions=obstacle_solutions,
                progress_indicators=guidance.progress_indicators,
                session_structure=session_structure,
                deepening_practices=guidance.deepening_techniques,
                scriptural_wisdom=scriptural_wisdom
            )
            
        except Exception as e:
            logger.error(f"Error processing dhyana query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> DhyanaResponse:
        """Create fallback response when processing fails"""
        return DhyanaResponse(
            dhyana_level="beginner",
            meditation_guidance="Begin simply with awareness of breath. Meditation is not about stopping thoughts but developing a peaceful relationship with them. Consistency matters more than duration.",
            practice_methods=[
                "5-10 minute daily breath awareness",
                "Simple mindfulness of body sensations",
                "Basic mantra repetition (Om or So Hum)",
                "Walking meditation in nature"
            ],
            concentration_training=[
                "Count breaths from 1 to 10, then repeat",
                "Focus on the sensation of breath at nostrils",
                "Use a simple word or phrase as anchor",
                "Practice single-pointed attention for short periods"
            ],
            daily_integration=[
                "Set specific daily meditation time",
                "Create quiet, dedicated meditation space",
                "Practice mindful breathing during daily activities",
                "Keep meditation journal to track experience"
            ],
            obstacle_solutions={
                "restlessness": "Begin with movement meditation or walking",
                "dullness": "Sit more upright with stronger posture"
            },
            progress_indicators=[
                "Sitting still for chosen duration becomes easier",
                "Noticing mind wandering more quickly",
                "Experiencing moments of calm and stillness",
                "Feeling more centered throughout the day"
            ],
            session_structure={
                "type": "mindfulness",
                "duration": "20 minutes",
                "preparation": ["Sit comfortably", "Take three deep breaths"],
                "main_practice": "Observe breath naturally without controlling",
                "integration": ["Sit quietly before opening eyes"],
                "common_experiences": ["Mind wandering and returning to breath"]
            },
            deepening_practices=[
                "Gradually extend sitting time",
                "Experiment with different meditation objects",
                "Join meditation group or find teacher",
                "Read authentic meditation instructions"
            ],
            scriptural_wisdom="Katha Upanishad: 'When the five senses are stilled, when the mind is at rest, when the intellect wavers not - that is called the highest state.'"
        )


# Global instance
_dhyana_module = None

def get_dhyana_module() -> DhyanaModule:
    """Get global Dhyana module instance"""
    global _dhyana_module
    if _dhyana_module is None:
        _dhyana_module = DhyanaModule()
    return _dhyana_module

# Factory function for easy access
def create_dhyana_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> DhyanaResponse:
    """Factory function to create dhyana guidance"""
    import asyncio
    module = get_dhyana_module()
    return asyncio.run(module.process_dhyana_query(query, user_context))
