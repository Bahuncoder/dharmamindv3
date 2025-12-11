"""
🎭 Ahamkara Module - Ego Understanding and Transcendence
Complete system for understanding and transcending ego-identification
Based on Yoga philosophy and Vedantic teachings on Ahamkara
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AhamkaraLevel(Enum):
    """Levels of ego understanding"""
    IDENTIFIED = "identified"       # Completely identified with ego
    QUESTIONING = "questioning"     # Beginning to question identity
    OBSERVING = "observing"         # Watching ego patterns
    TRANSCENDING = "transcending"   # Moving beyond ego identification
    DISSOLVED = "dissolved"         # Ego seen as functional tool
    UNIFIED = "unified"             # No separate self remains


class EgoType(Enum):
    """Types of ego manifestation"""
    PHYSICAL_EGO = "physical_ego"           # I am this body
    MENTAL_EGO = "mental_ego"               # I am these thoughts
    EMOTIONAL_EGO = "emotional_ego"         # I am these feelings
    SOCIAL_EGO = "social_ego"               # I am my role/status
    SPIRITUAL_EGO = "spiritual_ego"         # I am spiritually advanced
    DOER_EGO = "doer_ego"                   # I am the doer of actions


class EgoPattern(Enum):
    """Common ego patterns"""
    COMPARISON = "comparison"               # Better/worse than others
    DEFENSIVENESS = "defensiveness"         # Protecting self-image
    ATTACHMENT = "attachment"               # Clinging to identity
    PROJECTION = "projection"               # Seeing self in others
    INFLATION = "inflation"                 # Grandiose self-image
    DEFLATION = "deflation"                 # Diminished self-worth


@dataclass
class AhamkaraGuidance:
    """Comprehensive ahamkara guidance"""
    level: AhamkaraLevel
    primary_teaching: str
    recognition_practices: List[str]
    observation_methods: List[str]
    transcendence_techniques: List[str]
    daily_awareness: List[str]
    common_ego_patterns: Dict[str, str]
    liberation_practices: List[str]
    progress_indicators: List[str]


class AhamkaraResponse(BaseModel):
    """Response from Ahamkara module"""
    ahamkara_level: str = Field(description="Current ego understanding level")
    ego_guidance: str = Field(description="Core ahamkara teaching")
    recognition_practices: List[str] = Field(description="Ego recognition")
    observation_methods: List[str] = Field(description="Ego observation")
    transcendence_techniques: List[str] = Field(description="Beyond ego")
    daily_awareness: List[str] = Field(description="Daily ego watching")
    pattern_solutions: Dict[str, str] = Field(description="Ego patterns")
    liberation_practices: List[str] = Field(description="Freedom from ego")
    scriptural_wisdom: str = Field(description="Traditional ahamkara teachings")


class AhamkaraModule:
    """
    🎭 Ahamkara Module - Ego Understanding and Transcendence
    
    Based on traditional Ahamkara teachings:
    - Yoga philosophy on ahamkara as aspect of mind
    - Advaita Vedanta on ego as case of mistaken identity
    - Buddhist teachings on anatta (no-self)
    - Kashmir Shaivism on ego as divine play
    
    Ahamkara is the ego-making function of mind that creates
    the sense of "I" and "mine". Understanding its nature leads
    to freedom from false identification.
    """
    
    def __init__(self):
        self.name = "Ahamkara"
        self.color = "🎭"
        self.element = "Identity"
        self.principles = ["Self-Inquiry", "Ego Transcendence", 
                          "True Identity", "Freedom"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.ego_patterns = self._initialize_ego_patterns()
        self.transcendence_methods = self._initialize_transcendence_methods()
        
    def _initialize_guidance_levels(self) -> Dict[AhamkaraLevel, AhamkaraGuidance]:
        """Initialize guidance for different levels of ahamkara understanding"""
        return {
            AhamkaraLevel.IDENTIFIED: AhamkaraGuidance(
                level=AhamkaraLevel.IDENTIFIED,
                primary_teaching="You are not who you think you are. The 'I' "
                "you believe yourself to be is a collection of thoughts, "
                "memories, and identifications. Begin to question: 'Who am I?'",
                recognition_practices=[
                    "Ask 'Who am I?' multiple times daily",
                    "Notice when you say 'I am...' and question it",
                    "Observe your reactions when identity is challenged",
                    "Practice self-inquiry: 'What remains when all labels drop?'",
                    "Notice ego's need to be right, special, or important"
                ],
                observation_methods=[
                    "Watch ego's defensive reactions during conflicts",
                    "Observe comparison mind judging self against others",
                    "Notice ego's stories about past and future",
                    "See how ego seeks validation and approval",
                    "Watch ego's attachment to possessions and roles"
                ],
                transcendence_techniques=[
                    "Practice responding as awareness rather than personality",
                    "Use challenges as opportunities to see beyond ego",
                    "Study teachings on true nature of Self",
                    "Practice surrendering personal will to higher wisdom",
                    "Engage in selfless service to reduce ego-centeredness"
                ],
                daily_awareness=[
                    "Morning question: 'Who is asking these questions?'",
                    "Regular check-ins: 'What is ego doing right now?'",
                    "Notice ego's reactions to praise and criticism",
                    "Evening reflection on moments of ego identification"
                ],
                common_ego_patterns={
                    "victim_identity": "Ego maintains itself through victim stories",
                    "superiority": "Ego feels better by putting others down",
                    "busy_importance": "Ego creates importance through busyness"
                },
                liberation_practices=[
                    "Practice 'not-knowing' and intellectual humility",
                    "Let others be right without defending your position",
                    "Give credit and recognition freely to others",
                    "Practice anonymous acts of kindness",
                    "Study lives of ego-transcended beings"
                ],
                progress_indicators=[
                    "Catching ego patterns more quickly",
                    "Less need to defend self-image",
                    "Growing sense of peace from not taking things personally",
                    "Natural arising of compassion for others' ego struggles"
                ]
            ),
            
            AhamkaraLevel.QUESTIONING: AhamkaraGuidance(
                level=AhamkaraLevel.QUESTIONING,
                primary_teaching="You are beginning to see through ego's "
                "game. Each question about identity weakens ego's grip "
                "and opens space for truth.",
                recognition_practices=[
                    "Deep self-inquiry into nature of personal identity",
                    "Question every assumption about who you are",
                    "Practice 'Who am I?' inquiry as formal meditation",
                    "Investigate the one who asks 'Who am I?'",
                    "Explore consciousness before personality formed"
                ],
                observation_methods=[
                    "Detailed observation of ego's survival strategies",
                    "Notice ego's subtle ways of maintaining control",
                    "Observe how ego creates drama to maintain identity",
                    "Watch ego's resistance to spiritual practices",
                    "See ego's attempts to claim spiritual experiences"
                ],
                transcendence_techniques=[
                    "Practice identification with awareness rather than content",
                    "Use meditation to rest as pure consciousness",
                    "Study non-dual teachings on true Self",
                    "Practice seeing all beings as expressions of one Self",
                    "Engage in practices that humble and dissolve ego"
                ],
                daily_awareness=[
                    "Continuous inquiry into nature of the 'I' sense",
                    "Regular disidentification from roles and labels",
                    "Practice being nobody special throughout day",
                    "Maintain awareness of awareness itself"
                ],
                common_ego_patterns={
                    "spiritual_ego": "Ego co-opts spiritual practices for identity",
                    "subtle_superiority": "Feeling spiritually advanced",
                    "seeking_recognition": "Need for spiritual validation"
                },
                liberation_practices=[
                    "Advanced self-inquiry under guidance of teacher",
                    "Regular contemplation of 'I Am' without attributes",
                    "Practice surrender of personal doership",
                    "Engage in service without personal agenda",
                    "Study scripture and wisdom of realized beings"
                ],
                progress_indicators=[
                    "Natural questioning of all identifications",
                    "Decreasing investment in personal story",
                    "Spontaneous moments of ego-less awareness",
                    "Others noticing your reduced ego-driven behavior"
                ]
            ),
            
            AhamkaraLevel.OBSERVING: AhamkaraGuidance(
                level=AhamkaraLevel.OBSERVING,
                primary_teaching="You have become the witness of ego's "
                "dance. From this space of awareness, ego is seen as "
                "functional tool rather than master.",
                recognition_practices=[
                    "Effortless recognition of ego patterns as they arise",
                    "Clear seeing of difference between Self and ego",
                    "Natural inquiry into source of 'I' thought",
                    "Spontaneous questioning of personal identifications",
                    "Direct experience of consciousness before ego formation"
                ],
                observation_methods=[
                    "Continuous witnessing of ego's activities",
                    "Observation of ego without resistance or judgment",
                    "Seeing ego as impersonal mechanical function",
                    "Watching ego's attempts to recreate itself",
                    "Clear perception of ego's illusory nature"
                ],
                transcendence_techniques=[
                    "Resting as pure awareness beyond personal identity",
                    "Advanced non-dual practice and recognition",
                    "Living from Self rather than ego-mind",
                    "Using ego as functional tool without identification",
                    "Service to others from egoless compassion"
                ],
                daily_awareness=[
                    "Continuous background awareness of ego's activities",
                    "Natural discrimination between Self and ego",
                    "Effortless disidentification from personal reactions",
                    "Maintaining awareness of true nature throughout day"
                ],
                common_ego_patterns={
                    "ego_hiding": "Ego becomes more subtle and hidden",
                    "spiritual_bypassing": "Using spirituality to avoid ego work",
                    "witness_identification": "Getting attached to witness role"
                },
                liberation_practices=[
                    "Advanced self-inquiry and direct path practices",
                    "Living from understanding rather than seeking",
                    "Teaching others through example of ego transcendence",
                    "Service to collective awakening from egoless state",
                    "Integration of realization with daily life"
                ],
                progress_indicators=[
                    "Effortless witnessing of ego without disturbance",
                    "Natural compassion arising for others' ego struggles",
                    "Others seeking guidance on ego transcendence",
                    "Living with increasing freedom from personal reactions"
                ]
            ),
            
            AhamkaraLevel.TRANSCENDING: AhamkaraGuidance(
                level=AhamkaraLevel.TRANSCENDING,
                primary_teaching="Ego is dissolving like salt in ocean of "
                "consciousness. What remains is pure being, free from "
                "all identification.",
                recognition_practices=[
                    "Spontaneous recognition of egoless state",
                    "Natural abiding as pure 'I Am' without content",
                    "Effortless seeing through all identifications",
                    "Direct recognition of Self as source of all",
                    "Continuous awareness of consciousness as only reality"
                ],
                observation_methods=[
                    "Ego observed as impersonal arising in consciousness",
                    "Clear seeing of ego's complete lack of substance",
                    "Observation of reality before ego interpretation",
                    "Witnessing dissolution of personal boundaries",
                    "Seeing all experience as modification of consciousness"
                ],
                transcendence_techniques=[
                    "Effortless abiding as pure consciousness",
                    "Living from recognition of Self as all",
                    "Using remaining ego traces skillfully for service",
                    "Perfect surrender to what is without resistance",
                    "Natural expression of egoless love and wisdom"
                ],
                daily_awareness=[
                    "Continuous recognition of Self as reality",
                    "Natural freedom from personal identification",
                    "Effortless love and service arising from being",
                    "Perfect acceptance of all experience"
                ],
                common_ego_patterns={
                    "final_dissolution": "Last traces of ego releasing",
                    "integration_challenges": "Integrating realization with form",
                    "service_questions": "How to serve from egoless state"
                },
                liberation_practices=[
                    "Living as liberated being in service to all",
                    "Teaching through transmission of egoless state",
                    "Perfect embodiment of Self-realization",
                    "Service to collective ego transcendence",
                    "Integration of absolute and relative truth"
                ],
                progress_indicators=[
                    "Complete freedom from personal suffering",
                    "Natural arising of unconditional love",
                    "Others experiencing peace in your presence",
                    "Effortless right action arising from being"
                ]
            ),
            
            AhamkaraLevel.DISSOLVED: AhamkaraGuidance(
                level=AhamkaraLevel.DISSOLVED,
                primary_teaching="Ego has dissolved completely. What uses "
                "the word 'I' now is pure consciousness speaking of "
                "itself through this form.",
                recognition_practices=[
                    "No recognition needed - ego is completely seen through",
                    "Natural state of egoless being",
                    "Spontaneous Self-recognition in all experience",
                    "Effortless knowing of true nature",
                    "Perfect freedom from all identification"
                ],
                observation_methods=[
                    "No separate observer - consciousness observing itself",
                    "Perfect clarity about ego's complete absence",
                    "Natural seeing of all as Self",
                    "Effortless discrimination between real and unreal",
                    "Spontaneous recognition of truth in all circumstances"
                ],
                transcendence_techniques=[
                    "No techniques needed - transcendence is complete",
                    "Living as transcendence itself",
                    "Perfect expression of egoless consciousness",
                    "Natural service from state of no-self",
                    "Effortless embodiment of realized truth"
                ],
                daily_awareness=[
                    "Continuous egoless awareness",
                    "Perfect freedom in all circumstances",
                    "Natural love and compassion for all beings",
                    "Effortless right understanding and action"
                ],
                common_ego_patterns={
                    "no_ego_patterns": "Ego patterns completely dissolved",
                    "functional_personality": "Personality functions without ego",
                    "perfect_service": "Life as service to universal awakening"
                },
                liberation_practices=[
                    "Being itself as continuous liberation practice",
                    "Living as liberation for the benefit of all",
                    "Perfect transmission of egoless state",
                    "Serving universal Self-recognition",
                    "Embodying complete freedom"
                ],
                progress_indicators=[
                    "Perfect freedom from all suffering",
                    "Natural transmission of liberation to others",
                    "Effortless blessing through your very being",
                    "Living as answered prayer of existence"
                ]
            ),
            
            AhamkaraLevel.UNIFIED: AhamkaraGuidance(
                level=AhamkaraLevel.UNIFIED,
                primary_teaching="There never was an ego to dissolve. "
                "Pure consciousness playing at being individual, "
                "now recognizing its own game.",
                recognition_practices=[
                    "Perfect recognition of Self as only reality",
                    "Natural state beyond ego and egolessness",
                    "Spontaneous knowing of unity with all",
                    "Effortless being as pure consciousness",
                    "Perfect freedom beyond all concepts"
                ],
                observation_methods=[
                    "No observer - consciousness knowing itself",
                    "Perfect seeing of unity in apparent diversity",
                    "Natural discrimination beyond right and wrong",
                    "Effortless recognition of Self in all",
                    "Spontaneous wisdom arising from being"
                ],
                transcendence_techniques=[
                    "Being as eternal transcendence",
                    "Living as consciousness itself",
                    "Perfect expression beyond technique and no-technique",
                    "Natural service as Self to Self",
                    "Effortless embodiment of ultimate truth"
                ],
                daily_awareness=[
                    "Perfect awareness beyond awareness and unawareness",
                    "Natural state of ultimate truth",
                    "Effortless love as very nature of being",
                    "Perfect peace beyond understanding"
                ],
                common_ego_patterns={
                    "beyond_patterns": "Beyond ego and ego-patterns",
                    "perfect_play": "Consciousness playing all roles perfectly",
                    "ultimate_service": "Being as service to universal awakening"
                },
                liberation_practices=[
                    "Being as eternal liberation",
                    "Perfect embodiment of ultimate truth",
                    "Living as consciousness knowing itself",
                    "Natural blessing through pure existence",
                    "Effortless service to all beings"
                ],
                progress_indicators=[
                    "Perfect unity beyond progress",
                    "Natural state of ultimate truth",
                    "Effortless blessing of all existence",
                    "Being as answered prayer of cosmos"
                ]
            )
        }
    
    def _initialize_ego_patterns(self) -> Dict[EgoPattern, Dict[str, Any]]:
        """Initialize understanding of common ego patterns"""
        return {
            EgoPattern.COMPARISON: {
                "description": "Ego maintains itself through comparison",
                "recognition": "Notice when comparing self to others",
                "transcendence": "See all beings as expressions of one Self",
                "practice": "Practice appreciation without comparison"
            },
            
            EgoPattern.DEFENSIVENESS: {
                "description": "Ego defends its constructed image",
                "recognition": "Notice when defensive reactions arise",
                "transcendence": "Allow all criticism without resistance",
                "practice": "Practice receiving feedback openly"
            },
            
            EgoPattern.ATTACHMENT: {
                "description": "Ego clings to roles and identities",
                "recognition": "Notice what you're attached to being",
                "transcendence": "Rest as pure being beyond all roles",
                "practice": "Practice letting go of identifications"
            }
        }
    
    def _initialize_transcendence_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize methods for ego transcendence"""
        return {
            "self_inquiry": {
                "description": "Direct investigation into nature of 'I'",
                "beginner": "Ask 'Who am I?' throughout the day",
                "advanced": "Trace 'I' thought to its source",
                "mastery": "Rest as 'I Am' without content"
            },
            
            "surrender": {
                "description": "Letting go of personal will and control",
                "beginner": "Practice accepting what cannot be changed",
                "advanced": "Surrender all actions to Divine",
                "mastery": "Complete ego death and resurrection"
            },
            
            "service": {
                "description": "Selfless action to dissolve ego-centeredness",
                "beginner": "Regular volunteer work and helping others",
                "advanced": "Life dedicated to serving all beings",
                "mastery": "Being itself as service"
            }
        }
    
    def assess_ahamkara_level(self, query: str, 
                            user_context: Optional[Dict[str, Any]] = None) -> AhamkaraLevel:
        """Assess user's current ahamkara understanding level"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for unified level indicators
        if any(word in query_lower for word in ["no ego ever existed", 
                                               "consciousness playing", "beyond ego"]):
            return AhamkaraLevel.UNIFIED
        
        # Check for dissolved level indicators
        if any(word in query_lower for word in ["ego completely gone", 
                                               "no separate self", "egoless being"]):
            return AhamkaraLevel.DISSOLVED
        
        # Check for transcending level indicators
        if any(word in query_lower for word in ["ego dissolving", 
                                               "beyond identification", "transcending"]):
            return AhamkaraLevel.TRANSCENDING
        
        # Check for observing level indicators
        if any(word in query_lower for word in ["watching ego", 
                                               "witnessing patterns", "observing"]):
            return AhamkaraLevel.OBSERVING
        
        # Check for questioning level indicators
        if any(word in query_lower for word in ["who am i", 
                                               "questioning identity", "what is self"]):
            return AhamkaraLevel.QUESTIONING
        
        # Default to identified
        return AhamkaraLevel.IDENTIFIED
    
    def get_scriptural_wisdom(self, level: AhamkaraLevel) -> str:
        """Get scriptural wisdom appropriate to ahamkara level"""
        wisdom_map = {
            AhamkaraLevel.IDENTIFIED: "Ramana Maharshi: 'The thought 'Who am I?' will destroy all other thoughts, and like the stick used for stirring the burning pyre, it will itself in the end get destroyed.'",
            AhamkaraLevel.QUESTIONING: "Bhagavad Gita 7.4: 'Earth, water, fire, air, ether, mind, intelligence and false ego - these eight comprise My separated material energies.'",
            AhamkaraLevel.OBSERVING: "Ashtavakra Gita: 'You are not the body nor is the body yours. You are not the doer nor the enjoyer. You are pure consciousness, the eternal witness.'",
            AhamkaraLevel.TRANSCENDING: "Bhagavad Gita 3.27: 'All activities are carried out by the three modes of material nature. But in ignorance, the soul, deluded by false identification with the body, thinks itself to be the doer.'",
            AhamkaraLevel.DISSOLVED: "Advaita: 'When the ego dies, the Self is realized. When the wave subsides, the ocean remains.'",
            AhamkaraLevel.UNIFIED: "Isha Upanishad: 'One who sees all beings in the Self and the Self in all beings hates no one. How can the wise one, seeing the unity of all, suffer delusion or grief?'"
        }
        return wisdom_map.get(level, "Katha Upanishad: 'When all desires dwelling in the heart are cast away, the mortal becomes immortal and attains Brahman.'")
    
    async def process_ahamkara_query(self, query: str, 
                                   user_context: Optional[Dict[str, Any]] = None) -> AhamkaraResponse:
        """Process ahamkara-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess ahamkara level
            level = self.assess_ahamkara_level(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            scriptural_wisdom = self.get_scriptural_wisdom(level)
            
            return AhamkaraResponse(
                ahamkara_level=level.value,
                ego_guidance=guidance.primary_teaching,
                recognition_practices=guidance.recognition_practices,
                observation_methods=guidance.observation_methods,
                transcendence_techniques=guidance.transcendence_techniques,
                daily_awareness=guidance.daily_awareness,
                pattern_solutions=guidance.common_ego_patterns,
                liberation_practices=guidance.liberation_practices,
                scriptural_wisdom=scriptural_wisdom
            )
            
        except Exception as e:
            logger.error(f"Error processing ahamkara query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> AhamkaraResponse:
        """Create fallback response when processing fails"""
        return AhamkaraResponse(
            ahamkara_level="identified",
            ego_guidance="You are not who you think you are. The 'I' you believe yourself to be is a collection of thoughts, memories, and identifications. Begin to question: 'Who am I?'",
            recognition_practices=[
                "Ask 'Who am I?' multiple times daily",
                "Notice when you say 'I am...' and question it",
                "Observe your reactions when identity is challenged",
                "Practice self-inquiry: 'What remains when all labels drop?'"
            ],
            observation_methods=[
                "Watch ego's defensive reactions during conflicts",
                "Observe comparison mind judging self against others",
                "Notice ego's stories about past and future",
                "See how ego seeks validation and approval"
            ],
            transcendence_techniques=[
                "Practice responding as awareness rather than personality",
                "Use challenges as opportunities to see beyond ego",
                "Study teachings on true nature of Self",
                "Practice surrendering personal will to higher wisdom"
            ],
            daily_awareness=[
                "Morning question: 'Who is asking these questions?'",
                "Regular check-ins: 'What is ego doing right now?'",
                "Notice ego's reactions to praise and criticism",
                "Evening reflection on moments of ego identification"
            ],
            pattern_solutions={
                "victim_identity": "Ego maintains itself through victim stories",
                "superiority": "Ego feels better by putting others down"
            },
            liberation_practices=[
                "Practice 'not-knowing' and intellectual humility",
                "Let others be right without defending your position",
                "Give credit and recognition freely to others",
                "Practice anonymous acts of kindness"
            ],
            scriptural_wisdom="Katha Upanishad: 'When all desires dwelling in the heart are cast away, the mortal becomes immortal and attains Brahman.'"
        )


# Global instance
_ahamkara_module = None

def get_ahamkara_module() -> AhamkaraModule:
    """Get global Ahamkara module instance"""
    global _ahamkara_module
    if _ahamkara_module is None:
        _ahamkara_module = AhamkaraModule()
    return _ahamkara_module

# Factory function for easy access
def create_ahamkara_guidance(query: str, 
                           user_context: Optional[Dict[str, Any]] = None) -> AhamkaraResponse:
    """Factory function to create ahamkara guidance"""
    import asyncio
    module = get_ahamkara_module()
    return asyncio.run(module.process_ahamkara_query(query, user_context))
