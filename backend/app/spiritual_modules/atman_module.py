"""
Atman Module - Soul Awareness and Self-Realization System
=========================================================

Provides guidance on recognizing the true Self beyond body, mind, and ego.
Based on Upanishads, Vedanta, and self-inquiry teachings for soul realization.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class SoulAwarenessLevel(Enum):
    """Levels of soul awareness"""
    BODY_IDENTIFICATION = "body_identification"
    MIND_IDENTIFICATION = "mind_identification"
    SOUL_GLIMPSE = "soul_glimpse"
    SOUL_AWARENESS = "soul_awareness"
    ATMA_BRAHMA_UNITY = "atma_brahma_unity"

class EgoObstacle(Enum):
    """Common obstacles to soul realization"""
    BODY_ATTACHMENT = "body_attachment"
    MENTAL_IDENTIFICATION = "mental_identification"
    DESIRE_SEEKING = "desire_seeking"
    FEAR_OF_DISSOLUTION = "fear_of_dissolution"
    INTELLECTUAL_UNDERSTANDING = "intellectual_understanding"
    SPIRITUAL_EGO = "spiritual_ego"

@dataclass
class AtmanInsight:
    """Insight about soul awareness"""
    level: SoulAwarenessLevel
    teaching: str
    practice: str
    scripture_reference: str
    spiritual_guidance: str
    meditation: str

@dataclass
class SelfInquiryPractice:
    """Self-inquiry practice guidance"""
    question: str
    method: str
    duration: str
    signs_of_progress: List[str]
    common_distractions: List[str]
    deepening_techniques: List[str]

class AtmanModule:
    """
    Atman Module - Soul Awareness and Self-Realization System
    
    Helps users recognize their true Self beyond body, mind, and ego.
    Based on Upanishads, Vedanta, and self-inquiry teachings.
    Provides practices for direct experience of the soul (Atman).
    """
    
    def __init__(self):
        self.module_name = "Atman Module"
        self.element = "Pure Consciousness"
        self.color = "Golden White"
        self.mantra = "Aham Brahmasmi"  # I am Brahman
        self.deity = "Brahman (Formless Absolute)"
        self.principles = ["Self-Knowledge", "Non-Dual Awareness", "Witness Consciousness", "Divine Identity"]
        self.teachings_database = self._initialize_teachings()
        self.self_inquiry_practices = self._initialize_self_inquiry()
        self.ego_obstacles = self._initialize_ego_obstacles()
        logger.info(f"Initialized {self.module_name} with soul awareness guidance")
    
    def _initialize_teachings(self) -> Dict[SoulAwarenessLevel, AtmanInsight]:
        """Initialize soul awareness teachings"""
        return {
            SoulAwarenessLevel.BODY_IDENTIFICATION: AtmanInsight(
                level=SoulAwarenessLevel.BODY_IDENTIFICATION,
                teaching="You are not this body. The body is a temporary vessel for the eternal soul.",
                practice="Body witness meditation - observe the body without identifying with it",
                scripture_reference="Katha Upanishad 1.3.3-4: The body is the chariot, intellect the charioteer",
                spiritual_guidance="Begin by understanding that you are the observer of the body, not the body itself",
                meditation="Sit quietly and observe bodily sensations without attachment"
            ),
            
            SoulAwarenessLevel.MIND_IDENTIFICATION: AtmanInsight(
                level=SoulAwarenessLevel.MIND_IDENTIFICATION,
                teaching="You are not the mind or thoughts. You are the witness of mental activities.",
                practice="Thought witnessing - observe thoughts without being caught in them",
                scripture_reference="Mundaka Upanishad 3.1.1: Two birds sit on the same tree, one acts, one observes",
                spiritual_guidance="Practice being the silent witness of your thoughts and emotions",
                meditation="Watch thoughts arise and pass like clouds in the sky of consciousness"
            ),
            
            SoulAwarenessLevel.SOUL_GLIMPSE: AtmanInsight(
                level=SoulAwarenessLevel.SOUL_GLIMPSE,
                teaching="Moments of peace and joy come from touching your true nature - the soul.",
                practice="Self-inquiry: Ask 'Who am I?' repeatedly until you reach pure awareness",
                scripture_reference="Chandogya Upanishad 6.8.7: Tat tvam asi - Thou art That",
                spiritual_guidance="In moments of deep peace, you glimpse your true Self",
                meditation="Ramana's self-inquiry: 'Who am I?' meditation"
            ),
            
            SoulAwarenessLevel.SOUL_AWARENESS: AtmanInsight(
                level=SoulAwarenessLevel.SOUL_AWARENESS,
                teaching="The soul is your true identity - pure consciousness, bliss, and existence.",
                practice="Abiding as awareness itself, beyond all mental modifications",
                scripture_reference="Brihadaranyaka Upanishad 1.4.10: Aham Brahmasmi - I am Brahman",
                spiritual_guidance="Rest in the knowledge 'I am the Self of all beings'",
                meditation="Meditation on 'I AM' without attributes or descriptions"
            ),
            
            SoulAwarenessLevel.ATMA_BRAHMA_UNITY: AtmanInsight(
                level=SoulAwarenessLevel.ATMA_BRAHMA_UNITY,
                teaching="Complete realization that individual soul and universal consciousness are one.",
                practice="Spontaneous recognition of unity in all experiences",
                scripture_reference="Mandukya Upanishad: 'All this is indeed Brahman'",
                spiritual_guidance="Live from the understanding that all is One consciousness",
                meditation="Natural state of samadhi - no separation between meditator and meditation"
            )
        }
    
    def _initialize_self_inquiry(self) -> List[SelfInquiryPractice]:
        """Initialize self-inquiry practices"""
        return [
            SelfInquiryPractice(
                question="Who am I?",
                method="Trace the 'I' thought back to its source in pure awareness",
                duration="15-45 minutes daily",
                signs_of_progress=[
                    "Thoughts slow down naturally",
                    "Sense of peaceful emptiness",
                    "Identification with thoughts weakens",
                    "Awareness of awareness itself"
                ],
                common_distractions=[
                    "Getting caught in thought stories",
                    "Trying to think the answer",
                    "Becoming frustrated with the process",
                    "Seeking experiences rather than understanding"
                ],
                deepening_techniques=[
                    "Ask 'Who is aware of this thought?'",
                    "Return to the feeling of 'I' before thoughts",
                    "Notice the space in which thoughts appear",
                    "Rest in the 'I AM' presence"
                ]
            ),
            
            SelfInquiryPractice(
                question="What am I?",
                method="Investigate the nature of your being beyond all descriptions",
                duration="20-60 minutes",
                signs_of_progress=[
                    "Recognition of pure existence",
                    "Freedom from all definitions",
                    "Natural peace and contentment",
                    "Unity with all experience"
                ],
                common_distractions=[
                    "Trying to conceptualize the answer",
                    "Seeking mystical experiences",
                    "Comparing with spiritual descriptions",
                    "Doubting the simplicity"
                ],
                deepening_techniques=[
                    "Notice what remains when all thoughts stop",
                    "Investigate the awareness aware of awareness",
                    "Rest in being itself",
                    "Recognize the changeless background"
                ]
            )
        ]
    
    def _initialize_ego_obstacles(self) -> Dict[EgoObstacle, Dict[str, Any]]:
        """Initialize ego obstacles and their remedies"""
        return {
            EgoObstacle.BODY_ATTACHMENT: {
                "description": "Believing you are the physical body",
                "remedies": [
                    "Practice body witnessing meditation",
                    "Study the temporary nature of the body",
                    "Cultivate awareness of awareness itself",
                    "Remember: 'I have a body, but I am not the body'"
                ],
                "teaching": "The body is like clothes for the soul - useful but not your identity"
            },
            
            EgoObstacle.MENTAL_IDENTIFICATION: {
                "description": "Believing you are your thoughts and emotions",
                "remedies": [
                    "Practice thought witnessing",
                    "Ask 'Who is aware of this thought?'",
                    "Cultivate the observer perspective",
                    "Study the changeable nature of mind"
                ],
                "teaching": "Thoughts are like clouds passing through the sky of consciousness"
            },
            
            EgoObstacle.DESIRE_SEEKING: {
                "description": "Seeking fulfillment through external objects",
                "remedies": [
                    "Investigate the source of desire",
                    "Practice contentment with what is",
                    "Seek happiness in your true nature",
                    "Understand desires as pointers to the Self"
                ],
                "teaching": "What you seek through desires is actually your own true nature"
            },
            
            EgoObstacle.FEAR_OF_DISSOLUTION: {
                "description": "Fear of losing personal identity",
                "remedies": [
                    "Understand that the real You never changes",
                    "Study teachings on the eternal nature of the Self",
                    "Practice gradual surrender",
                    "Realize that ego dissolution is ego expansion"
                ],
                "teaching": "You don't lose yourself in realization; you find your true Self"
            },
            
            EgoObstacle.INTELLECTUAL_UNDERSTANDING: {
                "description": "Staying only in conceptual knowledge",
                "remedies": [
                    "Move from thinking to being",
                    "Practice direct investigation",
                    "Spend time in silence",
                    "Apply understanding in daily life"
                ],
                "teaching": "The moon can be described, but you must see it directly"
            },
            
            EgoObstacle.SPIRITUAL_EGO: {
                "description": "Pride in spiritual attainments",
                "remedies": [
                    "Practice humility and surrender",
                    "Remember that realization is your natural state",
                    "Avoid comparing spiritual experiences",
                    "Serve others without attachment to being a teacher"
                ],
                "teaching": "True spirituality is the absence of the one who is spiritual"
            }
        }
    
    async def process_atman_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process soul awareness queries and provide guidance"""
        context = user_context or {}
        
        # Assess current soul awareness level
        awareness_level = self._assess_soul_awareness(query, context)
        
        # Get relevant teaching
        insight = self.teachings_database.get(awareness_level, self.teachings_database[SoulAwarenessLevel.SOUL_GLIMPSE])
        
        # Identify obstacles
        obstacles = self._identify_ego_obstacles(query, context)
        
        # Generate self-inquiry guidance
        inquiry_guidance = self._get_self_inquiry_guidance(awareness_level)
        
        # Get Upanishad wisdom
        upanishad_wisdom = self._get_relevant_upanishad_teaching(query)
        
        return {
            "query": query,
            "soul_awareness_level": awareness_level.value,
            "atman_teaching": insight.teaching,
            "self_inquiry_practice": insight.practice,
            "meditation_technique": insight.meditation,
            "spiritual_guidance": insight.spiritual_guidance,
            "scripture_reference": insight.scripture_reference,
            "upanishad_wisdom": upanishad_wisdom,
            "ego_obstacles": obstacles,
            "inquiry_guidance": inquiry_guidance,
            "atma_brahma_unity": self._explain_atma_brahma_unity(awareness_level),
            "daily_practices": self._get_daily_practices(awareness_level),
            "practical_reminders": self._get_practical_reminders()
        }
    
    def _assess_soul_awareness(self, query: str, context: Dict[str, Any]) -> SoulAwarenessLevel:
        """Assess user's current level of soul awareness"""
        query_lower = query.lower()
        
        # Check for advanced understanding
        if any(term in query_lower for term in ["brahman", "unity", "oneness", "non-dual", "advaita"]):
            return SoulAwarenessLevel.ATMA_BRAHMA_UNITY
        
        # Check for soul awareness
        if any(term in query_lower for term in ["soul", "atman", "true self", "witness", "awareness"]):
            return SoulAwarenessLevel.SOUL_AWARENESS
        
        # Check for glimpses
        if any(term in query_lower for term in ["peace", "bliss", "meditation", "silence"]):
            return SoulAwarenessLevel.SOUL_GLIMPSE
        
        # Check for mental identification
        if any(term in query_lower for term in ["thoughts", "mind", "emotions", "mental"]):
            return SoulAwarenessLevel.MIND_IDENTIFICATION
        
        # Default to body identification
        return SoulAwarenessLevel.BODY_IDENTIFICATION
    
    def _identify_ego_obstacles(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify ego obstacles present in the query"""
        obstacles = []
        query_lower = query.lower()
        
        obstacle_keywords = {
            EgoObstacle.BODY_ATTACHMENT: ["body", "physical", "appearance", "health", "pain"],
            EgoObstacle.MENTAL_IDENTIFICATION: ["thoughts", "thinking", "emotions", "mind", "mental"],
            EgoObstacle.DESIRE_SEEKING: ["want", "need", "desire", "seeking", "happiness", "fulfillment"],
            EgoObstacle.FEAR_OF_DISSOLUTION: ["afraid", "fear", "losing", "death", "disappear"],
            EgoObstacle.INTELLECTUAL_UNDERSTANDING: ["understand", "concept", "theory", "knowledge"],
            EgoObstacle.SPIRITUAL_EGO: ["enlightened", "advanced", "special", "superior", "teacher"]
        }
        
        for obstacle, keywords in obstacle_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                obstacle_info = self.ego_obstacles[obstacle]
                obstacles.append({
                    "obstacle": obstacle.value,
                    "description": obstacle_info["description"],
                    "remedies": obstacle_info["remedies"][:2],  # Top 2 remedies
                    "teaching": obstacle_info["teaching"]
                })
        
        return obstacles
    
    def _get_self_inquiry_guidance(self, level: SoulAwarenessLevel) -> Dict[str, Any]:
        """Get self-inquiry guidance appropriate for the level"""
        if level in [SoulAwarenessLevel.BODY_IDENTIFICATION, SoulAwarenessLevel.MIND_IDENTIFICATION]:
            practice = self.self_inquiry_practices[0]  # "Who am I?"
        else:
            practice = self.self_inquiry_practices[1]  # "What am I?"
        
        return {
            "question": practice.question,
            "method": practice.method,
            "duration": practice.duration,
            "signs_of_progress": practice.signs_of_progress[:3],
            "deepening_techniques": practice.deepening_techniques[:2]
        }
    
    def _get_relevant_upanishad_teaching(self, query: str) -> str:
        """Get relevant Upanishad teaching based on query"""
        teachings = {
            "identity": "Chandogya Upanishad: 'Tat tvam asi' - Thou art That divine essence",
            "consciousness": "Mandukya Upanishad: Consciousness has four states, the fourth is pure awareness",
            "self": "Brihadaranyaka Upanishad: 'Aham Brahmasmi' - I am Brahman",
            "unity": "Isha Upanishad: See the Self in all beings and all beings in the Self",
            "peace": "Katha Upanishad: When all desires are abandoned, the mortal becomes immortal",
            "knowledge": "Mundaka Upanishad: The knower of Brahman becomes Brahman",
            "bliss": "Taittiriya Upanishad: Brahman is existence, consciousness, and bliss"
        }
        
        query_lower = query.lower()
        for key, teaching in teachings.items():
            if key in query_lower:
                return teaching
                
        return "Svetasvatara Upanishad: The Self is the light of lights, beyond darkness"
    
    def _explain_atma_brahma_unity(self, level: SoulAwarenessLevel) -> str:
        """Explain Atma-Brahma unity based on awareness level"""
        explanations = {
            SoulAwarenessLevel.BODY_IDENTIFICATION: 
                "The individual soul (Atman) is your true identity beyond the body. It is connected to the universal consciousness (Brahman).",
            SoulAwarenessLevel.MIND_IDENTIFICATION:
                "Your soul is not separate from the universal soul. The sense of separation is created by the mind.",
            SoulAwarenessLevel.SOUL_GLIMPSE:
                "Atman (individual consciousness) and Brahman (universal consciousness) are like a wave and the ocean - apparently different but essentially one.",
            SoulAwarenessLevel.SOUL_AWARENESS:
                "The realization dawns that your true Self and the cosmic Self are one and the same consciousness.",
            SoulAwarenessLevel.ATMA_BRAHMA_UNITY:
                "Complete understanding that there is no difference between Atman and Brahman - individual and universal consciousness are one."
        }
        return explanations.get(level, explanations[SoulAwarenessLevel.BODY_IDENTIFICATION])
    
    def _get_daily_practices(self, level: SoulAwarenessLevel) -> List[str]:
        """Get daily practices appropriate for the awareness level"""
        base_practices = [
            "Morning self-inquiry meditation",
            "Evening reflection on the day's experiences as witness",
            "Study of Upanishads or Advaita Vedanta texts",
            "Practice seeing the same Self in all beings"
        ]
        
        level_specific = {
            SoulAwarenessLevel.BODY_IDENTIFICATION: [
                "Body witnessing meditation",
                "Practice 'I have a body, but I am not the body'"
            ],
            SoulAwarenessLevel.MIND_IDENTIFICATION: [
                "Thought witnessing practice",
                "Ask 'Who is aware of this thought?'"
            ],
            SoulAwarenessLevel.SOUL_GLIMPSE: [
                "Regular 'Who am I?' inquiry",
                "Cultivate moments of silence and peace"
            ],
            SoulAwarenessLevel.SOUL_AWARENESS: [
                "Abide as pure awareness",
                "Practice 'I AM' meditation"
            ],
            SoulAwarenessLevel.ATMA_BRAHMA_UNITY: [
                "Live from unity consciousness",
                "Spontaneous recognition of oneness"
            ]
        }
        
        return base_practices + level_specific.get(level, [])
    
    def _get_practical_reminders(self) -> List[str]:
        """Get practical reminders for soul awareness"""
        return [
            "You are the awareness in which all experience appears",
            "The Self you seek is the Self that seeks",
            "Peace is your nature, not something to be attained",
            "You are already that which you are seeking to become",
            "The witness of change is unchanging",
            "Love is the recognition of yourself in another",
            "Surrender is the recognition that you were never in control",
            "Enlightenment is the natural state, not an achievement"
        ]
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get current status of the Atman Module"""
        return {
            "name": self.module_name,
            "state": "active",
            "element": self.element,
            "color": self.color,
            "mantra": self.mantra,
            "governing_deity": self.deity,
            "core_principles": self.principles,
            "primary_functions": [
                "Soul awareness assessment",
                "Self-inquiry guidance",
                "Ego obstacle identification",
                "Atma-Brahma unity teaching"
            ],
            "wisdom_available": "Direct guidance for self-realization and recognition of your true nature"
        }
    
    async def daily_atman_practice(self, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Provide daily practices for soul awareness"""
        context = user_context or {}
        
        morning_practice = [
            "Begin with self-inquiry: 'Who am I?'",
            "Rest in the 'I AM' presence before thoughts arise",
            "Set intention to remain as the witness throughout the day"
        ]
        
        throughout_day = [
            "Remember: 'I am the awareness of this experience'",
            "Practice seeing the same Self in all beings",
            "When thoughts arise, ask 'Who is aware of this?'",
            "Return to the sense of 'I' behind all experiences"
        ]
        
        evening_practice = [
            "Reflect on the day as the witness of all experiences",
            "Practice gratitude for glimpses of your true nature",
            "Study spiritual texts on self-knowledge",
            "Rest in deep sleep as pure consciousness"
        ]
        
        return {
            "morning_practice": morning_practice,
            "throughout_day": throughout_day,
            "evening_practice": evening_practice,
            "weekly_focus": "Deepen self-inquiry practice and study Advaita Vedanta",
            "monthly_goal": "Strengthen identification with pure awareness rather than body-mind",
            "ultimate_reminder": "You are already the Self you are seeking to realize"
        }

# Global instance for easy import
atman_module = AtmanModule()

def get_atman_module():
    """Get the global atman module instance"""
    return atman_module
