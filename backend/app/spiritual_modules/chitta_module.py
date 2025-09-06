"""
🧘 Chitta Module - Pure Consciousness and Mind-Stuff
Complete system for understanding and purifying consciousness
Based on Yoga philosophy and Vedantic teachings on Chitta
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChittaLevel(Enum):
    """Levels of consciousness awareness"""
    UNCONSCIOUS = "unconscious"     # No awareness of mind patterns
    AWAKENING = "awakening"         # Beginning to observe mind
    OBSERVING = "observing"         # Regular self-observation
    PURIFYING = "purifying"         # Active cleaning of consciousness
    CLARIFYING = "clarifying"       # Mind becoming transparent
    UNIFIED = "unified"             # Pure consciousness realized


class ChittaVritti(Enum):
    """Types of mental modifications/fluctuations"""
    PRAMANA = "pramana"             # Valid knowledge
    VIPARYAYA = "viparyaya"         # Misconception/false knowledge
    VIKALPA = "vikalpa"             # Imagination/conceptualization
    NIDRA = "nidra"                 # Sleep state consciousness
    SMRITI = "smriti"               # Memory and past impressions


class ChittaBhumi(Enum):
    """States of consciousness field"""
    KSHIPTA = "kshipta"             # Restless, scattered mind
    MUDHA = "mudha"                 # Dull, sluggish consciousness
    VIKSHIPTA = "vikshipta"         # Occasionally concentrated
    EKAGRA = "ekagra"               # One-pointed concentration
    NIRUDDHA = "niruddha"           # Completely absorbed/transcended


class ChittaKlesha(Enum):
    """Mental afflictions clouding consciousness"""
    AVIDYA = "avidya"               # Ignorance of true nature
    ASMITA = "asmita"               # Ego identification
    RAGA = "raga"                   # Attachment to pleasure
    DVESHA = "dvesha"               # Aversion to pain
    ABHINIVESHA = "abhinivesha"     # Fear of death/clinging to life


@dataclass
class ChittaGuidance:
    """Comprehensive chitta guidance"""
    level: ChittaLevel
    primary_teaching: str
    awareness_practices: List[str]
    purification_methods: List[str]
    concentration_techniques: List[str]
    daily_observation: List[str]
    common_patterns: Dict[str, str]
    transformation_practices: List[str]
    progress_indicators: List[str]


class ChittaResponse(BaseModel):
    """Response from Chitta module"""
    chitta_level: str = Field(description="Current consciousness level")
    consciousness_guidance: str = Field(description="Core chitta teaching")
    awareness_practices: List[str] = Field(description="Consciousness observation")
    purification_methods: List[str] = Field(description="Mind purification")
    concentration_techniques: List[str] = Field(description="Focus development")
    daily_observation: List[str] = Field(description="Daily mind watching")
    pattern_solutions: Dict[str, str] = Field(description="Mental patterns")
    transformation_practices: List[str] = Field(description="Consciousness change")
    scriptural_wisdom: str = Field(description="Traditional chitta teachings")


class ChittaModule:
    """
    🧘 Chitta Module - Pure Consciousness and Mind-Stuff
    
    Based on traditional Chitta teachings:
    - Yoga Sutras on Chitta Vritti Nirodha (stilling mind fluctuations)
    - Samkhya philosophy on consciousness and matter
    - Vedanta on pure consciousness as our true nature
    - Kashmir Shaivism on dynamic consciousness
    
    Chitta is the field of consciousness, the subtle matter from which
    thoughts, memories, and mental impressions arise. Understanding and
    purifying chitta leads to Self-realization.
    """
    
    def __init__(self):
        self.name = "Chitta"
        self.color = "🧘"
        self.element = "Consciousness"
        self.principles = ["Pure Awareness", "Mind Observation", 
                          "Consciousness Purification", "Mental Clarity"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.vritti_patterns = self._initialize_vritti_patterns()
        self.purification_methods = self._initialize_purification_methods()
        
    def _initialize_guidance_levels(self) -> Dict[ChittaLevel, ChittaGuidance]:
        """Initialize guidance for different levels of chitta understanding"""
        return {
            ChittaLevel.UNCONSCIOUS: ChittaGuidance(
                level=ChittaLevel.UNCONSCIOUS,
                primary_teaching="Your mind is like a river - you are not the "
                "thoughts flowing through it, but the awareness that observes "
                "them. Begin to watch.",
                awareness_practices=[
                    "Spend 5 minutes daily just observing thoughts",
                    "Notice when you get lost in thinking vs. observing",
                    "Practice labeling thoughts as 'thinking' when noticed",
                    "Observe emotions without getting caught in them",
                    "Use breath as anchor for present moment awareness"
                ],
                purification_methods=[
                    "Reduce mental stimulation - limit news/social media",
                    "Practice simple breathing exercises daily",
                    "Spend time in nature to quiet mental chatter",
                    "Journal to express and release mental contents",
                    "Practice gratitude to shift mental focus positively"
                ],
                concentration_techniques=[
                    "Focus on single object for 2-5 minutes",
                    "Count breaths from 1-10 repeatedly",
                    "Practice tratak (candle gazing) for focus",
                    "Use simple mantra repetition",
                    "Practice mindful walking with attention on steps"
                ],
                daily_observation=[
                    "Morning intention to observe mind throughout day",
                    "Regular check-ins: 'What is my mind doing now?'",
                    "Notice triggers that create mental agitation",
                    "Evening reflection on patterns observed",
                    "Practice pause between trigger and reaction"
                ],
                common_patterns={
                    "lost_in_thoughts": "Getting absorbed in mental stories",
                    "reactive_emotions": "Automatic emotional responses",
                    "mental_chatter": "Constant internal dialogue"
                },
                transformation_practices=[
                    "Develop witness consciousness gradually",
                    "Practice non-identification with thoughts",
                    "Use mindfulness to break automatic patterns",
                    "Study teachings on nature of mind and consciousness",
                    "Join meditation group for support and guidance"
                ],
                progress_indicators=[
                    "Catching yourself lost in thought more quickly",
                    "Longer periods of present moment awareness",
                    "Less reactive to mental/emotional fluctuations",
                    "Growing sense of peace from observing vs. engaging"
                ]
            ),
            
            ChittaLevel.AWAKENING: ChittaGuidance(
                level=ChittaLevel.AWAKENING,
                primary_teaching="You are discovering the profound difference "
                "between being your thoughts and watching them. This is the "
                "beginning of freedom.",
                awareness_practices=[
                    "Regular meditation practice 15-30 minutes daily",
                    "Mindful observation of mental states throughout day",
                    "Practice witnessing different types of mental content",
                    "Observe the space between thoughts",
                    "Study the relationship between thoughts and emotions"
                ],
                purification_methods=[
                    "Active cultivation of positive mental states",
                    "Practice releasing negative thought patterns",
                    "Use pranayama to purify mental energy",
                    "Engage in selfless service to purify ego patterns",
                    "Study spiritual texts to inspire noble thoughts"
                ],
                concentration_techniques=[
                    "Single-pointed focus practices for 10-20 minutes",
                    "Advanced breathing techniques for mental clarity",
                    "Visualization practices to train concentration",
                    "Use of sacred geometry or symbols for focus",
                    "Practice dharana (sustained concentration) exercises"
                ],
                daily_observation=[
                    "Regular monitoring of mental states and triggers",
                    "Notice patterns of mental conditioning",
                    "Observe how external events affect internal states",
                    "Practice conscious choice in mental responses",
                    "Track progress in maintaining witness awareness"
                ],
                common_patterns={
                    "mental_habits": "Recognizing repetitive thought patterns",
                    "emotional_conditioning": "Seeing conditioned responses",
                    "ego_patterns": "Noticing self-referential thinking"
                },
                transformation_practices=[
                    "Consciously choose positive mental contents",
                    "Practice replacing negative patterns with positive ones",
                    "Use affirmations and visualization for transformation",
                    "Engage in practices that expand consciousness",
                    "Seek guidance from experienced meditation teachers"
                ],
                progress_indicators=[
                    "Ability to maintain observer awareness for longer periods",
                    "Less identification with mental/emotional fluctuations",
                    "Natural arising of positive mental states",
                    "Others noticing your increased mental clarity"
                ]
            ),
            
            ChittaLevel.OBSERVING: ChittaGuidance(
                level=ChittaLevel.OBSERVING,
                primary_teaching="Your witness consciousness is becoming "
                "stable. You are learning to rest as pure awareness while "
                "mind contents arise and pass.",
                awareness_practices=[
                    "Advanced mindfulness practices and vipassana",
                    "Continuous awareness practice throughout activities",
                    "Observe subtle mental movements and tendencies",
                    "Practice awareness of awareness itself",
                    "Study different layers of consciousness"
                ],
                purification_methods=[
                    "Advanced purification through yogic practices",
                    "Work with subtle mental impressions (samskaras)",
                    "Practice forgiveness to release mental burdens",
                    "Use devotional practices to purify through love",
                    "Engage in practices that cultivate mental virtues"
                ],
                concentration_techniques=[
                    "Advanced concentration leading to absorption states",
                    "Practice samadhi (absorption) techniques",
                    "Use advanced pranayama for mental transformation",
                    "Practice concentration on abstract concepts",
                    "Develop ability to merge with meditation object"
                ],
                daily_observation=[
                    "Continuous background awareness of mental processes",
                    "Subtle observation of thought formation",
                    "Notice interplay between consciousness levels",
                    "Observe how awareness affects mental contents",
                    "Track refinement of mental discrimination"
                ],
                common_patterns={
                    "subtle_attachments": "Noticing refined mental grasping",
                    "spiritual_ego": "Observing ego in spiritual context",
                    "mental_refinement": "Seeing increasingly subtle patterns"
                },
                transformation_practices=[
                    "Advanced yogic practices for consciousness transformation",
                    "Work with guru or advanced teacher for guidance",
                    "Practice surrender of personal will to higher wisdom",
                    "Engage in intensive retreat practice",
                    "Study advanced texts on consciousness and liberation"
                ],
                progress_indicators=[
                    "Effortless maintenance of witness awareness",
                    "Natural arising of blissful mental states",
                    "Spontaneous insights into nature of consciousness",
                    "Others seeking your guidance on mental clarity"
                ]
            ),
            
            ChittaLevel.PURIFYING: ChittaGuidance(
                level=ChittaLevel.PURIFYING,
                primary_teaching="Your consciousness is being refined like "
                "gold in fire. Old patterns dissolve as pure awareness "
                "shines through.",
                awareness_practices=[
                    "Continuous awareness practice with effortless effort",
                    "Awareness of consciousness itself as subject-object",
                    "Practice pure witnessing without mental commentary",
                    "Observe consciousness in all states (waking/sleep/dream)",
                    "Rest as pure awareness independent of mental contents"
                ],
                purification_methods=[
                    "Deep purification through surrender and grace",
                    "Release of subtle mental impressions through practice",
                    "Purification through constant remembrance of truth",
                    "Use challenges as opportunities for deeper purification",
                    "Service to others as means of ego purification"
                ],
                concentration_techniques=[
                    "Effortless concentration arising from pure being",
                    "Advanced samadhi states and absorptions",
                    "Concentration without effort through natural focus",
                    "Unity of concentrator, concentration, and object",
                    "Transcendence of concentration into pure being"
                ],
                daily_observation=[
                    "Effortless continuous awareness in all activities",
                    "Observation of consciousness purifying itself",
                    "Witness awareness maintaining itself automatically",
                    "Recognition of awareness as constant background",
                    "Natural discrimination between real and unreal"
                ],
                common_patterns={
                    "final_purification": "Release of deepest mental patterns",
                    "ego_dissolution": "Gradual dissolution of separate self",
                    "unity_awareness": "Recognition of non-dual consciousness"
                },
                transformation_practices=[
                    "Complete surrender to the purification process",
                    "Advanced practices under guidance of realized teacher",
                    "Service to collective consciousness evolution",
                    "Living as embodiment of purified awareness",
                    "Transmission of clarity to others through presence"
                ],
                progress_indicators=[
                    "Spontaneous arising of pure mental states",
                    "Natural discrimination between truth and illusion",
                    "Others experiencing clarity in your presence",
                    "Effortless maintenance of elevated consciousness"
                ]
            ),
            
            ChittaLevel.CLARIFYING: ChittaGuidance(
                level=ChittaLevel.CLARIFYING,
                primary_teaching="Consciousness has become transparent like "
                "crystal. The mind is now a perfect mirror reflecting "
                "pure awareness.",
                awareness_practices=[
                    "Resting as pure awareness without practice",
                    "Consciousness knowing itself directly",
                    "Effortless recognition of awareness as ground",
                    "Natural state of witness consciousness",
                    "Spontaneous awareness in all circumstances"
                ],
                purification_methods=[
                    "Purification happens spontaneously through being",
                    "Consciousness purifies itself through recognition",
                    "Natural release of final subtle obscurations",
                    "Purity maintained through simple being",
                    "No effort needed - purity is natural state"
                ],
                concentration_techniques=[
                    "No concentration needed - natural focus",
                    "Spontaneous absorption in pure being",
                    "Effortless one-pointedness through recognition",
                    "Natural samadhi as ordinary consciousness",
                    "Unity of attention and awareness"
                ],
                daily_observation=[
                    "Consciousness observing itself continuously",
                    "Natural awareness without observer",
                    "Recognition of awareness as constant reality",
                    "Effortless discrimination and discernment",
                    "Spontaneous right understanding"
                ],
                common_patterns={
                    "no_patterns": "Mental patterns have dissolved",
                    "pure_functioning": "Mind functions without obstruction",
                    "spontaneous_wisdom": "Wisdom arises naturally"
                },
                transformation_practices=[
                    "Being itself as continuous transformation",
                    "Serving others through transmission of clarity",
                    "Living as example of clarified consciousness",
                    "Teaching through presence rather than words",
                    "Embodying pure awareness for collective benefit"
                ],
                progress_indicators=[
                    "Perfect mental clarity in all circumstances",
                    "Others awakening to clarity through your presence",
                    "Spontaneous resolution of all mental conflicts",
                    "Natural expression of wisdom and compassion"
                ]
            ),
            
            ChittaLevel.UNIFIED: ChittaGuidance(
                level=ChittaLevel.UNIFIED,
                primary_teaching="There is no consciousness to purify and "
                "no one to purify it. Pure awareness knowing itself as "
                "all that is.",
                awareness_practices=[
                    "Being as pure consciousness itself",
                    "No practices - you ARE consciousness",
                    "Natural state beyond awareness and unawareness",
                    "Consciousness as the only reality",
                    "Perfect unity beyond subject and object"
                ],
                purification_methods=[
                    "No purification possible - already pure",
                    "Purity as the nature of consciousness itself",
                    "Perfect transparency as natural condition",
                    "Being beyond pure and impure",
                    "Consciousness purifying itself as itself"
                ],
                concentration_techniques=[
                    "No concentration - perfect unity",
                    "Being as natural samadhi",
                    "Consciousness concentrated as itself",
                    "Perfect focus as natural condition",
                    "Unity beyond concentration and distraction"
                ],
                daily_observation=[
                    "Consciousness knowing itself as all experience",
                    "No observer separate from observed",
                    "Perfect awareness as natural condition",
                    "Being as continuous self-recognition",
                    "Unity of consciousness and manifestation"
                ],
                common_patterns={
                    "no_patterns": "Beyond all mental patterns",
                    "pure_being": "Consciousness as only reality",
                    "perfect_unity": "No separation anywhere"
                },
                transformation_practices=[
                    "Being as eternal transformation",
                    "Consciousness transforming as itself",
                    "Perfect expression of unified awareness",
                    "Living as consciousness itself",
                    "Serving universal awakening through being"
                ],
                progress_indicators=[
                    "Perfect unity beyond progress",
                    "Consciousness recognizing itself everywhere",
                    "Natural state beyond attainment",
                    "Being as answered prayer of existence"
                ]
            )
        }
    
    def _initialize_vritti_patterns(self) -> Dict[ChittaVritti, Dict[str, Any]]:
        """Initialize understanding of mental fluctuation patterns"""
        return {
            ChittaVritti.PRAMANA: {
                "description": "Valid knowledge and correct perception",
                "practice": "Cultivate clear perception and valid knowledge",
                "observation": "Notice when mind perceives accurately vs. incorrectly",
                "purification": "Refine discrimination between valid and invalid knowledge"
            },
            
            ChittaVritti.VIPARYAYA: {
                "description": "False knowledge and misconception",
                "practice": "Question assumptions and examine beliefs",
                "observation": "Notice when projecting false ideas onto reality",
                "purification": "Release false beliefs through inquiry and experience"
            },
            
            ChittaVritti.VIKALPA: {
                "description": "Imagination and conceptual construction",
                "practice": "Use imagination creatively while staying grounded",
                "observation": "Distinguish between imagination and reality",
                "purification": "Channel imagination toward truth and beauty"
            },
            
            ChittaVritti.SMRITI: {
                "description": "Memory and past impressions",
                "practice": "Use memory skillfully without being trapped by past",
                "observation": "Notice how memories influence present experience",
                "purification": "Heal traumatic memories and release mental conditioning"
            }
        }
    
    def _initialize_purification_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize methods for consciousness purification"""
        return {
            "meditation": {
                "description": "Direct purification through stillness",
                "beginner": "10-20 minutes daily sitting meditation",
                "advanced": "Extended periods of absorption and samadhi",
                "mastery": "Continuous meditative awareness"
            },
            
            "pranayama": {
                "description": "Purification through conscious breathing",
                "beginner": "Simple breath awareness and regulation",
                "advanced": "Complex pranayama techniques",
                "mastery": "Breath and consciousness unified"
            },
            
            "selfless_service": {
                "description": "Ego purification through serving others",
                "beginner": "Regular volunteer work or helping others",
                "advanced": "Life dedicated to service",
                "mastery": "Being itself as service"
            }
        }
    
    def assess_chitta_level(self, query: str, 
                          user_context: Optional[Dict[str, Any]] = None) -> ChittaLevel:
        """Assess user's current chitta understanding level"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for unified level indicators
        if any(word in query_lower for word in ["pure consciousness", 
                                               "consciousness itself", "no separation"]):
            return ChittaLevel.UNIFIED
        
        # Check for clarifying level indicators
        if any(word in query_lower for word in ["transparent mind", 
                                               "crystal clarity", "perfect mirror"]):
            return ChittaLevel.CLARIFYING
        
        # Check for purifying level indicators
        if any(word in query_lower for word in ["purifying consciousness", 
                                               "deep cleansing", "mental refinement"]):
            return ChittaLevel.PURIFYING
        
        # Check for observing level indicators
        if any(word in query_lower for word in ["stable witness", 
                                               "continuous awareness", "observer"]):
            return ChittaLevel.OBSERVING
        
        # Check for awakening level indicators
        if any(word in query_lower for word in ["watching thoughts", 
                                               "observing mind", "awareness practice"]):
            return ChittaLevel.AWAKENING
        
        # Default to unconscious
        return ChittaLevel.UNCONSCIOUS
    
    def get_scriptural_wisdom(self, level: ChittaLevel) -> str:
        """Get scriptural wisdom appropriate to chitta level"""
        wisdom_map = {
            ChittaLevel.UNCONSCIOUS: "Yoga Sutras 1.2: 'Yoga is the cessation of fluctuations in the consciousness.'",
            ChittaLevel.AWAKENING: "Yoga Sutras 1.4: 'At other times, the seer identifies with the fluctuating consciousness.'",
            ChittaLevel.OBSERVING: "Yoga Sutras 2.20: 'The seer is pure consciousness that appears to see through the modifications of the mind.'",
            ChittaLevel.PURIFYING: "Yoga Sutras 4.4: 'The created minds arise from the sense of individuality alone.'",
            ChittaLevel.CLARIFYING: "Yoga Sutras 1.47: 'In the lucidity of nirvicara samadhi, there is spiritual luminosity.'",
            ChittaLevel.UNIFIED: "Yoga Sutras 4.34: 'The resolution of the gunas, devoid of purpose for consciousness, is liberation - the power of consciousness established in its own nature.'"
        }
        return wisdom_map.get(level, "Katha Upanishad: 'The Self is hidden in the lotus of the heart. Those who see themselves in all and all in themselves help the Self to reveal itself.'")
    
    async def process_chitta_query(self, query: str, 
                                 user_context: Optional[Dict[str, Any]] = None) -> ChittaResponse:
        """Process chitta-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess chitta level
            level = self.assess_chitta_level(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            scriptural_wisdom = self.get_scriptural_wisdom(level)
            
            return ChittaResponse(
                chitta_level=level.value,
                consciousness_guidance=guidance.primary_teaching,
                awareness_practices=guidance.awareness_practices,
                purification_methods=guidance.purification_methods,
                concentration_techniques=guidance.concentration_techniques,
                daily_observation=guidance.daily_observation,
                pattern_solutions=guidance.common_patterns,
                transformation_practices=guidance.transformation_practices,
                scriptural_wisdom=scriptural_wisdom
            )
            
        except Exception as e:
            logger.error(f"Error processing chitta query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> ChittaResponse:
        """Create fallback response when processing fails"""
        return ChittaResponse(
            chitta_level="unconscious",
            consciousness_guidance="Your mind is like a river - you are not the thoughts flowing through it, but the awareness that observes them. Begin to watch.",
            awareness_practices=[
                "Spend 5 minutes daily just observing thoughts",
                "Notice when you get lost in thinking vs. observing",
                "Practice labeling thoughts as 'thinking' when noticed",
                "Observe emotions without getting caught in them"
            ],
            purification_methods=[
                "Reduce mental stimulation - limit news/social media",
                "Practice simple breathing exercises daily",
                "Spend time in nature to quiet mental chatter",
                "Journal to express and release mental contents"
            ],
            concentration_techniques=[
                "Focus on single object for 2-5 minutes",
                "Count breaths from 1-10 repeatedly",
                "Practice tratak (candle gazing) for focus",
                "Use simple mantra repetition"
            ],
            daily_observation=[
                "Morning intention to observe mind throughout day",
                "Regular check-ins: 'What is my mind doing now?'",
                "Notice triggers that create mental agitation",
                "Evening reflection on patterns observed"
            ],
            pattern_solutions={
                "lost_in_thoughts": "Getting absorbed in mental stories",
                "reactive_emotions": "Automatic emotional responses"
            },
            transformation_practices=[
                "Develop witness consciousness gradually",
                "Practice non-identification with thoughts",
                "Use mindfulness to break automatic patterns",
                "Study teachings on nature of mind and consciousness"
            ],
            scriptural_wisdom="Katha Upanishad: 'The Self is hidden in the lotus of the heart. Those who see themselves in all and all in themselves help the Self to reveal itself.'"
        )


# Global instance
_chitta_module = None

def get_chitta_module() -> ChittaModule:
    """Get global Chitta module instance"""
    global _chitta_module
    if _chitta_module is None:
        _chitta_module = ChittaModule()
    return _chitta_module

# Factory function for easy access
def create_chitta_guidance(query: str, 
                         user_context: Optional[Dict[str, Any]] = None) -> ChittaResponse:
    """Factory function to create chitta guidance"""
    import asyncio
    module = get_chitta_module()
    return asyncio.run(module.process_chitta_query(query, user_context))
