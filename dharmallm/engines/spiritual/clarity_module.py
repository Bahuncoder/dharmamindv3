"""
🔍 Clarity Module - Viveka (Spiritual Discrimination) and Clear Vision
Complete system for developing discernment between Maya and Reality
Trains in Viveka, Vairagya, and clear perception of truth
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ClarityLevel(Enum):
    """Levels of spiritual discernment and clarity"""
    CONFUSED = "confused"             # Lost in Maya, unclear vision
    QUESTIONING = "questioning"       # Beginning to question reality
    DISCERNING = "discerning"        # Developing Viveka (discrimination)
    CLEAR = "clear"                  # Consistent clear perception
    ILLUMINATED = "illuminated"       # Established in clear vision

class ClarityAspect(Enum):
    """Aspects of clarity to develop"""
    VIVEKA = "viveka"                 # Discrimination between real and unreal
    VAIRAGYA = "vairagya"            # Dispassion toward temporary things
    BUDDHI = "buddhi"                # Pure intelligence beyond emotions
    INTUITION = "intuition"           # Direct knowing beyond reasoning
    PRESENCE = "presence"             # Clear awareness in the moment
    DISCRIMINATION = "discrimination"  # Ability to distinguish truth from falsehood

class ClarityObstacle(Enum):
    """Common obstacles to clear perception"""
    MAYA_CONFUSION = "maya_confusion"           # Illusion and delusion
    EMOTIONAL_REACTIVITY = "emotional_reactivity" # Emotions clouding judgment
    MENTAL_NOISE = "mental_noise"               # Overthinking and mental chatter
    ATTACHMENT = "attachment"                   # Clinging to outcomes
    FEAR_BASED_THINKING = "fear_based_thinking" # Fear distorting perception
    EGO_IDENTIFICATION = "ego_identification"   # Mistaking ego for self
    SOCIAL_CONDITIONING = "social_conditioning" # External programming
    SENSORY_DISTRACTION = "sensory_distraction" # Overwhelm from external stimuli

class ClarityPractice(Enum):
    """Practices for developing clarity"""
    VIVEKA_MEDITATION = "viveka_meditation"     # Discrimination practice
    WITNESS_CONSCIOUSNESS = "witness_consciousness" # Sakshi Bhava
    INQUIRY_PRACTICE = "inquiry_practice"       # Self-inquiry and questioning
    MINDFULNESS = "mindfulness"                 # Present moment awareness
    DETACHMENT_PRACTICE = "detachment_practice" # Vairagya cultivation
    BUDDHI_DEVELOPMENT = "buddhi_development"   # Pure intelligence training
    SILENCE_PRACTICE = "silence_practice"       # Mauna for clarity
    SCRIPTURAL_STUDY = "scriptural_study"       # Study of wisdom texts

@dataclass
class ClarityGuidance:
    """Comprehensive clarity guidance"""
    level: ClarityLevel
    primary_teaching: str
    viveka_practices: List[str]
    vairagya_practices: List[str]
    daily_clarity_routine: List[str]
    obstacle_solutions: Dict[str, str]
    discrimination_exercises: List[str]
    scriptural_references: List[str]
    practical_applications: List[str]
    progress_indicators: List[str]

@dataclass
class ClarityInsight:
    """Insight for developing clarity"""
    aspect: ClarityAspect
    teaching: str
    practice_method: str
    immediate_application: str
    daily_integration: str

class ClarityAssessment(BaseModel):
    """Assessment of current clarity level"""
    current_level: str = Field(description="Current level of clarity")
    viveka_strength: str = Field(description="Level of discrimination ability")
    main_obstacles: List[str] = Field(description="Primary obstacles to clarity")
    recommended_practices: List[str] = Field(description="Suggested practices")

class ClarityResponse(BaseModel):
    """Response from Clarity module"""
    clarity_level: str = Field(description="Current clarity level assessment")
    viveka_teaching: str = Field(description="Teaching on spiritual discrimination")
    vairagya_guidance: str = Field(description="Guidance on detachment and dispassion")
    daily_practices: List[str] = Field(description="Daily clarity development practices")
    discrimination_exercises: List[str] = Field(description="Exercises for developing discernment")
    obstacle_solutions: Dict[str, str] = Field(description="Solutions for clarity obstacles")
    scriptural_wisdom: str = Field(description="Relevant scriptural guidance")
    practical_applications: List[str] = Field(description="Ways to apply clarity in daily life")
    progress_indicators: List[str] = Field(description="Signs of developing clarity")
    meditation_guidance: str = Field(description="Specific meditation guidance for clarity")

class ClarityModule:
    """
    🔍 Clarity Module - Viveka (Spiritual Discrimination) and Clear Vision
    
    Based on authentic Advaita and Kashmir Shaivism teachings:
    - Adi Shankara's Viveka Chudamani (Crest Jewel of Discrimination)
    - Patanjali's guidance on Buddhi and discrimination
    - Ramana Maharshi's self-inquiry method
    - Kashmir Shaivism's Pratyabhijna (Recognition)
    
    Develops the ability to distinguish:
    - Real from unreal (Sat-Asat Viveka)
    - Eternal from temporary (Nitya-Anitya Viveka)
    - Self from not-Self (Atma-Anatma Viveka)
    """
    
    def __init__(self):
        self.name = "Clarity"
        self.color = "🔍"
        self.element = "Viveka (Discrimination)"
        self.principles = ["Discrimination", "Detachment", "Direct Knowing", "Presence"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.clarity_practices = self._initialize_clarity_practices()
        self.discrimination_exercises = self._initialize_discrimination_exercises()
        self.obstacle_solutions = self._initialize_obstacle_solutions()
        
    def _initialize_guidance_levels(self) -> Dict[ClarityLevel, ClarityGuidance]:
        """Initialize guidance for different levels of clarity"""
        return {
            ClarityLevel.CONFUSED: ClarityGuidance(
                level=ClarityLevel.CONFUSED,
                primary_teaching="Begin by recognizing that confusion is the starting point of wisdom. The very recognition that you are confused is the first glimmer of clarity.",
                viveka_practices=[
                    "Daily question: 'What is real and what is just appearance?'",
                    "Practice distinguishing thoughts from the thinker",
                    "Observe the difference between emotions and awareness",
                    "Study basic discrimination between body, mind, and consciousness"
                ],
                vairagya_practices=[
                    "Practice letting go of one small attachment daily",
                    "Observe how things change and pass away",
                    "Cultivate contentment with what you have",
                    "Reduce excessive desires and wants"
                ],
                daily_clarity_routine=[
                    "Morning: 10 minutes of witness consciousness practice",
                    "Midday: Pause and ask 'What is real in this moment?'",
                    "Evening: Review the day - what was real, what was reaction?",
                    "Before sleep: Practice gratitude and release the day"
                ],
                obstacle_solutions={
                    "Overwhelming confusion": "Start with simple discrimination - breath vs. thoughts",
                    "Too much thinking": "Practice watching thoughts without engaging",
                    "Emotional reactivity": "Learn to pause before reacting to situations"
                },
                discrimination_exercises=[
                    "Distinguish between the observer and the observed",
                    "Notice the difference between thoughts and awareness",
                    "Practice seeing the permanent vs. temporary in experiences"
                ],
                scriptural_references=[
                    "Bhagavad Gita 2.69: 'What is night for all beings is the time of awakening for the self-controlled'",
                    "Viveka Chudamani: 'The real nature of things is to be known by discrimination'"
                ],
                practical_applications=[
                    "When confused, pause and ask 'What do I actually know for certain?'",
                    "In conflicts, distinguish between facts and interpretations",
                    "Before making decisions, separate emotions from clear thinking"
                ],
                progress_indicators=[
                    "Moments of spontaneous clarity in daily life",
                    "Increased ability to pause before reacting",
                    "Growing awareness of thoughts and emotions as temporary"
                ]
            ),
            
            ClarityLevel.QUESTIONING: ClarityGuidance(
                level=ClarityLevel.QUESTIONING,
                primary_teaching="Questioning is the beginning of wisdom. Use doubt as a tool for discovery. Question everything until you reach the unquestionable truth of your own being.",
                viveka_practices=[
                    "Practice Neti-Neti (Not this, not this) inquiry",
                    "Question the reality of thoughts and mental concepts",
                    "Distinguish between direct experience and beliefs",
                    "Study the four-fold discrimination of Vedanta"
                ],
                vairagya_practices=[
                    "Practice witnessing desires without immediately acting",
                    "Cultivate inner contentment independent of circumstances",
                    "Release attachment to being right or wrong",
                    "Practice equanimity in success and failure"
                ],
                daily_clarity_routine=[
                    "Morning: Self-inquiry meditation - 'Who am I?'",
                    "Throughout day: Question automatic thoughts and reactions",
                    "Evening: Discrimination practice - real vs. unreal in the day",
                    "Night: Practice Yoga Nidra for deepening awareness"
                ],
                obstacle_solutions={
                    "Doubt leading to paralysis": "Use doubt as inquiry tool, not stopping point",
                    "Mental loops": "Practice witnessing the questioner itself",
                    "Spiritual confusion": "Return to simple direct experience"
                },
                discrimination_exercises=[
                    "Ramana's 'Who am I?' inquiry practice",
                    "Distinguish between the perceiver and perceptions",
                    "Practice seeing through the stories mind creates"
                ],
                scriptural_references=[
                    "Katha Upanishad: 'When all desires in the heart are abandoned, the mortal becomes immortal'",
                    "Mandukya Upanishad: Teaching on the four states of consciousness"
                ],
                practical_applications=[
                    "Question assumptions before making important decisions",
                    "Use conflicts as opportunities to practice discrimination",
                    "In relationships, distinguish between person and projections"
                ],
                progress_indicators=[
                    "Ability to hold questions without immediately seeking answers",
                    "Growing comfort with uncertainty and not-knowing",
                    "Spontaneous insights arising from inquiry practice"
                ]
            ),
            
            ClarityLevel.DISCERNING: ClarityGuidance(
                level=ClarityLevel.DISCERNING,
                primary_teaching="You are developing the eye of wisdom (Jnana Chaksu). Trust your deepening ability to discriminate truth from falsehood, real from apparent.",
                viveka_practices=[
                    "Advanced Atma-Anatma Viveka (Self vs. not-Self discrimination)",
                    "Practice distinguishing between awareness and its contents",
                    "Develop discrimination between ego-mind and Buddhi",
                    "Study and apply the discriminations of Kashmir Shaivism"
                ],
                vairagya_practices=[
                    "Cultivate detachment from outcomes while remaining engaged",
                    "Practice inner surrender while taking appropriate action",
                    "Develop equanimity toward praise and criticism",
                    "Release attachment to spiritual experiences"
                ],
                daily_clarity_routine=[
                    "Morning: Extended Viveka meditation and self-inquiry",
                    "Midday: Practice presence and discrimination in activities",
                    "Evening: Deep contemplation on the nature of reality",
                    "Integration: Apply discrimination in all relationships and decisions"
                ],
                obstacle_solutions={
                    "Spiritual pride": "Remember that clarity is grace, not personal achievement",
                    "Bypass of emotions": "Include feelings in awareness without being controlled by them",
                    "Intellectual understanding only": "Emphasize direct experience over concepts"
                },
                discrimination_exercises=[
                    "Practice of Sakshi Bhava (witness consciousness)",
                    "Discrimination between different levels of mind",
                    "Advanced inquiry into the nature of the 'I'"
                ],
                scriptural_references=[
                    "Viveka Chudamani: 'Discrimination is the foremost among all means to liberation'",
                    "Bhagavad Gita 7.19: 'After many births, the wise one takes refuge in Me'"
                ],
                practical_applications=[
                    "Use discrimination in complex life decisions",
                    "Help others develop their own clarity without imposing",
                    "Apply Viveka in understanding spiritual teachings"
                ],
                progress_indicators=[
                    "Consistent ability to distinguish between ego reactions and wisdom",
                    "Natural arising of appropriate responses in challenging situations",
                    "Growing alignment between understanding and living"
                ]
            ),
            
            ClarityLevel.CLEAR: ClarityGuidance(
                level=ClarityLevel.CLEAR,
                primary_teaching="You abide more and more in clear seeing. The clouds of ignorance are dispersing. Rest in the natural state of awareness that you are.",
                viveka_practices=[
                    "Effortless discrimination arising spontaneously",
                    "Recognition of awareness as your true nature",
                    "Dissolution of subtle subject-object duality",
                    "Integration of relative and absolute understanding"
                ],
                vairagya_practices=[
                    "Natural detachment without forcing or suppressing",
                    "Spontaneous contentment and inner fullness",
                    "Freedom from the compulsion to change anything",
                    "Effortless letting go as understanding deepens"
                ],
                daily_clarity_routine=[
                    "Living from presence rather than scheduled practices",
                    "Continuous recognition of awareness in all activities",
                    "Natural meditation throughout the day",
                    "Serving others from overflow of clarity"
                ],
                obstacle_solutions={
                    "Subtle attachment to clarity": "Let go of even the achievement of clarity",
                    "Responsibility of clear seeing": "Serve spontaneously without burden",
                    "Others' expectations": "Remain authentic to your own understanding"
                },
                discrimination_exercises=[
                    "Effortless abiding in witness consciousness",
                    "Spontaneous recognition of the Self in all",
                    "Natural dissolution of problems through clear seeing"
                ],
                scriptural_references=[
                    "Ashtavakra Gita: 'You are the one witness of everything and are always free'",
                    "Advaita teaching: 'I am That (Tat Tvam Asi)'"
                ],
                practical_applications=[
                    "Live as an example of clarity without claiming special status",
                    "Offer guidance naturally when requested",
                    "Handle complex situations with spontaneous wisdom"
                ],
                progress_indicators=[
                    "Effortless peace regardless of external circumstances",
                    "Natural wisdom arising to meet each situation",
                    "Freedom from the need to maintain or defend any position"
                ]
            ),
            
            ClarityLevel.ILLUMINATED: ClarityGuidance(
                level=ClarityLevel.ILLUMINATED,
                primary_teaching="You are established in Sahaja Samadhi - the natural state. Clarity is no longer something you have, but what you are. You are the light of awareness itself.",
                viveka_practices=[
                    "No separation between practice and being",
                    "Discrimination happens through you, not by you",
                    "Clear seeing is your natural function",
                    "You are the Viveka by which others discriminate"
                ],
                vairagya_practices=[
                    "Complete freedom without rejecting anything",
                    "Natural non-attachment as expression of fullness",
                    "Spontaneous right action without personal agenda",
                    "Love without possessiveness or condition"
                ],
                daily_clarity_routine=[
                    "No routine - life itself is the practice",
                    "Continuous recognition of the Self in all",
                    "Spontaneous service as divine play",
                    "Teaching through presence and being"
                ],
                obstacle_solutions={
                    "Appearance of obstacles": "Even obstacles are recognized as divine play",
                    "Others' projections": "Remain unaffected while compassionately responsive",
                    "Responsibility of embodiment": "Serve naturally without sense of burden"
                },
                discrimination_exercises=[
                    "You are the awareness in which all discrimination appears",
                    "Recognition that seeker, seeking, and sought are one",
                    "Natural effortless abiding as the Self"
                ],
                scriptural_references=[
                    "Ribhu Gita: 'I am the Self, you are the Self, all this is nothing but the Self'",
                    "Kashmir Shaivism: 'I am Shiva, this world is my divine play'"
                ],
                practical_applications=[
                    "Embody clarity for the benefit of all beings",
                    "Serve as a clear mirror for others' self-recognition",
                    "Live as divine grace in human form"
                ],
                progress_indicators=[
                    "No sense of progress - you are what you sought",
                    "Natural compassion without effort or intention",
                    "Others recognize their own clarity in your presence"
                ]
            )
        }
    
    def _initialize_clarity_practices(self) -> Dict[ClarityAspect, List[str]]:
        """Initialize specific practices for each aspect of clarity"""
        return {
            ClarityAspect.VIVEKA: [
                "Daily discrimination between real and unreal",
                "Neti-Neti (Not this, not this) practice",
                "Study of Viveka Chudamani with application",
                "Practice distinguishing Seer from seen"
            ],
            
            ClarityAspect.VAIRAGYA: [
                "Witnessing desires without compulsive action",
                "Practicing contentment with present circumstances",
                "Releasing attachment to outcomes",
                "Cultivating inner satisfaction independent of externals"
            ],
            
            ClarityAspect.BUDDHI: [
                "Developing pure intelligence beyond emotional reactivity",
                "Practicing discernment in decision-making",
                "Strengthening intuitive wisdom",
                "Cultivating inner knowing beyond mental concepts"
            ],
            
            ClarityAspect.INTUITION: [
                "Sitting in silence to access inner knowing",
                "Trusting first insights before mental analysis",
                "Developing sensitivity to subtle guidance",
                "Practicing immediate recognition of truth"
            ],
            
            ClarityAspect.PRESENCE: [
                "Maintaining awareness in all activities",
                "Practicing witness consciousness throughout the day",
                "Returning attention to the present moment",
                "Developing continuous mindfulness"
            ],
            
            ClarityAspect.DISCRIMINATION: [
                "Daily practice of separating facts from interpretations",
                "Distinguishing between ego reactions and wisdom responses",
                "Learning to discern truth from conditioning",
                "Practicing clear perception in relationships"
            ]
        }
    
    def _initialize_discrimination_exercises(self) -> List[str]:
        """Initialize specific discrimination exercises"""
        return [
            "Who Am I? (Ko'ham) inquiry practice",
            "Witnessing thoughts without identification",
            "Distinguishing between awareness and its contents",
            "Practicing Sakshi Bhava (witness consciousness)",
            "Separating the observer from observed",
            "Discriminating between reactions and responses",
            "Recognizing the changeless in the changing",
            "Distinguishing between person and presence",
            "Seeing through mental stories and narratives",
            "Practicing direct perception beyond concepts"
        ]
    
    def _initialize_obstacle_solutions(self) -> Dict[ClarityObstacle, Dict[str, Any]]:
        """Initialize solutions for clarity obstacles"""
        return {
            ClarityObstacle.MAYA_CONFUSION: {
                "description": "Caught in illusion and unable to see clearly",
                "solutions": [
                    "Start with simple discrimination exercises",
                    "Study teachings on Maya and its nature",
                    "Practice witnessing thoughts and emotions",
                    "Seek guidance from clear teachers"
                ],
                "practices": ["Daily Viveka meditation", "Study of Advaita texts"],
                "mantras": ["सत्यम् (Satyam - Truth)", "विवेक (Viveka - Discrimination)"]
            },
            
            ClarityObstacle.EMOTIONAL_REACTIVITY: {
                "description": "Emotions clouding clear perception and judgment",
                "solutions": [
                    "Practice witnessing emotions without suppression",
                    "Develop space between stimulus and response",
                    "Learn to include emotions in awareness",
                    "Cultivate emotional equanimity"
                ],
                "practices": ["Emotion witnessing meditation", "Pranayama for stability"],
                "mantras": ["शांति (Shanti - Peace)", "साक्षी (Sakshi - Witness)"]
            },
            
            ClarityObstacle.MENTAL_NOISE: {
                "description": "Overthinking and mental chatter preventing clear perception",
                "solutions": [
                    "Practice meditation to calm the mind",
                    "Learn to distinguish between useful and useless thinking",
                    "Develop skill in attention management",
                    "Cultivate periods of mental silence"
                ],
                "practices": ["Trataka (candle gazing)", "Mantra repetition"],
                "mantras": ["मौन (Mauna - Silence)", "ॐ (Om - Primordial sound)"]
            },
            
            ClarityObstacle.ATTACHMENT: {
                "description": "Clinging to outcomes and possessions clouding judgment",
                "solutions": [
                    "Practice Vairagya (dispassion)",
                    "Study teachings on non-attachment",
                    "Cultivate inner contentment",
                    "Practice letting go regularly"
                ],
                "practices": ["Detachment meditation", "Gratitude practice"],
                "mantras": ["वैराग्य (Vairagya - Dispassion)", "संतोष (Santosha - Contentment)"]
            }
        }
    
    def assess_clarity_level(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> ClarityLevel:
        """Assess user's current level of clarity"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for illuminated level indicators
        if any(word in query_lower for word in ["established", "natural state", "sahaja", "embodied clarity"]):
            return ClarityLevel.ILLUMINATED
        
        # Check for clear level indicators
        if any(word in query_lower for word in ["consistent clarity", "clear seeing", "established practice"]):
            return ClarityLevel.CLEAR
        
        # Check for discerning level indicators  
        if any(word in query_lower for word in ["discrimination", "viveka", "developing clarity", "deepening"]):
            return ClarityLevel.DISCERNING
        
        # Check for questioning level indicators
        if any(word in query_lower for word in ["questioning", "doubting", "inquiry", "who am i"]):
            return ClarityLevel.QUESTIONING
        
        # Default to confused level
        return ClarityLevel.CONFUSED
    
    def identify_clarity_obstacles(self, query: str, context: Dict[str, Any]) -> List[ClarityObstacle]:
        """Identify clarity obstacles mentioned in query"""
        obstacles = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["confused", "illusion", "maya", "unclear"]):
            obstacles.append(ClarityObstacle.MAYA_CONFUSION)
        
        if any(word in query_lower for word in ["emotional", "reactive", "overwhelmed", "upset"]):
            obstacles.append(ClarityObstacle.EMOTIONAL_REACTIVITY)
        
        if any(word in query_lower for word in ["thinking", "mental", "overthinking", "analysis"]):
            obstacles.append(ClarityObstacle.MENTAL_NOISE)
        
        if any(word in query_lower for word in ["attached", "clinging", "possessive", "wanting"]):
            obstacles.append(ClarityObstacle.ATTACHMENT)
        
        if any(word in query_lower for word in ["fear", "anxiety", "worried", "scared"]):
            obstacles.append(ClarityObstacle.FEAR_BASED_THINKING)
        
        return obstacles if obstacles else [ClarityObstacle.MAYA_CONFUSION]
    
    def get_obstacle_solutions(self, obstacles: List[ClarityObstacle]) -> Dict[str, str]:
        """Get solutions for identified obstacles"""
        solutions = {}
        
        for obstacle in obstacles:
            obstacle_data = self.obstacle_solutions.get(obstacle, {})
            solutions[obstacle.value] = "; ".join(obstacle_data.get("solutions", ["Practice discrimination and seek guidance"]))
        
        return solutions
    
    def get_meditation_guidance(self, level: ClarityLevel) -> str:
        """Get specific meditation guidance for clarity level"""
        guidance_map = {
            ClarityLevel.CONFUSED: "Start with simple breath awareness. Notice the one who is aware of breathing.",
            ClarityLevel.QUESTIONING: "Practice self-inquiry: 'Who am I?' Keep returning to the questioner.",
            ClarityLevel.DISCERNING: "Abide as witness consciousness. Rest as awareness itself.",
            ClarityLevel.CLEAR: "Natural meditation - no effort needed, just abide as you are.",
            ClarityLevel.ILLUMINATED: "You are meditation itself. No separation between meditator and meditation."
        }
        return guidance_map.get(level, "Begin with simple awareness of awareness")
    
    async def process_clarity_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> ClarityResponse:
        """Process clarity-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess clarity aspects
            level = self.assess_clarity_level(query, context)
            obstacles = self.identify_clarity_obstacles(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            obstacle_solutions = self.get_obstacle_solutions(obstacles)
            meditation_guidance = self.get_meditation_guidance(level)
            
            # Select appropriate scriptural reference
            scriptural_wisdom = guidance.scriptural_references[0] if guidance.scriptural_references else "Tat Tvam Asi - Thou Art That"
            
            return ClarityResponse(
                clarity_level=level.value,
                viveka_teaching=guidance.primary_teaching,
                vairagya_guidance=f"Practice detachment: {guidance.vairagya_practices[0] if guidance.vairagya_practices else 'Witness without attachment'}",
                daily_practices=guidance.daily_clarity_routine,
                discrimination_exercises=guidance.discrimination_exercises,
                obstacle_solutions=obstacle_solutions,
                scriptural_wisdom=scriptural_wisdom,
                practical_applications=guidance.practical_applications,
                progress_indicators=guidance.progress_indicators,
                meditation_guidance=meditation_guidance
            )
            
        except Exception as e:
            logger.error(f"Error processing clarity query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> ClarityResponse:
        """Create fallback response when processing fails"""
        return ClarityResponse(
            clarity_level="confused",
            viveka_teaching="Begin by recognizing that confusion is the starting point of wisdom. The very recognition that you are confused is the first glimmer of clarity.",
            vairagya_guidance="Practice witnessing your experiences without immediately judging them as good or bad.",
            daily_practices=[
                "Morning: Ask 'Who am I?' and sit in silence",
                "Throughout day: Practice witnessing thoughts and emotions",
                "Evening: Distinguish between what is real and what is just mental interpretation"
            ],
            discrimination_exercises=self.discrimination_exercises,
            obstacle_solutions={
                "confusion": "Start with simple discrimination between thoughts and awareness",
                "overwhelm": "Focus on one moment at a time, rest in present awareness"
            },
            scriptural_wisdom="Viveka Chudamani: 'Discrimination is the foremost among all means to liberation'",
            practical_applications=[
                "Before reacting, pause and ask what is actually happening",
                "Distinguish between facts and your interpretations",
                "Practice seeing the changeless awareness behind changing experiences"
            ],
            progress_indicators=[
                "Moments of spontaneous clarity",
                "Increased ability to pause before reacting",
                "Growing sense of inner peace and stability"
            ],
            meditation_guidance="Start with simple breath awareness. Notice the one who is aware of breathing."
        )
    
    def get_clarity_insight(self, aspect: ClarityAspect) -> Optional[ClarityInsight]:
        """Get specific insight about a clarity aspect"""
        aspect_teachings = {
            ClarityAspect.VIVEKA: "Discrimination between real and unreal is the foundation of wisdom",
            ClarityAspect.VAIRAGYA: "Detachment brings freedom and natural contentment",
            ClarityAspect.BUDDHI: "Pure intelligence operates beyond emotional reactivity",
            ClarityAspect.INTUITION: "Direct knowing arises when the mind is still and receptive"
        }
        
        teaching = aspect_teachings.get(aspect, "Cultivate clear seeing in all circumstances")
        practices = self.clarity_practices.get(aspect, ["Practice mindful awareness"])
        
        return ClarityInsight(
            aspect=aspect,
            teaching=teaching,
            practice_method=practices[0] if practices else "Practice mindful awareness",
            immediate_application="Apply this understanding to your current situation",
            daily_integration="Integrate this wisdom into your daily routine"
        )

# Global instance
_clarity_module = None

def get_clarity_module() -> ClarityModule:
    """Get global Clarity module instance"""
    global _clarity_module
    if _clarity_module is None:
        _clarity_module = ClarityModule()
    return _clarity_module

# Factory function for easy access
def create_clarity_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> ClarityResponse:
    """Factory function to create clarity guidance"""
    import asyncio
    module = get_clarity_module()
    return asyncio.run(module.process_clarity_query(query, user_context))
