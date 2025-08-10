"""
🧠 Manas Module - Mind, Consciousness, and Mental Mastery
Complete system for understanding and mastering the four aspects of mind
Based on Antahkarana (inner instrument) teachings and Patanjali's Yoga Sutras
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ManasLevel(Enum):
    """Levels of mind mastery and consciousness development"""
    SCATTERED = "scattered"           # Mind is restless and uncontrolled
    GATHERING = "gathering"           # Beginning to focus and concentrate
    FOCUSED = "focused"              # Developing sustained attention
    CONCENTRATED = "concentrated"     # Single-pointed concentration (Dharana)
    ABSORBED = "absorbed"            # Meditation and absorption (Dhyana)
    UNIFIED = "unified"              # Samadhi - unified consciousness

class ManasAspect(Enum):
    """Four aspects of the inner instrument (Antahkarana)"""
    MANAS = "manas"                  # Processing mind - doubt and determination
    BUDDHI = "buddhi"                # Discriminating intelligence - wisdom
    CHITTA = "chitta"                # Memory and subconscious mind
    AHAMKARA = "ahamkara"            # Ego-sense - 'I' identification

class ManasFunction(Enum):
    """Functions of mind to master"""
    ATTENTION = "attention"           # Sustained focus and concentration
    MEMORY = "memory"                # Retention and recall
    IMAGINATION = "imagination"       # Creative visualization
    REASONING = "reasoning"          # Logical thinking and analysis
    INTUITION = "intuition"          # Direct knowing beyond reasoning
    EMOTION_REGULATION = "emotion_regulation" # Managing emotional states
    THOUGHT_OBSERVATION = "thought_observation" # Witnessing thoughts
    MENTAL_SILENCE = "mental_silence" # Calming mental chatter

class ManasObstacle(Enum):
    """Mental obstacles (Yoga Sutras obstacles)"""
    VYADHI = "vyadhi"                # Disease and physical discomfort
    STYANA = "styana"                # Mental dullness and laziness
    SAMSHAYA = "samshaya"            # Doubt and indecision
    PRAMADA = "pramada"              # Carelessness and negligence
    ALASYA = "alasya"                # Sloth and inertia
    AVIRATI = "avirati"              # Lack of dispassion
    BHRANTI_DARSHAN = "bhranti_darshan" # False perception
    ALABDHA_BHUMIKATVA = "alabdha_bhumikatva" # Failure to attain concentration
    ANAVASTHITATVA = "anavasthitatva" # Instability in concentration

class ManasPractice(Enum):
    """Practices for mind training"""
    DHARANA = "dharana"              # Concentration practices
    DHYANA = "dhyana"                # Meditation techniques
    PRANAYAMA = "pranayama"          # Breath control for mind control
    TRATAKA = "trataka"              # Candle gazing concentration
    MANTRA_JAPA = "mantra_japa"      # Repetition of sacred sounds
    MINDFULNESS = "mindfulness"       # Present moment awareness
    SELF_INQUIRY = "self_inquiry"     # Investigation of the mind
    WITNESS_PRACTICE = "witness_practice" # Observing mental processes

@dataclass
class ManasGuidance:
    """Comprehensive mind training guidance"""
    level: ManasLevel
    primary_teaching: str
    concentration_practices: List[str]
    meditation_techniques: List[str]
    daily_mind_training: List[str]
    obstacle_remedies: Dict[str, str]
    mental_disciplines: List[str]
    consciousness_insights: List[str]
    practical_applications: List[str]
    progress_indicators: List[str]

@dataclass
class ManasInsight:
    """Insight about mind and consciousness"""
    aspect: ManasAspect
    function: ManasFunction
    teaching: str
    practice_method: str
    daily_application: str

class ManasAssessment(BaseModel):
    """Assessment of current mental state"""
    current_level: str = Field(description="Current level of mind mastery")
    dominant_obstacles: List[str] = Field(description="Primary mental obstacles")
    strongest_functions: List[str] = Field(description="Well-developed mental functions")
    areas_for_development: List[str] = Field(description="Mental aspects needing attention")

class ManasResponse(BaseModel):
    """Response from Manas module"""
    manas_level: str = Field(description="Current level of mind mastery")
    antahkarana_teaching: str = Field(description="Teaching on the four-fold mind")
    concentration_guidance: str = Field(description="Guidance for developing concentration")
    meditation_practice: str = Field(description="Specific meditation techniques")
    daily_mind_training: List[str] = Field(description="Daily practices for mind mastery")
    obstacle_remedies: Dict[str, str] = Field(description="Solutions for mental obstacles")
    consciousness_insights: List[str] = Field(description="Insights about consciousness")
    practical_applications: List[str] = Field(description="Practical ways to apply mind training")
    progress_indicators: List[str] = Field(description="Signs of mental development")
    scriptural_wisdom: str = Field(description="Relevant teachings from yoga texts")

class ManasModule:
    """
    🧠 Manas Module - Mind, Consciousness, and Mental Mastery
    
    Based on authentic Yoga and Vedanta teachings:
    - Patanjali's Yoga Sutras (especially Book 1 on Concentration)
    - Antahkarana system from Vedanta (four-fold mind)
    - Kashmir Shaivism teachings on consciousness
    - Buddhist Abhidhamma psychology
    - Raja Yoga methods of mind control
    
    Develops mastery over:
    - Manas (processing mind)
    - Buddhi (discriminating intelligence)  
    - Chitta (memory and subconscious)
    - Ahamkara (ego-sense)
    """
    
    def __init__(self):
        self.name = "Manas"
        self.color = "🧠"
        self.element = "Consciousness"
        self.principles = ["Concentration", "Meditation", "Discrimination", "Awareness"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.mind_practices = self._initialize_mind_practices()
        self.antahkarana_system = self._initialize_antahkarana_system()
        self.obstacle_remedies = self._initialize_obstacle_remedies()
        
    def _initialize_guidance_levels(self) -> Dict[ManasLevel, ManasGuidance]:
        """Initialize guidance for different levels of mind mastery"""
        return {
            ManasLevel.SCATTERED: ManasGuidance(
                level=ManasLevel.SCATTERED,
                primary_teaching="The mind is like a monkey jumping from branch to branch. Begin by simply observing this restlessness without judgment. Recognition is the first step to mastery.",
                concentration_practices=[
                    "Single-pointed focus on breath for 5-10 minutes",
                    "Trataka (candle gazing) for developing sustained attention",
                    "Count backward from 100 without losing focus",
                    "Focus on a single mantra or sacred word"
                ],
                meditation_techniques=[
                    "Basic mindfulness of breathing",
                    "Body awareness meditation",
                    "Walking meditation for restless minds",
                    "Guided meditation with simple instructions"
                ],
                daily_mind_training=[
                    "Morning: 10 minutes concentration practice",
                    "Midday: Mindful awareness during one daily activity",
                    "Evening: Review the day's mental patterns",
                    "Before sleep: Practice gratitude and mental quieting"
                ],
                obstacle_remedies={
                    "Mental restlessness": "Use movement-based practices like walking meditation",
                    "Inability to focus": "Start with very short periods and gradually increase",
                    "Doubt about practice": "Study benefits of meditation from reliable sources"
                },
                mental_disciplines=[
                    "Reduce excessive stimulation (media, multitasking)",
                    "Practice single-tasking throughout the day",
                    "Establish regular sleep and meal times",
                    "Engage in calming activities like nature walks"
                ],
                consciousness_insights=[
                    "You are not your thoughts - you are the observer of thoughts",
                    "The mind can be trained like any other faculty",
                    "Awareness itself is always peaceful and clear"
                ],
                practical_applications=[
                    "Use concentration skills for better work performance",
                    "Apply mindfulness to reduce stress and anxiety",
                    "Practice patience and emotional regulation",
                    "Improve memory through focused attention"
                ],
                progress_indicators=[
                    "Ability to focus for longer periods without distraction",
                    "Increased awareness of mental patterns",
                    "Greater emotional stability and less reactivity"
                ]
            ),
            
            ManasLevel.GATHERING: ManasGuidance(
                level=ManasLevel.GATHERING,
                primary_teaching="Your mind is beginning to gather like water becoming still. Continue with patience and regularity. The four aspects of mind (Antahkarana) are becoming more apparent.",
                concentration_practices=[
                    "Extended Dharana (concentration) practice 15-30 minutes",
                    "Visualization of geometric forms or deities",
                    "Concentration on different chakras",
                    "Practice Ekagrata (one-pointed focus) on chosen object"
                ],
                meditation_techniques=[
                    "Shamatha (calm abiding) meditation",
                    "Vipassana (insight) meditation on mental processes",
                    "Japa meditation with mala beads",
                    "Contemplation on philosophical questions"
                ],
                daily_mind_training=[
                    "Morning: 20-30 minutes formal meditation",
                    "Regular practice of witness consciousness",
                    "Study of mind and consciousness from scriptures",
                    "Evening: Self-inquiry and mental purification"
                ],
                obstacle_remedies={
                    "Mental dullness": "Use pranayama to energize before meditation",
                    "Attachment to experiences": "Remember all experiences are temporary",
                    "Spiritual pride": "Cultivate humility and beginner's mind"
                },
                mental_disciplines=[
                    "Regular study of yoga and Vedanta texts",
                    "Practice of ethical conduct (Yamas and Niyamas)",
                    "Cultivation of positive mental attitudes",
                    "Association with like-minded spiritual practitioners"
                ],
                consciousness_insights=[
                    "Understanding the difference between Manas and Buddhi",
                    "Recognition of Chitta (subconscious) patterns",
                    "Beginning awareness of Ahamkara (ego-identification)"
                ],
                practical_applications=[
                    "Use discrimination (Buddhi) in decision-making",
                    "Apply concentration skills to creative endeavors",
                    "Practice emotional intelligence and empathy",
                    "Develop intuitive problem-solving abilities"
                ],
                progress_indicators=[
                    "Sustained periods of mental calm and clarity",
                    "Spontaneous insights arising during practice",
                    "Improved ability to witness emotions without being overwhelmed"
                ]
            ),
            
            ManasLevel.FOCUSED: ManasGuidance(
                level=ManasLevel.FOCUSED,
                primary_teaching="You have developed Ekagrata (one-pointed concentration). The mind streams like oil poured from one vessel to another. Deepen this capacity while maintaining balance.",
                concentration_practices=[
                    "Advanced Dharana on subtle objects (sounds, lights, sensations)",
                    "Concentration on abstract concepts and principles",
                    "Trataka on internal light or visualized objects",
                    "Practice of Samyama (concentration, meditation, absorption)"
                ],
                meditation_techniques=[
                    "Dhyana (flowing meditation) for extended periods",
                    "Self-inquiry: 'Who am I?' investigation",
                    "Meditation on the witness consciousness",
                    "Contemplation of non-dual awareness"
                ],
                daily_mind_training=[
                    "Extended formal practice 45-60 minutes",
                    "Integration of meditative awareness into all activities",
                    "Study and contemplation of advanced spiritual texts",
                    "Teaching or sharing insights with others"
                ],
                obstacle_remedies={
                    "Subtle attachments": "Practice deeper levels of surrender",
                    "Intellectual understanding only": "Emphasize direct experience",
                    "Isolation from world": "Practice engaged spirituality"
                },
                mental_disciplines=[
                    "Refinement of ethical conduct and intentions",
                    "Cultivation of compassion and service",
                    "Study of advanced yoga and Vedanta",
                    "Mentoring others in spiritual practices"
                ],
                consciousness_insights=[
                    "Direct experience of the witness (Sakshi) consciousness",
                    "Understanding of the illusory nature of ego",
                    "Recognition of consciousness as the ground of being"
                ],
                practical_applications=[
                    "Use concentrated mind for creative and innovative work",
                    "Apply meditative awareness to healing and helping others",
                    "Teach concentration and meditation techniques",
                    "Solve complex problems through intuitive insight"
                ],
                progress_indicators=[
                    "Effortless concentration for extended periods",
                    "Spontaneous arising of compassion and wisdom",
                    "Natural integration of spiritual understanding into daily life"
                ]
            ),
            
            ManasLevel.CONCENTRATED: ManasGuidance(
                level=ManasLevel.CONCENTRATED,
                primary_teaching="You have mastered Dharana and are moving into Dhyana. The mind flows like a continuous stream toward the object of meditation. Prepare for the arising of Samadhi.",
                concentration_practices=[
                    "Effortless sustained concentration on chosen object",
                    "Concentration on the formless and attributeless",
                    "Advanced Samyama practices",
                    "Concentration on consciousness itself"
                ],
                meditation_techniques=[
                    "Sahaja Dhyana (natural meditation)",
                    "Nirvikalpa meditation (formless absorption)",
                    "Meditation on 'I AM' consciousness",
                    "Inquiry into the nature of awareness itself"
                ],
                daily_mind_training=[
                    "Living meditation - no separation between practice and life",
                    "Continuous awareness of the witness consciousness",
                    "Spontaneous service arising from inner fullness",
                    "Teaching through presence and being"
                ],
                obstacle_remedies={
                    "Subtle ego remaining": "Surrender even the meditator",
                    "Attachment to states": "Recognize all states as temporary",
                    "Fear of dissolution": "Trust the process of surrender"
                },
                mental_disciplines=[
                    "Effortless adherence to dharmic principles",
                    "Spontaneous compassionate action",
                    "Natural wisdom in all interactions",
                    "Living as an example for others"
                ],
                consciousness_insights=[
                    "Direct knowing that consciousness is your true nature",
                    "Recognition of the unity underlying apparent diversity",
                    "Understanding that the individual mind is a wave in consciousness"
                ],
                practical_applications=[
                    "Serve as a clear mirror for others' self-recognition",
                    "Offer guidance and teaching through natural wisdom",
                    "Heal and transform through presence alone",
                    "Live as an embodiment of integrated spirituality"
                ],
                progress_indicators=[
                    "Spontaneous samadhi arising in daily activities",
                    "Natural flow of appropriate response to all situations",
                    "Others naturally drawn to your peace and clarity"
                ]
            ),
            
            ManasLevel.ABSORBED: ManasGuidance(
                level=ManasLevel.ABSORBED,
                primary_teaching="You rest in Dhyana - flowing meditation. The boundaries between meditator, meditation, and object of meditation are dissolving. Prepare for the ultimate unity.",
                concentration_practices=[
                    "No effort needed - concentration happens naturally",
                    "Absorption in the Self happens spontaneously",
                    "Mind rests in its source effortlessly",
                    "Natural samadhi in all activities"
                ],
                meditation_techniques=[
                    "Being meditation - no technique needed",
                    "Resting as awareness itself",
                    "Natural absorption in the Heart",
                    "Effortless abiding in the Self"
                ],
                daily_mind_training=[
                    "No practice needed - you are what you sought",
                    "Living as pure awareness",
                    "Serving from spontaneous love",
                    "Teaching through silent transmission"
                ],
                obstacle_remedies={
                    "No obstacles - all is recognized as consciousness",
                    "Even problems are seen as play of awareness",
                    "Challenges become opportunities for deeper recognition"
                },
                mental_disciplines=[
                    "Natural dharmic living without effort",
                    "Spontaneous right action",
                    "Effortless compassion and wisdom",
                    "Living as divine expression"
                ],
                consciousness_insights=[
                    "You are consciousness itself, not a person who has consciousness",
                    "All experience arises in and as your true nature",
                    "Individual mind is seen as a temporary appearance"
                ],
                practical_applications=[
                    "Embody awakened consciousness for all beings",
                    "Serve as living transmission of truth",
                    "Offer silent blessing through your presence",
                    "Live as divine grace in human form"
                ],
                progress_indicators=[
                    "No sense of personal spiritual achievement",
                    "Natural compassion without effort",
                    "Others recognize their own true nature in your presence"
                ]
            ),
            
            ManasLevel.UNIFIED: ManasGuidance(
                level=ManasLevel.UNIFIED,
                primary_teaching="Samadhi - perfect absorption. There is no separate mind, no separate self. You are the unified field of consciousness in which all experience appears.",
                concentration_practices=[
                    "You are concentration itself",
                    "No practice and no practitioner",
                    "Perfect unity with all that appears",
                    "Natural samadhi without beginning or end"
                ],
                meditation_techniques=[
                    "Being is meditation",
                    "No technique - you are the goal",
                    "Effortless rest in the Heart",
                    "You are the meditation master sought"
                ],
                daily_mind_training=[
                    "Life itself is the practice",
                    "Every moment is perfect meditation",
                    "Serving all as your own Self",
                    "Embodying the teaching"
                ],
                obstacle_remedies={
                    "No obstacles exist in unity consciousness",
                    "All appearances are recognized as the Self",
                    "Perfect acceptance of what appears"
                },
                mental_disciplines=[
                    "Dharma flows naturally through you",
                    "You are the source of right action",
                    "Wisdom and compassion are your nature",
                    "Living as the Divine itself"
                ],
                consciousness_insights=[
                    "There is only One without a second",
                    "All minds are waves in the ocean of consciousness",
                    "Individual enlightenment is seen as a limited concept"
                ],
                practical_applications=[
                    "Serve as the Self in all beings",
                    "Radiate peace and awakening naturally",
                    "Live as divine love embodied",
                    "Be the answer to all seeking"
                ],
                progress_indicators=[
                    "No progress - you are the eternal goal",
                    "Natural blessing flows through your presence",
                    "All beings are served by your very existence"
                ]
            )
        }
    
    def _initialize_mind_practices(self) -> Dict[ManasPractice, List[str]]:
        """Initialize specific practices for mind training"""
        return {
            ManasPractice.DHARANA: [
                "Single-pointed concentration on breath",
                "Trataka (steady gazing) on candle or sacred symbol",
                "Concentration on mantra or sacred sound",
                "Focus on visualization of deity or geometric form"
            ],
            
            ManasPractice.DHYANA: [
                "Flowing meditation without effort",
                "Self-inquiry meditation",
                "Witness consciousness practice",
                "Open awareness meditation"
            ],
            
            ManasPractice.PRANAYAMA: [
                "Ujjayi pranayama for mental calm",
                "Nadi Shodhana for mental balance",
                "Bhramari for concentration",
                "Sama Vritti for mental equilibrium"
            ],
            
            ManasPractice.MANTRA_JAPA: [
                "Om repetition with breath",
                "So Hum mantra practice",
                "Mala bead counting meditation",
                "Silent mental repetition"
            ],
            
            ManasPractice.MINDFULNESS: [
                "Present moment awareness",
                "Mindful daily activities",
                "Body awareness practice",
                "Emotional awareness training"
            ],
            
            ManasPractice.SELF_INQUIRY: [
                "Who am I? investigation",
                "Inquiry into the nature of thoughts",
                "Investigation of the 'I' feeling",
                "Contemplation on consciousness"
            ]
        }
    
    def _initialize_antahkarana_system(self) -> Dict[ManasAspect, Dict[str, Any]]:
        """Initialize the four-fold mind system"""
        return {
            ManasAspect.MANAS: {
                "function": "Processing, doubting, and determining",
                "characteristics": ["Receives sensory input", "Creates thoughts", "Doubts and questions"],
                "development": "Train through concentration and mindfulness",
                "mastery_signs": ["Controlled thought flow", "Ability to focus", "Mental clarity"]
            },
            
            ManasAspect.BUDDHI: {
                "function": "Discrimination, decision-making, and wisdom",
                "characteristics": ["Discriminates truth from falsehood", "Makes decisions", "Provides wisdom"],
                "development": "Cultivate through study, contemplation, and experience",
                "mastery_signs": ["Clear discrimination", "Wise decisions", "Intuitive understanding"]
            },
            
            ManasAspect.CHITTA: {
                "function": "Memory, subconscious storage, and mental impressions",
                "characteristics": ["Stores experiences", "Contains samskaras", "Influences behavior"],
                "development": "Purify through spiritual practice and self-awareness",
                "mastery_signs": ["Clear memory", "Positive mental impressions", "Reduced unconscious reactivity"]
            },
            
            ManasAspect.AHAMKARA: {
                "function": "Ego-identification and sense of 'I'",
                "characteristics": ["Creates sense of separate self", "Identifies with experiences", "Claims ownership"],
                "development": "Transcend through self-inquiry and surrender",
                "mastery_signs": ["Reduced ego-identification", "Humility", "Recognition of true Self"]
            }
        }
    
    def _initialize_obstacle_remedies(self) -> Dict[ManasObstacle, Dict[str, Any]]:
        """Initialize remedies for mental obstacles"""
        return {
            ManasObstacle.VYADHI: {
                "description": "Disease and physical discomfort affecting mental practice",
                "remedies": [
                    "Maintain good physical health through yoga and proper diet",
                    "Adapt practice to physical limitations",
                    "Use illness as opportunity for deeper surrender",
                    "Practice patience and acceptance"
                ],
                "practices": ["Gentle yoga", "Healing meditation", "Pranayama for health"],
                "affirmations": ["My body is temporary, I am eternal consciousness"]
            },
            
            ManasObstacle.STYANA: {
                "description": "Mental dullness, lethargy, and lack of enthusiasm",
                "remedies": [
                    "Energize through pranayama and physical movement",
                    "Vary meditation practices to maintain interest",
                    "Study inspiring spiritual texts",
                    "Seek guidance from experienced practitioners"
                ],
                "practices": ["Bhastrika pranayama", "Walking meditation", "Chanting"],
                "affirmations": ["I am alert and aware consciousness"]
            },
            
            ManasObstacle.SAMSHAYA: {
                "description": "Doubt about practice, teachings, or one's ability",
                "remedies": [
                    "Study authentic spiritual texts and teachings",
                    "Seek guidance from qualified teachers",
                    "Start with simple practices and build confidence",
                    "Remember that doubt is also a thought to be witnessed"
                ],
                "practices": ["Study of scriptures", "Satsang with practitioners", "Self-inquiry"],
                "affirmations": ["I trust the wisdom of the ancient teachings"]
            },
            
            ManasObstacle.PRAMADA: {
                "description": "Carelessness and lack of attention in practice",
                "remedies": [
                    "Establish regular practice schedule",
                    "Create dedicated practice space",
                    "Cultivate reverence and devotion in practice",
                    "Remember the importance of spiritual development"
                ],
                "practices": ["Formal sitting meditation", "Ritual and ceremony", "Mindful preparation"],
                "affirmations": ["I approach spiritual practice with dedication and reverence"]
            }
        }
    
    def assess_manas_level(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> ManasLevel:
        """Assess user's current level of mind mastery"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for unified level indicators
        if any(word in query_lower for word in ["samadhi", "unified", "no separate self", "pure consciousness"]):
            return ManasLevel.UNIFIED
        
        # Check for absorbed level indicators
        if any(word in query_lower for word in ["absorbed", "dhyana", "natural meditation", "effortless"]):
            return ManasLevel.ABSORBED
        
        # Check for concentrated level indicators
        if any(word in query_lower for word in ["concentrated", "dharana mastery", "sustained focus", "one-pointed"]):
            return ManasLevel.CONCENTRATED
        
        # Check for focused level indicators
        if any(word in query_lower for word in ["focused", "concentration", "meditation practice", "developing"]):
            return ManasLevel.FOCUSED
        
        # Check for gathering level indicators
        if any(word in query_lower for word in ["gathering", "improving", "practicing", "studying"]):
            return ManasLevel.GATHERING
        
        # Default to scattered
        return ManasLevel.SCATTERED
    
    def identify_mental_obstacles(self, query: str, context: Dict[str, Any]) -> List[ManasObstacle]:
        """Identify mental obstacles mentioned in query"""
        obstacles = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["sick", "illness", "physical", "pain"]):
            obstacles.append(ManasObstacle.VYADHI)
        
        if any(word in query_lower for word in ["dull", "lazy", "lethargic", "tired"]):
            obstacles.append(ManasObstacle.STYANA)
        
        if any(word in query_lower for word in ["doubt", "uncertain", "questioning", "skeptical"]):
            obstacles.append(ManasObstacle.SAMSHAYA)
        
        if any(word in query_lower for word in ["careless", "distracted", "unfocused", "scattered"]):
            obstacles.append(ManasObstacle.PRAMADA)
        
        return obstacles if obstacles else [ManasObstacle.PRAMADA]
    
    def get_obstacle_remedies(self, obstacles: List[ManasObstacle]) -> Dict[str, str]:
        """Get remedies for identified obstacles"""
        remedies = {}
        
        for obstacle in obstacles:
            obstacle_data = self.obstacle_remedies.get(obstacle, {})
            remedies[obstacle.value] = "; ".join(obstacle_data.get("remedies", ["Practice patience and regular meditation"]))
        
        return remedies
    
    def get_antahkarana_teaching(self, level: ManasLevel) -> str:
        """Get teaching about the four-fold mind appropriate to level"""
        if level in [ManasLevel.SCATTERED, ManasLevel.GATHERING]:
            return "Learn about the four aspects of mind: Manas (processing), Buddhi (discrimination), Chitta (memory), and Ahamkara (ego). Begin by observing these different functions."
        elif level in [ManasLevel.FOCUSED, ManasLevel.CONCENTRATED]:
            return "Develop mastery over each aspect of Antahkarana. Use Buddhi to discriminate, purify Chitta through practice, and begin to transcend Ahamkara identification."
        else:
            return "Rest as the awareness in which all four aspects of mind appear. You are not the mind but the consciousness that witnesses it."
    
    async def process_manas_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> ManasResponse:
        """Process mind-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess mental aspects
            level = self.assess_manas_level(query, context)
            obstacles = self.identify_mental_obstacles(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            obstacle_remedies = self.get_obstacle_remedies(obstacles)
            antahkarana_teaching = self.get_antahkarana_teaching(level)
            
            # Select appropriate scriptural reference
            scriptural_references = [
                "Yoga Sutras 1.14: 'Practice is firmly grounded when it is cultivated continuously for an extended period with dedication'",
                "Bhagavad Gita 6.35: 'The mind is restless and difficult to control, but it can be mastered through practice and dispassion'",
                "Katha Upanishad: 'When the five senses are still, when the mind is still, when the intellect is still - that is the highest state'"
            ]
            scriptural_wisdom = scriptural_references[min(level.value.__hash__() % len(scriptural_references), len(scriptural_references)-1)]
            
            return ManasResponse(
                manas_level=level.value,
                antahkarana_teaching=antahkarana_teaching,
                concentration_guidance=f"Develop concentration through: {guidance.concentration_practices[0] if guidance.concentration_practices else 'breath awareness'}",
                meditation_practice=f"Practice: {guidance.meditation_techniques[0] if guidance.meditation_techniques else 'mindfulness meditation'}",
                daily_mind_training=guidance.daily_mind_training,
                obstacle_remedies=obstacle_remedies,
                consciousness_insights=guidance.consciousness_insights,
                practical_applications=guidance.practical_applications,
                progress_indicators=guidance.progress_indicators,
                scriptural_wisdom=scriptural_wisdom
            )
            
        except Exception as e:
            logger.error(f"Error processing manas query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> ManasResponse:
        """Create fallback response when processing fails"""
        return ManasResponse(
            manas_level="scattered",
            antahkarana_teaching="The mind has four aspects: Manas (processing), Buddhi (discrimination), Chitta (memory), and Ahamkara (ego). Begin by simply observing these functions.",
            concentration_guidance="Start with simple breath awareness for 5-10 minutes daily",
            meditation_practice="Practice mindfulness of breathing - simply observe the breath without controlling it",
            daily_mind_training=[
                "Morning: 10 minutes concentration practice",
                "Midday: Mindful awareness during daily activities",
                "Evening: Review mental patterns and thoughts",
                "Before sleep: Practice gratitude and mental quieting"
            ],
            obstacle_remedies={
                "scattered_mind": "Start with short practices and gradually increase duration",
                "doubt": "Study authentic teachings and seek guidance from experienced practitioners"
            },
            consciousness_insights=[
                "You are the observer of thoughts, not the thoughts themselves",
                "The mind can be trained through regular practice",
                "Awareness is naturally peaceful and clear"
            ],
            practical_applications=[
                "Use concentration skills for better work performance",
                "Apply mindfulness to reduce stress and anxiety",
                "Practice emotional regulation through witness consciousness"
            ],
            progress_indicators=[
                "Increased ability to focus for longer periods",
                "Greater awareness of mental patterns",
                "Improved emotional stability"
            ],
            scriptural_wisdom="Yoga Sutras 1.2: 'Yoga is the cessation of fluctuations in the mind'"
        )
    
    def get_manas_insight(self, aspect: ManasAspect, function: ManasFunction) -> Optional[ManasInsight]:
        """Get specific insight about mind aspect and function"""
        aspect_data = self.antahkarana_system.get(aspect, {})
        
        return ManasInsight(
            aspect=aspect,
            function=function,
            teaching=aspect_data.get("function", "Develop awareness of this mental function"),
            practice_method=aspect_data.get("development", "Practice regular meditation and self-observation"),
            daily_application="Apply this understanding in your daily interactions and decisions"
        )

# Global instance
_manas_module = None

def get_manas_module() -> ManasModule:
    """Get global Manas module instance"""
    global _manas_module
    if _manas_module is None:
        _manas_module = ManasModule()
    return _manas_module

# Factory function for easy access
def create_manas_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> ManasResponse:
    """Factory function to create manas guidance"""
    import asyncio
    module = get_manas_module()
    return asyncio.run(module.process_manas_query(query, user_context))
