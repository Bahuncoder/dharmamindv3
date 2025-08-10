"""
🔥 Tapas Module - Sacred Discipline and Spiritual Austerity
Complete Tapas system based on authentic dharmic traditions
Develops spiritual willpower, discipline, and transformative practice
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TapasType(Enum):
    """Types of spiritual discipline"""
    PHYSICAL = "physical"      # Body-based practices (asana, fasting, etc.)
    MENTAL = "mental"         # Mind training and meditation
    SPIRITUAL = "spiritual"   # Soul-focused disciplines
    EMOTIONAL = "emotional"   # Heart purification practices
    SOCIAL = "social"        # Service and relationship practices
    SPEECH = "speech"        # Control of speech and truthfulness
    SENSORY = "sensory"      # Sense control and withdrawal

class TapasLevel(Enum):
    """Levels of disciplinary practice"""
    BEGINNER = "beginner"         # Starting spiritual practices
    DEVELOPING = "developing"     # Building consistency
    ESTABLISHED = "established"   # Strong practice foundation
    ADVANCED = "advanced"         # Deep spiritual discipline
    MASTER = "master"            # Teaching and guiding others

class TapasIntensity(Enum):
    """Intensity levels of tapas practice"""
    GENTLE = "gentle"           # Mild, sustainable practices
    MODERATE = "moderate"       # Regular, consistent discipline
    INTENSE = "intense"         # Rigorous, challenging practices
    EXTREME = "extreme"         # Only for advanced practitioners

class ObstacleType(Enum):
    """Common obstacles in tapas practice"""
    LAZINESS = "laziness"           # Inertia and procrastination
    ATTACHMENT = "attachment"       # Clinging to results
    PRIDE = "pride"                # Spiritual ego and comparison
    DOUBT = "doubt"                # Questioning the practice
    DISTRACTION = "distraction"    # Mental wandering
    IMPATIENCE = "impatience"      # Wanting quick results
    PHYSICAL_RESISTANCE = "physical_resistance"  # Body complaints

@dataclass
class TapasGuidance:
    """Comprehensive guidance for tapas development"""
    level: TapasLevel
    focus_areas: List[str]
    practices: List[str]
    mantras: List[str]
    challenges: List[str]
    milestones: List[str]
    scripture_references: List[str]
    daily_routine: List[str]

@dataclass
class TapasPractice:
    """Specific tapas practice structure"""
    name: str
    type: TapasType
    intensity: TapasIntensity
    duration: str
    instructions: List[str]
    benefits: List[str]
    precautions: List[str]
    variations: List[str]

class TapasInsight(BaseModel):
    """Insight from tapas module"""
    practice_type: str = Field(description="Type of tapas practice")
    intensity_level: str = Field(description="Recommended intensity")
    duration: str = Field(description="Practice duration")
    benefit: str = Field(description="Primary benefit")

class TapasResponse(BaseModel):
    """Response from Tapas module"""
    tapas_level: str = Field(description="Current level of spiritual discipline")
    recommended_practices: List[str] = Field(description="Recommended tapas practices")
    intensity_guidance: str = Field(description="Guidance on practice intensity")
    daily_routine: List[str] = Field(description="Daily discipline routine")
    obstacle_management: Dict[str, str] = Field(description="How to overcome obstacles")
    progress_indicators: List[str] = Field(description="Signs of progress in tapas")
    scriptural_wisdom: str = Field(description="Relevant scriptural guidance")
    practical_steps: List[str] = Field(description="Practical implementation steps")
    transformation_goals: List[str] = Field(description="Goals for spiritual transformation")

class TapasModule:
    """
    🔥 Tapas Module - The Sacred Fire of Discipline
    
    Based on authentic Hindu and yogic traditions of spiritual discipline
    Develops willpower, overcomes obstacles, and transforms desires
    Guides sustainable spiritual practice and inner purification
    """
    
    def __init__(self):
        self.name = "Tapas"
        self.color = "🔥"
        self.element = "Fire/Discipline"
        self.mantra = "TAPAS"
        self.deity = "Agni"  # Fire deity representing transformation
        self.principles = ["Discipline", "Willpower", "Purification", "Transformation"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.practices_database = self._initialize_practices()
        self.obstacle_solutions = self._initialize_obstacle_solutions()
        
    def _initialize_guidance_levels(self) -> Dict[TapasLevel, TapasGuidance]:
        """Initialize guidance for different levels of tapas practice"""
        return {
            TapasLevel.BEGINNER: TapasGuidance(
                level=TapasLevel.BEGINNER,
                focus_areas=[
                    "Establishing daily routine",
                    "Basic meditation practice",
                    "Simple dietary discipline",
                    "Regular sleep schedule"
                ],
                practices=[
                    "10-15 minutes daily meditation",
                    "Simple breathing exercises (pranayama)",
                    "Mindful eating practices",
                    "Early rising (before sunrise if possible)",
                    "Evening reflection and gratitude"
                ],
                mantras=[
                    "ॐ गं गणपतये नमः (Om Gam Ganapataye Namaha)",
                    "ॐ नमः शिवाय (Om Namah Shivaya)",
                    "ॐ (Om)"
                ],
                challenges=[
                    "Inconsistency in practice",
                    "Lack of motivation",
                    "External distractions",
                    "Physical resistance"
                ],
                milestones=[
                    "Completing 21 days of consistent practice",
                    "Natural arising of practice desire",
                    "Increased mental clarity",
                    "Better emotional regulation"
                ],
                scripture_references=[
                    "Bhagavad Gita 6.16: 'Yoga is not for one who eats too much or too little'",
                    "Yoga Sutras 2.46: 'Asana should be steady and comfortable'",
                    "Bhagavad Gita 6.17: 'Moderate in eating, sleeping, working and recreation'"
                ],
                daily_routine=[
                    "4:30-5:30 AM: Wake up and morning ablutions",
                    "5:30-6:00 AM: Meditation and pranayama",
                    "Evening: Reflection and gratitude practice",
                    "Night: Early sleep (by 10 PM)"
                ]
            ),
            
            TapasLevel.DEVELOPING: TapasGuidance(
                level=TapasLevel.DEVELOPING,
                focus_areas=[
                    "Deepening meditation practice",
                    "Periodic fasting",
                    "Speech control and truthfulness",
                    "Overcoming negative habits"
                ],
                practices=[
                    "30-45 minutes daily meditation",
                    "Weekly fasting (Ekadashi or chosen day)",
                    "Mauna (silence) practice",
                    "Scripture study (Svadhyaya)",
                    "Simple living and voluntary restraint"
                ],
                mantras=[
                    "गायत्री मंत्र (Gayatri Mantra)",
                    "ॐ तत् सत् (Om Tat Sat)",
                    "महामृत्युंजय मंत्र (Mahamrityunjaya Mantra)"
                ],
                challenges=[
                    "Spiritual dryness periods",
                    "Physical discomfort during practice",
                    "Mental resistance and doubt",
                    "Balancing practice with worldly duties"
                ],
                milestones=[
                    "Sustaining practice for 3-6 months",
                    "Natural reduction in desires",
                    "Increased tolerance for discomfort",
                    "Growing compassion and patience"
                ],
                scripture_references=[
                    "Bhagavad Gita 17.14: 'Tapas of the body includes worship of deities, the pure-hearted, teachers, and the wise'",
                    "Yoga Sutras 2.43: 'Through tapas, impurities are destroyed and perfection of body and senses is attained'",
                    "Bhagavad Gita 17.15: 'Speech that is truthful, pleasant, beneficial, and used for study is called tapas of speech'"
                ],
                daily_routine=[
                    "4:00-5:00 AM: Extended meditation and pranayama",
                    "Morning: Scripture study",
                    "Midday: Mindful work and service",
                    "Evening: Reflection and mantra practice"
                ]
            ),
            
            TapasLevel.ESTABLISHED: TapasGuidance(
                level=TapasLevel.ESTABLISHED,
                focus_areas=[
                    "Advanced meditation techniques",
                    "Regular retreat practice",
                    "Teaching and serving others",
                    "Inner purification (chitta shuddhi)"
                ],
                practices=[
                    "1-2 hours daily meditation",
                    "Monthly silent retreats",
                    "Intensive pranayama practices",
                    "Seva (selfless service)",
                    "Advanced fasting practices"
                ],
                mantras=[
                    "ॐ नमो भगवते वासुदेवाय (Om Namo Bhagavate Vasudevaya)",
                    "सोऽहम् (So'ham - I am That)",
                    "शिवोऽहम् (Shivo'ham - I am Shiva)"
                ],
                challenges=[
                    "Spiritual pride and ego",
                    "Attachment to spiritual experiences",
                    "Isolation from worldly responsibilities",
                    "Subtle desires for recognition"
                ],
                milestones=[
                    "Natural spontaneous meditation states",
                    "Equanimity in pleasure and pain",
                    "Ability to guide others effectively",
                    "Witnessing consciousness stabilization"
                ],
                scripture_references=[
                    "Bhagavad Gita 17.16: 'Mental serenity, gentleness, silence, self-control, and purity of thought are called tapas of the mind'",
                    "Yoga Sutras 3.4: 'The practice of all three (dharana, dhyana, samadhi) together is samyama'",
                    "Bhagavad Gita 6.20-21: 'When the mind is completely absorbed in yoga, one finds supreme satisfaction'"
                ],
                daily_routine=[
                    "3:30-6:00 AM: Intensive meditation practice",
                    "Morning: Teaching or service work",
                    "Afternoon: Study and contemplation",
                    "Evening: Community satsang or solitary practice"
                ]
            ),
            
            TapasLevel.ADVANCED: TapasGuidance(
                level=TapasLevel.ADVANCED,
                focus_areas=[
                    "Transcending discipline through love",
                    "Spontaneous samadhi states",
                    "Serving as spiritual guide",
                    "Living in constant awareness"
                ],
                practices=[
                    "Continuous awareness practice",
                    "Spontaneous meditation",
                    "Effortless service to all beings",
                    "Teaching through presence",
                    "Living meditation"
                ],
                mantras=[
                    "अहं ब्रह्मास्मि (Aham Brahmasmi - I am Brahman)",
                    "तत्त्वमसि (Tat Tvam Asi - That Thou Art)",
                    "Beyond mantras - pure silence"
                ],
                challenges=[
                    "Maintaining humility despite realization",
                    "Relating to those at earlier stages",
                    "Balancing solitude with accessibility",
                    "Avoiding spiritual materialism"
                ],
                milestones=[
                    "Effortless maintenance of awareness",
                    "Natural compassionate action",
                    "Teaching through mere presence",
                    "Recognition by authentic lineages"
                ],
                scripture_references=[
                    "Bhagavad Gita 4.19: 'One whose actions are devoid of desires and whose actions are burned by the fire of knowledge'",
                    "Yoga Sutras 4.31: 'When all coverings and impurities are removed, knowledge becomes infinite'",
                    "Bhagavad Gita 18.54: 'Having become Brahman, serene in the Self, one neither grieves nor desires'"
                ],
                daily_routine=[
                    "Natural awakening with cosmic rhythm",
                    "Spontaneous periods of meditation and service",
                    "Teaching and guidance as called",
                    "Living in constant remembrance"
                ]
            ),
            
            TapasLevel.MASTER: TapasGuidance(
                level=TapasLevel.MASTER,
                focus_areas=[
                    "Embodying the teachings",
                    "Establishing spiritual lineages",
                    "Serving humanity's evolution",
                    "Pure being-consciousness-bliss"
                ],
                practices=[
                    "Living as the teaching itself",
                    "Compassionate presence",
                    "Silent transmission of wisdom",
                    "Establishing centers of learning",
                    "Writing and preserving wisdom"
                ],
                mantras=[
                    "Beyond all mantras - pure existence",
                    "Silence as the highest teaching",
                    "Being itself as mantra"
                ],
                challenges=[
                    "Remaining in the world while transcendent",
                    "Managing disciples and institutions",
                    "Preserving authenticity in transmission",
                    "Balancing universal love with human limitations"
                ],
                milestones=[
                    "Complete ego dissolution",
                    "Universal compassion without attachment",
                    "Establishment of lasting spiritual institutions",
                    "Recognition as authentic master"
                ],
                scripture_references=[
                    "Bhagavad Gita 3.20: 'Through action alone, Janaka and others attained perfection'",
                    "Yoga Sutras 4.34: 'Kaivalya is the return of the gunas to their source'",
                    "Bhagavad Gita 4.34: 'Seek that knowledge by prostrating, by questions, and by service'"
                ],
                daily_routine=[
                    "Living in timeless awareness",
                    "Responding to the needs of the moment",
                    "Teaching through being",
                    "Serving the divine plan"
                ]
            )
        }
    
    def _initialize_practices(self) -> Dict[TapasType, List[TapasPractice]]:
        """Initialize specific tapas practices by type"""
        return {
            TapasType.PHYSICAL: [
                TapasPractice(
                    name="Sunrise Meditation",
                    type=TapasType.PHYSICAL,
                    intensity=TapasIntensity.GENTLE,
                    duration="15-30 minutes",
                    instructions=[
                        "Wake before sunrise",
                        "Face east and sit comfortably",
                        "Begin with three Om chants",
                        "Meditate on the rising sun's energy",
                        "Feel the light entering your heart"
                    ],
                    benefits=[
                        "Aligns with natural rhythms",
                        "Increases vital energy",
                        "Develops early rising habit",
                        "Connects with cosmic forces"
                    ],
                    precautions=[
                        "Start gradually if not used to early rising",
                        "Dress warmly in cold weather",
                        "Don't strain the eyes looking at sun"
                    ],
                    variations=[
                        "Indoor practice if outdoor not possible",
                        "Visualize sun if cloudy",
                        "Include pranayama breathing"
                    ]
                ),
                TapasPractice(
                    name="Fasting Practice",
                    type=TapasType.PHYSICAL,
                    intensity=TapasIntensity.MODERATE,
                    duration="12-24 hours",
                    instructions=[
                        "Choose appropriate day (Ekadashi recommended)",
                        "Eat light dinner before fast",
                        "Drink only water during fast",
                        "Engage in spiritual practices",
                        "Break fast mindfully with simple food"
                    ],
                    benefits=[
                        "Purifies digestive system",
                        "Develops willpower",
                        "Increases spiritual sensitivity",
                        "Breaks attachment to food"
                    ],
                    precautions=[
                        "Start with shorter fasts",
                        "Avoid if health conditions exist",
                        "Listen to body's signals",
                        "Consult doctor if needed"
                    ],
                    variations=[
                        "Fruit fast for beginners",
                        "Water fast for experienced",
                        "Sunrise to sunset fast"
                    ]
                )
            ],
            
            TapasType.MENTAL: [
                TapasPractice(
                    name="Concentration Meditation",
                    type=TapasType.MENTAL,
                    intensity=TapasIntensity.MODERATE,
                    duration="20-60 minutes",
                    instructions=[
                        "Choose single point of focus (breath, mantra, image)",
                        "Sit in stable, comfortable position",
                        "Gently bring mind back when it wanders",
                        "Don't fight thoughts, just return to focus",
                        "End with gratitude and dedication"
                    ],
                    benefits=[
                        "Develops mental concentration",
                        "Reduces mental chatter",
                        "Increases willpower",
                        "Prepares for deeper meditation"
                    ],
                    precautions=[
                        "Don't strain or force concentration",
                        "Be patient with wandering mind",
                        "Take breaks if intense"
                    ],
                    variations=[
                        "Trataka (candle gazing)",
                        "Breath counting",
                        "Mantra repetition"
                    ]
                )
            ],
            
            TapasType.SPEECH: [
                TapasPractice(
                    name="Mauna (Noble Silence)",
                    type=TapasType.SPEECH,
                    intensity=TapasIntensity.MODERATE,
                    duration="1 hour to full day",
                    instructions=[
                        "Choose specific time period for silence",
                        "Inform others of your practice",
                        "Avoid all verbal communication",
                        "Use mind to observe thoughts without expressing",
                        "Return to speech mindfully"
                    ],
                    benefits=[
                        "Conserves mental energy",
                        "Develops inner listening",
                        "Reduces reactivity",
                        "Deepens contemplation"
                    ],
                    precautions=[
                        "Inform family/colleagues beforehand",
                        "Keep emergency communication available",
                        "Start with shorter periods"
                    ],
                    variations=[
                        "Partial silence (essential communication only)",
                        "Written communication allowed",
                        "Complete silence including gestures"
                    ]
                )
            ]
        }
    
    def _initialize_obstacle_solutions(self) -> Dict[ObstacleType, Dict[str, Any]]:
        """Initialize solutions for common tapas obstacles"""
        return {
            ObstacleType.LAZINESS: {
                "causes": ["Tamas (inertia)", "Lack of motivation", "Poor lifestyle"],
                "solutions": [
                    "Start with very small commitments",
                    "Create accountability with others",
                    "Study inspiring texts and biographies",
                    "Improve diet and sleep patterns"
                ],
                "mantras": ["ॐ गं गणपतये नमः", "ॐ नमः शिवाय"],
                "practices": ["Early rising", "Energizing pranayama", "Cold water practices"]
            },
            
            ObstacleType.ATTACHMENT: {
                "causes": ["Desire for results", "Spiritual materialism", "Ego involvement"],
                "solutions": [
                    "Practice selfless action (nishkama karma)",
                    "Offer all results to the Divine",
                    "Study teachings on non-attachment",
                    "Cultivate witness consciousness"
                ],
                "mantras": ["ॐ तत् सत्", "ईश्वरप्रणिधान"],
                "practices": ["Karma yoga", "Bhakti practices", "Self-inquiry"]
            },
            
            ObstacleType.PRIDE: {
                "causes": ["Spiritual ego", "Comparison with others", "Achievement attachment"],
                "solutions": [
                    "Practice humility and service",
                    "Remember all comes from Divine grace",
                    "Study stories of humble saints",
                    "Avoid spiritual competition"
                ],
                "mantras": ["दासोऽहम् (I am a servant)", "न अहम् (Not I)"],
                "practices": ["Seva (service)", "Prostration", "Beginner's mind"]
            },
            
            ObstacleType.DOUBT: {
                "causes": ["Intellectual questions", "Lack of experience", "Past failures"],
                "solutions": [
                    "Study authentic scriptures",
                    "Seek guidance from experienced teachers",
                    "Focus on direct experience",
                    "Start with practices that show quick results"
                ],
                "mantras": ["श्रद्धा (Faith)", "गुरु मंत्र"],
                "practices": ["Satsang", "Scripture study", "Simple breath practices"]
            }
        }
    
    def assess_tapas_level(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> TapasLevel:
        """Assess user's current level of spiritual discipline"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for advanced indicators
        if any(word in query_lower for word in ["master", "teaching", "guiding others", "advanced practice"]):
            return TapasLevel.ADVANCED
        
        # Check for established practice indicators
        if any(word in query_lower for word in ["retreat", "intensive", "years of practice", "established"]):
            return TapasLevel.ESTABLISHED
        
        # Check for developing practice indicators
        if any(word in query_lower for word in ["deepening", "consistent", "months", "building"]):
            return TapasLevel.DEVELOPING
        
        # Default to beginner for new practitioners
        return TapasLevel.BEGINNER
    
    def identify_primary_obstacles(self, query: str, context: Dict[str, Any]) -> List[ObstacleType]:
        """Identify primary obstacles mentioned in query"""
        obstacles = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["lazy", "procrastination", "motivation"]):
            obstacles.append(ObstacleType.LAZINESS)
        
        if any(word in query_lower for word in ["results", "achievement", "success"]):
            obstacles.append(ObstacleType.ATTACHMENT)
        
        if any(word in query_lower for word in ["better than", "advanced", "superior"]):
            obstacles.append(ObstacleType.PRIDE)
        
        if any(word in query_lower for word in ["doubt", "question", "uncertain"]):
            obstacles.append(ObstacleType.DOUBT)
        
        if any(word in query_lower for word in ["distracted", "mind wanders", "focus"]):
            obstacles.append(ObstacleType.DISTRACTION)
        
        return obstacles if obstacles else [ObstacleType.LAZINESS]  # Default
    
    def recommend_practices(self, level: TapasLevel, query: str) -> List[str]:
        """Recommend specific practices based on level and query"""
        guidance = self.guidance_levels.get(level)
        if not guidance:
            return ["Daily meditation", "Simple breathing exercises", "Regular routine"]
        
        base_practices = guidance.practices
        
        # Add query-specific recommendations
        query_lower = query.lower()
        additional_practices = []
        
        if "concentration" in query_lower:
            additional_practices.append("Single-pointed meditation practice")
        
        if "discipline" in query_lower:
            additional_practices.append("Cold water practices for willpower")
        
        if "fasting" in query_lower:
            additional_practices.append("Weekly Ekadashi fasting")
        
        return base_practices + additional_practices
    
    def get_obstacle_solutions(self, obstacles: List[ObstacleType]) -> Dict[str, str]:
        """Get solutions for identified obstacles"""
        solutions = {}
        
        for obstacle in obstacles:
            obstacle_data = self.obstacle_solutions.get(obstacle, {})
            solutions[obstacle.value] = "; ".join(obstacle_data.get("solutions", ["Practice patience and persistence"]))
        
        return solutions
    
    def get_progress_indicators(self, level: TapasLevel) -> List[str]:
        """Get progress indicators for current level"""
        guidance = self.guidance_levels.get(level)
        return guidance.milestones if guidance else [
            "Increased consistency in practice",
            "Greater mental clarity",
            "Improved emotional regulation",
            "Natural desire for spiritual growth"
        ]
    
    def get_transformation_goals(self, level: TapasLevel) -> List[str]:
        """Get transformation goals for current level"""
        if level == TapasLevel.BEGINNER:
            return [
                "Establish regular spiritual practice",
                "Develop basic self-discipline",
                "Reduce negative habits",
                "Increase mental clarity"
            ]
        elif level == TapasLevel.DEVELOPING:
            return [
                "Deepen concentration and focus",
                "Overcome major obstacles",
                "Develop emotional equanimity",
                "Cultivate spiritual experiences"
            ]
        elif level == TapasLevel.ESTABLISHED:
            return [
                "Attain spontaneous meditation states",
                "Serve others through teaching",
                "Purify subtle mental impressions",
                "Develop witnessing consciousness"
            ]
        else:  # Advanced/Master
            return [
                "Live in constant awareness",
                "Embody the teachings naturally",
                "Guide others to liberation",
                "Serve universal consciousness"
            ]
    
    async def process_tapas_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> TapasResponse:
        """Process tapas-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess current level
            level = self.assess_tapas_level(query, context)
            guidance = self.guidance_levels.get(level)
            
            if not guidance:
                return self._create_fallback_response()
            
            # Identify obstacles
            obstacles = self.identify_primary_obstacles(query, context)
            
            # Get recommendations
            recommended_practices = self.recommend_practices(level, query)
            obstacle_solutions = self.get_obstacle_solutions(obstacles)
            progress_indicators = self.get_progress_indicators(level)
            transformation_goals = self.get_transformation_goals(level)
            
            # Determine intensity guidance
            if level in [TapasLevel.BEGINNER, TapasLevel.DEVELOPING]:
                intensity_guidance = "Start gently and build gradually. Consistency is more important than intensity."
            else:
                intensity_guidance = "Maintain steady, disciplined practice while remaining flexible and compassionate with yourself."
            
            # Select relevant scripture
            scripture_wisdom = guidance.scripture_references[0] if guidance.scripture_references else "तपस्वी भव - Be disciplined! (Traditional)"
            
            return TapasResponse(
                tapas_level=level.value,
                recommended_practices=recommended_practices,
                intensity_guidance=intensity_guidance,
                daily_routine=guidance.daily_routine,
                obstacle_management=obstacle_solutions,
                progress_indicators=progress_indicators,
                scriptural_wisdom=scripture_wisdom,
                practical_steps=[
                    "Start with small, achievable practices",
                    "Maintain consistent daily routine",
                    "Gradually increase intensity and duration",
                    "Seek guidance from experienced practitioners",
                    "Balance effort with surrender"
                ],
                transformation_goals=transformation_goals
            )
            
        except Exception as e:
            logger.error(f"Error processing tapas query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> TapasResponse:
        """Create fallback response when processing fails"""
        return TapasResponse(
            tapas_level="beginner",
            recommended_practices=[
                "Daily 10-15 minute meditation",
                "Simple breathing exercises",
                "Regular sleep schedule",
                "Mindful eating practices"
            ],
            intensity_guidance="Start gently and build gradually. Consistency is more important than intensity.",
            daily_routine=[
                "Early rising (before sunrise)",
                "Morning meditation and pranayama",
                "Mindful daily activities",
                "Evening reflection and gratitude"
            ],
            obstacle_management={
                "laziness": "Start with very small commitments and create accountability",
                "doubt": "Study authentic scriptures and seek experienced guidance"
            },
            progress_indicators=[
                "Increased consistency in practice",
                "Greater mental clarity",
                "Improved emotional regulation",
                "Natural desire for spiritual growth"
            ],
            scriptural_wisdom="तपस्वी भव - Be disciplined! Practice tapas for spiritual transformation.",
            practical_steps=[
                "Choose one simple practice to begin",
                "Set realistic goals and timeframes",
                "Create supportive environment",
                "Track progress without attachment",
                "Celebrate small victories"
            ],
            transformation_goals=[
                "Establish regular spiritual practice",
                "Develop basic self-discipline",
                "Reduce negative habits",
                "Increase mental clarity and peace"
            ]
        )
    
    def get_tapas_insight(self, practice_type: TapasType) -> Optional[TapasInsight]:
        """Get specific insight about a tapas practice type"""
        practices = self.practices_database.get(practice_type, [])
        if not practices:
            return None
        
        practice = practices[0]  # Get first practice as example
        return TapasInsight(
            practice_type=practice.type.value,
            intensity_level=practice.intensity.value,
            duration=practice.duration,
            benefit=practice.benefits[0] if practice.benefits else "Develops spiritual discipline"
        )

# Global instance
_tapas_module = None

def get_tapas_module() -> TapasModule:
    """Get global Tapas module instance"""
    global _tapas_module
    if _tapas_module is None:
        _tapas_module = TapasModule()
    return _tapas_module

# Factory function for easy access
def create_tapas_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> TapasResponse:
    """Factory function to create tapas guidance"""
    import asyncio
    module = get_tapas_module()
    return asyncio.run(module.process_tapas_query(query, user_context))
