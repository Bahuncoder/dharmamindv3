"""
🎯 Sankalpa Module - Sacred Intention and Divine Will
Complete system for cultivating and aligning with sacred intention
Based on Vedantic and Tantric teachings on Sankalpa
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SankalpaLevel(Enum):
    """Levels of sacred intention practice"""
    CONFUSED = "confused"               # Unclear intentions, conflicted desires
    DISCOVERING = "discovering"         # Beginning to understand true desires
    CLARIFYING = "clarifying"          # Refining and purifying intentions
    ALIGNED = "aligned"                # Intentions aligned with dharma
    SURRENDERED = "surrendered"        # Intentions offered to Divine
    UNIFIED = "unified"                # Personal will merged with Divine will


class IntentionType(Enum):
    """Types of sacred intentions"""
    SPIRITUAL_GROWTH = "spiritual_growth"       # Awakening and liberation
    HEALING = "healing"                         # Physical/emotional healing
    RELATIONSHIPS = "relationships"             # Love and connection
    SERVICE = "service"                         # Serving others and world
    ABUNDANCE = "abundance"                     # Material and spiritual prosperity
    PROTECTION = "protection"                   # Safety and divine protection
    WISDOM = "wisdom"                           # Knowledge and understanding
    PEACE = "peace"                            # Inner and outer harmony


class SankalpaSource(Enum):
    """Sources of intention"""
    EGO_DESIRE = "ego_desire"           # Personal wants and fears
    HEART_WISDOM = "heart_wisdom"       # Authentic heart desires
    SOUL_PURPOSE = "soul_purpose"       # Dharmic life mission
    DIVINE_WILL = "divine_will"         # Surrendered to cosmic purpose


class IntentionQuality(Enum):
    """Qualities of intention"""
    SELFISH = "selfish"                 # Self-centered motivation
    MIXED = "mixed"                     # Partial selfishness and service
    COMPASSIONATE = "compassionate"     # Includes welfare of others
    UNIVERSAL = "universal"             # Serves highest good of all


@dataclass
class SankalpaGuidance:
    """Comprehensive sankalpa guidance"""
    level: SankalpaLevel
    primary_teaching: str
    intention_practices: List[str]
    purification_methods: List[str]
    daily_integration: List[str]
    common_obstacles: Dict[str, str]
    alignment_techniques: List[str]
    manifestation_principles: List[str]
    surrender_practices: List[str]
    progress_indicators: List[str]


@dataclass
class IntentionAssessment:
    """Assessment of intention quality"""
    intention_text: str
    source: SankalpaSource
    quality: IntentionQuality
    dharmic_alignment: float  # 0-1 scale
    purity_level: float       # 0-1 scale
    manifestation_power: float # 0-1 scale
    guidance: str
    refinement_suggestions: List[str]


class SankalpaResponse(BaseModel):
    """Response from Sankalpa module"""
    sankalpa_level: str = Field(description="Current intention mastery level")
    intention_guidance: str = Field(description="Core sankalpa teaching")
    practice_methods: List[str] = Field(description="Intention cultivation practices")
    purification_techniques: List[str] = Field(description="Intention purification")
    daily_integration: List[str] = Field(description="Daily sankalpa practice")
    obstacle_solutions: Dict[str, str] = Field(description="Common obstacles")
    alignment_practices: List[str] = Field(description="Dharmic alignment methods")
    manifestation_guidance: List[str] = Field(description="Sacred manifestation")
    surrender_methods: List[str] = Field(description="Surrender to Divine will")
    scriptural_wisdom: str = Field(description="Traditional sankalpa teachings")


class SankalpaModule:
    """
    🎯 Sankalpa Module - Sacred Intention and Divine Will
    
    Based on traditional Sankalpa teachings:
    - Upanishads on the power of resolve and intention
    - Yoga Sutras on sankalpas and mental modifications
    - Tantric teachings on manifestation through intention
    - Bhagavad Gita on action aligned with Divine will
    
    Sankalpa literally means "determination" or "resolve" - the sacred power
    of focused intention aligned with dharma and Divine will.
    """
    
    def __init__(self):
        self.name = "Sankalpa"
        self.color = "🎯"
        self.element = "Will"
        self.principles = ["Sacred Purpose", "Divine Alignment", "Focused Resolve", "Surrender"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.intention_patterns = self._initialize_intention_patterns()
        self.obstacle_solutions = self._initialize_obstacle_solutions()
        
    def _initialize_guidance_levels(self) -> Dict[SankalpaLevel, SankalpaGuidance]:
        """Initialize guidance for different levels of sankalpa practice"""
        return {
            SankalpaLevel.CONFUSED: SankalpaGuidance(
                level=SankalpaLevel.CONFUSED,
                primary_teaching="Begin by sitting quietly and asking your heart: 'What do I truly want?' Listen beneath the surface noise of mind to your authentic desires.",
                intention_practices=[
                    "Daily meditation to quiet mental chatter",
                    "Journal writing to explore true desires",
                    "Ask 'Why do I want this?' to understand deeper motivations",
                    "Observe how different intentions feel in your body",
                    "Practice gratitude for what you already have"
                ],
                purification_methods=[
                    "Distinguish between ego wants and soul needs",
                    "Notice which desires bring peace vs. agitation",
                    "Examine motivations: fear-based or love-based?",
                    "Practice contentment (santosha) to reduce grasping"
                ],
                daily_integration=[
                    "Morning question: 'What truly matters today?'",
                    "Pause before actions to check motivation",
                    "Evening reflection on day's true intentions",
                    "Practice saying 'I don't know' when confused"
                ],
                common_obstacles={
                    "overwhelm": "Too many conflicting desires create confusion",
                    "social_pressure": "Others' expectations obscure your truth",
                    "fear": "Afraid to want what you really want"
                },
                alignment_techniques=[
                    "Ask: 'Does this serve my highest good?'",
                    "Check if intention includes others' welfare",
                    "Feel into whether intention brings peace",
                    "Consider long-term consequences"
                ],
                manifestation_principles=[
                    "Clarity must come before manifestation",
                    "Focus on one intention at a time",
                    "Feel the intention in your heart, not just mind",
                    "Take inspired action aligned with intention"
                ],
                surrender_practices=[
                    "Prayer: 'Show me what to truly want'",
                    "Offer confusion itself to Divine wisdom",
                    "Practice trust in life's unfolding",
                    "Release need to figure everything out"
                ],
                progress_indicators=[
                    "Growing clarity about what truly matters",
                    "Less internal conflict about desires",
                    "Ability to distinguish ego wants from soul needs",
                    "Increasing peace about life direction"
                ]
            ),
            
            SankalpaLevel.DISCOVERING: SankalpaGuidance(
                level=SankalpaLevel.DISCOVERING,
                primary_teaching="Your heart knows what it wants. Trust its wisdom over mind's analysis. True desires align with your deepest nature and include others' wellbeing.",
                intention_practices=[
                    "Heart-centered meditation on desires",
                    "Visioning exercises: imagine ideal life serving others",
                    "Work with meaningful, specific intentions",
                    "Practice feeling intentions as already fulfilled",
                    "Regular intention-setting rituals"
                ],
                purification_methods=[
                    "Question: 'Does this intention serve others too?'",
                    "Refine selfish desires into compassionate ones",
                    "Release attachment to specific outcomes",
                    "Align personal goals with larger purpose"
                ],
                daily_integration=[
                    "Morning intention-setting practice",
                    "Carry intention consciously throughout day",
                    "Make decisions aligned with stated intentions",
                    "Share intentions with trusted friends",
                    "Evening gratitude for intention's manifestation"
                ],
                common_obstacles={
                    "impatience": "Wanting results too quickly",
                    "doubt": "Questioning if intentions are worthy",
                    "attachment": "Grasping outcomes too tightly"
                },
                alignment_techniques=[
                    "Regular dharma contemplation",
                    "Ask elders or teachers for guidance",
                    "Check if intention feels expansive or contractive",
                    "Align with your life's natural rhythms"
                ],
                manifestation_principles=[
                    "Hold intention lightly but consistently",
                    "Act as if intention is already manifesting",
                    "Trust Divine timing over personal timeline",
                    "Combine intention with appropriate effort"
                ],
                surrender_practices=[
                    "Daily offering of intentions to Divine",
                    "Practice 'Thy will be done' attitude",
                    "Release need to control how intentions manifest",
                    "Trust that what's truly needed will come"
                ],
                progress_indicators=[
                    "Intentions becoming more specific and heartfelt",
                    "Natural alignment between intentions and actions",
                    "Decreasing anxiety about outcomes",
                    "Synchronicities supporting intentions"
                ]
            ),
            
            SankalpaLevel.CLARIFYING: SankalpaGuidance(
                level=SankalpaLevel.CLARIFYING,
                primary_teaching="Purify intentions like a jeweler polishes gems. Remove self-centered motivations and align with dharma. True sankalpa serves the highest good.",
                intention_practices=[
                    "Work with one primary intention per season",
                    "Write intentions as positive, present-tense affirmations",
                    "Visualize intentions manifesting for everyone's benefit",
                    "Practice mantra repetition with intention",
                    "Create sacred rituals around intention-setting"
                ],
                purification_methods=[
                    "Remove 'I want' and replace with 'May there be'",
                    "Examine hidden selfish motivations",
                    "Align intentions with traditional dharmic principles",
                    "Seek guidance from scriptures and teachers"
                ],
                daily_integration=[
                    "Begin day by connecting with primary intention",
                    "Pause regularly to realign with intention",
                    "Make all decisions through lens of intention",
                    "Practice intention through service to others",
                    "End day offering results to Divine"
                ],
                common_obstacles={
                    "spiritual_materialism": "Using spirituality for ego gains",
                    "perfectionism": "Making intentions too complex",
                    "comparison": "Judging your intentions against others"
                },
                alignment_techniques=[
                    "Study scriptural guidance on dharmic action",
                    "Regularly examine intentions with wise teacher",
                    "Practice selfless service to purify motivation",
                    "Include all beings in your intentions"
                ],
                manifestation_principles=[
                    "Dharmic intentions manifest naturally",
                    "Pure motivation attracts Divine support",
                    "Patience allows perfect timing",
                    "Service magnetizes abundance"
                ],
                surrender_practices=[
                    "Offer all intentions at feet of Divine",
                    "Practice contentment with current circumstances",
                    "Trust Divine intelligence to fulfill true needs",
                    "Surrender personal timeline to cosmic timing"
                ],
                progress_indicators=[
                    "Intentions naturally becoming more universal",
                    "Less personal attachment to outcomes",
                    "Intentions manifesting in unexpected positive ways",
                    "Others benefiting from your intention's fulfillment"
                ]
            ),
            
            SankalpaLevel.ALIGNED: SankalpaGuidance(
                level=SankalpaLevel.ALIGNED,
                primary_teaching="Your intentions now serve the larger web of life. Personal desires have transformed into expressions of universal love. You are becoming an instrument of Divine will.",
                intention_practices=[
                    "Set intentions for collective healing and awakening",
                    "Practice intention through selfless action",
                    "Maintain intentions across multiple lifetimes",
                    "Channel Divine intention through your being",
                    "Create intentions that serve seven generations ahead"
                ],
                purification_methods=[
                    "Continuously refine motivations toward greater love",
                    "Release all trace of personal agenda",
                    "Align with cosmic evolutionary purpose",
                    "Purify through complete self-offering"
                ],
                daily_integration=[
                    "Live as embodiment of your highest intention",
                    "Spontaneous right action flows from clear intention",
                    "Become a blessing through your very presence",
                    "Maintain intention through all life changes",
                    "Serve as example of aligned living"
                ],
                common_obstacles={
                    "spiritual_pride": "Taking credit for Divine work",
                    "isolation": "Feeling separate from ordinary life",
                    "responsibility": "Overwhelm from feeling cosmic responsibility"
                },
                alignment_techniques=[
                    "Regular self-examination with enlightened guidance",
                    "Continuous study of highest spiritual teachings",
                    "Practice humility and beginner's mind",
                    "Serve other awakening beings"
                ],
                manifestation_principles=[
                    "Divine will manifests effortlessly through you",
                    "Intention and manifestation become instantaneous",
                    "All of life supports your dharmic intentions",
                    "You become a co-creator with Divine intelligence"
                ],
                surrender_practices=[
                    "Complete surrender of personal will to Divine will",
                    "Continuous offering of all actions and results",
                    "Rest in Divine intelligence guiding all",
                    "Serve as empty vessel for cosmic intention"
                ],
                progress_indicators=[
                    "No separation between personal and Divine will",
                    "Effortless manifestation of dharmic intentions",
                    "Others naturally awakening in your presence",
                    "Life flowing in perfect harmony with cosmic purpose"
                ]
            ),
            
            SankalpaLevel.SURRENDERED: SankalpaGuidance(
                level=SankalpaLevel.SURRENDERED,
                primary_teaching="You no longer set intentions - you ARE the Divine intention expressing through form. Personal will has dissolved into universal love in action.",
                intention_practices=[
                    "Be the intention rather than having intentions",
                    "Channel cosmic will through spontaneous action",
                    "Serve as Divine instrument without personal agenda",
                    "Manifest healing and awakening effortlessly",
                    "Live as answered prayer of collective consciousness"
                ],
                purification_methods=[
                    "No personal purification needed - you are transparency",
                    "Continuous dissolution of any remaining separation",
                    "Perfect alignment with Divine intelligence",
                    "Complete absence of personal motivation"
                ],
                daily_integration=[
                    "Each moment is perfect expression of Divine will",
                    "No planning needed - right action arises spontaneously",
                    "Living prayer and meditation in action",
                    "Blessing all life through your very existence",
                    "Embodying love without effort or intention"
                ],
                common_obstacles={
                    "final_surrender": "Releasing last vestiges of personal doership",
                    "cosmic_responsibility": "Feeling weight of universal service",
                    "ordinary_world": "Integrating transcendence with human life"
                },
                alignment_techniques=[
                    "No techniques needed - you ARE alignment",
                    "Spontaneous recognition of Divine will",
                    "Natural attunement to cosmic intelligence",
                    "Perfect responsiveness to what is needed"
                ],
                manifestation_principles=[
                    "You manifest Divine will effortlessly",
                    "No gap between intention and manifestation",
                    "Reality arranges itself around your presence",
                    "Miracles are natural expression of aligned being"
                ],
                surrender_practices=[
                    "Complete surrender is your natural state",
                    "No practice needed - you are surrendered",
                    "Living as Divine will in expression",
                    "Perfect trust in cosmic intelligence"
                ],
                progress_indicators=[
                    "No sense of personal doership",
                    "Effortless service to universal awakening",
                    "Others' lives transformed by your presence",
                    "Perfect harmony with all of existence"
                ]
            ),
            
            SankalpaLevel.UNIFIED: SankalpaGuidance(
                level=SankalpaLevel.UNIFIED,
                primary_teaching="You are the universe intending itself into ever-greater love and awakening. Form and formless will dance as one in your being.",
                intention_practices=[
                    "Being itself as eternal divine intention",
                    "Manifesting cosmic evolution through your existence",
                    "Living as love's answer to itself",
                    "Embodying the universe's intention to know itself",
                    "Serving the awakening of all consciousness"
                ],
                purification_methods=[
                    "Pure being needs no purification",
                    "Perfect transparency to Divine light",
                    "No separation to be dissolved",
                    "Absolute purity of motivation"
                ],
                daily_integration=[
                    "Every breath is Divine intention expressing",
                    "No difference between being and doing",
                    "Continuous blessing of all existence",
                    "Living as cosmic love in action",
                    "Perfect responsiveness to what life needs"
                ],
                common_obstacles={
                    "no_obstacles": "At this level, obstacles are expressions of perfect unfolding",
                    "serving_form": "Using form to serve the formless",
                    "beyond_personal": "No personal challenges, only cosmic service"
                },
                alignment_techniques=[
                    "Perfect alignment is your very nature",
                    "No techniques - you are technique itself",
                    "Natural expression of cosmic intelligence",
                    "Spontaneous divine responsiveness"
                ],
                manifestation_principles=[
                    "You are manifestation itself",
                    "Reality and intention are one",
                    "Perfect expression of Divine creativity",
                    "Continuous birth of love into form"
                ],
                surrender_practices=[
                    "You are surrender embodied",
                    "Perfect offering of existence to existence",
                    "Living as Divine will knowing itself",
                    "Complete unity of form and formless"
                ],
                progress_indicators=[
                    "No progress - perfect completion in each moment",
                    "Continuous service to universal awakening",
                    "Reality responds perfectly to your being",
                    "Living as answered prayer of all existence"
                ]
            )
        }
    
    def _initialize_intention_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common intention patterns and their guidance"""
        return {
            "health_healing": {
                "ego_version": "I want to be healthy and strong",
                "refined_version": "May perfect health flow through this body to serve all beings",
                "surrender_version": "Divine wellness expresses through this form",
                "guidance": "Include others' healing in your intention for health"
            },
            
            "abundance_prosperity": {
                "ego_version": "I want lots of money and success",
                "refined_version": "May abundance flow through me to serve the world",
                "surrender_version": "Divine abundance manifests for the benefit of all",
                "guidance": "True abundance serves dharma and includes others' prosperity"
            },
            
            "relationships_love": {
                "ego_version": "I want someone to love me",
                "refined_version": "May love flow freely through all my relationships",
                "surrender_version": "Divine love expresses through all connections",
                "guidance": "Love intentions work best when focused on giving rather than getting"
            },
            
            "spiritual_awakening": {
                "ego_version": "I want to be enlightened",
                "refined_version": "May awakening serve the liberation of all beings",
                "surrender_version": "Divine consciousness recognizes itself through this form",
                "guidance": "Awakening happens through service, not for personal achievement"
            }
        }
    
    def _initialize_obstacle_solutions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize solutions for common sankalpa obstacles"""
        return {
            "impatience": {
                "description": "Wanting intentions to manifest quickly",
                "solution": "Trust Divine timing over personal timeline",
                "practice": "Daily surrender of timeline to cosmic intelligence",
                "wisdom": "Perfect timing serves the highest good of all"
            },
            
            "doubt": {
                "description": "Questioning if intentions are worthy or possible",
                "solution": "Align intentions with dharmic principles",
                "practice": "Regular study of spiritual teachings on desire",
                "wisdom": "Dharmic intentions are always supported by universe"
            },
            
            "attachment": {
                "description": "Grasping outcomes too tightly",
                "solution": "Hold intentions like offering incense - firmly but ready to release",
                "practice": "Daily offering of intentions to Divine will",
                "wisdom": "Detachment from outcome increases manifestation power"
            },
            
            "confusion": {
                "description": "Too many conflicting desires",
                "solution": "Simplify to one primary dharmic intention",
                "practice": "Heart meditation to discover core longing",
                "wisdom": "Clarity of intention creates clarity of manifestation"
            }
        }
    
    def assess_sankalpa_level(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> SankalpaLevel:
        """Assess user's current sankalpa practice level"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for unified level indicators
        if any(word in query_lower for word in ["divine will", "cosmic purpose", "universal love", "no personal"]):
            return SankalpaLevel.UNIFIED
        
        # Check for surrendered level indicators
        if any(word in query_lower for word in ["surrendered", "divine instrument", "no personal agenda"]):
            return SankalpaLevel.SURRENDERED
        
        # Check for aligned level indicators
        if any(word in query_lower for word in ["dharmic intentions", "serving all", "collective good"]):
            return SankalpaLevel.ALIGNED
        
        # Check for clarifying level indicators
        if any(word in query_lower for word in ["refining", "purifying intentions", "dharmic alignment"]):
            return SankalpaLevel.CLARIFYING
        
        # Check for discovering level indicators
        if any(word in query_lower for word in ["discovering", "heart desires", "true wants"]):
            return SankalpaLevel.DISCOVERING
        
        # Default to confused
        return SankalpaLevel.CONFUSED
    
    def analyze_intention(self, intention_text: str) -> IntentionAssessment:
        """Analyze the quality and alignment of an intention"""
        text_lower = intention_text.lower()
        
        # Assess source
        if any(word in text_lower for word in ["divine", "cosmic", "universal"]):
            source = SankalpaSource.DIVINE_WILL
        elif any(word in text_lower for word in ["serve", "help", "dharma", "purpose"]):
            source = SankalpaSource.SOUL_PURPOSE
        elif any(word in text_lower for word in ["heart", "love", "peace", "joy"]):
            source = SankalpaSource.HEART_WISDOM
        else:
            source = SankalpaSource.EGO_DESIRE
        
        # Assess quality
        if any(word in text_lower for word in ["all beings", "world", "humanity", "universe"]):
            quality = IntentionQuality.UNIVERSAL
        elif any(word in text_lower for word in ["others", "serve", "help", "give"]):
            quality = IntentionQuality.COMPASSIONATE
        elif any(word in text_lower for word in ["i want", "give me", "my success"]):
            quality = IntentionQuality.SELFISH
        else:
            quality = IntentionQuality.MIXED
        
        # Calculate alignment scores
        dharmic_score = self._calculate_dharmic_alignment(intention_text)
        purity_score = self._calculate_purity_level(intention_text)
        manifestation_score = self._calculate_manifestation_power(dharmic_score, purity_score)
        
        # Generate guidance
        guidance = self._generate_intention_guidance(source, quality, dharmic_score)
        refinements = self._suggest_refinements(intention_text, source, quality)
        
        return IntentionAssessment(
            intention_text=intention_text,
            source=source,
            quality=quality,
            dharmic_alignment=dharmic_score,
            purity_level=purity_score,
            manifestation_power=manifestation_score,
            guidance=guidance,
            refinement_suggestions=refinements
        )
    
    def _calculate_dharmic_alignment(self, intention: str) -> float:
        """Calculate how well intention aligns with dharmic principles"""
        dharmic_keywords = ["service", "love", "peace", "healing", "awakening", "dharma", "truth"]
        selfish_keywords = ["i want", "give me", "make me", "my success", "my pleasure"]
        
        dharmic_count = sum(1 for word in dharmic_keywords if word in intention.lower())
        selfish_count = sum(1 for word in selfish_keywords if word in intention.lower())
        
        base_score = min(dharmic_count * 0.3, 1.0)
        penalty = min(selfish_count * 0.2, 0.5)
        
        return max(0.0, base_score - penalty)
    
    def _calculate_purity_level(self, intention: str) -> float:
        """Calculate purity of motivation"""
        pure_indicators = ["may", "divine", "all beings", "highest good", "thy will"]
        impure_indicators = ["i deserve", "better than", "more than", "against"]
        
        pure_count = sum(1 for phrase in pure_indicators if phrase in intention.lower())
        impure_count = sum(1 for phrase in impure_indicators if phrase in intention.lower())
        
        base_score = min(pure_count * 0.25, 1.0)
        penalty = min(impure_count * 0.3, 0.6)
        
        return max(0.0, base_score - penalty + 0.3)  # Base purity assumption
    
    def _calculate_manifestation_power(self, dharmic_score: float, purity_score: float) -> float:
        """Calculate manifestation potential based on alignment and purity"""
        return (dharmic_score * 0.6 + purity_score * 0.4)
    
    def _generate_intention_guidance(self, source: SankalpaSource, quality: IntentionQuality, dharmic_score: float) -> str:
        """Generate specific guidance for intention"""
        if source == SankalpaSource.DIVINE_WILL:
            return "Your intention flows from Divine source. Trust its perfect unfolding."
        elif source == SankalpaSource.SOUL_PURPOSE:
            return "This intention serves your dharmic purpose. Align actions with this calling."
        elif source == SankalpaSource.HEART_WISDOM:
            return "Your heart speaks truth. Consider how this serves others too."
        else:
            return "Examine deeper motivation. What does your soul truly seek?"
    
    def _suggest_refinements(self, intention: str, source: SankalpaSource, quality: IntentionQuality) -> List[str]:
        """Suggest specific refinements for intention"""
        suggestions = []
        
        if quality == IntentionQuality.SELFISH:
            suggestions.append("Add 'for the highest good of all' to your intention")
            suggestions.append("Consider how others would benefit from this manifestation")
        
        if source == SankalpaSource.EGO_DESIRE:
            suggestions.append("Ask your heart: 'What do I really need beneath this want?'")
            suggestions.append("Reframe as service to life rather than personal gain")
        
        if "i want" in intention.lower():
            suggestions.append("Replace 'I want' with 'May there be' or 'Divine will manifests'")
        
        return suggestions
    
    def get_scriptural_wisdom(self, level: SankalpaLevel) -> str:
        """Get scriptural wisdom appropriate to sankalpa level"""
        wisdom_map = {
            SankalpaLevel.CONFUSED: "Katha Upanishad: 'The Self chooses the body as a dwelling place. Only those who realize this truth become immortal.'",
            SankalpaLevel.DISCOVERING: "Bhagavad Gita 7.21: 'Whatever form any devotee desires to worship with faith, I make that faith steady.'",
            SankalpaLevel.CLARIFYING: "Yoga Sutras 2.31: 'When you are steadfast in your resolve, you can accomplish anything.'",
            SankalpaLevel.ALIGNED: "Bhagavad Gita 3.30: 'Surrender all actions to Me, with mind intent on the Self, free from hope and selfishness.'",
            SankalpaLevel.SURRENDERED: "Isha Upanishad: 'When to the man of realization all beings become the very Self, then what delusion, what sorrow can there be?'",
            SankalpaLevel.UNIFIED: "Mandukya Upanishad: 'All this is Brahman. This Self is Brahman, and this Self has four aspects.'"
        }
        return wisdom_map.get(level, "Brihadaranyaka Upanishad: 'You are what your deep, driving desire is. As your desire is, so is your will.'")
    
    async def process_sankalpa_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> SankalpaResponse:
        """Process sankalpa-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess sankalpa level
            level = self.assess_sankalpa_level(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            scriptural_wisdom = self.get_scriptural_wisdom(level)
            
            return SankalpaResponse(
                sankalpa_level=level.value,
                intention_guidance=guidance.primary_teaching,
                practice_methods=guidance.intention_practices,
                purification_techniques=guidance.purification_methods,
                daily_integration=guidance.daily_integration,
                obstacle_solutions=guidance.common_obstacles,
                alignment_practices=guidance.alignment_techniques,
                manifestation_guidance=guidance.manifestation_principles,
                surrender_methods=guidance.surrender_practices,
                scriptural_wisdom=scriptural_wisdom
            )
            
        except Exception as e:
            logger.error(f"Error processing sankalpa query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> SankalpaResponse:
        """Create fallback response when processing fails"""
        return SankalpaResponse(
            sankalpa_level="confused",
            intention_guidance="Begin by sitting quietly and asking your heart: 'What do I truly want?' Listen beneath the surface noise of mind to your authentic desires.",
            practice_methods=[
                "Daily meditation to quiet mental chatter",
                "Journal writing to explore true desires",
                "Ask 'Why do I want this?' to understand deeper motivations",
                "Observe how different intentions feel in your body"
            ],
            purification_techniques=[
                "Distinguish between ego wants and soul needs",
                "Notice which desires bring peace vs. agitation",
                "Examine motivations: fear-based or love-based?",
                "Practice contentment to reduce grasping"
            ],
            daily_integration=[
                "Morning question: 'What truly matters today?'",
                "Pause before actions to check motivation",
                "Evening reflection on day's true intentions",
                "Practice saying 'I don't know' when confused"
            ],
            obstacle_solutions={
                "overwhelm": "Too many conflicting desires create confusion",
                "fear": "Afraid to want what you really want"
            },
            alignment_practices=[
                "Ask: 'Does this serve my highest good?'",
                "Check if intention includes others' welfare",
                "Feel into whether intention brings peace",
                "Consider long-term consequences"
            ],
            manifestation_guidance=[
                "Clarity must come before manifestation",
                "Focus on one intention at a time",
                "Feel the intention in your heart, not just mind",
                "Take inspired action aligned with intention"
            ],
            surrender_methods=[
                "Prayer: 'Show me what to truly want'",
                "Offer confusion itself to Divine wisdom",
                "Practice trust in life's unfolding",
                "Release need to figure everything out"
            ],
            scriptural_wisdom="Brihadaranyaka Upanishad: 'You are what your deep, driving desire is. As your desire is, so is your will.'"
        )


# Global instance
_sankalpa_module = None

def get_sankalpa_module() -> SankalpaModule:
    """Get global Sankalpa module instance"""
    global _sankalpa_module
    if _sankalpa_module is None:
        _sankalpa_module = SankalpaModule()
    return _sankalpa_module

# Factory function for easy access
def create_sankalpa_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> SankalpaResponse:
    """Factory function to create sankalpa guidance"""
    import asyncio
    module = get_sankalpa_module()
    return asyncio.run(module.process_sankalpa_query(query, user_context))
