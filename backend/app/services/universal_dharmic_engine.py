"""
Universal Dharmic Engine for DharmaMind platform

Central dharmic wisdom engine that integrates all spiritual traditions
and provides unified dharmic guidance.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class DharmicTradition(str, Enum):
    """Supported dharmic traditions"""
    HINDUISM = "hinduism"
    BUDDHISM = "buddhism"
    JAINISM = "jainism"
    SIKHISM = "sikhism"
    VEDANTA = "vedanta"
    YOGA = "yoga"
    TANTRA = "tantra"
    UNIVERSAL = "universal"

class DharmicPrinciple(str, Enum):
    """Universal dharmic principles"""
    AHIMSA = "ahimsa"  # Non-violence
    SATYA = "satya"    # Truthfulness
    ASTEYA = "asteya"  # Non-stealing
    BRAHMACHARYA = "brahmacharya"  # Energy conservation/celibacy
    APARIGRAHA = "aparigraha"  # Non-possessiveness
    SAUCHA = "saucha"  # Cleanliness/purity
    SANTOSHA = "santosha"  # Contentment
    TAPAS = "tapas"    # Disciplined practice
    SVADHYAYA = "svadhyaya"  # Self-study
    ISHVARA_PRANIDHANA = "ishvara_pranidhana"  # Surrender to divine

class WisdomLevel(str, Enum):
    """Levels of wisdom understanding"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    SAGE = "sage"
    ENLIGHTENED = "enlightened"

class DharmicGuidanceRequest(BaseModel):
    """Request for dharmic guidance"""
    user_id: str = Field(..., description="User identifier")
    question: str = Field(..., description="User's question or situation")
    tradition: Optional[DharmicTradition] = Field(default=None, description="Preferred tradition")
    wisdom_level: WisdomLevel = Field(default=WisdomLevel.INTERMEDIATE, description="User's wisdom level")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class DharmicGuidanceResponse(BaseModel):
    """Response with dharmic guidance"""
    guidance_id: str = Field(..., description="Unique guidance identifier")
    
    # Core guidance
    primary_guidance: str = Field(..., description="Main dharmic guidance")
    supporting_wisdom: List[str] = Field(default_factory=list, description="Supporting wisdom")
    practical_steps: List[str] = Field(default_factory=list, description="Practical action steps")
    
    # Dharmic analysis
    relevant_principles: List[DharmicPrinciple] = Field(default_factory=list, description="Relevant dharmic principles")
    tradition_perspectives: Dict[str, str] = Field(default_factory=dict, description="Multi-tradition perspectives")
    
    # Wisdom integration
    deeper_inquiry: List[str] = Field(default_factory=list, description="Questions for deeper inquiry")
    contemplation_points: List[str] = Field(default_factory=list, description="Points for contemplation")
    
    # Personalization
    wisdom_level_applied: WisdomLevel = Field(..., description="Wisdom level used")
    dharmic_confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in dharmic alignment")
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")
    tradition_used: DharmicTradition = Field(..., description="Primary tradition used")

class DharmicEngine:
    """Universal Dharmic Engine for integrated spiritual wisdom"""
    
    def __init__(self):
        self.tradition_wisdom = self._initialize_wisdom_base()
        self.principle_mappings = self._initialize_principle_mappings()
        self.guidance_history: List[DharmicGuidanceResponse] = []
        
    def _initialize_wisdom_base(self) -> Dict[DharmicTradition, Dict[str, Any]]:
        """Initialize wisdom knowledge base"""
        return {
            DharmicTradition.HINDUISM: {
                "core_texts": ["Bhagavad Gita", "Upanishads", "Ramayana", "Mahabharata"],
                "key_concepts": ["dharma", "karma", "moksha", "ahimsa", "atman", "brahman"],
                "practices": ["yoga", "meditation", "pranayama", "devotion", "selfless service"],
                "guidance_style": "comprehensive and philosophical"
            },
            DharmicTradition.BUDDHISM: {
                "core_texts": ["Dhammapada", "Lotus Sutra", "Heart Sutra", "Diamond Sutra"],
                "key_concepts": ["four noble truths", "eightfold path", "impermanence", "interdependence"],
                "practices": ["mindfulness", "meditation", "compassion cultivation", "loving-kindness"],
                "guidance_style": "practical and mindfulness-based"
            },
            DharmicTradition.JAINISM: {
                "core_texts": ["Agamas", "Tattvartha Sutra"],
                "key_concepts": ["ahimsa", "anekantavada", "aparigraha", "karma purification"],
                "practices": ["non-violence", "self-restraint", "meditation", "right conduct"],
                "guidance_style": "ethical and non-violent"
            },
            DharmicTradition.YOGA: {
                "core_texts": ["Yoga Sutras", "Hatha Yoga Pradipika", "Bhagavad Gita"],
                "key_concepts": ["eight limbs", "union", "transformation", "self-realization"],
                "practices": ["asana", "pranayama", "dharana", "dhyana", "samadhi"],
                "guidance_style": "practical and transformative"
            }
        }
    
    def _initialize_principle_mappings(self) -> Dict[DharmicPrinciple, Dict[str, Any]]:
        """Initialize dharmic principle mappings"""
        return {
            DharmicPrinciple.AHIMSA: {
                "description": "Non-violence in thought, word, and action",
                "applications": ["compassionate communication", "ethical living", "environmental care"],
                "practices": ["loving-kindness meditation", "conscious consumption", "peaceful conflict resolution"]
            },
            DharmicPrinciple.SATYA: {
                "description": "Truthfulness and honesty",
                "applications": ["honest communication", "self-honesty", "authentic living"],
                "practices": ["self-reflection", "honest speech", "alignment of values and actions"]
            },
            DharmicPrinciple.ASTEYA: {
                "description": "Non-stealing and honesty",
                "applications": ["respect for others' property", "time consciousness", "energy conservation"],
                "practices": ["gratitude cultivation", "sharing resources", "conscious consumption"]
            },
            DharmicPrinciple.APARIGRAHA: {
                "description": "Non-possessiveness and contentment",
                "applications": ["simple living", "detachment from outcomes", "generosity"],
                "practices": ["gratitude practice", "letting go exercises", "sharing and giving"]
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the dharmic engine"""
        try:
            logger.info("Universal Dharmic Engine initialized with wisdom from multiple traditions")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Universal Dharmic Engine: {e}")
            return False
    
    async def provide_guidance(
        self,
        request: DharmicGuidanceRequest
    ) -> DharmicGuidanceResponse:
        """Provide comprehensive dharmic guidance"""
        try:
            guidance_id = f"guidance_{request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine tradition to use
            tradition = request.tradition or DharmicTradition.UNIVERSAL
            
            # Analyze the question for relevant principles
            relevant_principles = await self._analyze_for_principles(request.question)
            
            # Generate primary guidance
            primary_guidance = await self._generate_primary_guidance(
                request.question,
                tradition,
                request.wisdom_level,
                relevant_principles
            )
            
            # Generate supporting wisdom
            supporting_wisdom = await self._generate_supporting_wisdom(
                request.question,
                tradition,
                relevant_principles
            )
            
            # Generate practical steps
            practical_steps = await self._generate_practical_steps(
                request.question,
                relevant_principles,
                request.wisdom_level
            )
            
            # Get multi-tradition perspectives
            tradition_perspectives = await self._get_tradition_perspectives(
                request.question,
                relevant_principles
            )
            
            # Generate deeper inquiry questions
            deeper_inquiry = await self._generate_deeper_inquiry(
                request.question,
                tradition,
                request.wisdom_level
            )
            
            # Generate contemplation points
            contemplation_points = await self._generate_contemplation_points(
                request.question,
                relevant_principles
            )
            
            # Calculate dharmic confidence
            dharmic_confidence = await self._calculate_dharmic_confidence(
                request.question,
                relevant_principles,
                tradition
            )
            
            response = DharmicGuidanceResponse(
                guidance_id=guidance_id,
                primary_guidance=primary_guidance,
                supporting_wisdom=supporting_wisdom,
                practical_steps=practical_steps,
                relevant_principles=relevant_principles,
                tradition_perspectives=tradition_perspectives,
                deeper_inquiry=deeper_inquiry,
                contemplation_points=contemplation_points,
                wisdom_level_applied=request.wisdom_level,
                dharmic_confidence=dharmic_confidence,
                tradition_used=tradition
            )
            
            # Store guidance history
            self.guidance_history.append(response)
            
            # Limit history size
            if len(self.guidance_history) > 1000:
                self.guidance_history = self.guidance_history[-500:]
            
            logger.info(f"Generated dharmic guidance {guidance_id} for user {request.user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error providing dharmic guidance: {e}")
            # Return fallback guidance
            return DharmicGuidanceResponse(
                guidance_id=f"fallback_{request.user_id}_{datetime.now().strftime('%H%M%S')}",
                primary_guidance="In times of uncertainty, return to the fundamental dharmic principles: practice non-violence, speak truthfully, and cultivate compassion for all beings.",
                supporting_wisdom=["Every challenge is an opportunity for spiritual growth"],
                practical_steps=["Take time for quiet reflection", "Practice loving-kindness toward yourself and others"],
                relevant_principles=[DharmicPrinciple.AHIMSA, DharmicPrinciple.SATYA],
                wisdom_level_applied=request.wisdom_level,
                dharmic_confidence=0.7,
                tradition_used=DharmicTradition.UNIVERSAL
            )
    
    async def _analyze_for_principles(self, question: str) -> List[DharmicPrinciple]:
        """Analyze question to identify relevant dharmic principles"""
        relevant_principles = []
        question_lower = question.lower()
        
        # Simple keyword-based analysis (can be enhanced with NLP)
        principle_keywords = {
            DharmicPrinciple.AHIMSA: ["violence", "harm", "hurt", "anger", "conflict", "peace", "compassion"],
            DharmicPrinciple.SATYA: ["truth", "lie", "honest", "deception", "authentic", "real"],
            DharmicPrinciple.ASTEYA: ["steal", "take", "borrow", "property", "time", "energy"],
            DharmicPrinciple.APARIGRAHA: ["attachment", "desire", "wanting", "greed", "possession", "letting go"],
            DharmicPrinciple.SANTOSHA: ["content", "satisfaction", "happiness", "gratitude", "enough"]
        }
        
        for principle, keywords in principle_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                relevant_principles.append(principle)
        
        # Always include core principles if none found
        if not relevant_principles:
            relevant_principles = [DharmicPrinciple.AHIMSA, DharmicPrinciple.SATYA]
        
        return relevant_principles[:3]  # Limit to top 3
    
    async def _generate_primary_guidance(
        self,
        question: str,
        tradition: DharmicTradition,
        wisdom_level: WisdomLevel,
        principles: List[DharmicPrinciple]
    ) -> str:
        """Generate primary dharmic guidance"""
        
        # Base guidance templates by wisdom level
        if wisdom_level == WisdomLevel.BEGINNER:
            guidance_template = "In the light of dharmic wisdom, this situation calls for {principle_guidance}. Remember that every action has consequences, and choosing the path of righteousness, though sometimes challenging, leads to lasting peace and growth."
        elif wisdom_level == WisdomLevel.INTERMEDIATE:
            guidance_template = "The dharmic path here involves understanding {principle_guidance}. This situation offers an opportunity to deepen your practice of {principles} and align more closely with your highest values."
        elif wisdom_level == WisdomLevel.ADVANCED:
            guidance_template = "From a dharmic perspective, this presents a chance to embody {principle_guidance}. Consider how {principles} can be integrated not just as practices, but as expressions of your deepest understanding."
        else:  # SAGE or ENLIGHTENED
            guidance_template = "The dharmic essence here transcends conventional approaches. {principle_guidance} becomes a gateway to recognizing the fundamental unity underlying apparent diversity, where {principles} are natural expressions of awakened consciousness."
        
        # Generate principle-specific guidance
        principle_guidance = self._get_principle_guidance(principles[0] if principles else DharmicPrinciple.AHIMSA)
        principles_text = ", ".join([p.value for p in principles])
        
        return guidance_template.format(
            principle_guidance=principle_guidance,
            principles=principles_text
        )
    
    def _get_principle_guidance(self, principle: DharmicPrinciple) -> str:
        """Get specific guidance for a dharmic principle"""
        guidance_map = {
            DharmicPrinciple.AHIMSA: "practicing non-violence in thought, word, and action, approaching all beings with compassion and understanding",
            DharmicPrinciple.SATYA: "speaking and living your truth while being sensitive to the impact of your words and actions on others",
            DharmicPrinciple.ASTEYA: "honoring others' time, energy, and resources while being generous with your own gifts",
            DharmicPrinciple.APARIGRAHA: "releasing attachment to outcomes and finding contentment in the present moment",
            DharmicPrinciple.SANTOSHA: "cultivating gratitude and finding joy in what is, rather than constantly seeking what is not"
        }
        
        return guidance_map.get(principle, "following the path of righteousness and compassion")
    
    async def _generate_supporting_wisdom(
        self,
        question: str,
        tradition: DharmicTradition,
        principles: List[DharmicPrinciple]
    ) -> List[str]:
        """Generate supporting wisdom quotes and insights"""
        
        universal_wisdom = [
            "The path of dharma is not always easy, but it is always worth walking.",
            "In every moment, we have the choice to align with our highest values.",
            "True strength comes not from dominating others, but from mastering oneself.",
            "Compassion is not weakness; it is the greatest strength of all.",
            "Every challenge is an opportunity to deepen our understanding and practice."
        ]
        
        tradition_specific = {
            DharmicTradition.HINDUISM: [
                "As the Bhagavad Gita teaches: 'It is better to perform one's own dharma imperfectly than to perform another's dharma perfectly.'",
                "The Upanishads remind us: 'You are what your deep, driving desire is.'"
            ],
            DharmicTradition.BUDDHISM: [
                "The Buddha taught: 'Hatred does not cease by hatred, but only by love; this is the eternal rule.'",
                "As it is written: 'Better than a thousand hollow words is one word that brings peace.'"
            ],
            DharmicTradition.YOGA: [
                "The Yoga Sutras teach: 'Yoga is the cessation of fluctuations in the mind.'",
                "Practice creates the foundation for wisdom to arise naturally."
            ]
        }
        
        wisdom = universal_wisdom[:2]
        if tradition in tradition_specific:
            wisdom.extend(tradition_specific[tradition][:1])
        
        return wisdom
    
    async def _generate_practical_steps(
        self,
        question: str,
        principles: List[DharmicPrinciple],
        wisdom_level: WisdomLevel
    ) -> List[str]:
        """Generate practical action steps"""
        
        base_steps = [
            "Take time for quiet reflection on the situation",
            "Consider how your actions align with your deepest values",
            "Practice compassionate understanding for all involved"
        ]
        
        principle_steps = {
            DharmicPrinciple.AHIMSA: [
                "Choose words and actions that promote harmony rather than division",
                "Practice loving-kindness meditation for difficult relationships"
            ],
            DharmicPrinciple.SATYA: [
                "Speak truthfully while considering the impact of your words",
                "Examine where you might be deceiving yourself"
            ],
            DharmicPrinciple.APARIGRAHA: [
                "Practice gratitude for what you already have",
                "Let go of attachment to specific outcomes"
            ]
        }
        
        steps = base_steps.copy()
        for principle in principles[:2]:  # Top 2 principles
            if principle in principle_steps:
                steps.extend(principle_steps[principle])
        
        return steps[:5]  # Limit to 5 steps
    
    async def _get_tradition_perspectives(
        self,
        question: str,
        principles: List[DharmicPrinciple]
    ) -> Dict[str, str]:
        """Get perspectives from different traditions"""
        
        perspectives = {
            "Hindu": "Dharmic duty (svadharma) guides right action based on one's nature and circumstances.",
            "Buddhist": "Right action arises from understanding interdependence and the impermanent nature of all phenomena.",
            "Jain": "The principle of ahimsa extends to the subtlest levels of thought and intention.",
            "Yogic": "Union with the divine is achieved through ethical living and spiritual practice."
        }
        
        return perspectives
    
    async def _generate_deeper_inquiry(
        self,
        question: str,
        tradition: DharmicTradition,
        wisdom_level: WisdomLevel
    ) -> List[str]:
        """Generate questions for deeper spiritual inquiry"""
        
        universal_inquiries = [
            "What would love do in this situation?",
            "How can this challenge serve my spiritual growth?",
            "What is the deeper lesson being offered here?"
        ]
        
        advanced_inquiries = [
            "What beliefs or attachments are being challenged in this situation?",
            "How might this experience be serving the highest good of all involved?",
            "What aspects of my ego are being revealed through this challenge?"
        ]
        
        if wisdom_level in [WisdomLevel.ADVANCED, WisdomLevel.SAGE, WisdomLevel.ENLIGHTENED]:
            return universal_inquiries + advanced_inquiries[:2]
        else:
            return universal_inquiries
    
    async def _generate_contemplation_points(
        self,
        question: str,
        principles: List[DharmicPrinciple]
    ) -> List[str]:
        """Generate points for contemplation"""
        
        points = [
            "Reflect on how this situation mirrors your inner spiritual work",
            "Consider the interconnectedness of all beings in this situation",
            "Contemplate what this experience is teaching you about yourself"
        ]
        
        principle_points = {
            DharmicPrinciple.AHIMSA: "Meditate on how non-violence can be expressed in thought, word, and deed",
            DharmicPrinciple.SATYA: "Reflect on the relationship between truth and compassion",
            DharmicPrinciple.APARIGRAHA: "Contemplate the freedom that comes from non-attachment"
        }
        
        for principle in principles[:2]:
            if principle in principle_points:
                points.append(principle_points[principle])
        
        return points
    
    async def _calculate_dharmic_confidence(
        self,
        question: str,
        principles: List[DharmicPrinciple],
        tradition: DharmicTradition
    ) -> float:
        """Calculate confidence in dharmic alignment of guidance"""
        
        base_confidence = 0.8
        
        # Boost confidence for well-defined principles
        if len(principles) >= 2:
            base_confidence += 0.1
        
        # Boost for specific tradition knowledge
        if tradition != DharmicTradition.UNIVERSAL:
            base_confidence += 0.05
        
        # Simple question complexity analysis
        if len(question.split()) > 20:  # Complex question
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    async def get_user_guidance_history(self, user_id: str) -> List[DharmicGuidanceResponse]:
        """Get guidance history for a user"""
        return [
            guidance for guidance in self.guidance_history
            if guidance.guidance_id.startswith(f"guidance_{user_id}_")
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check engine health"""
        return {
            "status": "healthy",
            "engine": "universal_dharmic",
            "traditions_supported": len(self.tradition_wisdom),
            "principles_mapped": len(self.principle_mappings),
            "guidance_provided": len(self.guidance_history)
        }

# Global engine instance
_universal_dharmic_engine: Optional[DharmicEngine] = None

async def get_universal_dharmic_engine() -> DharmicEngine:
    """Get the global universal dharmic engine instance"""
    global _universal_dharmic_engine
    
    if _universal_dharmic_engine is None:
        _universal_dharmic_engine = DharmicEngine()
        await _universal_dharmic_engine.initialize()
    
    return _universal_dharmic_engine