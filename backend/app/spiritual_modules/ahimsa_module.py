"""
Ahimsa Module - Non-Violence and Compassion System
==================================================

Provides guidance on practicing ahimsa (non-violence) in thought, word, and action.
Focuses on cultivating compassion, preventing harm, and promoting peaceful coexistence
based on the fundamental dharmic principle of ahimsa.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

class ViolenceType(Enum):
    """Types of violence according to dharmic understanding"""
    PHYSICAL = "physical_harm"        # Physical violence to beings
    MENTAL = "mental_harm"           # Psychological/emotional harm  
    VERBAL = "verbal_harm"           # Harsh or hurtful speech
    ENVIRONMENTAL = "environmental"   # Harm to nature/environment
    SYSTEMATIC = "systematic"        # Structural/institutional harm
    SUBTLE = "subtle_harm"          # Unconscious or indirect harm

class AhimsaLevel(Enum):
    """Levels of ahimsa practice"""
    BASIC = "basic_non_harm"         # Avoiding obvious harm
    CONSCIOUS = "conscious_care"      # Mindful prevention of harm
    COMPASSIONATE = "active_compassion"  # Actively helping others
    UNIVERSAL = "universal_love"      # Love for all beings

@dataclass
class AhimsaInsight:
    """An insight about practicing non-violence"""
    situation: str
    violence_type: ViolenceType
    ahimsa_response: str
    compassionate_action: str
    dharmic_principle: str
    practical_steps: List[str]
    related_teachings: List[str]
    timestamp: datetime

@dataclass
class ConflictResolution:
    """Framework for resolving conflicts through ahimsa"""
    conflict_type: str
    root_causes: List[str]
    ahimsa_approach: str
    communication_strategy: str
    healing_actions: List[str]
    prevention_measures: List[str]
    success_indicators: List[str]

class AhimsaModule:
    """
    Ahimsa (Non-Violence) Module
    
    Provides guidance on practicing non-violence in all aspects of life,
    cultivating compassion, and resolving conflicts through peaceful means
    based on the fundamental dharmic principle of ahimsa.
    """
    
    def __init__(self):
        self.module_name = "Ahimsa Module"
        self.element = "Love/Compassion"
        self.color = "Green"
        self.mantra = "MAITRI"  # Loving-kindness
        self.deity = "Vishnu"  # Preserver, protector
        self.principles = ["Non-violence", "Compassion", "Universal Love", "Peaceful Resolution"]
        self.violence_indicators = self._initialize_violence_detection()
        self.compassion_practices = self._initialize_compassion_practices()
        self.conflict_resolution_frameworks = self._initialize_conflict_frameworks()
        logger.info(f"Initialized {self.module_name} with ahimsa guidance systems")
    
    def _initialize_violence_detection(self) -> Dict[ViolenceType, List[str]]:
        """Initialize violence detection patterns"""
        return {
            ViolenceType.PHYSICAL: [
                "physical aggression", "hitting", "pushing", "animal cruelty",
                "destroying property", "force", "coercion"
            ],
            ViolenceType.MENTAL: [
                "humiliation", "intimidation", "bullying", "manipulation",
                "gaslighting", "emotional abuse", "psychological pressure"
            ],
            ViolenceType.VERBAL: [
                "harsh speech", "insults", "threats", "aggressive language",
                "hate speech", "degrading comments", "verbal abuse"
            ],
            ViolenceType.ENVIRONMENTAL: [
                "pollution", "waste", "destruction of nature", "overconsumption",
                "habitat destruction", "species extinction", "climate damage"
            ],
            ViolenceType.SYSTEMATIC: [
                "discrimination", "inequality", "oppression", "exploitation",
                "unfair systems", "structural violence", "institutional bias"
            ],
            ViolenceType.SUBTLE: [
                "unconscious bias", "microaggressions", "passive aggression",
                "indirect harm", "neglect", "indifference", "thoughtlessness"
            ]
        }
    
    def _initialize_compassion_practices(self) -> Dict[AhimsaLevel, List[str]]:
        """Initialize compassion cultivation practices"""
        return {
            AhimsaLevel.BASIC: [
                "Avoid causing obvious harm to any being",
                "Practice gentle speech and kind words",
                "Be mindful of your actions' impact on others",
                "Choose non-violent solutions to problems"
            ],
            AhimsaLevel.CONSCIOUS: [
                "Actively look for ways to help rather than harm",
                "Practice empathy and understanding in conflicts",
                "Make conscious choices for non-violent products",
                "Cultivate patience and tolerance"
            ],
            AhimsaLevel.COMPASSIONATE: [
                "Actively work to reduce suffering in the world",
                "Practice metta (loving-kindness) meditation",
                "Engage in service to protect vulnerable beings",
                "Transform anger into compassion"
            ],
            AhimsaLevel.UNIVERSAL: [
                "See the divine in all beings",
                "Practice unconditional love and acceptance",
                "Work for peace and harmony at all levels",
                "Embody ahimsa as your natural state"
            ]
        }
    
    def _initialize_conflict_frameworks(self) -> List[ConflictResolution]:
        """Initialize conflict resolution frameworks"""
        return [
            ConflictResolution(
                conflict_type="Personal Disagreement",
                root_causes=["Misunderstanding", "Ego", "Different perspectives"],
                ahimsa_approach="Listen with compassion, speak truth gently",
                communication_strategy="Non-violent communication, active listening",
                healing_actions=["Apology if needed", "Finding common ground", "Mutual understanding"],
                prevention_measures=["Regular communication", "Empathy practice", "Respect boundaries"],
                success_indicators=["Mutual respect", "Understanding", "Peaceful resolution"]
            ),
            ConflictResolution(
                conflict_type="Environmental Harm",
                root_causes=["Ignorance", "Greed", "Disconnection from nature"],
                ahimsa_approach="Educate gently, model sustainable behavior",
                communication_strategy="Share knowledge with love, not judgment",
                healing_actions=["Environmental restoration", "Sustainable practices", "Nature connection"],
                prevention_measures=["Education", "Sustainable choices", "Environmental awareness"],
                success_indicators=["Reduced harm", "Increased awareness", "Sustainable actions"]
            ),
            ConflictResolution(
                conflict_type="Social Injustice",
                root_causes=["Ignorance", "Fear", "Historical conditioning"],
                ahimsa_approach="Work for justice through peaceful means",
                communication_strategy="Truth with compassion, non-violent resistance",
                healing_actions=["Education", "Dialogue", "Systemic change"],
                prevention_measures=["Awareness raising", "Inclusive practices", "Equal opportunities"],
                success_indicators=["Reduced discrimination", "Equal treatment", "Harmonious society"]
            )
        ]
    
    async def process_ahimsa_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process ahimsa-related queries and provide guidance"""
        context = user_context or {}
        
        # Assess situation for violence and suggest ahimsa response
        assessment = await self.assess_situation_for_violence(query, context)
        
        return {
            "query": query,
            "ahimsa_analysis": assessment,
            "non_violent_guidance": await self._generate_non_violent_guidance(query, context),
            "compassion_practices": await self.get_compassion_practices(context),
            "daily_ahimsa": await self.daily_ahimsa_practice(context),
            "dharmic_wisdom": self._get_ahimsa_wisdom(query),
            "practical_tools": self._get_practical_tools()
        }
    
    async def assess_situation_for_violence(self, situation: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Assess a situation for potential violence and suggest ahimsa response"""
        context = context or {}
        
        # Detect types of violence present
        violence_detected = []
        for violence_type, indicators in self.violence_indicators.items():
            for indicator in indicators:
                if indicator.lower() in situation.lower():
                    violence_detected.append(violence_type)
                    break
        
        # Generate ahimsa response
        ahimsa_guidance = await self._generate_ahimsa_response(situation, violence_detected, context)
        
        # Suggest compassionate actions
        compassionate_actions = self._suggest_compassionate_actions(violence_detected, context)
        
        # Provide conflict resolution if needed
        resolution_framework = self._get_conflict_resolution(violence_detected)
        
        return {
            "situation_analysis": {
                "violence_types_detected": [vt.value for vt in violence_detected],
                "severity": self._assess_severity(violence_detected),
                "urgency": self._assess_urgency(situation, context)
            },
            "ahimsa_guidance": ahimsa_guidance,
            "compassionate_actions": compassionate_actions,
            "conflict_resolution": resolution_framework,
            "dharmic_teachings": self._get_relevant_teachings(violence_detected),
            "practical_steps": self._generate_practical_steps(violence_detected)
        }
    
    async def _generate_ahimsa_response(self, situation: str, violence_types: List[ViolenceType], context: Dict[str, Any]) -> str:
        """Generate ahimsa-based response to the situation"""
        base_response = "🕊️ The path of ahimsa (non-violence) guides us to respond with compassion rather than react with harm. "
        
        if not violence_types:
            return base_response + "In this situation, continue to practice mindful awareness to ensure no subtle harm occurs. Let love and understanding guide your actions."
        
        specific_guidance = []
        
        if ViolenceType.PHYSICAL in violence_types:
            specific_guidance.append("Physical safety is paramount. Remove yourself and others from danger while seeking peaceful resolution.")
        
        if ViolenceType.MENTAL in violence_types:
            specific_guidance.append("Protect your mental wellbeing and that of others. Practice compassionate boundaries and seek support if needed.")
        
        if ViolenceType.VERBAL in violence_types:
            specific_guidance.append("Respond to harsh words with gentle truth. 'Satyam vada, priyam vada' - speak truth, speak kindly.")
        
        if ViolenceType.ENVIRONMENTAL in violence_types:
            specific_guidance.append("Honor Mother Earth by choosing sustainable actions. Every choice can be an act of love for all beings.")
        
        if ViolenceType.SYSTEMATIC in violence_types:
            specific_guidance.append("Work for justice through peaceful means. Transform systems with truth, love, and persistent non-violent action.")
        
        if ViolenceType.SUBTLE in violence_types:
            specific_guidance.append("Cultivate awareness of subtle harm. Practice mindfulness in all interactions and intentions.")
        
        return base_response + " ".join(specific_guidance)
    
    def _suggest_compassionate_actions(self, violence_types: List[ViolenceType], context: Dict[str, Any]) -> List[str]:
        """Suggest specific compassionate actions"""
        actions = []
        
        # Universal compassionate actions
        actions.extend([
            "Practice loving-kindness meditation for all involved",
            "Seek to understand rather than to be understood",
            "Respond with patience and gentleness"
        ])
        
        # Specific actions based on violence types
        for violence_type in violence_types:
            if violence_type == ViolenceType.PHYSICAL:
                actions.extend([
                    "Ensure physical safety of all beings",
                    "Seek professional help if violence is ongoing",
                    "Practice self-defense only as last resort for protection"
                ])
            elif violence_type == ViolenceType.MENTAL:
                actions.extend([
                    "Offer emotional support and listening ear",
                    "Practice forgiveness (not condoning, but freeing yourself)",
                    "Seek counseling or therapy when appropriate"
                ])
            elif violence_type == ViolenceType.VERBAL:
                actions.extend([
                    "Speak truthfully but with kindness",
                    "Use 'I' statements to express feelings without blame",
                    "Practice silence when words might cause more harm"
                ])
            elif violence_type == ViolenceType.ENVIRONMENTAL:
                actions.extend([
                    "Choose eco-friendly alternatives",
                    "Support conservation efforts",
                    "Practice gratitude for nature's gifts"
                ])
        
        return list(set(actions))  # Remove duplicates
    
    def _get_conflict_resolution(self, violence_types: List[ViolenceType]) -> Optional[ConflictResolution]:
        """Get appropriate conflict resolution framework"""
        if not violence_types:
            return None
        
        # Return most relevant framework
        if ViolenceType.ENVIRONMENTAL in violence_types:
            return next((cr for cr in self.conflict_resolution_frameworks 
                        if cr.conflict_type == "Environmental Harm"), None)
        elif ViolenceType.SYSTEMATIC in violence_types:
            return next((cr for cr in self.conflict_resolution_frameworks 
                        if cr.conflict_type == "Social Injustice"), None)
        else:
            return next((cr for cr in self.conflict_resolution_frameworks 
                        if cr.conflict_type == "Personal Disagreement"), None)
    
    def _assess_severity(self, violence_types: List[ViolenceType]) -> str:
        """Assess severity of violence detected"""
        if not violence_types:
            return "none"
        elif ViolenceType.PHYSICAL in violence_types:
            return "high"
        elif len(violence_types) > 2:
            return "medium"
        else:
            return "low"
    
    def _assess_urgency(self, situation: str, context: Dict[str, Any]) -> str:
        """Assess urgency of response needed"""
        urgent_indicators = ["emergency", "immediate", "danger", "crisis", "urgent", "now"]
        
        if any(indicator in situation.lower() for indicator in urgent_indicators):
            return "immediate"
        elif context.get("time_sensitive", False):
            return "soon"
        else:
            return "when_appropriate"
    
    def _get_relevant_teachings(self, violence_types: List[ViolenceType]) -> List[str]:
        """Get relevant dharmic teachings for the situation"""
        teachings = [
            "Ahimsa paramo dharma - Non-violence is the highest virtue",
            "Sarve bhavantu sukhinah - May all beings be happy",
            "Vasudhaiva kutumbakam - The world is one family"
        ]
        
        if ViolenceType.VERBAL in violence_types:
            teachings.append("Satyam vada, priyam vada - Speak truth, speak kindly")
        
        if ViolenceType.MENTAL in violence_types:
            teachings.append("Kshama viraaya bhushanam - Forgiveness is the ornament of the brave")
        
        if ViolenceType.ENVIRONMENTAL in violence_types:
            teachings.append("Mata bhumih putro aham prithivyah - Earth is my mother, I am her child")
        
        return teachings
    
    def _generate_practical_steps(self, violence_types: List[ViolenceType]) -> List[str]:
        """Generate practical steps for implementing ahimsa"""
        steps = [
            "Pause and breathe before reacting",
            "Ask 'How can I respond with love?'",
            "Practice the golden rule: treat others as you wish to be treated",
            "Seek win-win solutions that benefit all"
        ]
        
        if violence_types:
            steps.extend([
                "Address the root cause, not just symptoms",
                "Practice restorative rather than punitive justice",
                "Transform anger into compassion through understanding"
            ])
        
        return steps
    
    async def _generate_non_violent_guidance(self, query: str, context: Dict[str, Any]) -> str:
        """Generate specific non-violent guidance"""
        base_guidance = "The principle of ahimsa invites us to find the most compassionate response in every situation. "
        
        if "conflict" in query.lower():
            return base_guidance + "In conflicts, remember that everyone is fighting their own battles. Seek understanding first, then resolution."
        elif "anger" in query.lower():
            return base_guidance + "Anger is often a sign of unmet needs. Transform anger into compassion by understanding what you truly need."
        elif "environment" in query.lower():
            return base_guidance + "Every choice is an opportunity to honor or harm our Earth. Choose with love for all beings."
        else:
            return base_guidance + "Practice ahimsa by asking: 'How can I bring more love and less harm to this situation?'"
    
    async def get_compassion_practices(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get compassion cultivation practices"""
        level = context.get("ahimsa_level", AhimsaLevel.BASIC)
        practices = self.compassion_practices.get(level, self.compassion_practices[AhimsaLevel.BASIC])
        
        return {
            "current_level": level.value,
            "practices": practices,
            "meditation_guidance": {
                "loving_kindness": [
                    "Begin with yourself: 'May I be happy, peaceful, free from suffering'",
                    "Extend to loved ones",
                    "Include neutral people",
                    "Include difficult people",
                    "Extend to all beings"
                ],
                "duration": "15-30 minutes daily",
                "benefits": [
                    "Reduces anger and hostility",
                    "Increases emotional resilience",
                    "Improves relationships",
                    "Promotes inner peace"
                ]
            }
        }
    
    def _get_ahimsa_wisdom(self, query: str) -> str:
        """Get relevant ahimsa wisdom"""
        wisdom_map = {
            "anger": "When anger arises, remember: it is a teacher showing us our attachments. Transform it with understanding.",
            "conflict": "Every conflict is an opportunity to practice deeper compassion and understanding.",
            "environment": "The Earth is our mother. How we treat her reflects our spiritual understanding.",
            "relationships": "All relationships are mirrors for practicing unconditional love.",
            "default": "Ahimsa is not mere non-violence, but the positive state of love that draws all beings to itself."
        }
        
        for key, wisdom in wisdom_map.items():
            if key in query.lower():
                return wisdom
        
        return wisdom_map["default"]
    
    def _get_practical_tools(self) -> Dict[str, Any]:
        """Get practical tools for ahimsa practice"""
        return {
            "non_violent_communication": {
                "steps": [
                    "Observe without evaluating",
                    "Express feelings without blame",
                    "Identify underlying needs",
                    "Make specific requests, not demands"
                ],
                "example": "When I see... I feel... because I need... Would you be willing to...?"
            },
            "pause_practice": {
                "description": "Before reacting, pause and ask three questions",
                "questions": [
                    "Is it true?",
                    "Is it necessary?", 
                    "Is it kind?"
                ]
            },
            "compassion_mantras": [
                "May all beings be happy",
                "May all beings be free from suffering",
                "May all beings live in peace",
                "Gate gate paragate parasamgate bodhi svaha"
            ]
        }
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get current status of the Ahimsa Module"""
        return {
            "name": self.module_name,
            "state": "active",
            "element": self.element,
            "color": self.color,
            "mantra": self.mantra,
            "governing_deity": self.deity,
            "core_principles": self.principles,
            "primary_functions": [
                "Violence detection and prevention",
                "Compassion cultivation",
                "Conflict resolution through ahimsa",
                "Non-violent communication guidance"
            ],
            "wisdom_available": "Guidance for practicing non-violence in thought, word, and action"
        }
    
    async def daily_ahimsa_practice(self, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Provide daily ahimsa practice suggestions"""
        context = user_context or {}
        
        morning_practices = [
            "Set intention to cause no harm today",
            "Practice loving-kindness meditation",
            "Choose compassionate responses to anticipated challenges"
        ]
        
        throughout_day = [
            "Before speaking, ask: 'Is it true? Is it necessary? Is it kind?'",
            "Practice patience in difficult situations",
            "Look for opportunities to help rather than harm",
            "Choose non-violent options for food, products, entertainment"
        ]
        
        evening_reflection = [
            "Review the day for moments of violence or compassion",
            "Forgive yourself for any harm caused",
            "Appreciate moments when you chose love over fear",
            "Set intention for greater compassion tomorrow"
        ]
        
        return {
            "morning_practices": morning_practices,
            "throughout_day": throughout_day,
            "evening_reflection": evening_reflection,
            "weekly_focus": "Choose one area of your life to practice deeper ahimsa",
            "monthly_goal": "Expand your circle of compassion to include previously difficult relationships",
            "reminder": "Ahimsa begins with yourself - practice self-compassion as the foundation for universal love"
        }

# Global instance for easy import
ahimsa_module = AhimsaModule()

def get_ahimsa_module():
    """Get the global ahimsa module instance"""
    return ahimsa_module
