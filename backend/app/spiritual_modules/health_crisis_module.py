"""
🏥 Health Crisis Module - Spiritual Support in Medical Challenges
Dharmic approach to illness, healing, and conscious living through
health challenges. Based on Ayurvedic wisdom and spiritual perspectives.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HealthCrisisType(Enum):
    """Types of health crises"""
    ACUTE_ILLNESS = "acute_illness"
    CHRONIC_DISEASE = "chronic_disease"
    SURGERY = "surgery"
    MENTAL_HEALTH = "mental_health"
    TERMINAL_DIAGNOSIS = "terminal_diagnosis"
    RECOVERY = "recovery"
    FAMILY_ILLNESS = "family_illness"
    PREVENTIVE_CARE = "preventive_care"


class HealingApproach(Enum):
    """Different approaches to healing"""
    MEDICAL_SPIRITUAL = "medical_spiritual"  # Medical with spiritual
    AYURVEDIC = "ayurvedic"                  # Traditional Ayurvedic
    YOGIC = "yogic"                          # Yoga therapy
    DEVOTIONAL = "devotional"                # Bhakti and prayer
    KARMIC = "karmic"                        # Understanding as karma
    ACCEPTANCE = "acceptance"                # Surrendering to divine


@dataclass
class HealthGuidance:
    """Comprehensive health crisis guidance"""
    crisis_type: HealthCrisisType
    spiritual_perspective: str
    healing_approach: List[HealingApproach]
    daily_practices: List[str]
    mantras_prayers: List[str]
    dietary_guidance: List[str]
    emotional_support: List[str]
    family_guidance: List[str]
    meditation_practices: List[str]
    scriptural_wisdom: str
    practical_steps: List[str]


class HealthCrisisResponse(BaseModel):
    """Response from Health Crisis module"""
    crisis_type: str = Field(description="Type of health crisis")
    spiritual_perspective: str = Field(description="Dharmic view of illness")
    healing_guidance: List[str] = Field(
        description="Spiritual healing practices"
    )
    daily_support_practices: List[str] = Field(
        description="Daily practices"
    )
    emotional_healing: List[str] = Field(description="Emotional support")
    family_guidance: List[str] = Field(description="Guidance for family")
    prayer_mantras: List[str] = Field(
        description="Healing prayers and mantras"
    )
    dietary_recommendations: List[str] = Field(
        description="Healing foods"
    )
    medical_integration: List[str] = Field(
        description="Integrating medical/spiritual"
    )
    hope_wisdom: str = Field(description="Encouraging spiritual wisdom")


class HealthCrisisModule:
    """
    🏥 Health Crisis Module - Spiritual Support in Medical Challenges
    
    Based on dharmic understanding of health and illness:
    - Ayurvedic perspective on disease and healing
    - Karmic understanding of illness as spiritual opportunity
    - Integration of medical treatment with spiritual practice
    - Death preparation and conscious departure guidance
    - Family support during health crises
    
    Provides spiritual support while encouraging proper medical care.
    """
    
    def __init__(self):
        self.name = "Health Crisis"
        self.color = "🏥"
        self.element = "Healing Wisdom"
        self.principles = ["Divine Healing", "Medical Integration",
                           "Karmic Understanding", "Conscious Recovery"]
        self.guidance_types = self._initialize_health_guidance()
        self.healing_mantras = self._initialize_healing_mantras()
        self.dietary_healing = self._initialize_healing_foods()
        self.death_preparation = self._initialize_death_guidance()
    
    def _initialize_health_guidance(self) -> Dict[HealthCrisisType, HealthGuidance]:
        """Initialize guidance for different health crises"""
        return {
            HealthCrisisType.ACUTE_ILLNESS: HealthGuidance(
                crisis_type=HealthCrisisType.ACUTE_ILLNESS,
                spiritual_perspective=(
                    "Acute illness is often the body's way of forcing rest and "
                    "inner reflection. It can be a call to examine lifestyle, "
                    "stress levels, and spiritual practices. See it as divine "
                    "intervention to slow down and reconnect with what matters."
                ),
                healing_approach=[
                    HealingApproach.MEDICAL_SPIRITUAL,
                    HealingApproach.AYURVEDIC,
                    HealingApproach.DEVOTIONAL
                ],
                daily_practices=[
                    "Begin day with healing mantras and gratitude",
                    "Practice gentle pranayama if possible",
                    "Rest with awareness - conscious recuperation",
                    "Light meditation focusing on affected area",
                    "Evening prayer for healing and surrender"
                ],
                mantras_prayers=[
                    "Dhanvantari Mantra: 'Om Namo Bhagavate Vasudevaya "
                    "Dhanvantaraye Amrita Kalasha Hastaya Sarva Maya "
                    "Vinashaya Trailokya Nathaya Dhanvantari Maha Vishnave Namaha'",
                    "Mahamrityunjaya Mantra for healing and protection",
                    "Gayatri Mantra for overall divine blessing",
                    "Simple prayer: 'Divine Mother, heal my body, "
                    "mind and spirit'"
                ],
                dietary_guidance=[
                    "Light, easily digestible foods",
                    "Warm water and herbal teas",
                    "Fresh ginger for digestion and immunity",
                    "Avoid heavy, cold, or processed foods",
                    "Fast if guided and medically appropriate"
                ],
                emotional_support=[
                    "Accept illness as temporary and purposeful",
                    "Practice patience with recovery process",
                    "Ask for help from family and friends",
                    "Find meaning in the slowing down",
                    "Trust body's natural healing wisdom"
                ],
                family_guidance=[
                    "Provide loving care without anxiety",
                    "Support medical treatment decisions",
                    "Create peaceful healing environment",
                    "Include patient in family prayers",
                    "Maintain normal routines where possible"
                ],
                meditation_practices=[
                    "Body scan with healing light visualization",
                    "Loving-kindness meditation for self and body",
                    "Breath awareness to calm nervous system",
                    "Healing color meditation (golden light)"
                ],
                scriptural_wisdom=(
                    "Bhagavad Gita 2.47: 'You have the right to perform your "
                    "duties, but not to the fruits of action.' Apply this to "
                    "healing - do what you can, surrender the results."
                ),
                practical_steps=[
                    "Follow medical advice completely",
                    "Create healing space with spiritual objects",
                    "Maintain spiritual practices adapted to condition",
                    "Document insights and lessons from illness",
                    "Plan gradual return to normal activities"
                ]
            ),
            
            HealthCrisisType.CHRONIC_DISEASE: HealthGuidance(
                crisis_type=HealthCrisisType.CHRONIC_DISEASE,
                spiritual_perspective=(
                    "Chronic illness is a profound spiritual teacher, offering "
                    "opportunities for deep surrender, patience, and the "
                    "cultivation of inner strength. It calls us to find meaning "
                    "beyond physical comfort and to develop unwavering faith."
                ),
                healing_approach=[
                    HealingApproach.ACCEPTANCE,
                    HealingApproach.AYURVEDIC,
                    HealingApproach.YOGIC,
                    HealingApproach.KARMIC
                ],
                daily_practices=[
                    "Morning acceptance and gratitude practice",
                    "Gentle yoga adapted to physical limitations",
                    "Regular meditation for pain and symptom management",
                    "Energy conservation and mindful activity",
                    "Evening reflection on daily blessings"
                ],
                mantras_prayers=[
                    "Acceptance mantra: 'Thy will be done, Divine Mother'",
                    "Patience mantra: 'Om Gam Ganapataye Namaha' for removing obstacles",
                    "Strength mantra: 'Om Hum Hanumate Namaha'",
                    "Healing prayers to Dhanvantari"
                ],
                dietary_guidance=[
                    "Anti-inflammatory foods and spices",
                    "Regular meal times for digestive stability",
                    "Avoid foods that trigger symptoms",
                    "Nutrient-dense, healing foods",
                    "Mindful eating as spiritual practice"
                ],
                emotional_support=[
                    "Develop acceptance without giving up hope",
                    "Find new sources of meaning and purpose",
                    "Build community of others with similar challenges",
                    "Practice self-compassion for limitations",
                    "Celebrate small improvements and good days"
                ],
                family_guidance=[
                    "Learn about the condition to provide better support",
                    "Adapt family activities to include patient",
                    "Maintain hope while accepting reality",
                    "Share spiritual practices as family",
                    "Seek support for caregivers too"
                ],
                meditation_practices=[
                    "Pain meditation - observing without resistance",
                    "Loving-kindness for the diseased body parts",
                    "Surrender meditation - offering illness to Divine",
                    "Gratitude meditation for functioning body parts"
                ],
                scriptural_wisdom=(
                    "From Isavasya Upanishad: 'Whatever happens, happens for good. "
                    "What is happening, is happening for good. What will happen, "
                    "will also happen for good.' Trust in divine purpose."
                ),
                practical_steps=[
                    "Establish sustainable daily routine",
                    "Learn stress management techniques",
                    "Create support network of patients and caregivers",
                    "Explore complementary healing modalities",
                    "Maintain spiritual practices consistently"
                ]
            ),
            
            HealthCrisisType.TERMINAL_DIAGNOSIS: HealthGuidance(
                crisis_type=HealthCrisisType.TERMINAL_DIAGNOSIS,
                spiritual_perspective=(
                    "Terminal diagnosis is an invitation to prepare consciously "
                    "for the great transition. It offers time to complete "
                    "relationships, share wisdom, and prepare the soul for "
                    "its journey beyond this body."
                ),
                healing_approach=[
                    HealingApproach.ACCEPTANCE,
                    HealingApproach.DEVOTIONAL,
                    HealingApproach.KARMIC
                ],
                daily_practices=[
                    "Life review and gratitude practice",
                    "Forgiveness work - giving and receiving",
                    "Sharing wisdom and love with family",
                    "Prayer and devotional practices",
                    "Conscious breathing and meditation"
                ],
                mantras_prayers=[
                    "Transition mantra: 'Om Mani Padme Hum'",
                    "Surrender prayer: 'Into your hands I commend my spirit'",
                    "Peace mantra: 'Om Shanti Shanti Shanti'",
                    "Ram Nam for dying process"
                ],
                dietary_guidance=[
                    "Whatever brings comfort and nourishment",
                    "Sacred foods - prasadam when possible",
                    "Light, easy to digest meals",
                    "Blessed water and herbal teas"
                ],
                emotional_support=[
                    "Process grief and fear with compassion",
                    "Focus on love and connection",
                    "Complete unfinished business",
                    "Find meaning in the life lived",
                    "Prepare for conscious departure"
                ],
                family_guidance=[
                    "Create sacred space for dying process",
                    "Share spiritual practices together",
                    "Help complete practical and emotional affairs",
                    "Provide comfort without denying reality",
                    "Support conscious dying process"
                ],
                meditation_practices=[
                    "Death meditation - contemplating impermanence",
                    "Light meditation - merging with divine light",
                    "Breath meditation - releasing attachment",
                    "Love meditation - expanding heart beyond body"
                ],
                scriptural_wisdom=(
                    "Bhagavad Gita 2.20: 'The soul is neither born nor does it die. "
                    "It is not slain when the body is slain.' Death is transition, "
                    "not ending."
                ),
                practical_steps=[
                    "Complete advance directives and wills",
                    "Record messages for loved ones",
                    "Create legacy of wisdom and values",
                    "Arrange for spiritual support during dying",
                    "Prepare sacred environment for transition"
                ]
            )
        }
    
    def _initialize_healing_mantras(self) -> Dict[str, List[str]]:
        """Initialize healing mantras for different conditions"""
        return {
            "general_healing": [
                "Om Tryambakam Yajamahe Sugandhim Pushti Vardhanam",
                "Om Namo Bhagavate Vasudevaya",
                "Gayatri Mantra for divine healing energy"
            ],
            "pain_relief": [
                "Om Gam Ganapataye Namaha",
                "Om Hrim Shrim Klim Parameshwari Swaha"
            ],
            "mental_healing": [
                "Om Namah Shivaya",
                "So Hum - I am That"
            ]
        }
    
    def _initialize_healing_foods(self) -> Dict[str, List[str]]:
        """Initialize healing foods for different conditions"""
        return {
            "immune_support": [
                "Turmeric with warm milk",
                "Fresh ginger tea",
                "Amla (Indian gooseberry)",
                "Tulsi (holy basil) tea"
            ],
            "digestive_healing": [
                "Cumin-coriander-fennel tea",
                "Rice porridge with ghee",
                "Buttermilk with roasted cumin",
                "Cooked apples with cinnamon"
            ]
        }
    
    def _initialize_death_guidance(self) -> Dict[str, Any]:
        """Initialize guidance for conscious dying"""
        return {
            "preparation": [
                "Complete relationships and forgive",
                "Share wisdom and blessings",
                "Surrender attachments gradually",
                "Focus on divine connection"
            ],
            "process": [
                "Maintain awareness during transition",
                "Chant divine names",
                "Focus on light and love",
                "Release body with gratitude"
            ]
        }
    
    def assess_crisis_type(self, query: str, context: Dict[str, Any]) -> HealthCrisisType:
        """Assess the type of health crisis from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["terminal", "dying", "end stage", "hospice"]):
            return HealthCrisisType.TERMINAL_DIAGNOSIS
        elif any(word in query_lower for word in ["chronic", "long term", "permanent", "ongoing"]):
            return HealthCrisisType.CHRONIC_DISEASE
        elif any(word in query_lower for word in ["surgery", "operation", "procedure"]):
            return HealthCrisisType.SURGERY
        elif any(word in query_lower for word in ["depression", "anxiety", "mental", "emotional"]):
            return HealthCrisisType.MENTAL_HEALTH
        elif any(word in query_lower for word in ["family", "loved one", "relative"]):
            return HealthCrisisType.FAMILY_ILLNESS
        elif any(word in query_lower for word in ["recovery", "healing", "getting better"]):
            return HealthCrisisType.RECOVERY
        else:
            return HealthCrisisType.ACUTE_ILLNESS
    
    async def process_health_crisis_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> HealthCrisisResponse:
        """Process health crisis query and provide spiritual support"""
        try:
            context = user_context or {}
            
            # Assess crisis type
            crisis_type = self.assess_crisis_type(query, context)
            
            # Get guidance
            guidance = self.guidance_types.get(crisis_type)
            if not guidance:
                return self._create_fallback_response()
            
            return HealthCrisisResponse(
                crisis_type=crisis_type.value,
                spiritual_perspective=guidance.spiritual_perspective,
                healing_guidance=guidance.daily_practices,
                daily_support_practices=guidance.daily_practices,
                emotional_healing=guidance.emotional_support,
                family_guidance=guidance.family_guidance,
                prayer_mantras=guidance.mantras_prayers,
                dietary_recommendations=guidance.dietary_guidance,
                medical_integration=guidance.practical_steps,
                hope_wisdom=guidance.scriptural_wisdom
            )
            
        except Exception as e:
            logger.error(f"Error processing health crisis query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> HealthCrisisResponse:
        """Create fallback response when processing fails"""
        return HealthCrisisResponse(
            crisis_type="general_support",
            spiritual_perspective="Every health challenge is an opportunity for spiritual growth and deeper faith",
            healing_guidance=["Combine medical treatment with spiritual practice", "Trust in divine healing power"],
            daily_support_practices=["Morning prayer for healing", "Gentle spiritual practices", "Evening gratitude"],
            emotional_healing=["Accept help from others", "Practice patience with recovery", "Find meaning in the experience"],
            family_guidance=["Provide loving support", "Maintain hope and faith", "Create peaceful environment"],
            prayer_mantras=["Om Namo Bhagavate Vasudevaya", "Mahamrityunjaya Mantra", "Simple heartfelt prayers"],
            dietary_recommendations=["Nutritious, easy to digest foods", "Plenty of fluids", "Foods that bring comfort"],
            medical_integration=["Follow medical advice completely", "Communicate openly with healthcare providers"],
            hope_wisdom="The body may be challenged, but the spirit remains whole and connected to the Divine"
        )


# Global instance
_health_crisis_module = None

def get_health_crisis_module() -> HealthCrisisModule:
    """Get global Health Crisis module instance"""
    global _health_crisis_module
    if _health_crisis_module is None:
        _health_crisis_module = HealthCrisisModule()
    return _health_crisis_module

# Factory function for easy access
def create_health_crisis_guidance(query: str, context: Optional[Dict[str, Any]] = None) -> HealthCrisisResponse:
    """Create health crisis guidance response"""
    module = get_health_crisis_module()
    import asyncio
    return asyncio.run(module.process_health_crisis_query(query, context))
