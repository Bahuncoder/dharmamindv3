"""
🟢 Wellness Module - Holistic Health and Ayurvedic Living
Complete wellness system based on Ayurveda and traditional healing
Integrates mind-body-spirit health through natural principles
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class Dosha(Enum):
    """Three constitutional types in Ayurveda"""
    VATA = "vata"      # Air + Space (movement, nervous system)
    PITTA = "pitta"    # Fire + Water (metabolism, digestion)
    KAPHA = "kapha"    # Earth + Water (structure, immunity)

class WellnessAspect(Enum):
    """Aspects of holistic wellness"""
    PHYSICAL = "physical"     # Body health and vitality
    MENTAL = "mental"         # Mind clarity and emotional balance
    SPIRITUAL = "spiritual"   # Soul connection and purpose
    ENERGETIC = "energetic"   # Prana and chakra balance
    SOCIAL = "social"         # Relationships and community
    ENVIRONMENTAL = "environmental"  # Living space and nature connection

class HealthImbalance(Enum):
    """Common health imbalances"""
    STRESS = "stress"                    # Mental and physical tension
    DIGESTIVE_ISSUES = "digestion"       # Agni (digestive fire) problems
    SLEEP_DISORDERS = "sleep"            # Rest and recovery issues
    ENERGY_DEPLETION = "energy"          # Lack of vitality
    EMOTIONAL_INSTABILITY = "emotions"   # Mood and mental health
    CHRONIC_PAIN = "pain"               # Physical discomfort
    ADDICTION = "addiction"             # Dependency patterns

class Season(Enum):
    """Seasons for Ayurvedic living"""
    SPRING = "spring"    # Kapha season
    SUMMER = "summer"    # Pitta season
    AUTUMN = "autumn"    # Vata season
    WINTER = "winter"    # Kapha season

@dataclass
class DoshaProfile:
    """Individual dosha constitution and current state"""
    dominant_dosha: Dosha
    secondary_dosha: Optional[Dosha]
    current_imbalances: List[Dosha]
    strengths: List[str]
    tendencies: List[str]
    recommendations: List[str]

@dataclass
class WellnessGuidance:
    """Comprehensive wellness guidance"""
    dosha_focus: Dosha
    health_aspect: WellnessAspect
    daily_routine: List[str]
    dietary_guidelines: List[str]
    lifestyle_practices: List[str]
    seasonal_adjustments: Dict[Season, List[str]]
    healing_herbs: List[str]
    yoga_practices: List[str]
    meditation_techniques: List[str]
    warning_signs: List[str]

class WellnessInsight(BaseModel):
    """Insight from wellness module"""
    dosha_type: str = Field(description="Primary dosha type")
    imbalance_area: str = Field(description="Area of potential imbalance")
    recommendation: str = Field(description="Primary recommendation")
    practice: str = Field(description="Suggested practice")

class WellnessResponse(BaseModel):
    """Response from Wellness module"""
    dosha_assessment: str = Field(description="Primary dosha constitution")
    current_imbalances: List[str] = Field(description="Identified health imbalances")
    wellness_guidance: str = Field(description="Primary wellness guidance")
    daily_routine: List[str] = Field(description="Recommended daily routine")
    dietary_recommendations: List[str] = Field(description="Dietary guidance")
    lifestyle_practices: List[str] = Field(description="Lifestyle recommendations")
    healing_modalities: List[str] = Field(description="Suggested healing approaches")
    seasonal_guidance: str = Field(description="Current seasonal recommendations")
    prevention_strategies: List[str] = Field(description="Disease prevention strategies")
    spiritual_practices: List[str] = Field(description="Spiritual wellness practices")

class WellnessModule:
    """
    🟢 Wellness Module - Holistic Health and Ayurvedic Living
    
    Based on authentic Ayurveda, yoga, and traditional healing systems
    Provides personalized guidance for mind-body-spirit wellness
    Integrates seasonal living, dosha balancing, and spiritual health
    """
    
    def __init__(self):
        self.name = "Wellness"
        self.color = "🟢"
        self.element = "Life Force (Prana)"
        self.principles = ["Balance", "Prevention", "Natural Healing", "Mind-Body-Spirit Unity"]
        self.dosha_profiles = self._initialize_dosha_profiles()
        self.wellness_guidance = self._initialize_wellness_guidance()
        self.seasonal_practices = self._initialize_seasonal_practices()
        self.healing_modalities = self._initialize_healing_modalities()
        
    def _initialize_dosha_profiles(self) -> Dict[Dosha, DoshaProfile]:
        """Initialize dosha profiles and characteristics"""
        return {
            Dosha.VATA: DoshaProfile(
                dominant_dosha=Dosha.VATA,
                secondary_dosha=None,
                current_imbalances=[],
                strengths=[
                    "Creative and imaginative",
                    "Quick thinking and adaptable",
                    "Enthusiastic and energetic",
                    "Good at initiating projects",
                    "Spiritually inclined"
                ],
                tendencies=[
                    "Irregular appetite and digestion",
                    "Variable energy levels",
                    "Tendency toward anxiety and worry",
                    "Dry skin and hair",
                    "Light sleep and possible insomnia",
                    "Cold hands and feet",
                    "Preference for warm climates"
                ],
                recommendations=[
                    "Regular, warm, oily foods",
                    "Consistent daily routine",
                    "Adequate rest and relaxation",
                    "Gentle, grounding exercises",
                    "Warm oil massages",
                    "Meditation and pranayama",
                    "Avoiding excessive stimulation"
                ]
            ),
            
            Dosha.PITTA: DoshaProfile(
                dominant_dosha=Dosha.PITTA,
                secondary_dosha=None,
                current_imbalances=[],
                strengths=[
                    "Strong digestion and metabolism",
                    "Natural leadership abilities",
                    "Focused and determined",
                    "Good at planning and organizing",
                    "Sharp intellect and memory"
                ],
                tendencies=[
                    "Strong appetite, irritable when hungry",
                    "Tendency toward anger and impatience",
                    "Sensitive to heat",
                    "Prone to skin rashes and inflammation",
                    "Perfectionist tendencies",
                    "Competitive nature",
                    "Prone to heartburn and acidity"
                ],
                recommendations=[
                    "Cool, sweet, and bitter foods",
                    "Moderate exercise, avoid overheating",
                    "Cool environments and practices",
                    "Stress management techniques",
                    "Cooling pranayama (Sheetali, Sheetkari)",
                    "Moon gazing and water activities",
                    "Avoiding excessive heat and spicy foods"
                ]
            ),
            
            Dosha.KAPHA: DoshaProfile(
                dominant_dosha=Dosha.KAPHA,
                secondary_dosha=None,
                current_imbalances=[],
                strengths=[
                    "Strong immunity and endurance",
                    "Calm and steady temperament",
                    "Loyal and compassionate nature",
                    "Good long-term memory",
                    "Natural ability to nurture others"
                ],
                tendencies=[
                    "Slow digestion and metabolism",
                    "Tendency to gain weight easily",
                    "Prone to congestion and mucus",
                    "May experience depression or lethargy",
                    "Difficulty with change",
                    "Heavy sleep, hard to wake up",
                    "Preference for routine and comfort"
                ],
                recommendations=[
                    "Light, warm, spicy foods",
                    "Regular vigorous exercise",
                    "Stimulating and energizing practices",
                    "Dry brushing and detoxification",
                    "Energizing pranayama (Bhastrika, Kapalabhati)",
                    "Exposure to sunlight",
                    "Avoiding heavy, oily, and cold foods"
                ]
            )
        }
    
    def _initialize_wellness_guidance(self) -> Dict[WellnessAspect, WellnessGuidance]:
        """Initialize wellness guidance for different aspects"""
        return {
            WellnessAspect.PHYSICAL: WellnessGuidance(
                dosha_focus=Dosha.VATA,  # Physical structure primarily Vata-governed
                health_aspect=WellnessAspect.PHYSICAL,
                daily_routine=[
                    "5:30-6:00 AM: Wake up before sunrise",
                    "6:00-6:30 AM: Morning ablutions and oral hygiene",
                    "6:30-7:30 AM: Yoga asana practice",
                    "7:30-8:00 AM: Pranayama and meditation",
                    "8:00-9:00 AM: Nutritious breakfast",
                    "12:00-1:00 PM: Main meal of the day",
                    "6:00-7:00 PM: Light dinner",
                    "9:00-10:00 PM: Prepare for sleep"
                ],
                dietary_guidelines=[
                    "Eat fresh, seasonal, and locally grown foods",
                    "Main meal at noon when digestive fire is strongest",
                    "Avoid processed and packaged foods",
                    "Drink warm water throughout the day",
                    "Eat in a calm, peaceful environment",
                    "Chew food thoroughly and eat mindfully"
                ],
                lifestyle_practices=[
                    "Regular exercise appropriate to constitution",
                    "Daily oil massage (Abhyanga)",
                    "Adequate sleep (7-8 hours)",
                    "Spending time in nature",
                    "Proper breathing practices",
                    "Avoiding excessive screen time"
                ],
                seasonal_adjustments={
                    Season.SPRING: ["Detoxification practices", "Light, cleansing diet"],
                    Season.SUMMER: ["Cooling practices", "Hydrating foods"],
                    Season.AUTUMN: ["Grounding practices", "Warm, nourishing foods"],
                    Season.WINTER: ["Warming practices", "Building foods"]
                },
                healing_herbs=[
                    "Ashwagandha for stress and vitality",
                    "Turmeric for inflammation",
                    "Ginger for digestion",
                    "Brahmi for mental clarity"
                ],
                yoga_practices=[
                    "Sun Salutations (Surya Namaskara)",
                    "Standing poses for strength",
                    "Balancing poses for stability",
                    "Restorative poses for recovery"
                ],
                meditation_techniques=[
                    "Body awareness meditation",
                    "Breath observation",
                    "Walking meditation in nature"
                ],
                warning_signs=[
                    "Chronic fatigue or low energy",
                    "Digestive problems",
                    "Frequent illness",
                    "Physical pain or tension"
                ]
            ),
            
            WellnessAspect.MENTAL: WellnessGuidance(
                dosha_focus=Dosha.PITTA,  # Mental processes primarily Pitta-governed
                health_aspect=WellnessAspect.MENTAL,
                daily_routine=[
                    "Morning meditation for mental clarity",
                    "Mindful work practices",
                    "Regular breaks from mental activity",
                    "Evening reflection and gratitude",
                    "Quality sleep for mental restoration"
                ],
                dietary_guidelines=[
                    "Foods that support brain health (omega-3s, antioxidants)",
                    "Avoid excessive caffeine and stimulants",
                    "Regular meal times to stabilize blood sugar",
                    "Hydration for optimal brain function"
                ],
                lifestyle_practices=[
                    "Stress management techniques",
                    "Regular learning and mental challenges",
                    "Creative expression",
                    "Social connections and community",
                    "Time in nature for mental clarity"
                ],
                seasonal_adjustments={
                    Season.SPRING: ["Mental detox and clarity practices"],
                    Season.SUMMER: ["Cooling mental practices", "Avoid overheating mind"],
                    Season.AUTUMN: ["Grounding mental practices"],
                    Season.WINTER: ["Contemplative and introspective practices"]
                },
                healing_herbs=[
                    "Brahmi (Bacopa) for memory and cognition",
                    "Shankhpushpi for mental clarity",
                    "Jatamansi for calming mind",
                    "Mandukaparni for nervous system"
                ],
                yoga_practices=[
                    "Pranayama for mental balance",
                    "Meditation poses",
                    "Gentle flowing sequences",
                    "Yoga Nidra for deep relaxation"
                ],
                meditation_techniques=[
                    "Mindfulness meditation",
                    "Concentration practices",
                    "Loving-kindness meditation",
                    "Self-inquiry practices"
                ],
                warning_signs=[
                    "Chronic stress or anxiety",
                    "Depression or mood swings",
                    "Memory problems",
                    "Difficulty concentrating"
                ]
            ),
            
            WellnessAspect.SPIRITUAL: WellnessGuidance(
                dosha_focus=Dosha.KAPHA,  # Spiritual stability primarily Kapha-supported
                health_aspect=WellnessAspect.SPIRITUAL,
                daily_routine=[
                    "Morning spiritual practice",
                    "Offering gratitude throughout day",
                    "Mindful service to others",
                    "Evening contemplation",
                    "Regular study of spiritual texts"
                ],
                dietary_guidelines=[
                    "Sattvic (pure) foods that support clarity",
                    "Avoiding tamasic (heavy) and rajasic (stimulating) foods",
                    "Mindful eating as spiritual practice",
                    "Periodic fasting for purification"
                ],
                lifestyle_practices=[
                    "Regular spiritual study (Svadhyaya)",
                    "Selfless service (Seva)",
                    "Pilgrimage and sacred travel",
                    "Community spiritual practice (Satsang)",
                    "Living according to dharmic principles"
                ],
                seasonal_adjustments={
                    Season.SPRING: ["Renewal and rebirth practices"],
                    Season.SUMMER: ["Practices of divine love and devotion"],
                    Season.AUTUMN: ["Harvest gratitude and wisdom practices"],
                    Season.WINTER: ["Inner contemplation and silence practices"]
                },
                healing_herbs=[
                    "Tulsi for spiritual purification",
                    "Lotus for spiritual awakening",
                    "Sandalwood for meditation",
                    "Rose for heart opening"
                ],
                yoga_practices=[
                    "Bhakti yoga (devotional practices)",
                    "Karma yoga (selfless action)",
                    "Raja yoga (meditation)",
                    "Jnana yoga (self-inquiry)"
                ],
                meditation_techniques=[
                    "Mantra meditation",
                    "Devotional meditation",
                    "Self-realization practices",
                    "Unity consciousness meditation"
                ],
                warning_signs=[
                    "Spiritual dryness or disconnection",
                    "Loss of meaning or purpose",
                    "Ethical conflicts",
                    "Disconnection from values"
                ]
            )
        }
    
    def _initialize_seasonal_practices(self) -> Dict[Season, Dict[str, List[str]]]:
        """Initialize seasonal wellness practices"""
        return {
            Season.SPRING: {
                "dominant_dosha": ["Kapha"],
                "practices": [
                    "Detoxification and cleansing",
                    "Energizing exercise",
                    "Light, warming foods",
                    "Dry brushing and massage"
                ],
                "foods_to_favor": [
                    "Bitter and pungent tastes",
                    "Light, warm, cooked foods",
                    "Seasonal greens and sprouts",
                    "Detoxifying herbs and teas"
                ],
                "practices_to_avoid": [
                    "Heavy, oily foods",
                    "Excessive sleep",
                    "Sedentary lifestyle",
                    "Cold, damp environments"
                ]
            },
            
            Season.SUMMER: {
                "dominant_dosha": ["Pitta"],
                "practices": [
                    "Cooling practices and foods",
                    "Moderate exercise",
                    "Swimming and water activities",
                    "Moon gazing and cooling pranayama"
                ],
                "foods_to_favor": [
                    "Sweet, bitter, and astringent tastes",
                    "Cool, fresh, seasonal fruits",
                    "Coconut water and cooling drinks",
                    "Raw foods and salads"
                ],
                "practices_to_avoid": [
                    "Excessive heat and sun exposure",
                    "Spicy, hot foods",
                    "Intense, competitive activities",
                    "Anger and conflict"
                ]
            },
            
            Season.AUTUMN: {
                "dominant_dosha": ["Vata"],
                "practices": [
                    "Grounding and stabilizing routines",
                    "Warm oil massages",
                    "Gentle, restorative yoga",
                    "Meditation and pranayama"
                ],
                "foods_to_favor": [
                    "Sweet, sour, and salty tastes",
                    "Warm, cooked, nourishing foods",
                    "Root vegetables and grains",
                    "Warm milk and herbal teas"
                ],
                "practices_to_avoid": [
                    "Irregular schedules",
                    "Cold, raw foods",
                    "Excessive travel",
                    "Overstimulation"
                ]
            },
            
            Season.WINTER: {
                "dominant_dosha": ["Kapha", "Vata"],
                "practices": [
                    "Warming and energizing practices",
                    "Regular exercise",
                    "Building and nourishing foods",
                    "Indoor spiritual practices"
                ],
                "foods_to_favor": [
                    "Sweet, sour, and salty tastes",
                    "Warm, cooked, substantial foods",
                    "Warming spices",
                    "Nourishing soups and stews"
                ],
                "practices_to_avoid": [
                    "Excessive cold exposure",
                    "Light, cold foods",
                    "Sedentary lifestyle",
                    "Social isolation"
                ]
            }
        }
    
    def _initialize_healing_modalities(self) -> Dict[HealthImbalance, List[str]]:
        """Initialize healing approaches for common imbalances"""
        return {
            HealthImbalance.STRESS: [
                "Regular meditation and pranayama",
                "Ashwagandha and adaptogenic herbs",
                "Warm oil massage (Abhyanga)",
                "Restorative yoga practices",
                "Time in nature",
                "Adequate sleep and rest"
            ],
            
            HealthImbalance.DIGESTIVE_ISSUES: [
                "Eating according to constitution and season",
                "Digestive spices (ginger, cumin, coriander)",
                "Regular meal times",
                "Proper food combining",
                "Mindful eating practices",
                "Triphala for gentle cleansing"
            ],
            
            HealthImbalance.SLEEP_DISORDERS: [
                "Regular sleep schedule",
                "Evening relaxation routine",
                "Avoiding screens before bed",
                "Warm milk with calming herbs",
                "Gentle yoga and meditation",
                "Creating peaceful sleep environment"
            ],
            
            HealthImbalance.ENERGY_DEPLETION: [
                "Rejuvenating herbs (Ashwagandha, Shatavari)",
                "Balanced nutrition and hydration",
                "Appropriate exercise for constitution",
                "Stress reduction techniques",
                "Regular rest and recovery",
                "Spiritual practices for energy renewal"
            ],
            
            HealthImbalance.EMOTIONAL_INSTABILITY: [
                "Mind-calming herbs (Brahmi, Jatamansi)",
                "Regular meditation practice",
                "Supportive community and relationships",
                "Creative expression",
                "Professional counseling if needed",
                "Spiritual practices for emotional healing"
            ]
        }
    
    def assess_dosha_constitution(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dosha:
        """Assess primary dosha constitution from query"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Simple assessment based on keywords
        vata_indicators = ["anxious", "variable", "dry", "cold", "irregular", "creative", "movement"]
        pitta_indicators = ["hot", "intense", "focused", "competitive", "sharp", "angry", "digestion"]
        kapha_indicators = ["heavy", "slow", "steady", "congestion", "mucus", "stable", "calm"]
        
        vata_score = sum(1 for indicator in vata_indicators if indicator in query_lower)
        pitta_score = sum(1 for indicator in pitta_indicators if indicator in query_lower)
        kapha_score = sum(1 for indicator in kapha_indicators if indicator in query_lower)
        
        if pitta_score > vata_score and pitta_score > kapha_score:
            return Dosha.PITTA
        elif kapha_score > vata_score and kapha_score > pitta_score:
            return Dosha.KAPHA
        else:
            return Dosha.VATA  # Default to Vata as most common modern imbalance
    
    def identify_health_imbalances(self, query: str, context: Dict[str, Any]) -> List[HealthImbalance]:
        """Identify health imbalances mentioned in query"""
        imbalances = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["stress", "anxiety", "tension", "overwhelm"]):
            imbalances.append(HealthImbalance.STRESS)
        
        if any(word in query_lower for word in ["digestion", "stomach", "bloating", "gas"]):
            imbalances.append(HealthImbalance.DIGESTIVE_ISSUES)
        
        if any(word in query_lower for word in ["sleep", "insomnia", "tired", "fatigue"]):
            imbalances.append(HealthImbalance.SLEEP_DISORDERS)
        
        if any(word in query_lower for word in ["energy", "vitality", "exhausted", "weak"]):
            imbalances.append(HealthImbalance.ENERGY_DEPLETION)
        
        if any(word in query_lower for word in ["emotional", "mood", "depression", "angry"]):
            imbalances.append(HealthImbalance.EMOTIONAL_INSTABILITY)
        
        return imbalances if imbalances else [HealthImbalance.STRESS]  # Default
    
    def get_current_season(self) -> Season:
        """Get current season (simplified)"""
        current_month = datetime.now().month
        
        if current_month in [3, 4, 5]:
            return Season.SPRING
        elif current_month in [6, 7, 8]:
            return Season.SUMMER
        elif current_month in [9, 10, 11]:
            return Season.AUTUMN
        else:
            return Season.WINTER
    
    def get_dosha_recommendations(self, dosha: Dosha) -> List[str]:
        """Get basic recommendations for balancing a dosha"""
        profile = self.dosha_profiles.get(dosha)
        return profile.recommendations if profile else [
            "Follow regular daily routine",
            "Eat according to your constitution",
            "Practice appropriate exercise",
            "Manage stress effectively"
        ]
    
    def get_seasonal_guidance(self, season: Season) -> str:
        """Get guidance for current season"""
        seasonal_data = self.seasonal_practices.get(season, {})
        practices = seasonal_data.get("practices", [])
        return f"For {season.value}, focus on: " + "; ".join(practices[:3])
    
    def get_healing_approaches(self, imbalances: List[HealthImbalance]) -> List[str]:
        """Get healing approaches for identified imbalances"""
        approaches = []
        for imbalance in imbalances:
            modalities = self.healing_modalities.get(imbalance, [])
            approaches.extend(modalities[:2])  # Add first two approaches for each
        
        return list(set(approaches))  # Remove duplicates
    
    def get_prevention_strategies(self, dosha: Dosha) -> List[str]:
        """Get disease prevention strategies based on constitution"""
        if dosha == Dosha.VATA:
            return [
                "Maintain regular routine",
                "Stay warm and well-nourished",
                "Practice grounding activities",
                "Avoid excessive stimulation"
            ]
        elif dosha == Dosha.PITTA:
            return [
                "Avoid overheating and overworking",
                "Practice cooling activities",
                "Manage anger and stress",
                "Eat cooling, moderate foods"
            ]
        else:  # KAPHA
            return [
                "Stay active and energized",
                "Avoid heavy, cold foods",
                "Practice stimulating activities",
                "Maintain social connections"
            ]
    
    async def process_wellness_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> WellnessResponse:
        """Process wellness-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess constitution and imbalances
            primary_dosha = self.assess_dosha_constitution(query, context)
            health_imbalances = self.identify_health_imbalances(query, context)
            current_season = self.get_current_season()
            
            # Get guidance components
            dosha_profile = self.dosha_profiles.get(primary_dosha)
            wellness_aspect = WellnessAspect.PHYSICAL  # Default to physical wellness
            guidance = self.wellness_guidance.get(wellness_aspect)
            
            # Generate recommendations
            dosha_recommendations = self.get_dosha_recommendations(primary_dosha)
            seasonal_guidance = self.get_seasonal_guidance(current_season)
            healing_approaches = self.get_healing_approaches(health_imbalances)
            prevention_strategies = self.get_prevention_strategies(primary_dosha)
            
            # Prepare response
            return WellnessResponse(
                dosha_assessment=f"Primary constitution: {primary_dosha.value.title()}",
                current_imbalances=[imbalance.value for imbalance in health_imbalances],
                wellness_guidance=f"🟢 For {primary_dosha.value} constitution, focus on {', '.join(dosha_profile.strengths[:2]) if dosha_profile else 'balance and harmony'}. {seasonal_guidance}",
                daily_routine=guidance.daily_routine if guidance else [
                    "Wake up early (before sunrise)",
                    "Morning spiritual practice",
                    "Mindful meals at regular times",
                    "Appropriate exercise",
                    "Evening relaxation and early sleep"
                ],
                dietary_recommendations=guidance.dietary_guidelines if guidance else [
                    "Eat fresh, seasonal foods",
                    "Main meal at midday",
                    "Drink warm water",
                    "Avoid processed foods"
                ],
                lifestyle_practices=dosha_recommendations,
                healing_modalities=healing_approaches,
                seasonal_guidance=seasonal_guidance,
                prevention_strategies=prevention_strategies,
                spiritual_practices=[
                    "Daily meditation practice",
                    "Pranayama (breathing exercises)",
                    "Yoga appropriate to constitution",
                    "Connection with nature",
                    "Gratitude and mindfulness"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error processing wellness query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> WellnessResponse:
        """Create fallback response when processing fails"""
        return WellnessResponse(
            dosha_assessment="Balanced constitution with focus on current needs",
            current_imbalances=["general_imbalance"],
            wellness_guidance="🟢 Focus on establishing a balanced daily routine with proper nutrition, exercise, rest, and spiritual practice.",
            daily_routine=[
                "Wake up early (5:30-6:00 AM)",
                "Morning meditation and pranayama",
                "Nutritious breakfast",
                "Mindful work and activities",
                "Healthy lunch (main meal)",
                "Appropriate exercise",
                "Light dinner",
                "Evening relaxation and early sleep"
            ],
            dietary_recommendations=[
                "Eat fresh, whole foods",
                "Follow natural eating rhythms",
                "Stay properly hydrated",
                "Avoid processed and junk foods",
                "Eat mindfully in peaceful environment"
            ],
            lifestyle_practices=[
                "Regular daily routine",
                "Adequate sleep (7-8 hours)",
                "Stress management",
                "Time in nature",
                "Social connections"
            ],
            healing_modalities=[
                "Yoga and meditation",
                "Ayurvedic herbs and therapies",
                "Proper nutrition",
                "Natural healing methods"
            ],
            seasonal_guidance="Live in harmony with natural seasons and cycles",
            prevention_strategies=[
                "Maintain balance in all activities",
                "Listen to your body's signals",
                "Practice preventive self-care",
                "Regular health check-ups"
            ],
            spiritual_practices=[
                "Daily meditation",
                "Pranayama breathing",
                "Yoga practice",
                "Gratitude and mindfulness",
                "Connection with nature"
            ]
        )
    
    def get_wellness_insight(self, dosha: Dosha) -> Optional[WellnessInsight]:
        """Get specific wellness insight for a dosha"""
        profile = self.dosha_profiles.get(dosha)
        if not profile:
            return None
        
        return WellnessInsight(
            dosha_type=dosha.value,
            imbalance_area=profile.tendencies[0] if profile.tendencies else "general imbalance",
            recommendation=profile.recommendations[0] if profile.recommendations else "maintain balance",
            practice="Daily routine and appropriate lifestyle"
        )

# Global instance
_wellness_module = None

def get_wellness_module() -> WellnessModule:
    """Get global Wellness module instance"""
    global _wellness_module
    if _wellness_module is None:
        _wellness_module = WellnessModule()
    return _wellness_module

# Factory function for easy access
def create_wellness_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> WellnessResponse:
    """Factory function to create wellness guidance"""
    import asyncio
    module = get_wellness_module()
    return asyncio.run(module.process_wellness_query(query, user_context))
