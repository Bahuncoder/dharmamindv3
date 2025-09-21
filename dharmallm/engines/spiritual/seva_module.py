"""
Seva Module - Service and Compassion Module
==========================================

Provides service-oriented guidance, compassion practices, and community engagement
for the DharmaMind system based on Seva (selfless service) principles.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SevaType(Enum):
    """Types of service/seva"""
    MANASIKA = "mental"          # Mental service through prayer, good thoughts
    VACHIKA = "verbal"           # Verbal service through teaching, counseling
    KAYIKA = "physical"          # Physical service through actions
    DRAVYA = "material"          # Material service through donations, resources

class ServiceArea(Enum):
    """Areas where service can be offered"""
    FAMILY = "family_service"
    COMMUNITY = "community_service"
    NATURE = "environmental_service"
    ANIMALS = "animal_welfare"
    EDUCATION = "knowledge_sharing"
    HEALTHCARE = "healing_service"
    SPIRITUAL = "spiritual_guidance"
    SOCIAL = "social_justice"

@dataclass
class SevaOpportunity:
    """A service opportunity or practice"""
    title: str
    description: str
    seva_type: SevaType
    area: ServiceArea
    time_commitment: str
    skills_needed: List[str]
    impact_description: str
    spiritual_benefit: str
    practical_steps: List[str]
    timestamp: datetime

@dataclass
class CompassionPractice:
    """A compassion cultivation practice"""
    name: str
    description: str
    duration: str
    steps: List[str]
    benefits: List[str]
    frequency: str

class SevaModule:
    """
    Seva (Service) Module
    
    Processes service-oriented queries and provides guidance on selfless service,
    compassion cultivation, and community engagement based on Karma Yoga principles.
    """
    
    def __init__(self):
        self.module_name = "Seva Module"
        self.element = "Air/Vayu"
        self.color = "Green"
        self.mantra = "YAM"
        self.deity = "Hanuman"
        self.principles = ["Selfless Service", "Compassion", "Humility", "Unity Consciousness"]
        
        # Initialize service wisdom
        self.seva_opportunities = self._initialize_seva_opportunities()
        self.compassion_practices = self._initialize_compassion_practices()
        self.service_principles = self._initialize_service_principles()
        
    def _initialize_seva_opportunities(self) -> List[SevaOpportunity]:
        """Initialize various seva opportunities"""
        return [
            SevaOpportunity(
                title="Family Service Practice",
                description="Transform daily family interactions into conscious service opportunities",
                seva_type=SevaType.KAYIKA,
                area=ServiceArea.FAMILY,
                time_commitment="Daily, integrated into routine",
                skills_needed=["Patience", "Active listening", "Empathy"],
                impact_description="Creates harmony at home and teaches service through example",
                spiritual_benefit="Dissolves ego through humble service to loved ones",
                practical_steps=[
                    "Serve meals with love and gratitude",
                    "Listen deeply when family members share concerns",
                    "Offer help before being asked",
                    "Practice forgiveness and understanding in conflicts",
                    "Express appreciation and gratitude regularly"
                ],
                timestamp=datetime.now()
            ),
            SevaOpportunity(
                title="Community Knowledge Sharing",
                description="Share your skills and knowledge to help others grow and learn",
                seva_type=SevaType.VACHIKA,
                area=ServiceArea.EDUCATION,
                time_commitment="2-4 hours per week",
                skills_needed=["Teaching ability", "Patience", "Subject expertise"],
                impact_description="Empowers others with knowledge and skills for better life",
                spiritual_benefit="Develops humility and joy in others' success",
                practical_steps=[
                    "Identify your areas of expertise and passion",
                    "Find local community centers, schools, or online platforms",
                    "Offer free workshops or mentoring sessions",
                    "Create helpful content or resources to share",
                    "Listen to students' needs and adapt your approach"
                ],
                timestamp=datetime.now()
            ),
            SevaOpportunity(
                title="Environmental Care Service",
                description="Serve Mother Earth through environmental conservation and care",
                seva_type=SevaType.KAYIKA,
                area=ServiceArea.NATURE,
                time_commitment="Flexible, from daily habits to weekend activities",
                skills_needed=["Environmental awareness", "Physical activity", "Community organizing"],
                impact_description="Protects environment for future generations",
                spiritual_benefit="Develops connection with all life and cosmic consciousness",
                practical_steps=[
                    "Reduce, reuse, and recycle in daily life",
                    "Plant trees or maintain a garden",
                    "Participate in community clean-up drives",
                    "Educate others about environmental issues",
                    "Support sustainable and eco-friendly practices"
                ],
                timestamp=datetime.now()
            ),
            SevaOpportunity(
                title="Compassionate Listening Service",
                description="Offer presence and compassionate listening to those in need",
                seva_type=SevaType.MANASIKA,
                area=ServiceArea.SOCIAL,
                time_commitment="1-2 hours per week",
                skills_needed=["Empathy", "Patience", "Non-judgmental attitude"],
                impact_description="Provides emotional support and reduces isolation",
                spiritual_benefit="Develops unconditional love and emotional maturity",
                practical_steps=[
                    "Practice deep listening without trying to solve problems",
                    "Offer presence without judgment or advice unless asked",
                    "Volunteer with counseling services or support groups",
                    "Be available for friends and family in times of need",
                    "Practice loving-kindness meditation to cultivate compassion"
                ],
                timestamp=datetime.now()
            )
        ]
    
    def _initialize_compassion_practices(self) -> List[CompassionPractice]:
        """Initialize compassion cultivation practices"""
        return [
            CompassionPractice(
                name="Loving-Kindness Meditation",
                description="Systematic cultivation of love and compassion for all beings",
                duration="15-30 minutes",
                steps=[
                    "Begin with yourself: 'May I be happy, may I be peaceful, may I be free from suffering'",
                    "Extend to loved ones: Send the same wishes to family and friends",
                    "Include neutral people: Extend compassion to acquaintances and strangers",
                    "Embrace difficult people: Send love even to those who challenge you",
                    "Expand to all beings: Include all living creatures in your compassion"
                ],
                benefits=[
                    "Reduces anger and resentment",
                    "Increases emotional resilience",
                    "Develops unconditional love",
                    "Improves relationships",
                    "Creates inner peace"
                ],
                frequency="Daily"
            ),
            CompassionPractice(
                name="Tonglen (Taking and Giving)",
                description="Breathe in suffering and breathe out relief and happiness",
                duration="10-20 minutes",
                steps=[
                    "Sit comfortably and breathe naturally",
                    "Visualize breathing in the suffering of others as dark smoke",
                    "Transform it in your heart with love and compassion",
                    "Breathe out relief, happiness, and peace as bright light",
                    "Start with your own suffering, then extend to others"
                ],
                benefits=[
                    "Transforms relationship with suffering",
                    "Develops courage and compassion",
                    "Reduces self-centeredness",
                    "Creates emotional strength",
                    "Deepens empathy for others"
                ],
                frequency="3-4 times per week"
            ),
            CompassionPractice(
                name="Random Acts of Kindness",
                description="Practice spontaneous compassion in daily life",
                duration="Throughout the day",
                steps=[
                    "Look for opportunities to help without being asked",
                    "Offer genuine compliments and appreciation",
                    "Practice patience in frustrating situations",
                    "Smile and make eye contact with people you meet",
                    "Hold space for others' emotions without trying to fix them"
                ],
                benefits=[
                    "Develops natural compassion",
                    "Creates positive karma",
                    "Improves mood and outlook",
                    "Builds connection with others",
                    "Makes service a natural habit"
                ],
                frequency="Daily opportunities"
            )
        ]
    
    def _initialize_service_principles(self) -> Dict[str, str]:
        """Initialize key principles of service"""
        return {
            "nishkama_seva": "Serve without expectation of reward or recognition",
            "humility": "Serve with humility, seeing yourself as an instrument of the Divine",
            "compassion": "Let love and compassion motivate all service activities",
            "unity": "See the Divine in all beings you serve",
            "gratitude": "Feel grateful for the opportunity to serve",
            "presence": "Be fully present and attentive when serving others",
            "non_judgment": "Serve without judging those who receive your service",
            "sustainability": "Ensure your service is sustainable and doesn't lead to burnout"
        }
    
    async def process_seva_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a service/compassion-based query
        
        Args:
            query: The user's question about service, compassion, or helping others
            context: Additional context about the user's situation
            
        Returns:
            Dict containing seva guidance, opportunities, and practices
        """
        try:
            # Analyze query intent
            query_intent = self._analyze_seva_intent(query)
            
            # Get relevant seva opportunities
            relevant_opportunities = self._get_relevant_opportunities(query, query_intent)
            
            # Get appropriate compassion practices
            relevant_practices = self._get_relevant_practices(query_intent)
            
            # Generate personalized guidance
            guidance = await self._generate_seva_guidance(query, query_intent, context or {})
            
            # Create service action plan
            action_plan = self._create_seva_action_plan(query_intent, context or {})
            
            return {
                "module": self.module_name,
                "query": query,
                "intent": query_intent,
                "guidance": guidance,
                "seva_opportunities": [self._opportunity_to_dict(opp) for opp in relevant_opportunities],
                "compassion_practices": [self._practice_to_dict(practice) for practice in relevant_practices],
                "action_plan": action_plan,
                "service_principles": self._get_relevant_principles(query_intent),
                "meditation_suggestion": self._get_seva_meditation(query_intent),
                "inspiration": self._get_seva_inspiration(query_intent),
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing seva query: {e}")
            return {
                "module": self.module_name,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_seva_intent(self, query: str) -> str:
        """Analyze the intent behind a seva/service query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["help", "serve", "volunteer", "contribute"]):
            return "service_opportunity"
        elif any(word in query_lower for word in ["compassion", "kindness", "love", "empathy"]):
            return "compassion_development"
        elif any(word in query_lower for word in ["community", "society", "world", "others"]):
            return "community_service"
        elif any(word in query_lower for word in ["family", "home", "parents", "children"]):
            return "family_service"
        elif any(word in query_lower for word in ["angry", "frustrated", "difficult", "conflict"]):
            return "compassion_challenge"
        elif any(word in query_lower for word in ["environment", "nature", "earth", "pollution"]):
            return "environmental_service"
        else:
            return "general_seva"
    
    def _get_relevant_opportunities(self, query: str, intent: str) -> List[SevaOpportunity]:
        """Get relevant seva opportunities based on query and intent"""
        relevant = []
        
        intent_to_area = {
            "family_service": ServiceArea.FAMILY,
            "community_service": ServiceArea.COMMUNITY,
            "environmental_service": ServiceArea.NATURE,
            "service_opportunity": ServiceArea.EDUCATION  # Default to education
        }
        
        target_area = intent_to_area.get(intent)
        
        if target_area:
            relevant = [opp for opp in self.seva_opportunities if opp.area == target_area]
        
        # If no specific match or need more options, add general opportunities
        if len(relevant) < 2:
            remaining = [opp for opp in self.seva_opportunities if opp not in relevant]
            relevant.extend(remaining[:2])
        
        return relevant[:3]  # Return top 3 most relevant
    
    def _get_relevant_practices(self, intent: str) -> List[CompassionPractice]:
        """Get relevant compassion practices based on intent"""
        if intent == "compassion_development":
            return self.compassion_practices
        elif intent == "compassion_challenge":
            # Focus on practices that help with difficult emotions
            return [p for p in self.compassion_practices if "anger" in p.description or "difficult" in p.description]
        else:
            # Return general compassion practices
            return self.compassion_practices[:2]
    
    async def _generate_seva_guidance(self, query: str, intent: str, context: Dict[str, Any]) -> str:
        """Generate personalized seva guidance"""
        guidance = f"💚 Regarding your inquiry about {query.lower()}, "
        
        intent_guidance = {
            "service_opportunity": "the path of seva (selfless service) offers countless opportunities to grow spiritually while helping others. Service purifies the heart and dissolves the ego.",
            "compassion_development": "compassion is the heart of all spiritual practice. As you cultivate love for all beings, you discover your own true nature of unlimited love.",
            "community_service": "serving your community is serving the Divine in all beings. Every act of service is an offering to the cosmic consciousness.",
            "family_service": "your family is your first community and greatest teacher. Serving them with love transforms ordinary interactions into spiritual practice.",
            "compassion_challenge": "difficult emotions are opportunities to deepen your compassion. Use these challenges to develop patience, understanding, and unconditional love.",
            "environmental_service": "caring for Mother Earth is dharmic duty. The environment sustains all life, and protecting it is serving all beings."
        }
        
        guidance += intent_guidance.get(intent, "service and compassion are the fastest paths to spiritual growth and inner peace.")
        
        guidance += "\n\n🌟 Remember: True seva is performed without expectation of reward, seeing the Divine in all beings you serve. Let love and gratitude motivate every action."
        
        return guidance
    
    def _create_seva_action_plan(self, intent: str, context: Dict[str, Any]) -> List[str]:
        """Create a practical seva action plan"""
        base_plan = [
            "🧘 Begin each day with meditation on compassion and service",
            "🙏 Offer your service activities to the Divine",
            "❤️ Approach all service with love, not duty"
        ]
        
        intent_specific = {
            "service_opportunity": [
                "🔍 Identify your skills and passions",
                "🌍 Research local service opportunities",
                "👥 Start with small, manageable commitments",
                "📈 Gradually expand your service activities"
            ],
            "compassion_development": [
                "🧘 Practice loving-kindness meditation daily",
                "💝 Perform random acts of kindness",
                "🤗 Practice forgiveness for yourself and others",
                "🌱 Cultivate patience in challenging situations"
            ],
            "family_service": [
                "🏠 Look for ways to help at home without being asked",
                "👂 Practice deep listening with family members",
                "💕 Express gratitude and appreciation regularly",
                "🕊️ Forgive family conflicts quickly"
            ],
            "environmental_service": [
                "🌱 Start with personal environmental practices",
                "🌳 Participate in local environmental initiatives",
                "📚 Educate yourself and others about environmental issues",
                "🤝 Connect with environmental organizations"
            ]
        }
        
        base_plan.extend(intent_specific.get(intent, [
            "🎯 Choose service activities aligned with your heart",
            "⏰ Create a sustainable service routine",
            "🤝 Connect with like-minded service communities"
        ]))
        
        base_plan.extend([
            "✨ Reflect on how service transforms your heart",
            "🙏 Practice gratitude for opportunities to serve"
        ])
        
        return base_plan
    
    def _get_relevant_principles(self, intent: str) -> List[str]:
        """Get relevant service principles"""
        all_principles = list(self.service_principles.items())
        
        # Always include core principles
        core = ["nishkama_seva", "humility", "compassion"]
        selected = [(k, v) for k, v in all_principles if k in core]
        
        # Add intent-specific principles
        if intent == "compassion_challenge":
            selected.append(("non_judgment", self.service_principles["non_judgment"]))
        elif intent == "family_service":
            selected.append(("presence", self.service_principles["presence"]))
        elif intent == "service_opportunity":
            selected.append(("sustainability", self.service_principles["sustainability"]))
        
        return [f"{k.replace('_', ' ').title()}: {v}" for k, v in selected]
    
    def _get_seva_meditation(self, intent: str) -> str:
        """Get appropriate meditation practice for seva intent"""
        meditations = {
            "compassion_development": "Practice loving-kindness meditation, starting with yourself and expanding to include all beings",
            "service_opportunity": "Meditate on seeing the Divine in all beings and offering your service to the cosmic consciousness",
            "family_service": "Contemplate how serving your family with love transforms ordinary actions into spiritual practice",
            "compassion_challenge": "Practice tonglen meditation - breathing in suffering and breathing out relief and love",
            "environmental_service": "Meditate on your connection with all life and Mother Earth, feeling gratitude for her gifts"
        }
        
        return meditations.get(intent, "Meditate on the unity of all beings and let that awareness inspire your service")
    
    def _get_seva_inspiration(self, intent: str) -> str:
        """Get inspirational message for seva practice"""
        inspirations = {
            "service_opportunity": "\"The best way to find yourself is to lose yourself in the service of others.\" - Gandhi",
            "compassion_development": "\"Compassion is not a relationship between the healer and the wounded. It's a relationship between equals.\" - Pema Chödrön",
            "family_service": "\"The family is the first essential cell of human society.\" - Serve with love at home first.",
            "environmental_service": "\"The Earth does not belong to us; we belong to the Earth. All things are connected.\"",
            "compassion_challenge": "\"Patience is not the ability to wait, but how you behave while waiting.\""
        }
        
        return inspirations.get(intent, "\"Service to others is the rent you pay for your room here on Earth.\" - Muhammad Ali")
    
    def _opportunity_to_dict(self, opportunity: SevaOpportunity) -> Dict[str, Any]:
        """Convert SevaOpportunity to dictionary"""
        return {
            "title": opportunity.title,
            "description": opportunity.description,
            "seva_type": opportunity.seva_type.value,
            "area": opportunity.area.value,
            "time_commitment": opportunity.time_commitment,
            "skills_needed": opportunity.skills_needed,
            "impact_description": opportunity.impact_description,
            "spiritual_benefit": opportunity.spiritual_benefit,
            "practical_steps": opportunity.practical_steps
        }
    
    def _practice_to_dict(self, practice: CompassionPractice) -> Dict[str, Any]:
        """Convert CompassionPractice to dictionary"""
        return {
            "name": practice.name,
            "description": practice.description,
            "duration": practice.duration,
            "steps": practice.steps,
            "benefits": practice.benefits,
            "frequency": practice.frequency
        }

# Factory function for module integration
def create_seva_module() -> SevaModule:
    """Create and return a Seva Module instance"""
    return SevaModule()

# Global instance
_seva_module = None

def get_seva_module() -> SevaModule:
    """Get global Seva module instance"""
    global _seva_module
    if _seva_module is None:
        _seva_module = SevaModule()
    return _seva_module
