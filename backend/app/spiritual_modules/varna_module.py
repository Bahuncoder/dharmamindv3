"""
🏛️ Varna Module - Social Harmony and Divine Order
Complete system for understanding dharmic social roles and conscious community
Based on authentic Varna principles beyond caste hierarchy
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VarnaLevel(Enum):
    """Levels of social harmony understanding"""
    CONFUSED = "confused"           # Confused about social role and duty
    SEEKING = "seeking"             # Seeking understanding of dharmic purpose
    LEARNING = "learning"           # Learning about natural inclinations
    ALIGNING = "aligning"           # Aligning with authentic dharmic role
    SERVING = "serving"             # Serving society through natural gifts
    HARMONIZING = "harmonizing"     # Creating harmony in social interactions


class VarnaType(Enum):
    """Natural inclinations and social roles"""
    BRAHMANA = "brahmana"           # Teacher, priest, scholar, wisdom keeper
    KSHATRIYA = "kshatriya"         # Leader, protector, warrior, organizer
    VAISHYA = "vaishya"             # Merchant, farmer, provider, creator
    SHUDRA = "shudra"               # Server, supporter, craftsperson, helper


class SocialChallenge(Enum):
    """Challenges in social harmony"""
    IDENTITY_CONFUSION = "identity_confusion"       # Unclear about natural role
    HIERARCHY_ATTACHMENT = "hierarchy_attachment"   # Attached to social status
    ROLE_RESISTANCE = "role_resistance"             # Resisting natural tendencies
    COMPARISON_TRAP = "comparison_trap"             # Comparing with others
    SERVICE_IMBALANCE = "service_imbalance"         # Not serving or over-serving
    COMMUNITY_DISCONNECT = "community_disconnect"   # Isolated from community


@dataclass
class VarnaGuidance:
    """Comprehensive varna guidance"""
    level: VarnaLevel
    primary_teaching: str
    role_understanding: List[str]
    dharmic_service: List[str]
    social_harmony: List[str]
    gift_development: List[str]
    community_building: List[str]
    integration_practices: List[str]
    progress_indicators: List[str]


class VarnaResponse(BaseModel):
    """Response from Varna module"""
    varna_level: str = Field(description="Current social harmony level")
    social_guidance: str = Field(description="Core varna teaching")
    role_understanding: List[str] = Field(description="Understand your role")
    dharmic_service: List[str] = Field(description="Serve through gifts")
    social_harmony: List[str] = Field(description="Create harmony")
    gift_development: List[str] = Field(description="Develop your gifts")
    community_building: List[str] = Field(description="Build community")
    integration_practices: List[str] = Field(description="Integrate wisdom")
    scriptural_wisdom: str = Field(description="Traditional varna teachings")


class VarnaModule:
    """
    🏛️ Varna Module - Social Harmony and Divine Order
    
    Based on authentic Varna teachings:
    - Bhagavad Gita's description of natural qualities and duties
    - Vedic understanding of social cooperation for collective welfare
    - Mahabharata's teachings on dharmic social responsibility
    - Traditional texts on natural aptitudes and social service
    - Modern understanding of conscious community building
    
    Varna represents the divine principle of social organization
    based on natural qualities and inclinations, not birth or
    hierarchy, serving the highest good of all beings.
    """
    
    def __init__(self):
        self.name = "Varna"
        self.color = "🏛️"
        self.element = "Social Harmony"
        self.principles = ["Natural Service", "Divine Order", 
                          "Conscious Community", "Dharmic Cooperation"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.varna_qualities = self._initialize_varna_qualities()
        self.service_methods = self._initialize_service_methods()
        
    def _initialize_guidance_levels(self) -> Dict[VarnaLevel, VarnaGuidance]:
        """Initialize guidance for different levels of social harmony"""
        return {
            VarnaLevel.CONFUSED: VarnaGuidance(
                level=VarnaLevel.CONFUSED,
                primary_teaching="Every being has unique gifts meant to serve "
                "the whole. Your confusion about your role is the beginning "
                "of discovering your authentic dharmic purpose.",
                role_understanding=[
                    "Observe what activities naturally energize you",
                    "Notice what problems you feel called to solve",
                    "Pay attention to what others seek from you",
                    "Reflect on your childhood interests and talents",
                    "Consider what contribution feels meaningful to you"
                ],
                dharmic_service=[
                    "Start serving wherever you feel called, without attachment",
                    "Volunteer in areas that interest or concern you",
                    "Offer your natural skills to help others",
                    "Practice seeing service as spiritual practice",
                    "Begin with small acts of daily kindness and helpfulness"
                ],
                social_harmony=[
                    "Practice appreciating different people's unique contributions",
                    "Avoid comparing your path with others' paths",
                    "Seek to understand rather than judge different approaches",
                    "Look for ways to support others in their dharmic roles",
                    "Create inclusive environments where all feel valued"
                ],
                gift_development=[
                    "Identify your natural talents and strengths",
                    "Study subjects that genuinely interest you",
                    "Practice skills that feel natural and enjoyable",
                    "Seek mentorship from those you admire",
                    "Experiment with different forms of creative expression"
                ],
                community_building=[
                    "Join groups aligned with your values and interests",
                    "Practice active listening and empathetic communication",
                    "Contribute your unique perspective to group discussions",
                    "Help connect people with complementary skills",
                    "Create opportunities for meaningful collaboration"
                ],
                integration_practices=[
                    "Daily reflection on how you served others",
                    "Regular assessment of your energy levels in different activities",
                    "Meditation on your life purpose and calling",
                    "Study of inspiring examples of dharmic service",
                    "Practice seeing all work as offering to the Divine"
                ],
                progress_indicators=[
                    "Increasing clarity about your natural inclinations",
                    "Growing sense of purpose and meaning in activities",
                    "Natural appreciation for others' different contributions",
                    "Spontaneous urges to help and serve others"
                ]
            ),
            
            VarnaLevel.SEEKING: VarnaGuidance(
                level=VarnaLevel.SEEKING,
                primary_teaching="Seeking your dharmic role with sincerity "
                "attracts the right opportunities and guidance. Trust the "
                "process of gradual unfolding of your authentic purpose.",
                role_understanding=[
                    "Deep exploration of your core values and motivations",
                    "Understanding the difference between ego desires and soul calling",
                    "Recognizing patterns in what consistently draws your attention",
                    "Studying the traditional qualities of different varnas",
                    "Seeking guidance from wise mentors and spiritual teachers"
                ],
                dharmic_service=[
                    "Expanding service activities to explore different expressions",
                    "Practicing service with awareness and presence",
                    "Offering your developing skills in structured ways",
                    "Learning from established service organizations",
                    "Balancing service with personal spiritual development"
                ],
                social_harmony=[
                    "Developing appreciation for the interdependence of all roles",
                    "Learning to see hierarchy as functional rather than personal",
                    "Practicing humble service regardless of recognition",
                    "Building bridges between different social groups",
                    "Advocating for inclusive and fair social structures"
                ],
                gift_development=[
                    "Systematic study and practice in areas of natural aptitude",
                    "Seeking formal or informal training to develop skills",
                    "Practicing discipline and dedication in gift cultivation",
                    "Finding teachers and examples who inspire excellence",
                    "Balancing skill development with spiritual growth"
                ],
                community_building=[
                    "Taking initiative in creating collaborative projects",
                    "Facilitating communication between different personality types",
                    "Organizing events that serve community needs",
                    "Developing leadership skills through service opportunities",
                    "Creating systems that support everyone's contribution"
                ],
                integration_practices=[
                    "Regular study of dharmic principles in social organization",
                    "Meditation on your unique role in cosmic evolution",
                    "Daily practices that align personal will with divine will",
                    "Journaling about insights from service experiences",
                    "Creating personal mission statement based on dharmic understanding"
                ],
                progress_indicators=[
                    "Clear sense of your natural varna inclinations",
                    "Growing skill and confidence in areas of natural aptitude",
                    "Recognition from others of your natural capabilities",
                    "Increasing harmony between personal fulfillment and service"
                ]
            ),
            
            VarnaLevel.LEARNING: VarnaGuidance(
                level=VarnaLevel.LEARNING,
                primary_teaching="Learning to embody your dharmic role with "
                "skill and wisdom. Your dedication to excellence in service "
                "becomes a spiritual practice and offering to the Divine.",
                role_understanding=[
                    "Comprehensive study of your varna's traditional duties and modern expressions",
                    "Understanding how your role serves the welfare of the whole",
                    "Learning to distinguish between essential duties and cultural additions",
                    "Studying examples of excellence in your chosen field of service",
                    "Understanding the spiritual dimensions of your worldly work"
                ],
                dharmic_service=[
                    "Developing mastery in your chosen form of service",
                    "Taking on increasing responsibility in service organizations",
                    "Mentoring others who are discovering their dharmic path",
                    "Creating innovative solutions to social problems",
                    "Establishing sustainable systems for ongoing service"
                ],
                social_harmony=[
                    "Acting as bridge and translator between different social groups",
                    "Modeling conscious leadership in your sphere of influence",
                    "Creating inclusive policies and practices in your organizations",
                    "Teaching others about dharmic principles of social organization",
                    "Working to transform oppressive hierarchies into service hierarchies"
                ],
                gift_development=[
                    "Achieving professional competence in your area of dharmic service",
                    "Developing teaching abilities to share your knowledge",
                    "Cultivating wisdom alongside technical skills",
                    "Learning to balance perfectionism with practical effectiveness",
                    "Integrating spiritual principles into professional practice"
                ],
                community_building=[
                    "Leading collaborative projects that serve community welfare",
                    "Creating educational programs in your area of expertise",
                    "Building networks of people committed to dharmic service",
                    "Establishing institutions that embody conscious principles",
                    "Training others in leadership and collaborative skills"
                ],
                integration_practices=[
                    "Advanced study of dharmic texts relevant to social organization",
                    "Regular practice of offering all work as service to the Divine",
                    "Daily alignment of personal goals with community welfare",
                    "Teaching and sharing knowledge as spiritual practice",
                    "Creating rituals that honor the sacred dimension of work"
                ],
                progress_indicators=[
                    "Recognition as competent and reliable in your field",
                    "Natural assumption of leadership responsibilities",
                    "Others seeking your guidance in dharmic service",
                    "Ability to balance personal needs with service demands"
                ]
            ),
            
            VarnaLevel.ALIGNING: VarnaGuidance(
                level=VarnaLevel.ALIGNING,
                primary_teaching="Alignment with your dharmic role brings "
                "effortless excellence and natural authority. Your service "
                "becomes a spontaneous expression of your authentic nature.",
                role_understanding=[
                    "Complete embodiment of your varna's essential qualities",
                    "Natural expression of dharmic leadership in your sphere",
                    "Understanding your role's place in cosmic evolution",
                    "Ability to adapt traditional roles to contemporary needs",
                    "Teaching others through example of aligned service"
                ],
                dharmic_service=[
                    "Effortless excellence in service arising from authentic alignment",
                    "Creation of lasting positive impact through your work",
                    "Establishment of institutions that continue your service",
                    "Innovation in methods while maintaining dharmic principles",
                    "Service that naturally attracts resources and support"
                ],
                social_harmony=[
                    "Natural ability to create cooperation between diverse groups",
                    "Modeling of new possibilities for conscious social organization",
                    "Creation of systems that honor everyone's dharmic contribution",
                    "Transformation of conflict into collaborative problem-solving",
                    "Establishment of communities based on dharmic principles"
                ],
                gift_development=[
                    "Mastery level skill in your area of dharmic service",
                    "Natural teaching ability and wisdom transmission",
                    "Integration of spiritual realization with worldly competence",
                    "Creation of new knowledge and methods in your field",
                    "Ability to bring out excellence in others"
                ],
                community_building=[
                    "Creation of self-sustaining communities and organizations",
                    "Training of next generation leaders in dharmic principles",
                    "Establishment of systems that outlast individual involvement",
                    "Building of networks that serve regional and global welfare",
                    "Creation of models for conscious civilization"
                ],
                integration_practices=[
                    "Living as embodiment of dharmic principles in social sphere",
                    "Transmission of wisdom through presence as much as words",
                    "Creation of practices and institutions for future generations",
                    "Integration of personal realization with social transformation",
                    "Being a bridge between spiritual wisdom and practical application"
                ],
                progress_indicators=[
                    "Recognition as authority and innovator in your field",
                    "Natural emergence as leader in conscious social transformation",
                    "Creation of lasting positive institutions and systems",
                    "Training of others who continue and expand your work"
                ]
            ),
            
            VarnaLevel.SERVING: VarnaGuidance(
                level=VarnaLevel.SERVING,
                primary_teaching="Service has become your spontaneous nature. "
                "You serve not from duty but from the overflow of love and "
                "the recognition of unity with all beings.",
                role_understanding=[
                    "Transcendence of personal identification with role",
                    "Understanding role as divine expression through human form",
                    "Natural adaptation to whatever service is needed",
                    "Seeing all roles as equally sacred expressions of the Divine",
                    "Teaching role understanding through presence and example"
                ],
                dharmic_service=[
                    "Service that naturally serves the highest good of all",
                    "Effortless manifestation of resources for service",
                    "Service that transforms not just conditions but consciousness",
                    "Creation of service that continues through others",
                    "Service as natural expression of unity consciousness"
                ],
                social_harmony=[
                    "Natural creation of harmony wherever you are present",
                    "Ability to see and serve the highest potential in all beings",
                    "Establishment of social structures based on love and wisdom",
                    "Transformation of social problems through conscious presence",
                    "Modeling of possibility for enlightened civilization"
                ],
                gift_development=[
                    "Gifts that serve transcendent purposes beyond personal achievement",
                    "Natural transmission of wisdom and capability to others",
                    "Development of new capacities as needed for service",
                    "Integration of multiple gifts in service of unity",
                    "Gifts that inspire and awaken others to their potential"
                ],
                community_building=[
                    "Creation of communities that embody highest human potential",
                    "Natural attraction of souls ready for dharmic transformation",
                    "Establishment of educational and spiritual institutions",
                    "Building of global networks for conscious evolution",
                    "Creation of new models for human civilization"
                ],
                integration_practices=[
                    "Being dharmic principle embodied in human form",
                    "Natural transmission of transformation through presence",
                    "Creation of energy fields that support awakening",
                    "Establishment of spiritual currents for future generations",
                    "Living as servant of cosmic evolution"
                ],
                progress_indicators=[
                    "Recognition as embodiment of dharmic ideals",
                    "Natural emergence as spiritual teacher and guide",
                    "Creation of transformative influence on human consciousness",
                    "Establishment of legacy that serves awakening of humanity"
                ]
            ),
            
            VarnaLevel.HARMONIZING: VarnaGuidance(
                level=VarnaLevel.HARMONIZING,
                primary_teaching="You have become harmony itself. Through "
                "your being, divine order naturally manifests in human "
                "relationships and social structures.",
                role_understanding=[
                    "Complete transcendence of individual role identification",
                    "Being the principle that creates harmony between all roles",
                    "Natural expression of whatever role serves the moment",
                    "Understanding beyond traditional varna framework",
                    "Embodiment of the unity that includes all diversity"
                ],
                dharmic_service=[
                    "Service that transcends individual action and becomes cosmic force",
                    "Natural alignment of all activities with divine will",
                    "Service that creates new possibilities for human evolution",
                    "Being service itself rather than one who serves",
                    "Service that operates through multiple dimensions simultaneously"
                ],
                social_harmony=[
                    "Being the source of harmony rather than creating harmony",
                    "Natural manifestation of divine order in all social interactions",
                    "Establishment of golden age consciousness in human communities",
                    "Transformation of society through presence rather than effort",
                    "Being the bridge between human and divine consciousness"
                ],
                gift_development=[
                    "Gifts that serve cosmic evolution beyond individual species",
                    "Natural manifestation of whatever capacities are needed",
                    "Development of gifts that transcend normal human limitations",
                    "Being the source from which gifts emerge in others",
                    "Gifts that serve multiple dimensional levels of existence"
                ],
                community_building=[
                    "Creation of communities that embody divine consciousness",
                    "Natural magnetism that draws souls ready for transformation",
                    "Establishment of centers for human spiritual evolution",
                    "Building of cosmic networks beyond individual planet",
                    "Being the seed of new evolutionary forms of community"
                ],
                integration_practices=[
                    "Being integration itself beyond all practice",
                    "Natural embodiment of all dharmic principles simultaneously",
                    "Spontaneous transmission of cosmic consciousness",
                    "Creation of new forms of spiritual practice for humanity",
                    "Being a doorway for divine consciousness to enter world"
                ],
                progress_indicators=[
                    "Recognition as embodiment of divine consciousness",
                    "Natural emergence as cosmic teacher and guide",
                    "Creation of transformative spiritual currents",
                    "Establishment of legacy that serves cosmic evolution"
                ]
            )
        }
    
    def _initialize_varna_qualities(self) -> Dict[VarnaType, Dict[str, Any]]:
        """Initialize qualities and expressions of different varnas"""
        return {
            VarnaType.BRAHMANA: {
                "core_qualities": ["Wisdom", "Teaching", "Purity", "Study", "Contemplation"],
                "natural_inclinations": [
                    "Love of learning and sharing knowledge",
                    "Interest in spiritual and philosophical questions",
                    "Natural ability to see patterns and meaning",
                    "Inclination toward teaching and guiding others"
                ],
                "service_expressions": [
                    "Teaching and education",
                    "Spiritual guidance and counseling",
                    "Research and knowledge preservation",
                    "Writing and communication of wisdom"
                ],
                "growth_challenges": [
                    "Avoiding intellectual pride",
                    "Balancing study with practical application",
                    "Serving diverse learning styles and backgrounds"
                ]
            },
            
            VarnaType.KSHATRIYA: {
                "core_qualities": ["Protection", "Leadership", "Courage", "Justice", "Organization"],
                "natural_inclinations": [
                    "Natural leadership and decision-making abilities",
                    "Strong sense of justice and protection of others",
                    "Ability to organize and coordinate group efforts",
                    "Courage in facing challenges and conflicts"
                ],
                "service_expressions": [
                    "Leadership in organizations and communities",
                    "Protection of vulnerable populations",
                    "Administration and governance",
                    "Conflict resolution and peace-making"
                ],
                "growth_challenges": [
                    "Avoiding abuse of power",
                    "Balancing firmness with compassion",
                    "Serving rather than dominating"
                ]
            },
            
            VarnaType.VAISHYA: {
                "core_qualities": ["Provision", "Creation", "Trade", "Abundance", "Sustainability"],
                "natural_inclinations": [
                    "Ability to create and manage material resources",
                    "Natural understanding of economic systems",
                    "Skill in organization and efficiency",
                    "Desire to provide for community needs"
                ],
                "service_expressions": [
                    "Business and entrepreneurship",
                    "Agriculture and food production",
                    "Trade and resource distribution",
                    "Innovation and technological development"
                ],
                "growth_challenges": [
                    "Avoiding excessive focus on material gain",
                    "Balancing profit with social welfare",
                    "Serving community needs rather than just personal success"
                ]
            },
            
            VarnaType.SHUDRA: {
                "core_qualities": ["Service", "Support", "Craftsmanship", "Devotion", "Assistance"],
                "natural_inclinations": [
                    "Natural desire to help and support others",
                    "Skill in practical and hands-on work",
                    "Ability to see and meet immediate needs",
                    "Devotion and loyalty in service"
                ],
                "service_expressions": [
                    "Craftsmanship and skilled trades",
                    "Support services and assistance",
                    "Healthcare and caring professions",
                    "Maintenance and practical problem-solving"
                ],
                "growth_challenges": [
                    "Recognizing the honor and importance of service",
                    "Developing self-worth independent of others' recognition",
                    "Balancing service with self-care"
                ]
            }
        }
    
    def _initialize_service_methods(self) -> Dict[str, List[str]]:
        """Initialize methods for dharmic service"""
        return {
            "discovering_gifts": [
                "Regular reflection on activities that energize you",
                "Seeking feedback from trusted friends and mentors",
                "Experimenting with different forms of service",
                "Studying and testing your natural aptitudes"
            ],
            "developing_skills": [
                "Formal education and training in your area of service",
                "Apprenticeship with accomplished practitioners",
                "Regular practice and disciplined effort",
                "Integration of spiritual principles with practical skills"
            ],
            "creating_harmony": [
                "Active listening and empathetic communication",
                "Seeking to understand different perspectives",
                "Building bridges between conflicting groups",
                "Creating inclusive processes and structures"
            ]
        }
    
    def assess_varna_level(self, query: str, 
                          user_context: Optional[Dict[str, Any]] = None) -> VarnaLevel:
        """Assess user's current social harmony level"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for harmonizing level indicators
        if any(word in query_lower for word in ["divine order", 
                                               "cosmic harmony", "unity consciousness"]):
            return VarnaLevel.HARMONIZING
        
        # Check for serving level indicators
        if any(word in query_lower for word in ["natural service", 
                                               "dharmic leadership", "transcendent purpose"]):
            return VarnaLevel.SERVING
        
        # Check for aligning level indicators
        if any(word in query_lower for word in ["aligned service", 
                                               "effortless excellence", "dharmic mastery"]):
            return VarnaLevel.ALIGNING
        
        # Check for learning level indicators
        if any(word in query_lower for word in ["developing skills", 
                                               "dharmic competence", "service mastery"]):
            return VarnaLevel.LEARNING
        
        # Check for seeking level indicators
        if any(word in query_lower for word in ["finding purpose", 
                                               "discovering gifts", "dharmic role"]):
            return VarnaLevel.SEEKING
        
        # Default to confused
        return VarnaLevel.CONFUSED
    
    def get_scriptural_wisdom(self, level: VarnaLevel) -> str:
        """Get scriptural wisdom appropriate to social harmony level"""
        wisdom_map = {
            VarnaLevel.CONFUSED: "Bhagavad Gita 18.41: 'The duties of brahmanas, kshatriyas, vaishyas, and shudras are distributed according to their natural qualities.'",
            VarnaLevel.SEEKING: "Bhagavad Gita 4.13: 'The four varnas were created by Me according to the division of guna and karma. Though I am their creator, know Me to be the non-doer and eternal.'",
            VarnaLevel.LEARNING: "Bhagavad Gita 18.45: 'A person can attain perfection by being devoted to their own natural work. Listen as I explain how one can become perfect by such devotion.'",
            VarnaLevel.ALIGNING: "Bhagavad Gita 3.35: 'Better is one's own dharma, though imperfectly performed, than the dharma of another well performed.'",
            VarnaLevel.SERVING: "Bhagavad Gita 18.46: 'A person can attain the highest goal by worshipping the Lord through the performance of their natural work.'",
            VarnaLevel.HARMONIZING: "Isha Upanishad 1: 'The entire universe is the creation and property of the Lord. Therefore, accept only what you need for sustenance, and do not covet what belongs to others.'"
        }
        return wisdom_map.get(level, "Bhagavad Gita 3.8: 'Perform your prescribed duties, for action is better than inaction.'")
    
    async def process_varna_query(self, query: str, 
                                user_context: Optional[Dict[str, Any]] = None) -> VarnaResponse:
        """Process social harmony query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess social harmony level
            level = self.assess_varna_level(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            scriptural_wisdom = self.get_scriptural_wisdom(level)
            
            return VarnaResponse(
                varna_level=level.value,
                social_guidance=guidance.primary_teaching,
                role_understanding=guidance.role_understanding,
                dharmic_service=guidance.dharmic_service,
                social_harmony=guidance.social_harmony,
                gift_development=guidance.gift_development,
                community_building=guidance.community_building,
                integration_practices=guidance.integration_practices,
                scriptural_wisdom=scriptural_wisdom
            )
            
        except Exception as e:
            logger.error(f"Error processing varna query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> VarnaResponse:
        """Create fallback response when processing fails"""
        return VarnaResponse(
            varna_level="confused",
            social_guidance="Every being has unique gifts meant to serve the whole. Your confusion about your role is the beginning of discovering your authentic dharmic purpose.",
            role_understanding=[
                "Observe what activities naturally energize you",
                "Notice what problems you feel called to solve",
                "Pay attention to what others seek from you",
                "Consider what contribution feels meaningful to you"
            ],
            dharmic_service=[
                "Start serving wherever you feel called, without attachment",
                "Volunteer in areas that interest or concern you",
                "Offer your natural skills to help others",
                "Practice seeing service as spiritual practice"
            ],
            social_harmony=[
                "Practice appreciating different people's unique contributions",
                "Avoid comparing your path with others' paths",
                "Seek to understand rather than judge different approaches",
                "Look for ways to support others in their dharmic roles"
            ],
            gift_development=[
                "Identify your natural talents and strengths",
                "Study subjects that genuinely interest you",
                "Practice skills that feel natural and enjoyable",
                "Seek mentorship from those you admire"
            ],
            community_building=[
                "Join groups aligned with your values and interests",
                "Practice active listening and empathetic communication",
                "Contribute your unique perspective to group discussions",
                "Help connect people with complementary skills"
            ],
            integration_practices=[
                "Daily reflection on how you served others",
                "Regular assessment of your energy levels in different activities",
                "Meditation on your life purpose and calling",
                "Study of inspiring examples of dharmic service"
            ],
            scriptural_wisdom="Bhagavad Gita 3.8: 'Perform your prescribed duties, for action is better than inaction.'"
        )


# Global instance
_varna_module = None

def get_varna_module() -> VarnaModule:
    """Get global Varna module instance"""
    global _varna_module
    if _varna_module is None:
        _varna_module = VarnaModule()
    return _varna_module

# Factory function for easy access
def create_varna_guidance(query: str, 
                        user_context: Optional[Dict[str, Any]] = None) -> VarnaResponse:
    """Factory function to create varna guidance"""
    import asyncio
    module = get_varna_module()
    return asyncio.run(module.process_varna_query(query, user_context))
