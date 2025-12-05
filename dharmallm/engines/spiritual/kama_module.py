"""
🌸 Kama Module - Desire Transformation and Sacred Relationships
Complete system for transforming desires into dharmic fulfillment
Based on authentic Kama principles beyond mere sensual pleasure
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class KamaLevel(Enum):
    """Levels of desire understanding and transformation"""
    DRIVEN = "driven"               # Driven by unconscious desires
    AWARE = "aware"                 # Becoming aware of desire patterns
    UNDERSTANDING = "understanding" # Understanding desire's true nature
    TRANSFORMING = "transforming"   # Actively transforming desires
    FULFILLED = "fulfilled"         # Desires aligned with dharma
    TRANSCENDENT = "transcendent"   # Beyond desire-based living


class DesireType(Enum):
    """Types of desires in human experience"""
    SURVIVAL = "survival"           # Basic needs: food, shelter, safety
    PLEASURE = "pleasure"           # Sensory enjoyments and comforts
    EMOTIONAL = "emotional"         # Love, connection, belonging
    SOCIAL = "social"               # Recognition, status, achievement
    CREATIVE = "creative"           # Self-expression, beauty, art
    SPIRITUAL = "spiritual"         # Truth, meaning, transcendence


class RelationshipType(Enum):
    """Types of sacred relationships"""
    SELF_RELATIONSHIP = "self_relationship"         # Relationship with self
    INTIMATE_PARTNERSHIP = "intimate_partnership"   # Marriage/life partnership
    FAMILY_BONDS = "family_bonds"                   # Parent-child, siblings
    FRIENDSHIPS = "friendships"                     # Close companions
    COMMUNITY = "community"                         # Social relationships
    DIVINE_RELATIONSHIP = "divine_relationship"     # Connection with Divine


@dataclass
class KamaGuidance:
    """Comprehensive kama guidance"""
    level: KamaLevel
    primary_teaching: str
    desire_understanding: List[str]
    transformation_practices: List[str]
    relationship_wisdom: List[str]
    fulfillment_methods: List[str]
    restraint_practices: List[str]
    integration_guidance: List[str]
    progress_indicators: List[str]


class KamaResponse(BaseModel):
    """Response from Kama module"""
    kama_level: str = Field(description="Current desire transformation level")
    desire_guidance: str = Field(description="Core kama teaching")
    desire_understanding: List[str] = Field(description="Understand desires")
    transformation_practices: List[str] = Field(description="Transform desires")
    relationship_wisdom: List[str] = Field(description="Sacred relationships")
    fulfillment_methods: List[str] = Field(description="True fulfillment")
    restraint_practices: List[str] = Field(description="Healthy boundaries")
    integration_guidance: List[str] = Field(description="Integrate wisdom")
    scriptural_wisdom: str = Field(description="Traditional kama teachings")


class KamaModule:
    """
    🌸 Kama Module - Desire Transformation and Sacred Relationships
    
    Based on authentic Kama teachings:
    - Kama Sutra's philosophy of dharmic pleasure
    - Vedic understanding of desire as spiritual energy
    - Tantric transformation of desire into devotion
    - Bhakti yoga's sublimation of love
    - Vedantic insights on the nature of desires
    
    Kama, as one of the four life goals (purusharthas), represents
    the transformation of desires from unconscious drives into
    conscious expressions of divine love and creativity.
    """
    
    def __init__(self):
        self.name = "Kama"
        self.color = "🌸"
        self.element = "Desire"
        self.principles = ["Sacred Pleasure", "Divine Love", 
                          "Conscious Relationship", "Transformed Desire"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.relationship_wisdom = self._initialize_relationship_wisdom()
        self.transformation_methods = self._initialize_transformation_methods()
        
    def _initialize_guidance_levels(self) -> Dict[KamaLevel, KamaGuidance]:
        """Initialize guidance for different levels of desire transformation"""
        return {
            KamaLevel.DRIVEN: KamaGuidance(
                level=KamaLevel.DRIVEN,
                primary_teaching="You are not your desires. They are visitors "
                "in the guest house of consciousness. Begin to observe them "
                "with curiosity rather than being controlled by them.",
                desire_understanding=[
                    "Notice when desires arise without immediately acting",
                    "Distinguish between needs and wants",
                    "Recognize emotional triggers behind desires",
                    "Understand the temporary nature of all cravings",
                    "See desires as pointing to deeper spiritual needs"
                ],
                transformation_practices=[
                    "Pause and breathe before acting on impulses",
                    "Ask: 'What am I really seeking through this desire?'",
                    "Practice delaying gratification for small desires",
                    "Transform wanting into appreciation for what is",
                    "Use desire energy for creative and spiritual pursuits"
                ],
                relationship_wisdom=[
                    "Relationships mirror your inner state",
                    "Needing others vs. freely choosing to be with them",
                    "Communication is key to resolving relationship conflicts",
                    "Forgiveness frees you from past relationship pain",
                    "Love yourself first to love others authentically"
                ],
                fulfillment_methods=[
                    "Find joy in simple, present-moment experiences",
                    "Cultivate gratitude for what you already have",
                    "Engage in activities that serve others",
                    "Connect with nature to feel natural abundance",
                    "Practice contentment independent of circumstances"
                ],
                restraint_practices=[
                    "Set healthy boundaries with unhealthy desires",
                    "Fast occasionally to understand hunger vs. craving",
                    "Practice celibacy periods to understand sexual energy",
                    "Limit consumption of things that increase agitation",
                    "Create sacred spaces free from excessive stimulation"
                ],
                integration_guidance=[
                    "Balance enjoyment with spiritual practice",
                    "Include others' wellbeing in your pleasure-seeking",
                    "Use periods of fulfillment to deepen spiritual practice",
                    "Transform selfish desires into selfless service",
                    "See all experiences as opportunities for growth"
                ],
                progress_indicators=[
                    "Less compulsive behavior around desires",
                    "Increased awareness before acting on impulses",
                    "More stable mood regardless of getting what you want",
                    "Natural consideration of others in your choices"
                ]
            ),
            
            KamaLevel.AWARE: KamaGuidance(
                level=KamaLevel.AWARE,
                primary_teaching="Awareness is the beginning of freedom. "
                "As you watch desires arise and pass, you discover the "
                "unchanging awareness that you truly are.",
                desire_understanding=[
                    "Detailed observation of desire's arising and passing",
                    "Understanding desire patterns and their root causes",
                    "Recognizing difference between healthy and unhealthy desires",
                    "Seeing how desires create suffering through attachment",
                    "Understanding desires as expressions of creative life force"
                ],
                transformation_practices=[
                    "Meditation on the impermanent nature of all desires",
                    "Inquiry into the source from which desires arise",
                    "Transforming sexual energy through breath and awareness",
                    "Using artistic expression to channel creative desires",
                    "Practicing conscious choice rather than unconscious reaction"
                ],
                relationship_wisdom=[
                    "Seeing others as mirrors reflecting your unconscious patterns",
                    "Learning to give and receive love without attachment",
                    "Understanding projection and taking back projections",
                    "Communicating needs clearly without manipulation",
                    "Practicing unconditional love starting with yourself"
                ],
                fulfillment_methods=[
                    "Finding fulfillment through creative self-expression",
                    "Experiencing joy through service to others",
                    "Discovering the bliss of deep meditation",
                    "Appreciating beauty in art, nature, and relationships",
                    "Cultivating inner richness independent of external circumstances"
                ],
                restraint_practices=[
                    "Conscious fasting to understand the difference between need and greed",
                    "Periods of sensory withdrawal to deepen inner awareness",
                    "Restraint in speech to cultivate mindful communication",
                    "Limiting entertainment that stimulates unhealthy desires",
                    "Practicing moderation in all pleasures"
                ],
                integration_guidance=[
                    "Integrating spiritual insights into daily relationship interactions",
                    "Balancing solitude and social connection consciously",
                    "Using pleasure as gateway to appreciation and gratitude",
                    "Transforming personal desires into compassionate action",
                    "Creating rituals that honor both human and divine aspects"
                ],
                progress_indicators=[
                    "Clear discrimination between healthy and unhealthy desires",
                    "Ability to enjoy pleasures without becoming addicted",
                    "More harmonious and conscious relationships",
                    "Natural arising of compassion and service orientation"
                ]
            ),
            
            KamaLevel.UNDERSTANDING: KamaGuidance(
                level=KamaLevel.UNDERSTANDING,
                primary_teaching="Understanding reveals desire as life energy "
                "seeking expression. When understood properly, desire becomes "
                "a doorway to divine love and creative expression.",
                desire_understanding=[
                    "Deep comprehension of desire as misdirected spiritual seeking",
                    "Understanding the role of desire in spiritual evolution",
                    "Recognizing desire as divine energy expressing through human form",
                    "Seeing all desires as ultimately seeking union with the Divine",
                    "Understanding how to work with desire rather than against it"
                ],
                transformation_practices=[
                    "Advanced pranayama to transform sexual energy into spiritual energy",
                    "Tantric practices that honor desire while transcending attachment",
                    "Bhakti practices that transform all desires into love for Divine",
                    "Creative practices that channel desire into beautiful expression",
                    "Service practices that transform selfish desires into selfless action"
                ],
                relationship_wisdom=[
                    "Understanding relationships as spiritual partnerships for growth",
                    "Seeing the Divine in your beloved and treating them as such",
                    "Creating conscious relationship agreements based on mutual spiritual growth",
                    "Understanding the difference between love and attachment",
                    "Using relationship challenges as opportunities for spiritual development"
                ],
                fulfillment_methods=[
                    "Finding ultimate fulfillment through self-realization",
                    "Experiencing fulfillment through devotional practices",
                    "Discovering fulfillment through creative self-expression",
                    "Finding fulfillment through serving something greater than yourself",
                    "Cultivating fulfillment through direct experience of unity"
                ],
                restraint_practices=[
                    "Conscious celibacy as spiritual practice when appropriate",
                    "Restraint practiced as devotion rather than suppression",
                    "Using restraint to build spiritual energy for higher purposes",
                    "Practicing restraint in consumption to deepen appreciation",
                    "Restraining automatic reactions to cultivate conscious response"
                ],
                integration_guidance=[
                    "Integrating understanding into all areas of life seamlessly",
                    "Teaching others through example of transformed desire",
                    "Creating sacred sexuality practices with spiritual partner",
                    "Establishing daily practices that honor both human and divine nature",
                    "Using all experiences as opportunities to deepen understanding"
                ],
                progress_indicators=[
                    "Natural integration of spiritual understanding into relationships",
                    "Others seeking guidance about desire and relationship issues",
                    "Effortless balance between enjoyment and spiritual practice",
                    "Creative expressions that inspire and uplift others"
                ]
            ),
            
            KamaLevel.TRANSFORMING: KamaGuidance(
                level=KamaLevel.TRANSFORMING,
                primary_teaching="You have become an alchemist of desire, "
                "transforming base wants into golden expressions of divine "
                "love. Your relationships become sacred temples.",
                desire_understanding=[
                    "Complete understanding of desire as creative force of universe",
                    "Recognition of your role as conscious director of desire energy",
                    "Understanding how personal transformation affects collective consciousness",
                    "Seeing all beings as expressions of divine desire for experience",
                    "Understanding the cosmic purpose behind human desires"
                ],
                transformation_practices=[
                    "Mastery of energy transformation through advanced yogic practices",
                    "Ability to transmute any desire into spiritual fuel",
                    "Advanced tantric practices that unite individual and cosmic consciousness",
                    "Teaching others the art of desire transformation",
                    "Creating new methods for conscious desire transformation"
                ],
                relationship_wisdom=[
                    "Relationships as conscious co-creation of divine love on earth",
                    "Ability to see and relate to the highest Self in all beings",
                    "Creating relationship dynamics that support mutual awakening",
                    "Understanding relationship as service to cosmic evolution",
                    "Modeling divine love through human relationship"
                ],
                fulfillment_methods=[
                    "Continuous fulfillment through serving divine will",
                    "Fulfillment through facilitating others' spiritual growth",
                    "Creative fulfillment through expressions that serve awakening",
                    "Fulfillment through conscious participation in cosmic evolution",
                    "Fulfillment through unity consciousness in all activities"
                ],
                restraint_practices=[
                    "Effortless restraint arising from fulfilled satisfaction",
                    "Using restraint as powerful tool for serving others",
                    "Restraint practiced from abundance rather than scarcity",
                    "Teaching restraint as pathway to true freedom",
                    "Restraint that enhances rather than diminishes life energy"
                ],
                integration_guidance=[
                    "Complete integration of understanding into life mission",
                    "Creating institutions that support conscious relationship",
                    "Training others in the art of desire transformation",
                    "Establishing communities based on transformed consciousness",
                    "Living as example of possibility for humanity"
                ],
                progress_indicators=[
                    "Recognition as teacher and guide in conscious relationship",
                    "Creation of lasting positive impact on others' relationships",
                    "Establishment of practices and communities that outlast your lifetime",
                    "Living as embodiment of divine love in human form"
                ]
            ),
            
            KamaLevel.FULFILLED: KamaGuidance(
                level=KamaLevel.FULFILLED,
                primary_teaching="Fulfillment has become your natural state. "
                "Desires arise and are fulfilled not for personal satisfaction "
                "but as expressions of cosmic play and divine service.",
                desire_understanding=[
                    "Perfect understanding of desire as divine energy expressing through form",
                    "Recognition that all desires ultimately seek return to Source",
                    "Understanding your role as facilitator of cosmic desire fulfillment",
                    "Seeing individual desires as movements in universal consciousness",
                    "Complete integration of personal and cosmic perspectives on desire"
                ],
                transformation_practices=[
                    "Effortless transformation of any energy into service",
                    "Natural alignment of all desires with cosmic will",
                    "Spontaneous transmission of transformation to others",
                    "Advanced teaching of transformation through pure presence",
                    "Creation of energy fields that automatically transform desires"
                ],
                relationship_wisdom=[
                    "All relationships experienced as Divine relating to Itself",
                    "Perfect balance of intimacy and freedom in all connections",
                    "Relationships that automatically serve awakening of all involved",
                    "Natural ability to bring out highest potential in others",
                    "Living as embodiment of unconditional divine love"
                ],
                fulfillment_methods=[
                    "Continuous fulfillment through being rather than doing",
                    "Fulfillment that is independent of circumstances",
                    "Natural overflow of fulfillment blessing all beings",
                    "Fulfillment through perfect unity with cosmic purpose",
                    "Being fulfillment itself rather than seeking it"
                ],
                restraint_practices=[
                    "Restraint and indulgence both arising from perfect balance",
                    "Natural restraint that serves optimal functioning",
                    "Restraint as expression of divine wisdom rather than personal effort",
                    "Using restraint to model optimal human functioning",
                    "Restraint that increases rather than decreases life force"
                ],
                integration_guidance=[
                    "Perfect integration expressed through every breath and action",
                    "Teaching integration through mere presence and being",
                    "Creating organizational structures that embody integration",
                    "Establishing legacy of integration for future generations",
                    "Being integration itself in service to cosmic evolution"
                ],
                progress_indicators=[
                    "Recognition as embodiment of fulfilled human potential",
                    "Natural transmission of fulfillment to all encountered beings",
                    "Establishment of enduring positive influence on human consciousness",
                    "Living as proof of possibility for human transformation"
                ]
            ),
            
            KamaLevel.TRANSCENDENT: KamaGuidance(
                level=KamaLevel.TRANSCENDENT,
                primary_teaching="You have transcended the very framework "
                "of desire and fulfillment. What acts through your form "
                "is pure divine love expressing itself for cosmic purposes.",
                desire_understanding=[
                    "Perfect transcendence of personal desire framework",
                    "Understanding that transcends individual and cosmic perspectives",
                    "Direct knowing of reality beyond desire and fulfillment",
                    "Recognition that even transcendence is movement in consciousness",
                    "Complete freedom from all conceptual frameworks about desire"
                ],
                transformation_practices=[
                    "Being transformation itself rather than practicing it",
                    "Natural emanation of transformative energy without effort",
                    "Direct transmission of transcendent understanding",
                    "Spontaneous creation of awakening through pure presence",
                    "Being the source of transformation for all beings"
                ],
                relationship_wisdom=[
                    "Complete transcendence of relationship framework",
                    "Recognition that separation never existed",
                    "Direct expression of love beyond personal and impersonal",
                    "Unity consciousness that includes and transcends all relationships",
                    "Being love itself rather than having relationships"
                ],
                fulfillment_methods=[
                    "Transcendence of fulfillment and non-fulfillment",
                    "Being fulfillment beyond the concept of fulfillment",
                    "Natural state beyond seeking and finding",
                    "Direct being without reference to fulfillment",
                    "Perfect contentment beyond satisfaction and dissatisfaction"
                ],
                restraint_practices=[
                    "Transcendence of restraint and non-restraint",
                    "Natural functioning beyond effort and effortlessness",
                    "Perfect balance beyond the concept of balance",
                    "Being that includes and transcends all practices",
                    "Natural state beyond discipline and indulgence"
                ],
                integration_guidance=[
                    "Being beyond integration and non-integration",
                    "Natural functioning that serves without intention",
                    "Direct embodiment of truth beyond understanding",
                    "Perfect expression beyond conscious and unconscious",
                    "Being itself as ultimate teaching"
                ],
                progress_indicators=[
                    "Complete transcendence of all progress frameworks",
                    "Being beyond achievement and non-achievement",
                    "Natural state beyond spiritual and non-spiritual",
                    "Perfect embodiment of what cannot be described"
                ]
            )
        }
    
    def _initialize_relationship_wisdom(self) -> Dict[RelationshipType, Dict[str, Any]]:
        """Initialize wisdom for different types of relationships"""
        return {
            RelationshipType.SELF_RELATIONSHIP: {
                "core_principle": "You cannot love others more than you love yourself",
                "practices": [
                    "Daily self-compassion and forgiveness practice",
                    "Regular check-ins with your inner emotional state",
                    "Honoring your needs and boundaries"
                ],
                "challenges": ["Self-criticism", "Perfectionism", "Neglecting self-care"],
                "growth_edge": "Unconditional self-acceptance and authentic self-expression"
            },
            
            RelationshipType.INTIMATE_PARTNERSHIP: {
                "core_principle": "Sacred partnership serves mutual awakening",
                "practices": [
                    "Daily appreciation and gratitude expressions",
                    "Conscious communication about needs and feelings",
                    "Shared spiritual practices and growth activities"
                ],
                "challenges": ["Attachment", "Projection", "Power struggles"],
                "growth_edge": "Loving without possessing, growing together in truth"
            },
            
            RelationshipType.FAMILY_BONDS: {
                "core_principle": "Family relationships teach unconditional love",
                "practices": [
                    "Honoring family members while maintaining healthy boundaries",
                    "Healing ancestral patterns through conscious awareness",
                    "Expressing love in ways family members can receive"
                ],
                "challenges": ["Old patterns", "Expectations", "Conditional love"],
                "growth_edge": "Loving family as they are while inspiring growth"
            }
        }
    
    def _initialize_transformation_methods(self) -> Dict[DesireType, Dict[str, Any]]:
        """Initialize transformation methods for different desire types"""
        return {
            DesireType.SURVIVAL: {
                "understanding": "Basic needs are sacred and honoring them supports spiritual growth",
                "transformation": "Meet needs consciously and with gratitude",
                "practices": [
                    "Mindful eating as spiritual practice",
                    "Creating sacred space in your home",
                    "Gratitude for body's needs and Earth's provision"
                ]
            },
            
            DesireType.PLEASURE: {
                "understanding": "Pleasure is a gift from Divine to be enjoyed consciously",
                "transformation": "Transform seeking pleasure into appreciating present beauty",
                "practices": [
                    "Conscious enjoyment of sensory experiences",
                    "Using pleasure as gateway to gratitude",
                    "Sharing pleasurable experiences with others"
                ]
            },
            
            DesireType.SPIRITUAL: {
                "understanding": "Spiritual desires point toward your true nature",
                "transformation": "Recognize that what you seek spiritually, you already are",
                "practices": [
                    "Self-inquiry into the nature of the seeker",
                    "Meditation on your true identity",
                    "Service as expression of spiritual understanding"
                ]
            }
        }
    
    def assess_kama_level(self, query: str, 
                        user_context: Optional[Dict[str, Any]] = None) -> KamaLevel:
        """Assess user's current desire transformation level"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for transcendent level indicators
        if any(word in query_lower for word in ["beyond desire", 
                                               "transcendent love", "cosmic consciousness"]):
            return KamaLevel.TRANSCENDENT
        
        # Check for fulfilled level indicators
        if any(word in query_lower for word in ["fulfilled naturally", 
                                               "divine service", "cosmic purpose"]):
            return KamaLevel.FULFILLED
        
        # Check for transforming level indicators
        if any(word in query_lower for word in ["transforming desires", 
                                               "sacred relationships", "divine love"]):
            return KamaLevel.TRANSFORMING
        
        # Check for understanding level indicators
        if any(word in query_lower for word in ["understanding desire", 
                                               "spiritual energy", "conscious relationship"]):
            return KamaLevel.UNDERSTANDING
        
        # Check for aware level indicators
        if any(word in query_lower for word in ["aware of desires", 
                                               "watching patterns", "observing wants"]):
            return KamaLevel.AWARE
        
        # Default to driven
        return KamaLevel.DRIVEN
    
    def get_scriptural_wisdom(self, level: KamaLevel) -> str:
        """Get scriptural wisdom appropriate to desire transformation level"""
        wisdom_map = {
            KamaLevel.DRIVEN: "Bhagavad Gita 2.62-63: 'While contemplating objects of the senses, attachment develops. From attachment comes desire, from desire comes anger.'",
            KamaLevel.AWARE: "Bhagavad Gita: 'One who can withdraw the senses from sense objects like a tortoise withdrawing its limbs attains steady wisdom.'",
            KamaLevel.UNDERSTANDING: "Rumi: 'In your light I learn how to love. In your beauty, how to make poems. You dance inside my chest where no one sees you, but sometimes I do, and that sight becomes this art, this music, this form.'",
            KamaLevel.TRANSFORMING: "Bhagavad Gita 7.11: 'I am the strength of the strong, devoid of passion and desire. I am sex life which is not contrary to dharma.'",
            KamaLevel.FULFILLED: "Vijnana Bhairava Tantra: 'At the moment of sexual pleasure, keep attention on the fire in the beginning, and, so continuing, avoid the embers in the end.'",
            KamaLevel.TRANSCENDENT: "Ashtavakra Gita: 'You are pure consciousness. The world is an appearance in you. How can there be desire when there is no other?'"
        }
        return wisdom_map.get(level, "Katha Upanishad: 'When all desires dwelling in the heart are cast away, the mortal becomes immortal and attains Brahman.'")
    
    async def process_kama_query(self, query: str, 
                               user_context: Optional[Dict[str, Any]] = None) -> KamaResponse:
        """Process desire-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess desire transformation level
            level = self.assess_kama_level(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            scriptural_wisdom = self.get_scriptural_wisdom(level)
            
            return KamaResponse(
                kama_level=level.value,
                desire_guidance=guidance.primary_teaching,
                desire_understanding=guidance.desire_understanding,
                transformation_practices=guidance.transformation_practices,
                relationship_wisdom=guidance.relationship_wisdom,
                fulfillment_methods=guidance.fulfillment_methods,
                restraint_practices=guidance.restraint_practices,
                integration_guidance=guidance.integration_guidance,
                scriptural_wisdom=scriptural_wisdom
            )
            
        except Exception as e:
            logger.error(f"Error processing kama query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> KamaResponse:
        """Create fallback response when processing fails"""
        return KamaResponse(
            kama_level="driven",
            desire_guidance="You are not your desires. They are visitors in the guest house of consciousness. Begin to observe them with curiosity rather than being controlled by them.",
            desire_understanding=[
                "Notice when desires arise without immediately acting",
                "Distinguish between needs and wants",
                "Recognize emotional triggers behind desires",
                "Understand the temporary nature of all cravings"
            ],
            transformation_practices=[
                "Pause and breathe before acting on impulses",
                "Ask: 'What am I really seeking through this desire?'",
                "Practice delaying gratification for small desires",
                "Transform wanting into appreciation for what is"
            ],
            relationship_wisdom=[
                "Relationships mirror your inner state",
                "Needing others vs. freely choosing to be with them",
                "Communication is key to resolving relationship conflicts",
                "Love yourself first to love others authentically"
            ],
            fulfillment_methods=[
                "Find joy in simple, present-moment experiences",
                "Cultivate gratitude for what you already have",
                "Engage in activities that serve others",
                "Connect with nature to feel natural abundance"
            ],
            restraint_practices=[
                "Set healthy boundaries with unhealthy desires",
                "Fast occasionally to understand hunger vs. craving",
                "Limit consumption of things that increase agitation",
                "Create sacred spaces free from excessive stimulation"
            ],
            integration_guidance=[
                "Balance enjoyment with spiritual practice",
                "Include others' wellbeing in your pleasure-seeking",
                "Transform selfish desires into selfless service",
                "See all experiences as opportunities for growth"
            ],
            scriptural_wisdom="Katha Upanishad: 'When all desires dwelling in the heart are cast away, the mortal becomes immortal and attains Brahman.'"
        )


# Global instance
_kama_module = None

def get_kama_module() -> KamaModule:
    """Get global Kama module instance"""
    global _kama_module
    if _kama_module is None:
        _kama_module = KamaModule()
    return _kama_module

# Factory function for easy access
def create_kama_guidance(query: str, 
                       user_context: Optional[Dict[str, Any]] = None) -> KamaResponse:
    """Factory function to create kama guidance"""
    import asyncio
    module = get_kama_module()
    return asyncio.run(module.process_kama_query(query, user_context))
