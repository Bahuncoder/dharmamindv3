"""
🛡️ Raksha Module - Divine Protection and Spiritual Security
Complete system for spiritual protection and energetic shielding
Based on traditional Raksha principles and divine protection practices
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RakshaLevel(Enum):
    """Levels of divine protection awareness"""
    VULNERABLE = "vulnerable"       # No protection awareness
    SEEKING = "seeking"             # Looking for protection
    LEARNING = "learning"           # Understanding protection methods
    PRACTICING = "practicing"       # Active protection practices
    PROTECTED = "protected"         # Strong spiritual protection
    GUARDIAN = "guardian"           # Protecting others through divine grace


class ThreatType(Enum):
    """Types of spiritual and energetic threats"""
    NEGATIVE_ENERGY = "negative_energy"         # Psychic attacks, bad vibes
    EVIL_EYE = "evil_eye"                      # Jealousy-based harm
    SPIRITUAL_ATTACK = "spiritual_attack"       # Deliberate occult harm
    ENTITY_ATTACHMENT = "entity_attachment"     # Negative entity influence
    ENVIRONMENTAL = "environmental"             # Toxic places/people
    KARMIC_DEBTS = "karmic_debts"              # Past life influences


class ProtectionMethod(Enum):
    """Methods of spiritual protection"""
    MANTRA = "mantra"                          # Sacred sound protection
    YANTRA = "yantra"                          # Sacred geometry shields
    KAVACH = "kavach"                          # Protective armor mantras
    DEITY_INVOCATION = "deity_invocation"      # Divine grace protection
    AURA_SHIELDING = "aura_shielding"         # Energy field protection
    SACRED_SPACE = "sacred_space"              # Consecrated environments


@dataclass
class RakshaGuidance:
    """Comprehensive raksha guidance"""
    level: RakshaLevel
    primary_teaching: str
    threat_recognition: List[str]
    protection_methods: List[str]
    daily_practices: List[str]
    emergency_protocols: List[str]
    strengthening_rituals: List[str]
    blessing_practices: List[str]
    progress_indicators: List[str]


class RakshaResponse(BaseModel):
    """Response from Raksha module"""
    raksha_level: str = Field(description="Current protection level")
    protection_guidance: str = Field(description="Core raksha teaching")
    threat_recognition: List[str] = Field(description="Identify threats")
    protection_methods: List[str] = Field(description="Active protection")
    daily_practices: List[str] = Field(description="Daily protection")
    emergency_protocols: List[str] = Field(description="Crisis protection")
    strengthening_rituals: List[str] = Field(description="Build protection")
    blessing_practices: List[str] = Field(description="Bless and protect")
    scriptural_wisdom: str = Field(description="Traditional protection teachings")


class RakshaModule:
    """
    🛡️ Raksha Module - Divine Protection and Spiritual Security
    
    Based on traditional Raksha teachings:
    - Vedic protection mantras and yantras
    - Tantric kavach (protective armor) practices  
    - Devotional surrender to protective deities
    - Ayurvedic energetic protection methods
    - Folk wisdom on warding off negative influences
    
    Raksha encompasses all forms of spiritual protection, from basic
    energy hygiene to advanced protective rituals and divine grace.
    """
    
    def __init__(self):
        self.name = "Raksha"
        self.color = "🛡️"
        self.element = "Protection"
        self.principles = ["Divine Grace", "Energetic Boundaries", 
                          "Sacred Defense", "Blessing Power"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.protection_mantras = self._initialize_protection_mantras()
        self.emergency_responses = self._initialize_emergency_responses()
        
    def _initialize_guidance_levels(self) -> Dict[RakshaLevel, RakshaGuidance]:
        """Initialize guidance for different levels of protection awareness"""
        return {
            RakshaLevel.VULNERABLE: RakshaGuidance(
                level=RakshaLevel.VULNERABLE,
                primary_teaching="You are never truly defenseless. Divine "
                "protection is your birthright. Begin with simple practices "
                "to strengthen your spiritual boundaries and call upon higher help.",
                threat_recognition=[
                    "Feeling drained after being around certain people",
                    "Sudden mood changes in specific locations",
                    "Recurring nightmares or disturbed sleep",
                    "Unexplained anxiety or fear without cause",
                    "Feeling watched or followed by unseen presence"
                ],
                protection_methods=[
                    "Daily recitation of protective mantras",
                    "Visualization of white light surrounding you",
                    "Carrying protective symbols or talismans",
                    "Regular cleansing baths with salt or sacred herbs",
                    "Invoking divine names when feeling threatened"
                ],
                daily_practices=[
                    "Morning protection prayer upon waking",
                    "Visualize golden shield around your aura",
                    "Chant 'Om Gam Ganapataye Namaha' for obstacles",
                    "End day with gratitude to protective forces",
                    "Cleanse energy field before sleep"
                ],
                emergency_protocols=[
                    "Immediately chant 'Hanuman Chalisa' or 'Durga Mantra'",
                    "Visualize Hanuman or Durga Ma protecting you",
                    "Call upon your chosen deity for immediate help",
                    "Burn sacred incense or camphor if available",
                    "Remove yourself from threatening situation"
                ],
                strengthening_rituals=[
                    "Weekly deep cleansing with turmeric bath",
                    "Light ghee lamp while chanting protection mantras",
                    "Create sacred space with rangoli or kolam",
                    "Offer flowers to protective deities",
                    "Fast on protective deity's special days"
                ],
                blessing_practices=[
                    "Bless your home with sacred water daily",
                    "Place tulsi or neem plants for protection",
                    "Hang protective symbols at entrances",
                    "Share food with others to increase good karma",
                    "Speak only truthful and kind words"
                ],
                progress_indicators=[
                    "Feeling more confident in challenging situations",
                    "Better sleep and fewer disturbing dreams",
                    "Increased awareness of energy around you",
                    "Natural urge to help protect others"
                ]
            ),
            
            RakshaLevel.SEEKING: RakshaGuidance(
                level=RakshaLevel.SEEKING,
                primary_teaching="Your seeking itself creates protection. "
                "The sincere desire for divine help attracts helpful forces. "
                "Learn to distinguish between real and imagined threats.",
                threat_recognition=[
                    "Discerning psychic attacks from emotional sensitivity",
                    "Recognizing energy vampires and toxic relationships",
                    "Identifying places with heavy negative vibrations",
                    "Understanding generational or karmic patterns",
                    "Detecting entity attachments vs. psychological issues"
                ],
                protection_methods=[
                    "Advanced mantra practices with specific deities",
                    "Creation and consecration of personal yantras",
                    "Aura strengthening through pranayama",
                    "Protective mudras during meditation",
                    "Invoking protective circles before spiritual work"
                ],
                daily_practices=[
                    "Detailed morning protection ritual",
                    "Hourly brief mantra repetition",
                    "Energy scanning and cleansing check-ins",
                    "Evening protection review and gratitude",
                    "Protective visualization before meals"
                ],
                emergency_protocols=[
                    "Specific mantras for different types of attacks",
                    "Rapid yantra drawing for immediate protection",
                    "Emergency contact with spiritual guide or guru",
                    "Protective breathing techniques",
                    "Sacred geometry visualization for shielding"
                ],
                strengthening_rituals=[
                    "Monthly fire ceremony (havan) for protection",
                    "Pilgrimage to protective deity temples",
                    "Advanced fasting and purification practices",
                    "Group protection rituals with spiritual community",
                    "Study of protective texts and scriptures"
                ],
                blessing_practices=[
                    "Blessing food and water before consumption",
                    "Protective mantras for family and loved ones",
                    "Creating protective energy for community",
                    "Teaching basic protection to others",
                    "Maintaining protective altar or shrine"
                ],
                progress_indicators=[
                    "Clear discrimination between real and false threats",
                    "Spontaneous protective responses to danger",
                    "Others seeking your guidance on protection",
                    "Ability to clear negative spaces effectively"
                ]
            ),
            
            RakshaLevel.LEARNING: RakshaGuidance(
                level=RakshaLevel.LEARNING,
                primary_teaching="Knowledge is protection. Understanding the "
                "subtle laws of energy and karma creates natural immunity. "
                "Study traditional texts while developing intuitive wisdom.",
                threat_recognition=[
                    "Sophisticated understanding of subtle energy dynamics",
                    "Recognition of protection needs in others",
                    "Ability to diagnose spiritual protection issues",
                    "Understanding of planetary and cosmic influences",
                    "Recognition of protection gaps in spiritual practices"
                ],
                protection_methods=[
                    "Complex kavach mantras and protective armor",
                    "Multi-layered yantra systems for different needs",
                    "Personalized protection based on birth chart",
                    "Advanced energy healing for protection",
                    "Coordination with spiritual guides and masters"
                ],
                daily_practices=[
                    "Sophisticated morning protection establishment",
                    "Regular energy field maintenance throughout day",
                    "Protective practices during all spiritual work",
                    "Teaching protection methods to students",
                    "Research and experimentation with new methods"
                ],
                emergency_protocols=[
                    "Advanced diagnostic skills for protection crises",
                    "Ability to perform protection for others remotely",
                    "Knowledge of when to refer to higher authorities",
                    "Sophisticated clearing and exorcism techniques",
                    "Emergency network of protection practitioners"
                ],
                strengthening_rituals=[
                    "Seasonal protection renewal ceremonies",
                    "Advanced tantric protection practices",
                    "Group leadership in protection rituals",
                    "Creation of protective devices and talismans",
                    "Advanced study with protection masters"
                ],
                blessing_practices=[
                    "Blessing and protecting spiritual communities",
                    "Creating protective fields for events and gatherings",
                    "Teaching comprehensive protection courses",
                    "Writing and sharing protection knowledge",
                    "Mentoring others in protection development"
                ],
                progress_indicators=[
                    "Natural protective presence that others feel",
                    "Ability to teach protection effectively",
                    "Recognition by spiritual community as protector",
                    "Success in protecting others from harm"
                ]
            ),
            
            RakshaLevel.PRACTICING: RakshaGuidance(
                level=RakshaLevel.PRACTICING,
                primary_teaching="Protection becomes effortless when aligned "
                "with divine will. Your practice creates protective fields "
                "that benefit all beings. Focus on service rather than fear.",
                threat_recognition=[
                    "Instantaneous recognition of all forms of spiritual danger",
                    "Ability to perceive protection needs before they manifest",
                    "Understanding of cosmic timing in protection work",
                    "Recognition of collective protection needs",
                    "Ability to work with protective hierarchies"
                ],
                protection_methods=[
                    "Spontaneous protective response to any situation",
                    "Ability to create protective fields through presence",
                    "Advanced work with protective deities and guides",
                    "Transmission of protection through touch or word",
                    "Creation of long-lasting protective environments"
                ],
                daily_practices=[
                    "Living as continuous protective presence",
                    "Natural integration of protection with all activities",
                    "Protecting through example and consciousness",
                    "Teaching protection through transmission",
                    "Maintaining global protection awareness"
                ],
                emergency_protocols=[
                    "Instantaneous response to protection emergencies",
                    "Ability to coordinate group protection efforts",
                    "Direct communication with protective forces",
                    "Intervention in collective spiritual crises",
                    "Protection work during natural disasters"
                ],
                strengthening_rituals=[
                    "Advanced tantric protection accomplishments",
                    "Leadership in major protection ceremonies",
                    "Creation of protection systems for institutions",
                    "Advanced initiation into protection mysteries",
                    "Direct discipleship with protection masters"
                ],
                blessing_practices=[
                    "Blessing protection for large groups automatically",
                    "Creating protection for future generations",
                    "Establishing permanent protective institutions",
                    "Training advanced protection practitioners",
                    "Writing definitive texts on protection methods"
                ],
                progress_indicators=[
                    "Recognition as protection master by peers",
                    "Ability to protect across time and space",
                    "Establishment of protection lineages",
                    "Documented success in major protection works"
                ]
            ),
            
            RakshaLevel.PROTECTED: RakshaGuidance(
                level=RakshaLevel.PROTECTED,
                primary_teaching="You have become a fortress of divine grace. "
                "Your very being radiates protection. Now serve as guardian "
                "for those still vulnerable to spiritual harm.",
                threat_recognition=[
                    "Perfect discrimination of real vs. illusory threats",
                    "Ability to see protection needs across dimensions",
                    "Understanding of cosmic cycles affecting protection",
                    "Recognition of evolutionary protection requirements",
                    "Perception of threats to spiritual evolution itself"
                ],
                protection_methods=[
                    "Effortless manifestation of perfect protection",
                    "Ability to protect through pure consciousness",
                    "Direct command of protective cosmic forces",
                    "Protection through blessing and grace transmission",
                    "Creation of protection across multiple lifetimes"
                ],
                daily_practices=[
                    "Being protection itself rather than practicing it",
                    "Natural emanation of protective force",
                    "Continuous blessing of all encountered beings",
                    "Teaching protection through mere presence",
                    "Maintaining cosmic protection awareness"
                ],
                emergency_protocols=[
                    "Immediate resolution of any protection crisis",
                    "Ability to prevent problems before they arise",
                    "Direct intervention through divine authority",
                    "Coordination of cosmic protection forces",
                    "Protection of collective human consciousness"
                ],
                strengthening_rituals=[
                    "Advanced cosmic protection ceremonies",
                    "Initiation of others into protection mysteries",
                    "Creation of protection for future ages",
                    "Direct work with cosmic protection councils",
                    "Establishment of protection for entire lineages"
                ],
                blessing_practices=[
                    "Automatic protection blessing for all beings",
                    "Creation of protection across time and space",
                    "Establishment of permanent protection institutions",
                    "Training of advanced protection masters",
                    "Divine protection through pure grace"
                ],
                progress_indicators=[
                    "Recognition as embodiment of divine protection",
                    "Ability to protect across multiple dimensions",
                    "Establishment of eternal protection systems",
                    "Direct communication with cosmic protection forces"
                ]
            ),
            
            RakshaLevel.GUARDIAN: RakshaGuidance(
                level=RakshaLevel.GUARDIAN,
                primary_teaching="You are divine protection incarnate. Through "
                "you, the cosmic protection force works to safeguard all "
                "beings. You are guardian of the dharma itself.",
                threat_recognition=[
                    "Perfect omniscient awareness of all threats",
                    "Ability to perceive threats across all dimensions",
                    "Understanding of threats to cosmic evolution",
                    "Recognition of protection needs of future ages",
                    "Perception of ultimate spiritual dangers"
                ],
                protection_methods=[
                    "Perfect effortless protection through divine will",
                    "Ability to protect entire worlds through consciousness",
                    "Direct embodiment of cosmic protection force",
                    "Protection through pure love and grace",
                    "Creation of eternal protection systems"
                ],
                daily_practices=[
                    "Being cosmic protection force incarnate",
                    "Natural protection of all existence",
                    "Continuous blessing of universal evolution",
                    "Teaching through pure divine transmission",
                    "Maintaining ultimate cosmic protection"
                ],
                emergency_protocols=[
                    "Instantaneous cosmic intervention capability",
                    "Prevention of threats across all dimensions",
                    "Direct embodiment of divine protective will",
                    "Command of all cosmic protection forces",
                    "Protection of cosmic dharma itself"
                ],
                strengthening_rituals=[
                    "Being the source of all protection rituals",
                    "Direct transmission of protection mastery",
                    "Creation of protection for cosmic cycles",
                    "Direct unity with cosmic protection source",
                    "Establishment of eternal protection dharma"
                ],
                blessing_practices=[
                    "Universal automatic protection blessing",
                    "Creation of protection across all time and space",
                    "Direct emanation of divine protective grace",
                    "Training of cosmic protection guardians",
                    "Being divine protection itself"
                ],
                progress_indicators=[
                    "Recognition as divine protection incarnation",
                    "Complete unity with cosmic protection force",
                    "Establishment of eternal dharma protection",
                    "Perfect embodiment of divine guardian presence"
                ]
            )
        }
    
    def _initialize_protection_mantras(self) -> Dict[ThreatType, Dict[str, Any]]:
        """Initialize protective mantras for different threat types"""
        return {
            ThreatType.NEGATIVE_ENERGY: {
                "primary_mantra": "Om Gam Ganapataye Namaha",
                "description": "Removes obstacles and negative influences",
                "practice": "108 times while visualizing Ganesha",
                "yantra": "Ganesha Yantra drawn with red kumkum"
            },
            
            ThreatType.EVIL_EYE: {
                "primary_mantra": "Om Dum Durgayei Namaha",
                "description": "Divine Mother's protection from jealousy",
                "practice": "21 times with camphor flame offering",
                "yantra": "Durga Yantra in copper or silver"
            },
            
            ThreatType.SPIRITUAL_ATTACK: {
                "primary_mantra": "Om Hanumate Namaha",
                "description": "Hanuman's power destroys all attacks",
                "practice": "Hanuman Chalisa daily with devotion",
                "yantra": "Hanuman Yantra facing south"
            },
            
            ThreatType.ENTITY_ATTACHMENT: {
                "primary_mantra": "Om Namah Shivaya",
                "description": "Shiva's power removes all attachments",
                "practice": "Maha Mrityunjaya Mantra 108 times",
                "yantra": "Shiva Yantra with bel leaves"
            },
            
            ThreatType.ENVIRONMENTAL: {
                "primary_mantra": "Om Shanti Shanti Shanti",
                "description": "Peace mantra for toxic environments",
                "practice": "Continuous recitation in negative spaces",
                "yantra": "Peace Yantra with white flowers"
            },
            
            ThreatType.KARMIC_DEBTS: {
                "primary_mantra": "Om Namo Narayanaya",
                "description": "Vishnu's grace for karmic clearing",
                "practice": "Daily with offering to Vishnu",
                "yantra": "Vishnu Yantra in gold or yellow"
            }
        }
    
    def _initialize_emergency_responses(self) -> Dict[str, Dict[str, Any]]:
        """Initialize emergency protection responses"""
        return {
            "psychic_attack": {
                "immediate": [
                    "Visualize mirror shield reflecting attack back",
                    "Chant 'Om Hram Hrim Hrum Hanamate Namaha'",
                    "Call upon your protective deity strongly"
                ],
                "followup": [
                    "Perform cleansing ritual with salt water",
                    "Strengthen protection with additional mantras",
                    "Seek guidance from spiritual teacher"
                ]
            },
            
            "entity_presence": {
                "immediate": [
                    "Assert your divine nature: 'I am child of Divine Light'",
                    "Visualize brilliant white light filling space",
                    "Command entity to leave in name of your deity"
                ],
                "followup": [
                    "Cleanse space with sage or camphor",
                    "Establish stronger protective boundaries",
                    "Consider professional spiritual help if needed"
                ]
            },
            
            "negative_space": {
                "immediate": [
                    "Mentally create protective bubble around yourself",
                    "Breathe protective light into your aura",
                    "Mentally offer prayers for peace in space"
                ],
                "followup": [
                    "Leave space as soon as appropriate",
                    "Cleanse your energy field thoroughly",
                    "Strengthen protection before returning"
                ]
            }
        }
    
    def assess_raksha_level(self, query: str, 
                          user_context: Optional[Dict[str, Any]] = None) -> RakshaLevel:
        """Assess user's current protection awareness level"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for guardian level indicators
        if any(word in query_lower for word in ["protecting others", 
                                               "cosmic protection", "divine guardian"]):
            return RakshaLevel.GUARDIAN
        
        # Check for protected level indicators
        if any(word in query_lower for word in ["strong protection", 
                                               "advanced protection", "protection master"]):
            return RakshaLevel.PROTECTED
        
        # Check for practicing level indicators
        if any(word in query_lower for word in ["protection practice", 
                                               "regular protection", "teaching protection"]):
            return RakshaLevel.PRACTICING
        
        # Check for learning level indicators
        if any(word in query_lower for word in ["learning protection", 
                                               "protection methods", "spiritual security"]):
            return RakshaLevel.LEARNING
        
        # Check for seeking level indicators
        if any(word in query_lower for word in ["need protection", 
                                               "spiritual attack", "negative energy"]):
            return RakshaLevel.SEEKING
        
        # Default to vulnerable
        return RakshaLevel.VULNERABLE
    
    def get_scriptural_wisdom(self, level: RakshaLevel) -> str:
        """Get scriptural wisdom appropriate to protection level"""
        wisdom_map = {
            RakshaLevel.VULNERABLE: "Hanuman Chalisa: 'Ram duare tum rakhvare, hokar na aagya binu paisare' - At Ram's door you are the guardian, none can enter without your permission.",
            RakshaLevel.SEEKING: "Devi Mahatmya: 'Sarva mangala mangalye shive sarvartha sadhike, sharanye tryambake gauri narayani namostute' - To the auspicious of all auspicious, we bow to the Divine Mother.",
            RakshaLevel.LEARNING: "Vishnu Sahasranama: 'Rakshita rakshitah rakshana rakshakah' - He is the protector, the protected, the protection, and the one who protects.",
            RakshaLevel.PRACTICING: "Shiva Kavach: 'Namaste astu bhagavan vishveshvaraya mahadevaya trayambakaya' - Salutations to the Lord, the ruler of the universe, the great God with three eyes.",
            RakshaLevel.PROTECTED: "Durga Saptashati: 'Ya devi sarva bhutesu shakti rupena samsthita' - The Goddess who resides in all beings in the form of power.",
            RakshaLevel.GUARDIAN: "Bhagavad Gita 9.22: 'Ananyas cintayanto mam ye janah paryupasate tesham nityabhiyuktanam yoga kshemam vahamy aham' - For those who worship Me with devotion, I provide what they lack and preserve what they have."
        }
        return wisdom_map.get(level, "Gayatri Mantra: 'Om bhur bhuvah swah tat savitur varenyam bhargo devasya dhimahi dhiyo yo nah prachodayat' - Divine light, guide and protect our consciousness.")
    
    async def process_raksha_query(self, query: str, 
                                 user_context: Optional[Dict[str, Any]] = None) -> RakshaResponse:
        """Process protection-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess protection level
            level = self.assess_raksha_level(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            scriptural_wisdom = self.get_scriptural_wisdom(level)
            
            return RakshaResponse(
                raksha_level=level.value,
                protection_guidance=guidance.primary_teaching,
                threat_recognition=guidance.threat_recognition,
                protection_methods=guidance.protection_methods,
                daily_practices=guidance.daily_practices,
                emergency_protocols=guidance.emergency_protocols,
                strengthening_rituals=guidance.strengthening_rituals,
                blessing_practices=guidance.blessing_practices,
                scriptural_wisdom=scriptural_wisdom
            )
            
        except Exception as e:
            logger.error(f"Error processing raksha query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> RakshaResponse:
        """Create fallback response when processing fails"""
        return RakshaResponse(
            raksha_level="vulnerable",
            protection_guidance="You are never truly defenseless. Divine protection is your birthright. Begin with simple practices to strengthen your spiritual boundaries and call upon higher help.",
            threat_recognition=[
                "Feeling drained after being around certain people",
                "Sudden mood changes in specific locations",
                "Recurring nightmares or disturbed sleep",
                "Unexplained anxiety or fear without cause"
            ],
            protection_methods=[
                "Daily recitation of protective mantras",
                "Visualization of white light surrounding you",
                "Carrying protective symbols or talismans",
                "Regular cleansing baths with salt or sacred herbs"
            ],
            daily_practices=[
                "Morning protection prayer upon waking",
                "Visualize golden shield around your aura",
                "Chant 'Om Gam Ganapataye Namaha' for obstacles",
                "End day with gratitude to protective forces"
            ],
            emergency_protocols=[
                "Immediately chant 'Hanuman Chalisa' or 'Durga Mantra'",
                "Visualize Hanuman or Durga Ma protecting you",
                "Call upon your chosen deity for immediate help",
                "Remove yourself from threatening situation"
            ],
            strengthening_rituals=[
                "Weekly deep cleansing with turmeric bath",
                "Light ghee lamp while chanting protection mantras",
                "Create sacred space with rangoli or kolam",
                "Offer flowers to protective deities"
            ],
            blessing_practices=[
                "Bless your home with sacred water daily",
                "Place tulsi or neem plants for protection",
                "Hang protective symbols at entrances",
                "Share food with others to increase good karma"
            ],
            scriptural_wisdom="Gayatri Mantra: 'Om bhur bhuvah swah tat savitur varenyam bhargo devasya dhimahi dhiyo yo nah prachodayat' - Divine light, guide and protect our consciousness."
        )


# Global instance
_raksha_module = None

def get_raksha_module() -> RakshaModule:
    """Get global Raksha module instance"""
    global _raksha_module
    if _raksha_module is None:
        _raksha_module = RakshaModule()
    return _raksha_module

# Factory function for easy access
def create_raksha_guidance(query: str, 
                         user_context: Optional[Dict[str, Any]] = None) -> RakshaResponse:
    """Factory function to create raksha guidance"""
    import asyncio
    module = get_raksha_module()
    return asyncio.run(module.process_raksha_query(query, user_context))
