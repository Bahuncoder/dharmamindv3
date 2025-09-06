"""
Bhakti Module - The Center of Divine Love and Devotion

This module represents the sacred path of Bhakti Yoga - the yoga of love and devotion.
It guides users in cultivating deep spiritual love, devotion, surrender, and divine connection.
Drawing from the Bhakti Sutras of Narada, the Ramayana, and other devotional traditions.

Key Functions:
- Assessing and nurturing devotional capacity
- Guiding various forms of bhakti practice
- Emotional healing through divine love
- Cultivating surrender and trust
- Building connection with the Divine

Sanskrit: भक्ति (Bhakti) - devotion, love, attachment to the Divine
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging

logger = logging.getLogger(__name__)

class DevotionType(Enum):
    """Types of devotional expressions"""
    SHRAVANA = "shravana"  # Listening to divine stories/names
    KIRTANA = "kirtana"  # Singing/chanting divine names
    SMARANA = "smarana"  # Remembering the Divine
    PADASEVANAM = "padasevanam"  # Service to divine feet/presence
    ARCHANA = "archana"  # Worship and ritual offerings
    VANDANA = "vandana"  # Prostration and reverence
    DASYA = "dasya"  # Servitude to the Divine
    SAKHYA = "sakhya"  # Friendship with the Divine
    ATMANIVEDANA = "atmanivedana"  # Self-surrender

class EmotionalState(Enum):
    """Emotional states in devotional practice"""
    LONGING = "longing"  # Viraha - separation from beloved
    JOY = "joy"  # Ananda - divine bliss
    SURRENDER = "surrender"  # Saranagati - complete letting go
    GRATITUDE = "gratitude"  # Kritajna - thankfulness
    COMPASSION = "compassion"  # Karuna - divine love for all
    REVERENCE = "reverence"  # Shraddha - deep respect
    HUMILITY = "humility"  # Vinaya - egoless devotion
    ECSTASY = "ecstasy"  # Mahabhava - highest devotional state

class DevotionalObstacle(Enum):
    """Common obstacles in bhakti practice"""
    DOUBT = "doubt"  # Lack of faith
    PRIDE = "pride"  # Ego preventing surrender
    ATTACHMENT = "attachment"  # Worldly attachments
    IMPATIENCE = "impatience"  # Wanting quick results
    COMPARISON = "comparison"  # Comparing one's devotion to others
    DRYNESS = "dryness"  # Feeling disconnected
    MECHANICAL = "mechanical"  # Going through motions without feeling
    DISTRACTION = "distraction"  # Mind wandering during practice

@dataclass
class DevotionalPractice:
    """Represents a specific bhakti practice"""
    name: str
    type: DevotionType
    description: str
    steps: List[str]
    benefits: List[str]
    frequency: str
    duration: str
    prerequisites: List[str] = field(default_factory=list)
    variations: List[str] = field(default_factory=list)
    traditional_sources: List[str] = field(default_factory=list)

@dataclass
class BhaktiResponse:
    """Response from Bhakti Module assessment"""
    devotional_state: str
    recommended_practices: List[str]
    emotional_guidance: str
    mantras: List[str]
    deity_connection: str
    obstacles_identified: List[str]
    surrender_guidance: str
    daily_integration: List[str]
    dharmic_wisdom: str

class BhaktiModule:
    """
    The Bhakti Module - Center of Divine Love and Devotion
    
    Governs devotional practices, emotional healing, surrender,
    and the cultivation of divine love in all its forms.
    """
    
    def __init__(self):
        self.module_name = "Bhakti Module"
        self.element = "Love"
        self.color = "Golden Rose"
        self.mantra = "Om Bhakti Namaha"
        self.deity = "Sri Krishna, Divine Mother, Hanuman"
        self.principles = [
            "Prema (Divine Love)",
            "Saranagati (Surrender)", 
            "Shraddha (Faith)",
            "Seva (Devotional Service)",
            "Satsang (Sacred Community)"
        ]
        
        # Nine forms of devotion (Navavidha Bhakti)
        self.devotional_practices = self._initialize_devotional_practices()
        
        # Emotional healing through bhakti
        self.emotional_remedies = self._initialize_emotional_remedies()
        
        # Obstacles and their solutions
        self.obstacle_solutions = self._initialize_obstacle_solutions()
        
        # Deity connections and their qualities
        self.deity_connections = self._initialize_deity_connections()
    
    def _initialize_devotional_practices(self) -> Dict[DevotionType, DevotionalPractice]:
        """Initialize the nine forms of devotional practice"""
        return {
            DevotionType.SHRAVANA: DevotionalPractice(
                name="Shravana - Divine Listening",
                type=DevotionType.SHRAVANA,
                description="Listening to divine stories, names, and teachings with complete attention",
                steps=[
                    "Choose sacred text or divine story",
                    "Create quiet, sacred space for listening", 
                    "Listen with full heart and attention",
                    "Reflect on deeper meanings",
                    "Allow stories to penetrate the heart"
                ],
                benefits=[
                    "Purification of consciousness",
                    "Development of divine qualities",
                    "Emotional healing through sacred stories",
                    "Increased faith and devotion"
                ],
                frequency="Daily",
                duration="15-30 minutes",
                traditional_sources=["Srimad Bhagavatam", "Ramayana", "Divine Name repetition"]
            ),
            DevotionType.KIRTANA: DevotionalPractice(
                name="Kirtana - Divine Singing",
                type=DevotionType.KIRTANA,
                description="Singing or chanting divine names and glories with devotion",
                steps=[
                    "Choose meaningful mantras or bhajans",
                    "Begin with slow, meditative chanting",
                    "Allow heart to open through sound",
                    "Increase intensity as devotion grows",
                    "End in silence to absorb the vibration"
                ],
                benefits=[
                    "Heart opening and emotional release",
                    "Divine presence invocation",
                    "Community bonding in group practice", 
                    "Transcendence of mental limitations"
                ],
                frequency="Daily",
                duration="20-45 minutes",
                variations=["Solo chanting", "Group kirtan", "Silent repetition"]
            ),
            DevotionType.SMARANA: DevotionalPractice(
                name="Smarana - Divine Remembrance",
                type=DevotionType.SMARANA,
                description="Constant remembrance of the Divine in all activities",
                steps=[
                    "Set intention to remember Divine throughout day",
                    "Use breath as reminder of divine presence",
                    "See divine presence in all beings and situations",
                    "Practice gratitude and offering of all actions",
                    "End day with divine appreciation"
                ],
                benefits=[
                    "Constant divine connection",
                    "Transformation of ordinary into sacred",
                    "Reduction of ego-centered thinking",
                    "Increased peace and surrender"
                ],
                frequency="Continuous",
                duration="All day",
                prerequisites=["Basic meditation practice", "Understanding of chosen deity/form"]
            ),
            DevotionType.ATMANIVEDANA: DevotionalPractice(
                name="Atmanivedana - Complete Self-Surrender",
                type=DevotionType.ATMANIVEDANA,
                description="Total surrender of individual will to Divine will",
                steps=[
                    "Recognize limitations of personal control",
                    "Offer entire being to Divine guidance",
                    "Release attachment to outcomes",
                    "Trust completely in Divine wisdom",
                    "Live as instrument of Divine will"
                ],
                benefits=[
                    "Ultimate freedom through surrender",
                    "Peace beyond personal struggles",
                    "Divine grace and protection",
                    "Liberation from ego-suffering"
                ],
                frequency="Continuous state",
                duration="Entire life",
                prerequisites=["Strong foundation in other devotional practices", "Mature spiritual understanding"]
            )
        }
    
    def _initialize_emotional_remedies(self) -> Dict[EmotionalState, Dict[str, Any]]:
        """Initialize emotional healing through bhakti"""
        return {
            EmotionalState.LONGING: {
                "description": "Sacred longing that draws the heart toward Divine",
                "healing_practices": [
                    "Embrace the longing as divine gift",
                    "Use longing to deepen prayer and meditation",
                    "Express longing through devotional songs",
                    "Find community with other seekers"
                ],
                "mantras": ["Om Namah Shivaya", "Hare Krishna", "Ma Ma Ma"],
                "wisdom": "The pain of separation from Divine is the very force that propels us toward reunion"
            },
            EmotionalState.JOY: {
                "description": "Divine joy arising from connection with the Beloved",
                "cultivation_practices": [
                    "Share joy through service to others",
                    "Express gratitude for divine blessings",
                    "Dance and sing in celebration",
                    "Use joy to inspire others' devotion"
                ],
                "mantras": ["Om Anandaya Namaha", "Jai Jai Jai"],
                "wisdom": "Divine joy is the natural state of the soul in union with its source"
            },
            EmotionalState.SURRENDER: {
                "description": "Complete letting go and trusting in Divine will",
                "deepening_practices": [
                    "Practice releasing control in small matters",
                    "Cultivate trust through remembrance of past grace",
                    "Offer all worries and concerns to Divine",
                    "Find peace in not knowing outcomes"
                ],
                "mantras": ["Tat Tvam Asi", "Thy Will Be Done", "Om Sharanam"],
                "wisdom": "In surrender we find the greatest strength - the power of Divine will working through us"
            },
            EmotionalState.GRATITUDE: {
                "description": "Deep appreciation for Divine grace and blessings",
                "expression_practices": [
                    "Daily gratitude meditation and prayer",
                    "Offering food and flowers in thanksgiving",
                    "Serving others as expression of gratitude",
                    "Seeing all experiences as divine gifts"
                ],
                "mantras": ["Dhanyawad", "Om Gratitude Namaha"],
                "wisdom": "Gratitude transforms ordinary moments into sacred communion with the Divine"
            }
        }
    
    def _initialize_obstacle_solutions(self) -> Dict[DevotionalObstacle, Dict[str, Any]]:
        """Initialize solutions for common devotional obstacles"""
        return {
            DevotionalObstacle.DOUBT: {
                "understanding": "Doubt is natural part of spiritual growth, not failure",
                "remedies": [
                    "Study lives of great devotees who overcame doubt",
                    "Start with small, achievable devotional practices",
                    "Seek guidance from experienced practitioners",
                    "Focus on experiences rather than beliefs"
                ],
                "supportive_practices": ["Scriptural study", "Satsang participation", "Prayer for faith"],
                "wisdom": "Doubt cleared by experience is stronger than blind faith"
            },
            DevotionalObstacle.PRIDE: {
                "understanding": "Ego resists surrender and devotion",
                "remedies": [
                    "Practice humility in daily interactions",
                    "Serve others without recognition",
                    "Remember dependence on Divine grace",
                    "Study teachings on ego dissolution"
                ],
                "supportive_practices": ["Seva", "Prostration", "Self-inquiry"],
                "wisdom": "Pride melts in the fire of genuine love for the Divine"
            },
            DevotionalObstacle.DRYNESS: {
                "understanding": "Periods of feeling disconnected are normal",
                "remedies": [
                    "Continue practices even when feeling dry",
                    "Try different forms of devotional expression",
                    "Seek inspiring company of devotees",
                    "Remember past experiences of connection"
                ],
                "supportive_practices": ["Group singing", "Sacred reading", "Nature communion"],
                "wisdom": "Dry periods often precede deeper spiritual breakthroughs"
            }
        }
    
    def _initialize_deity_connections(self) -> Dict[str, Dict[str, Any]]:
        """Initialize connections with different divine forms"""
        return {
            "Krishna": {
                "qualities": ["Divine love", "Joy", "Playfulness", "Wisdom", "Protection"],
                "mantras": ["Hare Krishna Hare Krishna Krishna Krishna Hare Hare", "Om Namo Bhagavate Vasudevaya"],
                "practices": ["Flute meditation", "Gopi bhava", "Leela contemplation"],
                "teachings": ["Bhagavad Gita", "Srimad Bhagavatam"],
                "guidance": "Approach with childlike love, surrender, and trust in divine protection"
            },
            "Divine Mother": {
                "qualities": ["Unconditional love", "Compassion", "Protection", "Nurturing", "Wisdom"],
                "mantras": ["Om Mata Namaha", "Ma Durga", "Jai Ma"],
                "practices": ["Mother devotion", "Protective prayers", "Offering to feminine divine"],
                "teachings": ["Devi Mahatmyam", "Lalita Sahasranama"],
                "guidance": "Approach as loving child seeking Mother's care and guidance"
            },
            "Hanuman": {
                "qualities": ["Devotion", "Strength", "Service", "Courage", "Humility"],
                "mantras": ["Hanuman Chalisa", "Om Hanumate Namaha", "Jai Bajrangbali"],
                "practices": ["Seva", "Strength practices", "Devotional service"],
                "teachings": ["Ramayana", "Hanuman Chalisa"],
                "guidance": "Embody selfless service and unwavering devotion to the Divine"
            }
        }
    
    async def assess_devotional_state(self, situation: str, context: Optional[Dict[str, Any]] = None) -> BhaktiResponse:
        """Assess current devotional state and provide guidance"""
        try:
            if context is None:
                context = {}
            
            # Analyze situation for devotional elements
            devotional_analysis = self._analyze_devotional_need(situation)
            
            # Identify obstacles
            obstacles = self._identify_obstacles(situation)
            
            # Recommend practices
            practices = self._recommend_practices(devotional_analysis, context)
            
            # Generate guidance
            guidance = self._generate_devotional_guidance(devotional_analysis, situation)
            
            return BhaktiResponse(
                devotional_state=devotional_analysis["state"],
                recommended_practices=practices,
                emotional_guidance=guidance,
                mantras=self._get_appropriate_mantras(devotional_analysis),
                deity_connection=self._suggest_deity_connection(devotional_analysis),
                obstacles_identified=obstacles,
                surrender_guidance=self._generate_surrender_guidance(devotional_analysis),
                daily_integration=self._suggest_daily_practices(devotional_analysis),
                dharmic_wisdom=self._get_bhakti_wisdom(devotional_analysis)
            )
            
        except Exception as e:
            logger.error(f"Error assessing devotional state: {e}")
            return self._create_fallback_response()
    
    def _analyze_devotional_need(self, situation: str) -> Dict[str, Any]:
        """Analyze situation for devotional elements"""
        situation_lower = situation.lower()
        
        analysis = {
            "state": "seeking",
            "emotional_tone": "neutral",
            "devotional_aspects": [],
            "recommended_type": DevotionType.SMARANA
        }
        
        # Detect emotional states
        if any(word in situation_lower for word in ["sad", "lonely", "lost", "empty"]):
            analysis["emotional_tone"] = "longing"
            analysis["recommended_type"] = DevotionType.KIRTANA
        elif any(word in situation_lower for word in ["grateful", "blessed", "happy", "joy"]):
            analysis["emotional_tone"] = "gratitude"
            analysis["recommended_type"] = DevotionType.ARCHANA
        elif any(word in situation_lower for word in ["confused", "difficult", "struggle"]):
            analysis["emotional_tone"] = "need_guidance"
            analysis["recommended_type"] = DevotionType.SHRAVANA
        
        # Detect devotional aspects
        if any(word in situation_lower for word in ["prayer", "devotion", "god", "divine"]):
            analysis["devotional_aspects"].append("already_devotional")
        
        return analysis
    
    def _identify_obstacles(self, situation: str) -> List[str]:
        """Identify devotional obstacles in the situation"""
        obstacles = []
        situation_lower = situation.lower()
        
        if any(word in situation_lower for word in ["doubt", "unsure", "believe"]):
            obstacles.append("doubt")
        if any(word in situation_lower for word in ["proud", "ego", "better than"]):
            obstacles.append("pride")
        if any(word in situation_lower for word in ["distracted", "busy", "no time"]):
            obstacles.append("distraction")
        if any(word in situation_lower for word in ["dry", "empty", "disconnected"]):
            obstacles.append("dryness")
        
        return obstacles
    
    def _recommend_practices(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Recommend specific devotional practices"""
        recommended_type = analysis["recommended_type"]
        practice = self.devotional_practices.get(recommended_type)
        
        if practice:
            return [practice.name] + practice.steps[:3]
        
        return ["Divine name repetition", "Gratitude prayer", "Heart-centered meditation"]
    
    def _generate_devotional_guidance(self, analysis: Dict[str, Any], situation: str) -> str:
        """Generate personalized devotional guidance"""
        base_guidance = "💖 The path of bhakti teaches us that love is both the means and the end. "
        
        emotional_tone = analysis.get("emotional_tone", "neutral")
        
        if emotional_tone == "longing":
            guidance = base_guidance + "Your longing is a sacred gift - it draws you closer to the Divine. Embrace this feeling as a doorway to deeper connection."
        elif emotional_tone == "gratitude":
            guidance = base_guidance + "Your gratitude opens the heart to receive even more divine blessings. Let this appreciation deepen your devotional practice."
        elif emotional_tone == "need_guidance":
            guidance = base_guidance + "In times of confusion, turn to the Divine with trust. Prayer and devotional practice will illuminate your path."
        else:
            guidance = base_guidance + "Every moment is an opportunity to remember the Divine presence within and around you."
        
        return guidance
    
    def _get_appropriate_mantras(self, analysis: Dict[str, Any]) -> List[str]:
        """Get mantras appropriate for the situation"""
        emotional_tone = analysis.get("emotional_tone", "neutral")
        
        mantra_map = {
            "longing": ["Om Namah Shivaya", "Hare Krishna", "Om Ma"],
            "gratitude": ["Om Ganesh Namaha", "Dhanyawad", "Om Gratitude"],
            "need_guidance": ["Om Gam Ganapataye Namaha", "Om Guru Om"],
            "neutral": ["Om", "So Hum", "Om Shanti"]
        }
        
        return mantra_map.get(emotional_tone, ["Om", "Om Namah Shivaya"])
    
    def _suggest_deity_connection(self, analysis: Dict[str, Any]) -> str:
        """Suggest appropriate deity connection"""
        emotional_tone = analysis.get("emotional_tone", "neutral")
        
        if emotional_tone == "longing":
            return "Krishna - for divine love and joy"
        elif emotional_tone == "gratitude":
            return "Divine Mother - for nurturing and blessings"
        elif emotional_tone == "need_guidance":
            return "Ganesha - for removing obstacles and clarity"
        else:
            return "Choose the divine form that naturally attracts your heart"
    
    def _generate_surrender_guidance(self, analysis: Dict[str, Any]) -> str:
        """Generate guidance on surrender"""
        return "Begin with small acts of surrender - offering your meal, your work, your worries to the Divine. Gradually, this attitude will transform your entire life into a devotional offering."
    
    def _suggest_daily_practices(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest daily devotional practices"""
        return [
            "Morning prayer or gratitude",
            "Offering your food before eating",
            "Brief evening reflection on the day's blessings",
            "Repeating a chosen mantra during routine activities",
            "Seeing the Divine in all beings you encounter"
        ]
    
    def _get_bhakti_wisdom(self, analysis: Dict[str, Any]) -> str:
        """Get relevant bhakti wisdom"""
        wisdom_quotes = [
            "Love is the bridge between you and everything - Rumi",
            "The path of love is not a tedious path - go to love, love comes quickly - Rumi",
            "When you make the two hearts into one, you will be made into the image of God - Gospel of Thomas",
            "Bhakti is a path for tender hearts - Ramakrishna",
            "God loves you not because you are worthy, but because God is Love - Divine Mother"
        ]
        
        import random
        return random.choice(wisdom_quotes)
    
    def _create_fallback_response(self) -> BhaktiResponse:
        """Create fallback response when processing fails"""
        return BhaktiResponse(
            devotional_state="seeking",
            recommended_practices=["Simple prayer", "Gratitude practice", "Divine name repetition"],
            emotional_guidance="💖 The heart that seeks the Divine is already on the sacred path of love.",
            mantras=["Om", "Om Namah Shivaya"],
            deity_connection="Follow your heart to the divine form that attracts you",
            obstacles_identified=[],
            surrender_guidance="Start with offering small things to the Divine with love",
            daily_integration=["Morning gratitude", "Offering food", "Evening prayer"],
            dharmic_wisdom="Love is the essence of all spiritual practice"
        )
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get current status of the Bhakti Module"""
        return {
            "name": self.module_name,
            "state": "active",
            "element": self.element,
            "color": self.color,
            "mantra": self.mantra,
            "governing_deity": self.deity,
            "core_principles": self.principles
        }

# Global instance
_bhakti_module = None

def get_bhakti_module() -> BhaktiModule:
    """Get global Bhakti module instance"""
    global _bhakti_module
    if _bhakti_module is None:
        _bhakti_module = BhaktiModule()
    return _bhakti_module
