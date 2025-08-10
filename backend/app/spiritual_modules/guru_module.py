"""
🟡 Guru Module - Spiritual Teacher and Inner Wisdom
Complete Guru-Tattva system based on authentic dharmic traditions
Guides the development of inner teacher, discipleship, and spiritual guidance
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class GuruType(Enum):
    """Types of Guru according to tradition"""
    DIKSHA_GURU = "diksha_guru"  # Initiating guru
    SIKSHA_GURU = "siksha_guru"  # Instructing guru
    VARTMA_PRADARSHAK = "vartma_pradarshak"  # Path-showing guru
    ANTARYAMI_GURU = "antaryami_guru"  # Inner guru (Paramatma)
    CHAITYA_GURU = "chaitya_guru"  # Guru in the heart

class DiscipleLevel(Enum):
    """Levels of discipleship development"""
    MUMUKSHU = "mumukshu"  # Seeker desiring liberation
    SADHAKA = "sadhaka"  # Practicing disciple
    SIDDHA = "siddha"  # Accomplished practitioner
    GURU = "guru"  # One who has become a teacher

class GuruPrinciple(Enum):
    """Essential Guru principles"""
    SHRADDHA = "shraddha"  # Faith and surrender
    SEVA = "seva"  # Selfless service
    SADHANA = "sadhana"  # Spiritual practice
    SURRENDER = "surrender"  # Complete submission
    GRATITUDE = "gratitude"  # Deep appreciation
    HUMILITY = "humility"  # Ego dissolution
    DISCRIMINATION = "discrimination"  # Spiritual discernment
    DEVOTION = "devotion"  # Pure love for guru

@dataclass
class GuruTeaching:
    """Structure for guru teachings"""
    principle: GuruPrinciple
    sanskrit_term: str
    teaching: str
    practice: str
    scripture_ref: str
    daily_application: List[str]
    obstacles: List[str]
    transcendence: str

@dataclass
class GuruGuidance:
    """Comprehensive guru guidance"""
    disciple_level: DiscipleLevel
    guru_type: GuruType
    main_teaching: str
    spiritual_practices: List[str]
    service_opportunities: List[str]
    study_materials: List[str]
    daily_sadhana: List[str]
    obstacles_to_overcome: Dict[str, str]
    progress_indicators: List[str]

class GuruInsight(BaseModel):
    """Insight from guru module"""
    teaching: str = Field(description="Core teaching")
    practice: str = Field(description="Recommended practice")
    scripture: str = Field(description="Scriptural reference")
    obstacle: str = Field(description="Potential obstacle")
    transcendence: str = Field(description="How to transcend")

class GuruResponse(BaseModel):
    """Response from Guru module"""
    disciple_level: str = Field(description="Current level of discipleship")
    guru_type_needed: str = Field(description="Type of guru guidance needed")
    main_teaching: str = Field(description="Primary teaching for current situation")
    spiritual_practices: List[str] = Field(description="Recommended spiritual practices")
    service_path: List[str] = Field(description="Ways to serve and grow")
    daily_sadhana: List[str] = Field(description="Daily spiritual discipline")
    scriptural_wisdom: str = Field(description="Relevant scriptural guidance")
    practical_steps: List[str] = Field(description="Practical steps for implementation")
    obstacles_solutions: Dict[str, str] = Field(description="Common obstacles and solutions")
    progress_signs: List[str] = Field(description="Signs of spiritual progress")
    surrender_guidance: str = Field(description="Guidance on surrender and humility")

class GuruModule:
    """
    🟡 Guru Module - The Sacred Teacher-Disciple Relationship
    
    Based on authentic Guru-Tattva from Vedic tradition
    Guides the development of proper guru-disciple relationship
    Supports both finding external guru and developing inner guru
    """
    
    def __init__(self):
        self.name = "Guru"
        self.color = "🟡"
        self.element = "Knowledge and Compassion"
        self.teachings = self._initialize_guru_teachings()
        self.guidance_levels = self._initialize_guidance_levels()
        
    def _initialize_guru_teachings(self) -> Dict[GuruPrinciple, GuruTeaching]:
        """Initialize authentic guru teachings"""
        return {
            GuruPrinciple.SHRADDHA: GuruTeaching(
                principle=GuruPrinciple.SHRADDHA,
                sanskrit_term="श्रद्धा (Shraddha)",
                teaching="Faith in the guru and the path is the foundation of spiritual growth. Without shraddha, no progress is possible.",
                practice="Daily meditation on guru's teachings, contemplation of guru's qualities, developing unshakeable faith",
                scripture_ref="Bhagavad Gita 4.39: 'One who has faith, who is devoted to spiritual practice, and has controlled the senses, attains knowledge'",
                daily_application=[
                    "Begin each day with gratitude to guru",
                    "Study guru's teachings with complete faith",
                    "Apply teachings without doubt or modification",
                    "Surrender personal preferences to guru's guidance"
                ],
                obstacles=["Doubt", "Intellectual pride", "Past experiences", "Social conditioning"],
                transcendence="Realize that guru is not the body but the divine principle manifesting through the teacher"
            ),
            
            GuruPrinciple.SEVA: GuruTeaching(
                principle=GuruPrinciple.SEVA,
                sanskrit_term="सेवा (Seva)",
                teaching="Selfless service to guru purifies the heart and develops humility. Through seva, ego dissolves naturally.",
                practice="Physical service, emotional support, spiritual service to guru and guru's mission",
                scripture_ref="Srimad Bhagavatam: 'One should worship the guru by serving him with body, mind, and words'",
                daily_application=[
                    "Offer physical service when possible",
                    "Support guru's teachings and mission",
                    "Serve fellow disciples with guru's spirit",
                    "Maintain guru's place with reverence"
                ],
                obstacles=["Ego", "Expectation of reward", "Selective service", "Pride in service"],
                transcendence="See guru in everyone and serve all as manifestations of guru principle"
            ),
            
            GuruPrinciple.SURRENDER: GuruTeaching(
                principle=GuruPrinciple.SURRENDER,
                sanskrit_term="शरणागति (Sharanagati)",
                teaching="Complete surrender to guru is the direct path to liberation. When 'I' dies, the Divine lives.",
                practice="Offering all actions, thoughts, and desires to guru; complete dependency on guru's will",
                scripture_ref="Bhagavad Gita 18.66: 'Abandon all varieties of dharma and surrender unto Me alone'",
                daily_application=[
                    "Begin each action with guru's permission",
                    "Offer all results to guru",
                    "Accept guru's will in all circumstances",
                    "Dissolve personal agenda in guru's mission"
                ],
                obstacles=["Fear of losing identity", "Control addiction", "Intellectual resistance", "Social pressure"],
                transcendence="Realize that true identity is not ego but the eternal soul connected to guru-tattva"
            ),
            
            GuruPrinciple.DEVOTION: GuruTeaching(
                principle=GuruPrinciple.DEVOTION,
                sanskrit_term="भक्ति (Bhakti)",
                teaching="Pure devotion to guru awakens divine love and accelerates spiritual evolution beyond all other methods.",
                practice="Constant remembrance of guru, emotional devotion, seeing guru as divine incarnation",
                scripture_ref="Guru Gita: 'Guru is Brahma, Guru is Vishnu, Guru is Maheshwara. Guru is the Supreme Absolute Itself'",
                daily_application=[
                    "Chant guru's name throughout the day",
                    "Meditate on guru's form and qualities",
                    "Feel guru's presence in all activities",
                    "Express love and gratitude constantly"
                ],
                obstacles=["Emotional dryness", "Intellectual barriers", "Comparing gurus", "Doubt in guru's divinity"],
                transcendence="Experience guru as the very Self, beyond all duality and separation"
            ),
            
            GuruPrinciple.DISCRIMINATION: GuruTeaching(
                principle=GuruPrinciple.DISCRIMINATION,
                sanskrit_term="विवेक (Viveka)",
                teaching="Spiritual discrimination helps distinguish between authentic guru and false teachers, truth and illusion.",
                practice="Study scriptures, observe guru's behavior, test teachings against tradition, inner discernment",
                scripture_ref="Mundaka Upanishad: 'Approach that guru who is learned in scriptures and established in Brahman'",
                daily_application=[
                    "Study authentic spiritual texts daily",
                    "Observe consistency between teaching and living",
                    "Seek signs of ego-dissolution in guru",
                    "Trust inner guidance while maintaining humility"
                ],
                obstacles=["Blind faith", "Intellectual pride", "Jumping between teachers", "Cynicism"],
                transcendence="Develop intuitive wisdom that recognizes divine authenticity beyond external appearances"
            ),
            
            GuruPrinciple.HUMILITY: GuruTeaching(
                principle=GuruPrinciple.HUMILITY,
                sanskrit_term="विनम्रता (Vinamrata)",
                teaching="Humility is the gateway to receiving guru's grace. The empty cup can be filled; the proud heart remains closed.",
                practice="Acknowledging ignorance, bowing to guru, accepting corrections, serving others humbly",
                scripture_ref="Bhagavad Gita 13.8: 'Humility, pridelessness... these are declared to be knowledge'",
                daily_application=[
                    "Prostrate before guru with complete humility",
                    "Accept all feedback as grace",
                    "Serve others without expecting recognition",
                    "Acknowledge dependence on guru's mercy"
                ],
                obstacles=["Spiritual pride", "Comparison with others", "Defensive ego", "False humility"],
                transcendence="Natural humility arising from realization of one's eternal position as servant of guru"
            ),
            
            GuruPrinciple.GRATITUDE: GuruTeaching(
                principle=GuruPrinciple.GRATITUDE,
                sanskrit_term="कृतज्ञता (Kritajna)",
                teaching="Gratitude to guru opens the heart to receive unlimited grace and transforms every moment into worship.",
                practice="Daily thanksgiving, expressing appreciation, sharing guru's glory, living as testament to guru's mercy",
                scripture_ref="From various Puranas: 'One who is grateful to guru receives unlimited blessings'",
                daily_application=[
                    "Begin and end day with gratitude to guru",
                    "Share stories of guru's compassion",
                    "Express thanks for both pleasant and difficult experiences",
                    "Live as evidence of guru's transforming power"
                ],
                obstacles=["Taking blessings for granted", "Focusing on what's lacking", "Complaining habit", "Spiritual materialism"],
                transcendence="Constant awareness of guru's infinite mercy in every aspect of existence"
            ),
            
            GuruPrinciple.SADHANA: GuruTeaching(
                principle=GuruPrinciple.SADHANA,
                sanskrit_term="साधना (Sadhana)",
                teaching="Consistent spiritual practice according to guru's instructions is the vehicle for transformation.",
                practice="Daily meditation, japa, study, service according to guru's specific guidance",
                scripture_ref="Bhagavad Gita 6.40: 'One who does spiritual practice is never overcome by evil'",
                daily_application=[
                    "Follow guru's prescribed daily practice exactly",
                    "Maintain consistency regardless of external circumstances",
                    "Gradually increase intensity with guru's permission",
                    "Offer all practice to guru's lotus feet"
                ],
                obstacles=["Inconsistency", "Modification of instructions", "Laziness", "Spiritual pride in practice"],
                transcendence="Practice becomes spontaneous expression of love for guru, beyond obligation or effort"
            )
        }
    
    def _initialize_guidance_levels(self) -> Dict[DiscipleLevel, GuruGuidance]:
        """Initialize guidance for different levels of discipleship"""
        return {
            DiscipleLevel.MUMUKSHU: GuruGuidance(
                disciple_level=DiscipleLevel.MUMUKSHU,
                guru_type=GuruType.VARTMA_PRADARSHAK,
                main_teaching="Develop sincere desire for liberation and seek authentic spiritual guidance",
                spiritual_practices=[
                    "Daily prayer for guidance in finding true guru",
                    "Study of authentic spiritual texts",
                    "Satsang with advanced practitioners",
                    "Self-inquiry into the nature of suffering",
                    "Cultivation of basic spiritual qualities"
                ],
                service_opportunities=[
                    "Serve in spiritual communities",
                    "Help other seekers find resources",
                    "Maintain spiritual spaces with devotion",
                    "Support genuine spiritual teachers"
                ],
                study_materials=[
                    "Bhagavad Gita",
                    "Guru Gita",
                    "Biographies of realized masters",
                    "Basic Vedanta texts"
                ],
                daily_sadhana=[
                    "Morning prayer for guru's grace",
                    "Study of spiritual texts (30 min)",
                    "Meditation or contemplation (20 min)",
                    "Evening gratitude practice"
                ],
                obstacles_to_overcome={
                    "Impatience in finding guru": "Trust divine timing and prepare yourself",
                    "Doubt about need for guru": "Study experiences of realized souls",
                    "Attachment to independence": "Recognize limitations of individual effort",
                    "Fear of surrender": "Start with small acts of trust and devotion"
                },
                progress_indicators=[
                    "Increasing desire for spiritual growth",
                    "Natural detachment from material pursuits",
                    "Attraction to spiritual company",
                    "Intuitive recognition of authentic teaching"
                ]
            ),
            
            DiscipleLevel.SADHAKA: GuruGuidance(
                disciple_level=DiscipleLevel.SADHAKA,
                guru_type=GuruType.SIKSHA_GURU,
                main_teaching="Develop deep relationship with guru through surrender, service, and sincere practice",
                spiritual_practices=[
                    "Daily meditation as taught by guru",
                    "Japa (repetition) of guru-given mantra",
                    "Study of texts recommended by guru",
                    "Regular satsang and spiritual discussion",
                    "Observance of spiritual disciplines"
                ],
                service_opportunities=[
                    "Direct service to guru when possible",
                    "Support guru's mission and teachings",
                    "Serve fellow disciples with humility",
                    "Share guru's teachings with sincere seekers"
                ],
                study_materials=[
                    "Texts specifically recommended by guru",
                    "Commentaries on primary scriptures",
                    "Biographies of guru's lineage",
                    "Works on guru-disciple relationship"
                ],
                daily_sadhana=[
                    "Early morning meditation (45-60 min)",
                    "Japa throughout the day",
                    "Study of guru's teachings (45 min)",
                    "Selfless service as worship",
                    "Evening surrender and gratitude"
                ],
                obstacles_to_overcome={
                    "Ego resistance to guru's guidance": "Practice immediate obedience to small instructions",
                    "Doubt during difficult times": "Remember guru's infinite compassion and wisdom",
                    "Comparison with other disciples": "Focus on your own relationship with guru",
                    "Attachment to spiritual experiences": "Offer all experiences to guru"
                },
                progress_indicators=[
                    "Natural love and devotion for guru",
                    "Ease in following guru's instructions",
                    "Increasing peace and clarity",
                    "Spontaneous service attitude"
                ]
            ),
            
            DiscipleLevel.SIDDHA: GuruGuidance(
                disciple_level=DiscipleLevel.SIDDHA,
                guru_type=GuruType.ANTARYAMI_GURU,
                main_teaching="Embody guru's teachings completely and prepare to guide others",
                spiritual_practices=[
                    "Constant awareness of guru's presence",
                    "Living as instrument of guru's will",
                    "Deep contemplation on guru-tattva",
                    "Spontaneous service and teaching",
                    "Integration of all activities as worship"
                ],
                service_opportunities=[
                    "Guide junior disciples with compassion",
                    "Represent guru's teachings authentically",
                    "Establish spiritual programs and centers",
                    "Write and speak on spiritual topics"
                ],
                study_materials=[
                    "Advanced Vedantic texts",
                    "Tantric and mystical literature",
                    "Comparative spiritual traditions",
                    "Contemporary spiritual challenges"
                ],
                daily_sadhana=[
                    "Constant remembrance of guru",
                    "Spontaneous meditation and prayer",
                    "Teaching and guiding others",
                    "Living as example of guru's grace"
                ],
                obstacles_to_overcome={
                    "Spiritual pride in accomplishments": "Remember all comes from guru's grace",
                    "Attachment to being 'advanced'": "Maintain beginner's mind and humility",
                    "Temptation to modify teachings": "Preserve tradition while adapting expression",
                    "Isolation from guru": "Maintain inner connection despite external distance"
                },
                progress_indicators=[
                    "Natural wisdom and compassion",
                    "Ability to guide others effectively",
                    "Unshakeable faith in all circumstances",
                    "Recognition by authentic teachers"
                ]
            ),
            
            DiscipleLevel.GURU: GuruGuidance(
                disciple_level=DiscipleLevel.GURU,
                guru_type=GuruType.DIKSHA_GURU,
                main_teaching="Become a transparent vessel for the divine teachings and guru's grace",
                spiritual_practices=[
                    "Complete identification with guru principle",
                    "Constant service to seekers and disciples",
                    "Maintenance of spiritual lineage",
                    "Continuous self-surrender to divine will",
                    "Living as embodiment of teachings"
                ],
                service_opportunities=[
                    "Initiate sincere seekers into spiritual life",
                    "Establish and guide spiritual communities",
                    "Preserve and transmit authentic teachings",
                    "Serve as bridge between human and divine"
                ],
                study_materials=[
                    "All sacred texts of the tradition",
                    "Responsibilities of spiritual teachers",
                    "Contemporary applications of ancient wisdom",
                    "Direct revelation through meditation"
                ],
                daily_sadhana=[
                    "Constant awareness of responsibility",
                    "Continuous prayer for disciples' welfare",
                    "Teaching through personal example",
                    "Sacrifice of personal preferences for divine will"
                ],
                obstacles_to_overcome={
                    "Spiritual materialism and ego": "Constant vigilance and self-surrender",
                    "Attachment to followers": "See all as belonging to the Divine",
                    "Compromise of teachings for popularity": "Maintain integrity regardless of consequences",
                    "Burden of responsibility": "Trust in divine support and guidance"
                },
                progress_indicators=[
                    "Complete selflessness in teaching",
                    "Natural wisdom flowing through words and actions",
                    "Transformation visible in disciples",
                    "Recognition by the lineage of authentic teachers"
                ]
            )
        }
    
    def assess_disciple_level(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> DiscipleLevel:
        """Assess user's current level of discipleship"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for guru/teacher indicators
        if any(word in query_lower for word in ["teach", "guide others", "guru responsibilities", "disciples"]):
            return DiscipleLevel.GURU
        
        # Check for advanced practitioner indicators
        if any(word in query_lower for word in ["advanced practice", "embodying", "living the teachings", "constant awareness"]):
            return DiscipleLevel.SIDDHA
        
        # Check for established practitioner indicators
        if any(word in query_lower for word in ["sadhana", "regular practice", "surrender to guru", "spiritual discipline"]):
            return DiscipleLevel.SADHAKA
        
        # Default to seeker level
        return DiscipleLevel.MUMUKSHU
    
    def determine_guru_type_needed(self, disciple_level: DiscipleLevel, query: str) -> GuruType:
        """Determine what type of guru guidance is needed"""
        if "inner guidance" in query.lower() or "heart" in query.lower():
            return GuruType.CHAITYA_GURU
        
        # Return appropriate guru type based on disciple level
        guidance = self.guidance_levels.get(disciple_level)
        return guidance.guru_type if guidance else GuruType.VARTMA_PRADARSHAK
    
    def get_practical_steps(self, disciple_level: DiscipleLevel, query: str) -> List[str]:
        """Get practical steps based on level and query"""
        guidance = self.guidance_levels.get(disciple_level)
        if not guidance:
            return [
                "Begin with sincere prayer for spiritual guidance",
                "Study authentic spiritual texts daily",
                "Seek association with genuine spiritual practitioners",
                "Develop basic spiritual qualities like humility and service"
            ]
        
        base_steps = [
            f"Practice {len(guidance.daily_sadhana)} daily spiritual disciplines",
            "Engage in selfless service opportunities",
            "Study recommended spiritual materials",
            "Maintain regular spiritual schedule"
        ]
        
        # Add specific steps based on query content
        if "surrender" in query.lower():
            base_steps.extend([
                "Practice offering each action to guru before beginning",
                "Accept all results as guru's grace",
                "Gradually release personal preferences and control"
            ])
        
        if "service" in query.lower():
            base_steps.extend([
                "Look for ways to serve guru's mission daily",
                "Serve fellow seekers with humility and compassion",
                "Offer physical, mental, and spiritual service"
            ])
        
        return base_steps[:8]  # Limit to manageable number
    
    def identify_obstacles_and_solutions(self, disciple_level: DiscipleLevel, query: str) -> Dict[str, str]:
        """Identify obstacles and provide solutions"""
        guidance = self.guidance_levels.get(disciple_level)
        base_obstacles = guidance.obstacles_to_overcome if guidance else {}
        
        # Add query-specific obstacles
        query_obstacles = {}
        if "doubt" in query.lower():
            query_obstacles["Spiritual doubt"] = "Study lives of realized souls, maintain regular practice despite feelings"
        if "ego" in query.lower():
            query_obstacles["Spiritual ego"] = "Practice selfless service, remember all progress comes from guru's grace"
        if "surrender" in query.lower():
            query_obstacles["Fear of surrender"] = "Start with small acts of trust, study benefits of surrender in scriptures"
        
        # Combine and limit
        all_obstacles = {**base_obstacles, **query_obstacles}
        return dict(list(all_obstacles.items())[:6])  # Limit to 6 most relevant
    
    def get_progress_indicators(self, disciple_level: DiscipleLevel) -> List[str]:
        """Get signs of spiritual progress for current level"""
        guidance = self.guidance_levels.get(disciple_level)
        return guidance.progress_indicators if guidance else [
            "Increasing desire for spiritual growth",
            "Natural detachment from material concerns",
            "Growing compassion for all beings",
            "Peace in difficult circumstances"
        ]
    
    def get_surrender_guidance(self, query: str, disciple_level: DiscipleLevel) -> str:
        """Provide specific guidance on surrender"""
        if disciple_level == DiscipleLevel.MUMUKSHU:
            return "Begin surrender by offering small daily actions to the Divine. Trust that grace will guide you to the right teacher."
        elif disciple_level == DiscipleLevel.SADHAKA:
            return "Deepen surrender by following guru's instructions exactly, even when the mind resists. See guru's will as divine will."
        elif disciple_level == DiscipleLevel.SIDDHA:
            return "Live in constant surrender, becoming a transparent instrument for divine work through guru's grace."
        else:  # GURU level
            return "Complete surrender means becoming one with the divine will, teaching through divine inspiration rather than personal knowledge."
    
    async def process_guru_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> GuruResponse:
        """Process guru-related query and provide comprehensive guidance"""
        try:
            # Assess current level and needs
            disciple_level = self.assess_disciple_level(query, user_context)
            guru_type_needed = self.determine_guru_type_needed(disciple_level, query)
            guidance = self.guidance_levels.get(disciple_level)
            
            if not guidance:
                return self._create_fallback_response()
            
            # Get specific guidance components
            practical_steps = self.get_practical_steps(disciple_level, query)
            obstacles_solutions = self.identify_obstacles_and_solutions(disciple_level, query)
            progress_signs = self.get_progress_indicators(disciple_level)
            surrender_guidance = self.get_surrender_guidance(query, disciple_level)
            
            # Select relevant scriptural wisdom
            if any(word in query.lower() for word in ["surrender", "devotion", "bhakti"]):
                scriptural_wisdom = "Bhagavad Gita 18.66: 'Abandon all dharmas and surrender unto Me alone. I shall deliver you from all sinful reactions.'"
            elif any(word in query.lower() for word in ["service", "seva"]):
                scriptural_wisdom = "Srimad Bhagavatam: 'One should worship the guru by serving with body, mind, and words.'"
            else:
                scriptural_wisdom = "Guru Gita: 'Guru is Brahma, Guru is Vishnu, Guru is Maheshwara. Guru is the Supreme Absolute Itself.'"
            
            return GuruResponse(
                disciple_level=disciple_level.value,
                guru_type_needed=guru_type_needed.value,
                main_teaching=guidance.main_teaching,
                spiritual_practices=guidance.spiritual_practices,
                service_path=guidance.service_opportunities,
                daily_sadhana=guidance.daily_sadhana,
                scriptural_wisdom=scriptural_wisdom,
                practical_steps=practical_steps,
                obstacles_solutions=obstacles_solutions,
                progress_signs=progress_signs,
                surrender_guidance=surrender_guidance
            )
            
        except Exception as e:
            logger.error(f"Error processing guru query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> GuruResponse:
        """Create fallback response when processing fails"""
        return GuruResponse(
            disciple_level="mumukshu",
            guru_type_needed="vartma_pradarshak",
            main_teaching="The spiritual path begins with sincere seeking and openness to divine guidance through a qualified teacher.",
            spiritual_practices=[
                "Daily prayer for spiritual guidance",
                "Study of authentic spiritual texts",
                "Association with sincere practitioners",
                "Cultivation of humility and service"
            ],
            service_path=[
                "Serve in spiritual communities",
                "Help other seekers find resources",
                "Practice kindness and compassion",
                "Support genuine spiritual teachers"
            ],
            daily_sadhana=[
                "Morning prayer and meditation",
                "Study spiritual texts",
                "Practice selfless service",
                "Evening gratitude and reflection"
            ],
            scriptural_wisdom="Mundaka Upanishad: 'To know That, approach a guru who is learned in scriptures and established in Brahman.'",
            practical_steps=[
                "Develop sincere desire for liberation",
                "Study traditional spiritual texts",
                "Seek association with authentic practitioners",
                "Prepare heart through service and humility"
            ],
            obstacles_solutions={
                "Doubt about need for guru": "Study experiences of realized souls in authentic texts",
                "Fear of surrender": "Begin with small acts of trust and devotion",
                "Spiritual pride": "Practice humility through selfless service",
                "Impatience": "Trust in divine timing while preparing sincerely"
            },
            progress_signs=[
                "Increasing desire for spiritual growth",
                "Natural detachment from material pursuits",
                "Attraction to spiritual company and teachings",
                "Growing compassion and inner peace"
            ],
            surrender_guidance="Begin surrender with simple daily offerings to the Divine, trusting that sincere seeking will attract appropriate guidance."
        )
    
    def get_guru_insight(self, principle: GuruPrinciple) -> Optional[GuruInsight]:
        """Get specific insight about a guru principle"""
        teaching = self.teachings.get(principle)
        if not teaching:
            return None
        
        return GuruInsight(
            teaching=teaching.teaching,
            practice=teaching.practice,
            scripture=teaching.scripture_ref,
            obstacle=teaching.obstacles[0] if teaching.obstacles else "Spiritual resistance",
            transcendence=teaching.transcendence
        )

# Global instance
_guru_module = None

def get_guru_module() -> GuruModule:
    """Get global Guru module instance"""
    global _guru_module
    if _guru_module is None:
        _guru_module = GuruModule()
    return _guru_module

# Factory function for easy access
def create_guru_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> GuruResponse:
    """Factory function to create guru guidance"""
    import asyncio
    module = get_guru_module()
    return asyncio.run(module.process_guru_query(query, user_context))
