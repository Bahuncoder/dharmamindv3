"""
Yoga Module - Complete Eight-Limbed Path System
Provides comprehensive guidance for Ashtanga Yoga practice including all eight limbs,
progressive development stages, obstacle management, and integration into daily life.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import asyncio


class YogaLimb(Enum):
    """The eight limbs of Ashtanga Yoga"""
    YAMA = "yama"           # Ethical restraints
    NIYAMA = "niyama"       # Observances
    ASANA = "asana"         # Physical postures
    PRANAYAMA = "pranayama" # Breath control
    PRATYAHARA = "pratyahara" # Withdrawal of senses
    DHARANA = "dharana"     # Concentration
    DHYANA = "dhyana"       # Meditation
    SAMADHI = "samadhi"     # Union/Absorption


class YogaLevel(Enum):
    """Progressive levels of yoga practice"""
    FOUNDATION = "foundation"
    DEVELOPING = "developing"
    INTEGRATING = "integrating"
    ADVANCED = "advanced"
    MASTER = "master"


class YogaObstacle(Enum):
    """Common obstacles in yoga practice (Antarayas)"""
    SICKNESS = "vyadhi"
    MENTAL_LAZINESS = "styana"
    DOUBT = "samshaya"
    CARELESSNESS = "pramada"
    SLOTH = "alasya"
    WORLDLY_MINDEDNESS = "avirati"
    FALSE_PERCEPTION = "bhranti_darshana"
    NON_ATTAINMENT = "alabdha_bhumikatva"
    INSTABILITY = "anavasthitatva"


class AsanaCategory(Enum):
    """Categories of yoga postures"""
    STANDING = "standing"
    SEATED = "seated"
    SUPINE = "supine"
    PRONE = "prone"
    BACKBENDS = "backbends"
    FORWARD_FOLDS = "forward_folds"
    TWISTS = "twists"
    INVERSIONS = "inversions"
    ARM_BALANCES = "arm_balances"
    RESTORATIVE = "restorative"


@dataclass
class YogaPractice:
    """Represents a specific yoga practice"""
    name: str
    limb: YogaLimb
    category: Optional[AsanaCategory]
    description: str
    steps: List[str]
    benefits: List[str]
    duration: str
    sutra_reference: Optional[str] = None


class YogaChakra:
    """
    The Yoga Chakra - Complete Eight-Limbed Path System
    
    Provides comprehensive guidance for Ashtanga Yoga practice, including:
    - All eight limbs with detailed practices
    - Progressive development stages
    - Obstacle identification and remedies
    - Personalized practice recommendations
    - Integration into daily life
    """
    
    def __init__(self):
        self.chakra_name = "Yoga Chakra"
        self.element = "Integration"
        self.color = "Violet"
        self.mantra = "Om Yogaya Namaha"
        self.deity = "Patanjali, Shiva as Adiyogi"
        self.principles = [
            "Ahimsa (Non-violence)",
            "Satya (Truthfulness)", 
            "Dharana (Concentration)",
            "Dhyana (Meditation)",
            "Samadhi (Union)"
        ]
        
        # Eight limbs practices
        self.ashtanga_practices = self._initialize_ashtanga_practices()
        
        # Obstacle remedies
        self.obstacle_remedies = self._initialize_obstacle_remedies()
        
        # Progressive development path
        self.development_stages = self._initialize_development_stages()
        
        # Asana sequences for different levels
        self.asana_sequences = self._initialize_asana_sequences()
    
    def _initialize_ashtanga_practices(self) -> Dict[YogaLimb, List[YogaPractice]]:
        """Initialize practices for each limb of Ashtanga Yoga"""
        return {
            YogaLimb.YAMA: [
                YogaPractice(
                    name="Ahimsa Practice",
                    limb=YogaLimb.YAMA,
                    category=None,
                    description="Practicing non-violence in thought, word, and action",
                    steps=[
                        "Observe thoughts for violent or harmful tendencies",
                        "Choose compassionate responses in difficult situations",
                        "Practice non-harm in diet and lifestyle choices",
                        "Cultivate loving-kindness toward all beings",
                        "Release anger and resentment through forgiveness"
                    ],
                    benefits=[
                        "Peace of mind and heart",
                        "Improved relationships",
                        "Reduced conflict and stress",
                        "Spiritual purification"
                    ],
                    duration="Continuous practice",
                    sutra_reference="Yoga Sutras 2.35: When ahimsa is established, hostility ceases"
                ),
                
                YogaPractice(
                    name="Satya Practice", 
                    limb=YogaLimb.YAMA,
                    category=None,
                    description="Practicing truthfulness and authenticity",
                    steps=[
                        "Speak only what is true and helpful",
                        "Avoid exaggeration and distortion",
                        "Live authentically according to your values",
                        "Practice honesty in self-reflection",
                        "Align actions with inner truth"
                    ],
                    benefits=[
                        "Clear communication",
                        "Trustworthy relationships", 
                        "Inner peace and authenticity",
                        "Spiritual power and clarity"
                    ],
                    duration="Continuous practice",
                    sutra_reference="Yoga Sutras 2.36: When truthfulness is established, actions bear fruit"
                ),
                
                YogaPractice(
                    name="Asteya Practice",
                    limb=YogaLimb.YAMA,
                    category=None,
                    description="Practicing non-stealing and non-coveting",
                    steps=[
                        "Avoid taking what is not freely given",
                        "Practice gratitude for what you have",
                        "Give full effort and attention in work",
                        "Respect others' time, energy, and resources",
                        "Cultivate contentment and non-attachment"
                    ],
                    benefits=[
                        "Abundance consciousness",
                        "Trustworthiness and integrity",
                        "Freedom from want and envy",
                        "Spiritual prosperity"
                    ],
                    duration="Continuous practice",
                    sutra_reference="Yoga Sutras 2.37: When non-stealing is established, all wealth comes"
                ),
                
                YogaPractice(
                    name="Brahmacharya Practice",
                    limb=YogaLimb.YAMA,
                    category=None,
                    description="Practicing energy conservation and appropriate use of vital force",
                    steps=[
                        "Use sexual energy consciously and appropriately",
                        "Conserve vital energy for spiritual growth",
                        "Practice moderation in all sensual pleasures",
                        "Channel energy toward higher purposes",
                        "Maintain purity in thought and action"
                    ],
                    benefits=[
                        "Increased vitality and energy",
                        "Enhanced spiritual power",
                        "Mental clarity and focus",
                        "Emotional stability"
                    ],
                    duration="Continuous practice",
                    sutra_reference="Yoga Sutras 2.38: When brahmacharya is established, vigor is gained"
                ),
                
                YogaPractice(
                    name="Aparigraha Practice",
                    limb=YogaLimb.YAMA,
                    category=None,
                    description="Practicing non-attachment and non-possessiveness",
                    steps=[
                        "Let go of excessive accumulation of possessions",
                        "Practice contentment with what you have",
                        "Avoid attachment to outcomes",
                        "Share generously with others",
                        "Find security in the Self rather than possessions"
                    ],
                    benefits=[
                        "Freedom from material anxiety",
                        "Increased generosity and openness",
                        "Clarity about life purpose",
                        "Spiritual lightness and joy"
                    ],
                    duration="Continuous practice",
                    sutra_reference="Yoga Sutras 2.39: When non-possessiveness is established, knowledge of past lives arises"
                )
            ],
            
            YogaLimb.NIYAMA: [
                YogaPractice(
                    name="Saucha Practice",
                    limb=YogaLimb.NIYAMA,
                    category=None,
                    description="Practicing cleanliness of body, mind, and environment",
                    steps=[
                        "Maintain physical cleanliness and hygiene",
                        "Keep living and practice spaces clean and organized",
                        "Practice mental cleanliness through positive thoughts",
                        "Eat pure, wholesome foods",
                        "Purify speech and associations"
                    ],
                    benefits=[
                        "Physical health and vitality",
                        "Mental clarity and peace",
                        "Spiritual purity",
                        "Conducive environment for practice"
                    ],
                    duration="Daily practice",
                    sutra_reference="Yoga Sutras 2.40: From cleanliness comes dispassion for one's own body"
                ),
                
                YogaPractice(
                    name="Santosha Practice",
                    limb=YogaLimb.NIYAMA,
                    category=None,
                    description="Practicing contentment and satisfaction",
                    steps=[
                        "Find joy and satisfaction in present circumstances",
                        "Practice gratitude for life's blessings",
                        "Avoid constant comparison with others",
                        "Appreciate simple pleasures",
                        "Cultivate inner peace independent of external conditions"
                    ],
                    benefits=[
                        "Inner peace and happiness",
                        "Reduced stress and anxiety",
                        "Improved relationships",
                        "Spiritual stability"
                    ],
                    duration="Continuous practice",
                    sutra_reference="Yoga Sutras 2.42: From contentment comes supreme happiness"
                ),
                
                YogaPractice(
                    name="Tapas Practice",
                    limb=YogaLimb.NIYAMA,
                    category=None,
                    description="Practicing disciplined effort and spiritual austerity",
                    steps=[
                        "Maintain regular spiritual practices despite difficulties",
                        "Practice self-discipline in daily routines",
                        "Persist through challenges and obstacles",
                        "Channel discomfort toward spiritual growth",
                        "Purify through conscious effort and dedication"
                    ],
                    benefits=[
                        "Increased willpower and determination",
                        "Purification of body and mind",
                        "Spiritual strength and resilience",
                        "Mastery over desires and impulses"
                    ],
                    duration="Regular practice",
                    sutra_reference="Yoga Sutras 2.43: From tapas comes perfection of body and senses"
                ),
                
                YogaPractice(
                    name="Svadhyaya Practice",
                    limb=YogaLimb.NIYAMA,
                    category=None,
                    description="Self-study and study of sacred texts",
                    steps=[
                        "Study yoga philosophy and sacred texts",
                        "Practice self-reflection and introspection",
                        "Repeat sacred mantras and prayers",
                        "Observe thoughts, emotions, and patterns",
                        "Seek wisdom through study and contemplation"
                    ],
                    benefits=[
                        "Deeper understanding of Self",
                        "Spiritual wisdom and insight",
                        "Connection with divine teachings",
                        "Self-awareness and growth"
                    ],
                    duration="Daily practice",
                    sutra_reference="Yoga Sutras 2.44: From self-study comes connection with chosen deity"
                ),
                
                YogaPractice(
                    name="Ishvara Pranidhana Practice",
                    limb=YogaLimb.NIYAMA,
                    category=None,
                    description="Surrender to the Divine",
                    steps=[
                        "Offer all actions to the Divine",
                        "Practice humility and surrender",
                        "Let go of ego and personal will",
                        "Trust in divine guidance and timing",
                        "Serve as instrument of higher purpose"
                    ],
                    benefits=[
                        "Freedom from ego struggles",
                        "Peace through surrender",
                        "Divine grace and guidance",
                        "Highest spiritual attainment"
                    ],
                    duration="Continuous practice",
                    sutra_reference="Yoga Sutras 2.45: From surrender to Ishvara comes samadhi"
                )
            ],
            
            YogaLimb.ASANA: [
                YogaPractice(
                    name="Foundation Postures",
                    limb=YogaLimb.ASANA,
                    category=AsanaCategory.STANDING,
                    description="Basic standing poses for strength and stability",
                    steps=[
                        "Mountain Pose (Tadasana) - establish alignment",
                        "Warrior poses - build strength and focus",
                        "Triangle pose - create length and balance",
                        "Tree pose - develop concentration and balance",
                        "Hold each pose with steady breath"
                    ],
                    benefits=[
                        "Physical strength and flexibility",
                        "Improved posture and alignment",
                        "Mental focus and concentration",
                        "Energy cultivation"
                    ],
                    duration="30-60 minutes daily",
                    sutra_reference="Yoga Sutras 2.46: Posture should be steady and comfortable"
                )
            ],
            
            YogaLimb.PRANAYAMA: [
                YogaPractice(
                    name="Basic Breath Control",
                    limb=YogaLimb.PRANAYAMA,
                    category=None,
                    description="Foundation breathing practices for life force control",
                    steps=[
                        "Natural breath observation",
                        "Three-part breath (Dirga)",
                        "Ocean breath (Ujjayi)",
                        "Alternate nostril breathing (Nadi Shodhana)",
                        "Breath retention (Kumbhaka) when ready"
                    ],
                    benefits=[
                        "Increased life force (prana)",
                        "Mental clarity and calm",
                        "Nervous system balance",
                        "Preparation for meditation"
                    ],
                    duration="15-30 minutes daily",
                    sutra_reference="Yoga Sutras 2.49: Pranayama is the cessation of inhalation and exhalation"
                )
            ],
            
            YogaLimb.PRATYAHARA: [
                YogaPractice(
                    name="Sense Withdrawal",
                    limb=YogaLimb.PRATYAHARA,
                    category=None,
                    description="Drawing attention inward from external distractions",
                    steps=[
                        "Practice in quiet, minimal stimulation environment",
                        "Close eyes and turn attention inward",
                        "Observe without engaging with external sounds",
                        "Release attachment to sensory experiences",
                        "Cultivate inner awareness and peace"
                    ],
                    benefits=[
                        "Reduced mental agitation",
                        "Increased inner awareness",
                        "Freedom from sensory addiction",
                        "Preparation for concentration"
                    ],
                    duration="10-20 minutes daily",
                    sutra_reference="Yoga Sutras 2.54: Pratyahara is the imitation of the mind by the senses"
                )
            ],
            
            YogaLimb.DHARANA: [
                YogaPractice(
                    name="Concentration Practice",
                    limb=YogaLimb.DHARANA,
                    category=None,
                    description="Focused attention on a single object or point",
                    steps=[
                        "Choose a concentration object (breath, mantra, image)",
                        "Focus attention completely on chosen object",
                        "When mind wanders, gently return to object",
                        "Gradually increase duration of focus",
                        "Develop one-pointed concentration"
                    ],
                    benefits=[
                        "Increased mental focus and clarity",
                        "Reduced mental scattered-ness",
                        "Enhanced memory and learning",
                        "Foundation for meditation"
                    ],
                    duration="15-45 minutes daily",
                    sutra_reference="Yoga Sutras 3.1: Dharana is binding consciousness to one place"
                )
            ],
            
            YogaLimb.DHYANA: [
                YogaPractice(
                    name="Meditation Practice",
                    limb=YogaLimb.DHYANA,
                    category=None,
                    description="Sustained awareness and effortless concentration",
                    steps=[
                        "Establish stable concentration first",
                        "Allow effortless flow of awareness",
                        "Maintain continuous attention without strain",
                        "Rest in pure awareness itself",
                        "Let meditation happen naturally"
                    ],
                    benefits=[
                        "Deep inner peace and stillness",
                        "Spiritual insight and wisdom",
                        "Connection with higher Self",
                        "Preparation for samadhi"
                    ],
                    duration="20-60 minutes daily",
                    sutra_reference="Yoga Sutras 3.2: Dhyana is sustained flow of consciousness toward object"
                )
            ],
            
            YogaLimb.SAMADHI: [
                YogaPractice(
                    name="Union and Absorption",
                    limb=YogaLimb.SAMADHI,
                    category=None,
                    description="Complete absorption and union with the Divine",
                    steps=[
                        "Surrender completely to the practice",
                        "Allow dissolution of subject-object duality",
                        "Rest in pure consciousness",
                        "Experience unity and oneness",
                        "Return with expanded awareness"
                    ],
                    benefits=[
                        "Direct experience of Truth",
                        "Liberation from suffering",
                        "Realization of true nature",
                        "Ultimate goal of yoga"
                    ],
                    duration="Grace of the Divine",
                    sutra_reference="Yoga Sutras 3.3: Samadhi is when only the object shines forth"
                )
            ]
        }
    
    def _initialize_obstacle_remedies(self) -> Dict[YogaObstacle, Dict[str, Any]]:
        """Initialize remedies for yoga obstacles as described in Yoga Sutras"""
        return {
            YogaObstacle.SICKNESS: {
                "description": "Physical illness disrupting practice",
                "remedies": [
                    "Maintain gentle, appropriate practice",
                    "Focus on breathing and meditation",
                    "Use illness as opportunity for surrender",
                    "Seek proper medical care when needed"
                ],
                "wisdom": "The body is temporary; use illness to deepen understanding of impermanence"
            },
            
            YogaObstacle.MENTAL_LAZINESS: {
                "description": "Lack of motivation and mental dullness",
                "remedies": [
                    "Start with small, manageable practices",
                    "Create inspiring environment and routine",
                    "Practice energizing pranayama",
                    "Study inspiring texts and teachings"
                ],
                "wisdom": "Laziness is overcome through patient, consistent effort"
            },
            
            YogaObstacle.DOUBT: {
                "description": "Questioning the path and losing faith",
                "remedies": [
                    "Study authenticated texts and teachings",
                    "Connect with experienced practitioners",
                    "Start with simple, tangible practices",
                    "Remember past positive experiences"
                ],
                "wisdom": "Doubt dissolves through direct experience and patient practice"
            },
            
            YogaObstacle.CARELESSNESS: {
                "description": "Lack of attention and mindfulness",
                "remedies": [
                    "Cultivate awareness in daily activities",
                    "Practice mindfulness meditation",
                    "Set regular reminders for mindful moments",
                    "Approach practice with reverence"
                ],
                "wisdom": "Mindfulness transforms ordinary moments into spiritual practice"
            },
            
            YogaObstacle.SLOTH: {
                "description": "Physical and mental heaviness",
                "remedies": [
                    "Practice activating pranayama",
                    "Engage in moderate physical exercise",
                    "Maintain regular sleep schedule",
                    "Eat light, sattvic foods"
                ],
                "wisdom": "Energy follows attention; direct awareness toward vitality"
            },
            
            YogaObstacle.WORLDLY_MINDEDNESS: {
                "description": "Excessive attachment to sensual pleasures",
                "remedies": [
                    "Practice moderation in all enjoyments",
                    "Cultivate detachment through study",
                    "Engage in service to others",
                    "Remember the temporary nature of pleasure"
                ],
                "wisdom": "True happiness comes from within, not from external objects"
            },
            
            YogaObstacle.FALSE_PERCEPTION: {
                "description": "Misunderstanding the nature of reality",
                "remedies": [
                    "Study authentic yoga philosophy",
                    "Practice discrimination between real and unreal",
                    "Seek guidance from realized teachers",
                    "Cultivate witness consciousness"
                ],
                "wisdom": "Clear perception comes through purification of consciousness"
            },
            
            YogaObstacle.NON_ATTAINMENT: {
                "description": "Feeling stuck or unable to progress",
                "remedies": [
                    "Practice patience and surrender",
                    "Focus on process rather than results",
                    "Adjust practice to current capacity",
                    "Remember that progress is not always visible"
                ],
                "wisdom": "The fruit comes in its own time; focus on sincere effort"
            },
            
            YogaObstacle.INSTABILITY: {
                "description": "Inability to maintain achieved states",
                "remedies": [
                    "Strengthen foundation practices",
                    "Practice consistency over intensity",
                    "Cultivate patience and long-term view",
                    "Use setbacks as learning opportunities"
                ],
                "wisdom": "Stability comes through steady, regular practice over time"
            }
        }
    
    def _initialize_development_stages(self) -> Dict[YogaLevel, Dict[str, Any]]:
        """Initialize progressive development stages"""
        return {
            YogaLevel.FOUNDATION: {
                "focus": "Establishing ethical foundation and basic practices",
                "practices": ["Basic yamas and niyamas", "Simple asana sequence", "Basic breathing"],
                "duration": "6 months - 2 years",
                "indicators": ["Regular practice habit", "Basic flexibility", "Emotional stability"]
            },
            
            YogaLevel.DEVELOPING: {
                "focus": "Deepening practice and developing internal awareness",
                "practices": ["Advanced pranayama", "Longer meditation", "More challenging asanas"],
                "duration": "2-5 years",
                "indicators": ["Steady breath control", "Improved concentration", "Sense withdrawal"]
            },
            
            YogaLevel.INTEGRATING: {
                "focus": "Living yoga principles naturally in all life situations",
                "practices": ["Constant awareness", "Service orientation", "Spontaneous practice"],
                "duration": "5-10 years",
                "indicators": ["Natural ethical behavior", "Sustained peace", "Selfless service"]
            },
            
            YogaLevel.ADVANCED: {
                "focus": "Mastery of all eight limbs and deep spiritual realization",
                "practices": ["Advanced meditation", "Spontaneous samadhi", "Teaching others"],
                "duration": "10+ years",
                "indicators": ["Siddhis (powers)", "Continuous awareness", "Wisdom embodiment"]
            },
            
            YogaLevel.MASTER: {
                "focus": "Complete embodiment and service to humanity",
                "practices": ["Living as example", "Guiding others", "Divine service"],
                "duration": "Lifetime achievement",
                "indicators": ["Egoless action", "Divine love", "Perfect peace"]
            }
        }
    
    def _initialize_asana_sequences(self) -> Dict[YogaLevel, List[str]]:
        """Initialize asana sequences for different levels"""
        return {
            YogaLevel.FOUNDATION: [
                "Mountain Pose (Tadasana)",
                "Forward Fold (Uttanasana)",
                "Warrior I (Virabhadrasana I)",
                "Downward Dog (Adho Mukha Svanasana)",
                "Child's Pose (Balasana)",
                "Seated Forward Fold (Paschimottanasana)",
                "Supine Twist (Supta Matsyendrasana)",
                "Corpse Pose (Savasana)"
            ],
            
            YogaLevel.DEVELOPING: [
                "Sun Salutation A & B",
                "Standing sequence with binds",
                "Seated poses with variations",
                "Basic backbends and arm balances",
                "Pranayama integration",
                "Longer relaxation"
            ],
            
            YogaLevel.INTEGRATING: [
                "Advanced sun salutations",
                "Challenging standing sequences",
                "Deep backbends and forward folds",
                "Arm balances and inversions",
                "Advanced pranayama",
                "Meditation integration"
            ],
            
            YogaLevel.ADVANCED: [
                "Spontaneous movement",
                "Advanced poses as appropriate",
                "Energy-based practice",
                "Long meditation sits",
                "Minimal physical practice needed"
            ],
            
            YogaLevel.MASTER: [
                "Practice as needed for teaching",
                "Demonstration poses",
                "Energy transmission",
                "Living embodiment"
            ]
        }
    
    async def process_yoga_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process yoga-related queries and provide comprehensive guidance"""
        context = user_context or {}
        
        # Analyze query for yoga focus
        query_lower = query.lower()
        focus_limb = self._determine_focus_limb(query_lower)
        user_level = self._assess_user_level(context)
        obstacles = self._identify_obstacles(query_lower, context)
        
        # Generate comprehensive response
        response = {
            "query": query,
            "yoga_analysis": await self._analyze_yoga_needs(query, context),
            "recommended_limb": focus_limb,
            "current_level": user_level,
            "practice_recommendations": await self._get_practice_recommendations(focus_limb, user_level),
            "daily_routine": self._get_daily_routine(user_level),
            "obstacles_identified": obstacles,
            "obstacle_remedies": self._get_obstacle_remedies(obstacles),
            "philosophical_guidance": await self._generate_philosophical_guidance(query, user_level, obstacles),
            "progressive_path": self._get_progressive_path(user_level),
            "mantras": self._get_appropriate_mantras(user_level, focus_limb),
            "yoga_wisdom": self._get_relevant_wisdom(user_level, obstacles),
            "practice_tips": [
                "Practice regularly, even if briefly",
                "Progress gradually without forcing",
                "Maintain awareness throughout practice",
                "Integrate yoga principles into daily life",
                "Seek guidance from qualified teachers"
            ]
        }
        
        return response
    
    def _determine_focus_limb(self, query: str) -> YogaLimb:
        """Determine which limb of yoga the query is focusing on"""
        limb_keywords = {
            YogaLimb.YAMA: ["ethics", "morality", "ahimsa", "non-violence", "truthfulness", "satya"],
            YogaLimb.NIYAMA: ["discipline", "cleanliness", "contentment", "study", "surrender"],
            YogaLimb.ASANA: ["posture", "pose", "flexibility", "strength", "physical"],
            YogaLimb.PRANAYAMA: ["breath", "breathing", "prana", "energy"],
            YogaLimb.PRATYAHARA: ["senses", "withdrawal", "distraction", "focus"],
            YogaLimb.DHARANA: ["concentration", "focus", "attention", "mind"],
            YogaLimb.DHYANA: ["meditation", "contemplation", "awareness"],
            YogaLimb.SAMADHI: ["union", "absorption", "enlightenment", "samadhi"]
        }
        
        for limb, keywords in limb_keywords.items():
            if any(keyword in query for keyword in keywords):
                return limb
        
        return YogaLimb.ASANA  # Default to asana for general queries
    
    def _assess_user_level(self, context: Dict[str, Any]) -> YogaLevel:
        """Assess user's current yoga level based on context"""
        experience = context.get("yoga_experience", "beginner")
        practice_years = context.get("practice_years", 0)
        
        if practice_years == 0 or experience == "beginner":
            return YogaLevel.FOUNDATION
        elif practice_years < 3:
            return YogaLevel.DEVELOPING
        elif practice_years < 7:
            return YogaLevel.INTEGRATING
        elif practice_years < 15:
            return YogaLevel.ADVANCED
        else:
            return YogaLevel.MASTER
    
    def _identify_obstacles(self, query: str, context: Dict[str, Any]) -> List[YogaObstacle]:
        """Identify obstacles mentioned or implied in query"""
        obstacle_keywords = {
            YogaObstacle.SICKNESS: ["sick", "illness", "health", "injury"],
            YogaObstacle.MENTAL_LAZINESS: ["lazy", "unmotivated", "dull", "tired"],
            YogaObstacle.DOUBT: ["doubt", "questioning", "faith", "unsure"],
            YogaObstacle.CARELESSNESS: ["distracted", "unfocused", "careless"],
            YogaObstacle.SLOTH: ["heavy", "sluggish", "lethargic"],
            YogaObstacle.WORLDLY_MINDEDNESS: ["attachment", "pleasure", "material"],
            YogaObstacle.FALSE_PERCEPTION: ["confused", "misunderstanding"],
            YogaObstacle.NON_ATTAINMENT: ["stuck", "progress", "plateau"],
            YogaObstacle.INSTABILITY: ["inconsistent", "unstable", "varying"]
        }
        
        identified = []
        for obstacle, keywords in obstacle_keywords.items():
            if any(keyword in query for keyword in keywords):
                identified.append(obstacle)
        
        return identified
    
    async def _analyze_yoga_needs(self, query: str, context: Dict[str, Any]) -> str:
        """Analyze the user's yoga needs based on query and context"""
        needs_analysis = "Based on your query, I see you're seeking guidance in yoga practice. "
        
        if "stress" in query.lower():
            needs_analysis += "Yoga can be particularly helpful for stress relief through pranayama and meditation. "
        
        if "physical" in query.lower() or "body" in query.lower():
            needs_analysis += "The physical aspect of yoga through asana practice will help build strength and flexibility. "
        
        if "spiritual" in query.lower() or "enlightenment" in query.lower():
            needs_analysis += "The spiritual dimension of yoga offers a complete path to self-realization. "
        
        needs_analysis += "Remember that yoga is a holistic practice integrating all aspects of being."
        
        return needs_analysis
    
    async def _get_practice_recommendations(self, focus_limb: YogaLimb, level: YogaLevel) -> List[str]:
        """Get specific practice recommendations"""
        practices = self.ashtanga_practices.get(focus_limb, [])
        
        if level == YogaLevel.FOUNDATION:
            return [practice.name for practice in practices[:2]]  # Basic practices
        elif level == YogaLevel.DEVELOPING:
            return [practice.name for practice in practices[:3]]  # Intermediate
        else:
            return [practice.name for practice in practices]  # All practices
    
    def _get_daily_routine(self, level: YogaLevel) -> Dict[str, List[str]]:
        """Get daily routine recommendations based on level"""
        if level == YogaLevel.FOUNDATION:
            return {
                "morning": [
                    "Simple sun salutation or basic poses",
                    "Basic breathing practice (5-10 minutes)",
                    "Brief meditation or relaxation"
                ],
                "evening": [
                    "Gentle stretching or restorative poses",
                    "Simple meditation or relaxation"
                ]
            }
        elif level == YogaLevel.DEVELOPING:
            return {
                "morning": [
                    "20-30 minutes asana practice",
                    "10-15 minutes pranayama",
                    "5-10 minutes meditation"
                ],
                "afternoon": [
                    "Mindful activities and breath awareness",
                    "Practice sense withdrawal during busy times",
                    "Brief concentration practice"
                ],
                "evening": [
                    "Gentle yoga or pranayama",
                    "Study of yoga texts",
                    "Longer meditation session"
                ]
            }
        elif level == YogaLevel.INTEGRATING:
            return {
                "morning": [
                    "45-60 minutes integrated practice",
                    "Advanced pranayama and meditation",
                    "Study and reflection"
                ],
                "throughout_day": [
                    "Constant remembrance and awareness",
                    "Living yoga principles naturally",
                    "Service and compassionate action"
                ],
                "evening": [
                    "Deep meditation practice",
                    "Contemplation and self-inquiry",
                    "Integration of day's experiences"
                ]
            }
        else:  # Advanced and Master
            return {
                "continuous": [
                    "Natural state of yoga throughout day",
                    "Spontaneous practice as needed",
                    "Teaching and serving others",
                    "Living as embodiment of yoga principles"
                ]
            }
    
    async def _generate_philosophical_guidance(self, situation: str, level: YogaLevel, obstacles: List[YogaObstacle]) -> str:
        """Generate philosophical guidance based on Yoga Sutras"""
        base_guidance = "🕉️ The path of yoga is the journey from fragmentation to wholeness, from separation to union. "
        
        if level == YogaLevel.FOUNDATION:
            level_guidance = "You are building the foundation of yoga through ethical living and physical practice. Remember that yoga begins with how we treat ourselves and others. "
        elif level == YogaLevel.DEVELOPING:
            level_guidance = "You are developing the internal practices that lead to mastery of the mind and breath. Stay consistent and patient as you build these essential skills. "
        elif level == YogaLevel.INTEGRATING:
            level_guidance = "You are integrating yoga into all aspects of life, living the principles naturally. Focus on serving others and maintaining constant awareness. "
        else:
            level_guidance = "You embody the highest teachings of yoga through your very being. Continue to serve as an example and guide for others on the path. "
        
        obstacle_guidance = ""
        if obstacles:
            obstacle_guidance = "The obstacles you're experiencing are normal parts of the yoga journey. "
            for obstacle in obstacles[:2]:
                remedy = self.obstacle_remedies.get(obstacle, {})
                wisdom = remedy.get("wisdom", "")
                if wisdom:
                    obstacle_guidance += f"{wisdom}. "
        
        sutra_guidance = "As Patanjali teaches us, 'Yoga chitta vritti nirodhah' - yoga is the stilling of the fluctuations of consciousness. Through practice and surrender, we find our true nature."
        
        return base_guidance + level_guidance + obstacle_guidance + sutra_guidance
    
    def _get_progressive_path(self, current_level: YogaLevel) -> List[str]:
        """Get progressive path from current level"""
        all_levels = [YogaLevel.FOUNDATION, YogaLevel.DEVELOPING, YogaLevel.INTEGRATING, YogaLevel.ADVANCED, YogaLevel.MASTER]
        current_index = all_levels.index(current_level)
        
        path = []
        for level in all_levels[current_index:]:
            stage_info = self.development_stages.get(level, {})
            path.append(f"{level.value.title()}: {stage_info.get('focus', '')}")
        
        return path
    
    def _get_obstacle_remedies(self, obstacles: List[YogaObstacle]) -> List[str]:
        """Get remedies for identified obstacles"""
        remedies = []
        for obstacle in obstacles:
            remedy_info = self.obstacle_remedies.get(obstacle, {})
            remedies.extend(remedy_info.get("remedies", [])[:2])  # Top 2 remedies per obstacle
        
        return remedies
    
    def _get_appropriate_mantras(self, level: YogaLevel, focus_limb: YogaLimb) -> List[str]:
        """Get mantras appropriate for level and focus"""
        mantras = ["Om"]  # Sacred mantra
        
        if focus_limb == YogaLimb.PRANAYAMA:
            mantras.extend(["So Hum", "Om Gam Ganapataye Namaha"])
        elif focus_limb == YogaLimb.DHYANA:
            mantras.extend(["Om Namah Shivaya", "Aham Brahmasmi"])
        elif focus_limb == YogaLimb.SAMADHI:
            mantras.extend(["Om Tat Sat", "Soham"])
        
        # Level-specific mantras
        if level in [YogaLevel.ADVANCED, YogaLevel.MASTER]:
            mantras.append("Tat Tvam Asi")
        
        return list(dict.fromkeys(mantras))[:4]
    
    def _get_relevant_wisdom(self, level: YogaLevel, obstacles: List[YogaObstacle]) -> str:
        """Get relevant wisdom from Yoga Sutras and tradition"""
        if obstacles:
            first_obstacle = obstacles[0]
            remedy = self.obstacle_remedies.get(first_obstacle, {})
            return remedy.get("wisdom", "")
        
        level_wisdom = {
            YogaLevel.FOUNDATION: "Abhyasa vairagyabhyam tat nirodhah - Through practice and detachment, the mind becomes still",
            YogaLevel.DEVELOPING: "Sthira sukham asanam - Posture should be steady and comfortable",
            YogaLevel.INTEGRATING: "Yoga karmasu kaushalam - Yoga is skill in action",
            YogaLevel.ADVANCED: "Tat tvam asi - Thou art That",
            YogaLevel.MASTER: "Aham Brahmasmi - I am the absolute reality"
        }
        
        return level_wisdom.get(level, "Yoga is the journey of the self, through the self, to the Self")
    
    async def create_yoga_practice_plan(self, goal: str = "general_yoga", duration: str = "30_days") -> Dict[str, Any]:
        """Create personalized yoga practice plan"""
        plans = {
            "general_yoga": {
                "focus": "Balanced development in all eight limbs",
                "daily_practices": [
                    "Morning asana practice (20-30 minutes)",
                    "Pranayama practice (10-15 minutes)",
                    "Meditation (10-20 minutes)",
                    "Ethical living throughout day"
                ],
                "weekly_focuses": [
                    "Week 1: Establishing routine and foundation",
                    "Week 2: Developing breath awareness and control",
                    "Week 3: Deepening concentration and meditation",
                    "Week 4: Integration and living yoga principles"
                ]
            },
            
            "stress_relief": {
                "focus": "Using yoga for stress management and relaxation",
                "daily_practices": [
                    "Gentle, restorative asana practice",
                    "Calming pranayama techniques",
                    "Mindfulness and meditation",
                    "Stress-reducing lifestyle choices"
                ],
                "weekly_focuses": [
                    "Week 1: Learning relaxation techniques",
                    "Week 2: Breath practices for calm",
                    "Week 3: Meditation for mental peace",
                    "Week 4: Integration into stressful situations"
                ]
            },
            
            "spiritual_growth": {
                "focus": "Using yoga for spiritual development and self-realization",
                "daily_practices": [
                    "Traditional spiritual practices",
                    "Study of yoga philosophy",
                    "Advanced pranayama and meditation",
                    "Service and compassionate living"
                ],
                "weekly_focuses": [
                    "Week 1: Purification through yamas and niyamas",
                    "Week 2: Energy cultivation through pranayama",
                    "Week 3: Mind training through concentration",
                    "Week 4: Deepening meditation and surrender"
                ]
            }
        }
        
        plan = plans.get(goal, plans["general_yoga"])
        
        return {
            "goal": goal,
            "duration": duration,
            "plan_overview": plan,
            "success_indicators": [
                "Increased flexibility and strength",
                "Better breath control and awareness",
                "Improved concentration and mental clarity",
                "Greater emotional stability",
                "Deeper sense of peace and wellbeing"
            ],
            "tracking_suggestions": [
                "Daily practice log",
                "Physical and mental state assessment",
                "Breath quality and capacity",
                "Meditation depth and duration"
            ],
            "support_resources": [
                "Qualified yoga teacher",
                "Yoga Sutras and classical texts",
                "Supportive practice community",
                "Regular workshops and retreats"
            ]
        }
    
    def get_chakra_status(self) -> Dict[str, Any]:
        """Get current status of the Yoga Chakra"""
        return {
            "name": self.chakra_name,
            "state": "active",
            "element": self.element,
            "color": self.color,
            "mantra": self.mantra,
            "governing_deity": self.deity,
            "core_principles": self.principles,
            "primary_functions": [
                "Complete yoga practice guidance",
                "Eight-limbed path progression",
                "Obstacle identification and remedies",
                "Integration of yoga in daily life"
            ],
            "wisdom_available": "Guidance for complete yoga practice from ethical foundations to highest realization"
        }
    
    async def daily_yoga_practice(self, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Provide daily yoga practice suggestions"""
        context = user_context or {}
        
        morning_practice = [
            "Begin with gratitude and intention setting",
            "Warm-up the body with gentle movements",
            "Practice asana sequence appropriate to your level",
            "Include pranayama for energy and focus",
            "End with brief meditation or relaxation"
        ]
        
        throughout_day = [
            "Live ethical principles in all interactions",
            "Maintain breath awareness during activities",
            "Practice sense withdrawal during overwhelming moments",
            "Apply yoga philosophy to challenges and decisions"
        ]
        
        evening_practice = [
            "Practice calming asanas or restorative poses",
            "Use breathing techniques to release day's tension",
            "Reflect on day's adherence to yoga principles",
            "End with meditation and surrender practice"
        ]
        
        return {
            "morning_practice": morning_practice,
            "throughout_day": throughout_day,
            "evening_practice": evening_practice,
            "weekly_practice": "Dedicate one day to deeper study and longer practice",
            "monthly_goal": "Assess progress and adjust practice as needed",
            "life_reminder": "Yoga is not just what you do on the mat, but how you live your life with awareness, compassion, and wisdom"
        }

# Global instance for easy import
yoga_module = YogaChakra()

def get_yoga_module():
    """Get the global yoga module instance"""
    return yoga_module
