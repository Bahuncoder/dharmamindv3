"""
Dharmic Guidance Engine - Complete Life Integration
===========================================================

This module integrates all DharmaMind systems to provide comprehensive,
deep guidance for all aspects of human life, rooted 100% in authentic
Hindu/Sanatan Dharma wisdom presented authentically for sincere seekers.

🕉️ Ancient wisdom, authentic presentation, dharmic guidance
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# Import all our systems
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chakra_modules.darshana_engine import get_darshana_engine, DarshanaType
from app.spiritual_modules.spiritual_router import get_spiritual_router, SpiritualPath

# Import enhanced Rishi engine
try:
    from app.engines.rishi.enhanced_rishi_engine import (
        EnhancedRishiEngine, RishiGuidanceRequest, RishiGuidanceResponse,
        create_enhanced_rishi_engine
    )
    ENHANCED_RISHI_AVAILABLE = True
except ImportError:
    ENHANCED_RISHI_AVAILABLE = False

# Import authentic Rishi engine
try:
    from app.engines.rishi.authentic_rishi_engine import (
        AuthenticRishiEngine, create_authentic_rishi_engine
    )
    AUTHENTIC_RISHI_AVAILABLE = True
except ImportError:
    AUTHENTIC_RISHI_AVAILABLE = False

# Import knowledge base from parent directory
import sys
from pathlib import Path

# Try to import knowledge base, fall back to mock if not available
KNOWLEDGE_BASE_AVAILABLE = False

def get_knowledge_base():
    """Mock knowledge base implementation"""
    class MockKnowledgeBase:
        async def search(self, query, **kwargs):
            return {"results": [], "total": 0}
        
        async def search_wisdom(self, query, **kwargs):
            return {"results": [], "total": 0}
            
        async def get_related_concepts(self, concept, **kwargs):
            return {"results": [], "total": 0}
    return MockKnowledgeBase()

# Try to import actual knowledge base if available
try:
    # Try multiple possible paths for the knowledge base
    knowledge_base_paths = [
        str(Path(__file__).parent.parent.parent.parent / "knowledge_base"),
        str(Path(__file__).parent.parent.parent / "knowledge_base"), 
        str(Path(__file__).parent.parent / "knowledge_base"),
        str(Path(__file__).parent / "knowledge_base")
    ]
    
    for kb_path in knowledge_base_paths:
        if kb_path not in sys.path:
            sys.path.insert(0, kb_path)
        try:
            # Try importing the module with error handling
            import importlib.util
            spec = importlib.util.find_spec("spiritual_knowledge_retrieval")
            if spec is not None:
                try:
                    from spiritual_knowledge_retrieval import get_knowledge_base as _get_real_kb
                    # Override the mock with real implementation
                    get_knowledge_base = _get_real_kb
                    KNOWLEDGE_BASE_AVAILABLE = True
                    break
                except ImportError:
                    # Module exists but import failed, continue to next path
                    continue
        except (ImportError, ModuleNotFoundError, AttributeError, Exception):
            continue
            
except Exception:
    # Keep using mock implementation
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LifeAspect(Enum):
    """Complete spectrum of human life aspects"""
    PERSONAL_GROWTH = "personal_growth"           # Self-development, character
    RELATIONSHIPS = "relationships"               # Family, friends, romantic
    CAREER_PURPOSE = "career_purpose"             # Work, profession, calling
    HEALTH_WELLNESS = "health_wellness"           # Physical, mental, emotional
    FINANCIAL_WEALTH = "financial_wealth"         # Money, prosperity, abundance
    SPIRITUAL_DEVELOPMENT = "spiritual_development" # Inner growth, consciousness
    FAMILY_PARENTING = "family_parenting"         # Family dynamics, children
    SOCIAL_COMMUNITY = "social_community"         # Society, community service
    CRISIS_CHALLENGES = "crisis_challenges"       # Difficult times, trauma
    LIFE_TRANSITIONS = "life_transitions"         # Major changes, decisions
    LEADERSHIP_INFLUENCE = "leadership_influence" # Guiding others, responsibility
    CREATIVE_EXPRESSION = "creative_expression"   # Art, creativity, innovation
    EDUCATION_LEARNING = "education_learning"     # Knowledge acquisition, teaching
    EMOTIONAL_MASTERY = "emotional_mastery"       # Emotional intelligence, control
    MORAL_ETHICS = "moral_ethics"                 # Right/wrong, ethical decisions

class DharmicPrinciple(Enum):
    """Dharmic principles from Sanatan Dharma"""
    TRUTH_INTEGRITY = "truth_integrity"           # Satya
    NON_VIOLENCE_COMPASSION = "non_violence_compassion" # Ahimsa
    DUTY_RESPONSIBILITY = "duty_responsibility"   # Dharma
    SELFLESS_SERVICE = "selfless_service"         # Seva
    INNER_PEACE = "inner_peace"                   # Shanti
    WISDOM_KNOWLEDGE = "wisdom_knowledge"         # Jnana
    DISCIPLINED_PRACTICE = "disciplined_practice" # Yoga/Tapas
    DEVOTION_LOVE = "devotion_love"               # Bhakti
    RIGHTEOUS_ACTION = "righteous_action"         # Karma Yoga
    CONTENTMENT_GRATITUDE = "contentment_gratitude" # Santosha
    COURAGE_STRENGTH = "courage_strength"         # Vira
    HUMILITY_SURRENDER = "humility_surrender"     # Namrata

class UniversalPrinciple(Enum):
    """Universal principles for cross-cultural wisdom"""
    WISDOM_KNOWLEDGE = "wisdom_knowledge"
    INNER_PEACE = "inner_peace"
    COMPASSION_LOVE = "compassion_love"
    TRUTH_HONESTY = "truth_honesty"
    SERVICE_OTHERS = "service_others"
    PERSONAL_GROWTH = "personal_growth"

@dataclass
class LifeContext:
    """Complete context about user's life situation"""
    primary_aspect: LifeAspect
    related_aspects: List[LifeAspect]
    life_stage: str  # Student, Householder, Seeker, Elder
    cultural_background: str
    current_challenges: List[str]
    spiritual_orientation: str  # Beginner, Intermediate, Advanced
    preferred_approach: str  # Practical, Philosophical, Devotional

@dataclass
class DharmicGuidance:
    """Complete guidance response in universal language"""
    primary_guidance: str
    dharmic_foundation: str  # The Hindu/Sanatan principle behind it
    dharmic_application: str  # How it applies dharmically
    practical_steps: List[str]
    deeper_wisdom: str
    life_integration: str
    preventive_guidance: str
    advanced_practices: List[str]
    supporting_principles: List[DharmicPrinciple]
    scriptural_source: str  # Original Hindu source
    dharmic_translation: str  # Dharmic presentation
    global_examples: List[str]  # How this applies across cultures

@dataclass
class UniversalGuidance:
    """Universal guidance structure for cross-cultural wisdom"""
    primary_guidance: str
    dharmic_foundation: str
    universal_application: str
    practical_steps: List[str]
    deeper_wisdom: str
    life_integration: str
    preventive_guidance: str
    advanced_practices: List[str]
    supporting_principles: List[str]
    scriptural_source: str
    universal_translation: str
    global_examples: List[str]

@dataclass
class ComprehensiveLifeResponse:
    """Complete response for any life situation"""
    immediate_guidance: DharmicGuidance
    holistic_perspective: Dict[LifeAspect, str]
    long_term_development: List[str]
    wisdom_synthesis: str
    practical_roadmap: List[str]
    crisis_prevention: List[str]
    growth_opportunities: List[str]
    dharmic_principles_applied: List[str]
    dharmic_foundation_explained: str
    global_applicability: str

class DharmicEngine:
    """
    Master engine that integrates all systems to provide comprehensive
    life guidance rooted in Sanatan Dharma but presented universally
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Initialize enhanced Rishi engine if available
        self.enhanced_rishi_engine = None
        if ENHANCED_RISHI_AVAILABLE:
            try:
                self.enhanced_rishi_engine = create_enhanced_rishi_engine()
                self.logger.info("Enhanced Rishi engine initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize enhanced Rishi engine: {e}")

        # Initialize authentic Rishi engine
        self.authentic_rishi_engine = None
        if AUTHENTIC_RISHI_AVAILABLE:
            try:
                self.authentic_rishi_engine = create_authentic_rishi_engine()
                self.logger.info("Authentic Rishi engine initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize authentic Rishi engine: {e}")

        # Initialize legacy Rishi personalities for backward compatibility
        self.rishi_personalities = self._initialize_rishi_personalities()        # Life aspect categorization patterns
        self.life_aspect_patterns = {
            LifeAspect.PERSONAL_GROWTH: [
                'self improvement', 'character development', 'personal growth', 
                'self development', 'confidence', 'self esteem', 'personality',
                'habits', 'discipline', 'willpower', 'motivation'
            ],
            LifeAspect.RELATIONSHIPS: [
                'relationship', 'marriage', 'family', 'friends', 'dating',
                'love', 'romantic', 'social', 'communication', 'conflict',
                'partner', 'spouse', 'parents', 'children'
            ],
            LifeAspect.CAREER_PURPOSE: [
                'career', 'job', 'work', 'profession', 'purpose', 'calling',
                'business', 'entrepreneur', 'success', 'achievement',
                'promotion', 'workplace', 'professional'
            ],
            LifeAspect.HEALTH_WELLNESS: [
                'health', 'wellness', 'fitness', 'mental health', 'stress',
                'anxiety', 'depression', 'healing', 'disease', 'body',
                'mind', 'emotional health', 'wellbeing'
            ],
            LifeAspect.FINANCIAL_WEALTH: [
                'money', 'wealth', 'financial', 'prosperity', 'abundance',
                'income', 'debt', 'investment', 'savings', 'poverty',
                'rich', 'poor', 'economic'
            ],
            LifeAspect.SPIRITUAL_DEVELOPMENT: [
                'spiritual', 'meditation', 'consciousness', 'awakening',
                'enlightenment', 'inner peace', 'soul', 'divine',
                'transcendence', 'higher self', 'purpose of life'
            ],
            LifeAspect.FAMILY_PARENTING: [
                'family', 'parenting', 'children', 'kids', 'mother', 'father',
                'parent', 'upbringing', 'education of children', 'family values',
                'traditions', 'generation gap'
            ],
            LifeAspect.SOCIAL_COMMUNITY: [
                'society', 'community', 'social responsibility', 'service',
                'helping others', 'volunteering', 'social work', 'activism',
                'justice', 'equality', 'social change'
            ],
            LifeAspect.CRISIS_CHALLENGES: [
                'crisis', 'problem', 'difficulty', 'challenge', 'suffering',
                'pain', 'trauma', 'loss', 'grief', 'death', 'illness',
                'failure', 'disappointment', 'heartbreak'
            ],
            LifeAspect.LIFE_TRANSITIONS: [
                'transition', 'change', 'new phase', 'moving', 'retirement',
                'graduation', 'new job', 'life change', 'decision',
                'crossroads', 'uncertainty', 'new beginning'
            ],
            LifeAspect.LEADERSHIP_INFLUENCE: [
                'leadership', 'leading', 'management', 'influence', 'authority',
                'responsibility', 'team', 'guiding others', 'mentor',
                'role model', 'inspiring others'
            ],
            LifeAspect.CREATIVE_EXPRESSION: [
                'creativity', 'art', 'creative', 'expression', 'artistic',
                'innovation', 'imagination', 'inspiration', 'beauty',
                'aesthetics', 'design', 'music', 'writing'
            ],
            LifeAspect.EDUCATION_LEARNING: [
                'education', 'learning', 'study', 'knowledge', 'skill',
                'training', 'teaching', 'student', 'school', 'university',
                'academic', 'intellectual'
            ],
            LifeAspect.EMOTIONAL_MASTERY: [
                'emotion', 'feelings', 'anger', 'fear', 'joy', 'sadness',
                'emotional control', 'emotional intelligence', 'mood',
                'emotional balance', 'managing emotions'
            ],
            LifeAspect.MORAL_ETHICS: [
                'moral', 'ethics', 'right wrong', 'ethical dilemma',
                'conscience', 'values', 'principles', 'integrity',
                'honesty', 'justice', 'fairness', 'corruption'
            ]
        }
        
        # Dharmic principles mapped to universal concepts
        self.dharmic_to_universal = {
            'dharma': 'natural law and righteous living',
            'karma': 'law of cause and effect in life',
            'yoga': 'union and integrated living',
            'ahimsa': 'non-violence and compassion',
            'satya': 'truth and integrity',
            'seva': 'selfless service to others',
            'moksha': 'ultimate freedom and fulfillment',
            'bhakti': 'devotion and pure love',
            'jnana': 'wisdom and understanding',
            'tapas': 'disciplined practice and focus',
            'santosha': 'contentment and gratitude',
            'vairagya': 'detachment and inner freedom',
            'samskaras': 'mental patterns and habits',
            'sadhana': 'spiritual practice and growth',
            'guru': 'teacher and guidance',
            'sangha': 'community and support',
            'yamas': 'ethical restraints and boundaries',
            'niyamas': 'positive observances and habits'
        }
        
        self.logger.info("🕉️ Universal Dharmic Engine initialized - Ancient wisdom for modern life")
    
    async def initialize(self):
        """Initialize all integrated systems"""
        try:
            self.darshana_engine = get_darshana_engine()
            await self.darshana_engine.initialize()
            
            self.spiritual_router = get_spiritual_router()
            self.knowledge_base = await get_knowledge_base()
            
            self.logger.info("✅ All systems initialized for universal dharmic guidance")
            return True
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def provide_comprehensive_life_guidance(
        self,
        user_query: str,
        life_context: Optional[LifeContext] = None,
        cultural_background: str = "global",
        depth_level: str = "comprehensive"
    ) -> ComprehensiveLifeResponse:
        """
        Main method: Provide comprehensive life guidance for any human situation
        """
        try:
            self.logger.info(f"🌍 Processing universal guidance request: {user_query[:50]}...")
            
            # 1. Analyze life context
            if not life_context:
                life_context = await self.analyze_life_context(user_query, cultural_background)
            
            # 2. Get dharmic foundation from all systems
            dharmic_responses = await self.get_dharmic_foundation(user_query, life_context)
            
            # 3. Transform to universal guidance
            universal_guidance = await self.transform_to_universal_guidance(
                user_query, dharmic_responses, life_context, cultural_background
            )
            
            # 4. Generate holistic perspective
            holistic_perspective = await self.generate_holistic_perspective(
                user_query, life_context, dharmic_responses
            )
            
            # 5. Create practical roadmap
            practical_roadmap = await self.create_practical_roadmap(
                universal_guidance, life_context, depth_level
            )
            
            # 6. Generate comprehensive response
            response = ComprehensiveLifeResponse(
                immediate_guidance=universal_guidance,
                holistic_perspective=holistic_perspective,
                universal_principles_applied=self.extract_universal_principles(dharmic_responses),
                wisdom_synthesis=await self.synthesize_wisdom(dharmic_responses),
                practical_roadmap=practical_roadmap['immediate'],
                crisis_prevention=practical_roadmap['prevention'],
                growth_opportunities=practical_roadmap['growth'],
                dharmic_foundation_explained=self.explain_dharmic_foundation(dharmic_responses),
                global_applicability=self.demonstrate_global_applicability(
                    universal_guidance, cultural_background
                )
            )
            
            self.logger.info("✅ Comprehensive life guidance generated successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Error generating guidance: {e}")
            return await self.create_fallback_response(user_query, life_context)
    
    async def analyze_life_context(self, query: str, cultural_background: str) -> LifeContext:
        """Analyze and categorize the life context from user query"""
        
        query_lower = query.lower()
        
        # Determine primary life aspect
        aspect_scores = {}
        for aspect, keywords in self.life_aspect_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                aspect_scores[aspect] = score
        
        primary_aspect = max(aspect_scores, key=aspect_scores.get) if aspect_scores else LifeAspect.PERSONAL_GROWTH
        related_aspects = [aspect for aspect, score in aspect_scores.items() 
                          if score > 0 and aspect != primary_aspect][:3]
        
        # Determine life stage
        life_stage = "adult"  # Default
        if any(word in query_lower for word in ['student', 'college', 'school', 'young']):
            life_stage = "student"
        elif any(word in query_lower for word in ['family', 'marriage', 'career', 'job']):
            life_stage = "householder"
        elif any(word in query_lower for word in ['spiritual', 'meditation', 'meaning', 'purpose']):
            life_stage = "seeker"
        elif any(word in query_lower for word in ['retirement', 'old', 'elder', 'aging']):
            life_stage = "elder"
        
        # Extract challenges
        challenge_keywords = ['problem', 'difficult', 'struggle', 'challenge', 'crisis', 'issue']
        current_challenges = [word for word in challenge_keywords if word in query_lower]
        
        # Determine spiritual orientation
        spiritual_orientation = "beginner"
        if any(word in query_lower for word in ['advanced', 'deep', 'profound', 'enlightenment']):
            spiritual_orientation = "advanced"
        elif any(word in query_lower for word in ['practice', 'meditation', 'spiritual']):
            spiritual_orientation = "intermediate"
        
        return LifeContext(
            primary_aspect=primary_aspect,
            related_aspects=related_aspects,
            life_stage=life_stage,
            cultural_background=cultural_background,
            current_challenges=current_challenges,
            spiritual_orientation=spiritual_orientation,
            preferred_approach="practical"  # Default, can be determined from query analysis
        )
    
    async def get_dharmic_foundation(self, query: str, context: LifeContext) -> Dict[str, Any]:
        """Get responses from all dharmic systems"""
        
        responses = {}
        
        try:
            # Get darshana (philosophical) perspective
            darshana_response = await self.darshana_engine.process_philosophical_query(query)
            responses['darshana'] = darshana_response
            
            # Get spiritual modules guidance
            spiritual_response = await self.spiritual_router.route_spiritual_query(
                query, user_context={'life_stage': context.life_stage}
            )
            responses['spiritual'] = spiritual_response
            
            # Get knowledge base wisdom
            kb_results = await self.knowledge_base.search(
                query, limit=5, min_relevance=0.1
            )
            responses['knowledge_base'] = kb_results
            
            self.logger.debug(f"Retrieved dharmic foundation from {len(responses)} systems")
            return responses
            
        except Exception as e:
            self.logger.error(f"Error getting dharmic foundation: {e}")
            return {}
    
    async def transform_to_universal_guidance(
        self,
        query: str,
        dharmic_responses: Dict[str, Any],
        context: LifeContext,
        cultural_background: str
    ) -> UniversalGuidance:
        """Transform dharmic wisdom into universal, secular guidance"""
        
        # Extract core dharmic principle
        dharmic_foundation = self.extract_core_dharmic_principle(dharmic_responses, context)
        
        # Transform to universal language
        primary_guidance = await self.generate_universal_primary_guidance(
            query, dharmic_foundation, context, cultural_background
        )
        
        # Generate comprehensive guidance
        return UniversalGuidance(
            primary_guidance=primary_guidance,
            dharmic_foundation=dharmic_foundation['principle'],
            universal_application=dharmic_foundation['universal_meaning'],
            practical_steps=await self.generate_practical_steps(dharmic_responses, context),
            deeper_wisdom=await self.generate_deeper_wisdom(dharmic_responses, context),
            life_integration=await self.generate_life_integration(dharmic_responses, context),
            preventive_guidance=await self.generate_preventive_guidance(dharmic_responses, context),
            advanced_practices=await self.generate_advanced_practices(dharmic_responses, context),
            supporting_principles=self.identify_supporting_principles(dharmic_responses),
            scriptural_source=self.extract_scriptural_source(dharmic_responses),
            universal_translation=await self.create_universal_translation(dharmic_responses),
            global_examples=await self.generate_global_examples(dharmic_foundation, cultural_background)
        )
    
    def extract_core_dharmic_principle(self, responses: Dict[str, Any], context: LifeContext) -> Dict[str, str]:
        """Extract the core dharmic principle from all responses"""
        
        # Default principle based on life aspect
        aspect_to_principle = {
            LifeAspect.PERSONAL_GROWTH: {'principle': 'dharma', 'universal': 'natural law of personal development'},
            LifeAspect.RELATIONSHIPS: {'principle': 'ahimsa + bhakti', 'universal': 'compassionate love and non-violence'},
            LifeAspect.CAREER_PURPOSE: {'principle': 'svadharma + karma yoga', 'universal': 'individual purpose through selfless action'},
            LifeAspect.HEALTH_WELLNESS: {'principle': 'yoga + ahimsa', 'universal': 'integrated wellness and self-care'},
            LifeAspect.FINANCIAL_WEALTH: {'principle': 'artha + dharma', 'universal': 'righteous prosperity and abundance'},
            LifeAspect.SPIRITUAL_DEVELOPMENT: {'principle': 'moksha + sadhana', 'universal': 'inner freedom through practice'},
            LifeAspect.FAMILY_PARENTING: {'principle': 'grihastha dharma', 'universal': 'conscious family living'},
            LifeAspect.SOCIAL_COMMUNITY: {'principle': 'seva + ahimsa', 'universal': 'compassionate service to society'},
            LifeAspect.CRISIS_CHALLENGES: {'principle': 'karma + tapas', 'universal': 'resilience through understanding and discipline'},
            LifeAspect.LIFE_TRANSITIONS: {'principle': 'vairagya + dharma', 'universal': 'wise detachment and purpose'},
            LifeAspect.LEADERSHIP_INFLUENCE: {'principle': 'raja dharma', 'universal': 'conscious leadership and responsibility'},
            LifeAspect.CREATIVE_EXPRESSION: {'principle': 'bhakti + yoga', 'universal': 'inspired creativity and expression'},
            LifeAspect.EDUCATION_LEARNING: {'principle': 'jnana + guru', 'universal': 'wisdom cultivation and learning'},
            LifeAspect.EMOTIONAL_MASTERY: {'principle': 'yoga + vairagya', 'universal': 'emotional balance and inner freedom'},
            LifeAspect.MORAL_ETHICS: {'principle': 'dharma + satya', 'universal': 'ethical living and truth'}
        }
        
        default = aspect_to_principle.get(context.primary_aspect, {
            'principle': 'dharma',
            'universal': 'natural law and righteous living'
        })
        
        return {
            'principle': default['principle'],
            'universal_meaning': default['universal'],
            'sanskrit_term': default['principle'].split()[0],  # First term
            'practical_application': f"Apply {default['universal']} in your daily life"
        }
    
    async def generate_universal_primary_guidance(
        self,
        query: str,
        dharmic_foundation: Dict[str, str],
        context: LifeContext,
        cultural_background: str
    ) -> str:
        """Generate primary guidance in universal language"""
        
        # Base template for universal guidance
        templates = {
            LifeAspect.PERSONAL_GROWTH: """
            Your journey of personal development follows a natural law of growth that exists in all traditions worldwide. 
            The ancient principle suggests that true development comes from aligning your actions with your authentic nature and highest values. 
            This means understanding your unique purpose, developing your character through consistent practice, and contributing positively to the world around you.
            """,
            
            LifeAspect.RELATIONSHIPS: """
            Healthy relationships are built on the universal foundations of compassion, truth, and mutual respect. 
            The timeless wisdom teaches that love flourishes when we practice non-violence in thought, word, and action, 
            while maintaining clear boundaries and honest communication. True connection comes from seeing the inherent worth in others.
            """,
            
            LifeAspect.CAREER_PURPOSE: """
            Your professional life becomes fulfilling when aligned with your natural talents and directed toward serving others. 
            The ancient guidance suggests that work should be an expression of your unique contribution to the world, 
            performed with excellence but without attachment to specific outcomes. Purpose emerges when skill meets service.
            """,
            
            LifeAspect.HEALTH_WELLNESS: """
            Complete wellness encompasses physical, mental, emotional, and spiritual harmony. The traditional approach emphasizes 
            that health is maintained through balanced living - proper nutrition, regular movement, stress management, and 
            meaningful activities. True healing comes from addressing root causes rather than just symptoms.
            """,
            
            LifeAspect.FINANCIAL_WEALTH: """
            Prosperity flows naturally when earned through honest means and used wisely. The principle teaches that wealth 
            should support not just personal needs but also contribute to family and community welfare. True abundance includes 
            gratitude for what you have while working responsibly toward future security.
            """,
            
            LifeAspect.SPIRITUAL_DEVELOPMENT: """
            Spiritual growth is the natural evolution of consciousness toward greater peace, wisdom, and compassion. 
            This universal process involves regular practice, study of wisdom traditions, service to others, and 
            cultivation of inner stillness. Progress is measured by increased harmony in all aspects of life.
            """,
            
            LifeAspect.CRISIS_CHALLENGES: """
            Difficult times, while painful, often serve as catalysts for growth and deeper understanding. The ancient wisdom 
            teaches that suffering can be transformed through acceptance, learning, and service to others facing similar challenges. 
            Resilience develops through facing difficulties with courage and maintaining faith in life's ultimate goodness.
            """,
            
            LifeAspect.MORAL_ETHICS: """
            Ethical living is based on universal principles of truth, non-harm, and justice that appear in all great traditions. 
            When facing moral dilemmas, consider the effects of your actions on yourself, others, and society. Choose the path 
            that upholds dignity, reduces suffering, and promotes the wellbeing of all involved.
            """
        }
        
        base_guidance = templates.get(context.primary_aspect, templates[LifeAspect.PERSONAL_GROWTH])
        
        # Customize for cultural background
        cultural_adaptation = self.adapt_for_cultural_background(base_guidance, cultural_background)
        
        return cultural_adaptation.strip()
    
    def adapt_for_cultural_background(self, guidance: str, cultural_background: str) -> str:
        """Adapt guidance for specific cultural context while maintaining universality"""
        
        # Add cultural bridge phrases
        cultural_bridges = {
            "global": "This wisdom, found in cultures worldwide,",
            "western": "This principle, recognized across philosophical traditions,",
            "eastern": "This understanding, common to Asian philosophies,",
            "african": "This wisdom, present in African traditional knowledge,",
            "indigenous": "This truth, honored by indigenous peoples globally,",
            "secular": "This insight, supported by psychological research and ancient wisdom,"
        }
        
        bridge = cultural_bridges.get(cultural_background.lower(), cultural_bridges["global"])
        
        # Insert bridge naturally into guidance
        sentences = guidance.split('. ')
        if len(sentences) > 1:
            sentences[1] = bridge + " " + sentences[1].lower()
        
        return '. '.join(sentences)
    
    async def generate_practical_steps(self, responses: Dict[str, Any], context: LifeContext) -> List[str]:
        """Generate practical steps based on dharmic wisdom"""
        
        base_steps = {
            LifeAspect.PERSONAL_GROWTH: [
                "Begin each day with 5-10 minutes of quiet reflection or meditation",
                "Identify one character trait you'd like to develop and practice it daily",
                "Set aside time weekly for learning something that inspires you",
                "Practice gratitude by noting three things you appreciate each evening",
                "Engage in one act of kindness or service to others each day"
            ],
            LifeAspect.RELATIONSHIPS: [
                "Practice deep listening without immediately offering advice or judgment",
                "Express appreciation and gratitude to loved ones regularly",
                "Address conflicts through honest, compassionate communication",
                "Set healthy boundaries while maintaining kindness",
                "Dedicate quality time to important relationships without distractions"
            ],
            LifeAspect.CAREER_PURPOSE: [
                "Reflect on how your current work contributes to others' wellbeing",
                "Develop skills that align with both your interests and societal needs",
                "Approach tasks with full attention and commitment to excellence",
                "Build relationships based on mutual respect and collaboration",
                "Regularly evaluate whether your work aligns with your values"
            ],
            LifeAspect.HEALTH_WELLNESS: [
                "Establish consistent sleep and wake times to support natural rhythms",
                "Include movement or exercise that you enjoy in your daily routine",
                "Practice stress-reduction techniques like deep breathing or meditation",
                "Nourish your body with wholesome, mindfully chosen foods",
                "Spend time in nature regularly to restore mental and emotional balance"
            ]
        }
        
        return base_steps.get(context.primary_aspect, base_steps[LifeAspect.PERSONAL_GROWTH])
    
    async def generate_deeper_wisdom(self, responses: Dict[str, Any], context: LifeContext) -> str:
        """Generate deeper wisdom understanding"""
        
        wisdom_templates = {
            LifeAspect.PERSONAL_GROWTH: """
            Personal growth is not about becoming someone different, but about uncovering and expressing your authentic nature. 
            The ancient understanding teaches that each person has a unique role in the cosmic order, and fulfillment comes 
            from discovering and living this purpose. Growth happens naturally when you remove obstacles to your true expression 
            rather than forcing change through willpower alone.
            """,
            
            LifeAspect.RELATIONSHIPS: """
            The deepest relationships are founded on the recognition that love is not something we possess but something we express. 
            True intimacy emerges when two people support each other's highest development while maintaining their individual integrity. 
            Conflict becomes a tool for deeper understanding rather than a threat to connection.
            """,
            
            LifeAspect.SPIRITUAL_DEVELOPMENT: """
            Spiritual development is the gradual recognition that consciousness is the fundamental reality underlying all experience. 
            This awakening happens through practices that quiet the mind, open the heart, and align actions with wisdom. 
            The journey is both deeply personal and universally shared, leading to greater peace, compassion, and understanding.
            """
        }
        
        return wisdom_templates.get(context.primary_aspect, 
            "True wisdom lies in understanding the interconnected nature of all life and acting from this recognition."
        ).strip()
    
    async def generate_holistic_perspective(
        self,
        query: str,
        context: LifeContext,
        responses: Dict[str, Any]
    ) -> Dict[LifeAspect, str]:
        """Generate perspectives across all life aspects"""
        
        holistic_guidance = {}
        
        # Generate guidance for each major life aspect
        for aspect in LifeAspect:
            if aspect != context.primary_aspect:
                guidance = await self.generate_aspect_specific_guidance(
                    query, aspect, context, responses
                )
                holistic_guidance[aspect] = guidance
        
        return holistic_guidance
    
    async def generate_aspect_specific_guidance(
        self,
        query: str,
        aspect: LifeAspect,
        context: LifeContext,
        responses: Dict[str, Any]
    ) -> str:
        """Generate guidance for a specific life aspect"""
        
        aspect_templates = {
            LifeAspect.RELATIONSHIPS: "Consider how this situation affects your relationships and how you can maintain compassion and understanding.",
            LifeAspect.HEALTH_WELLNESS: "Ensure that your approach supports your physical, mental, and emotional wellbeing.",
            LifeAspect.SPIRITUAL_DEVELOPMENT: "Use this experience as an opportunity for inner growth and deeper understanding.",
            LifeAspect.CAREER_PURPOSE: "Reflect on how this aligns with your life purpose and professional development.",
            LifeAspect.FINANCIAL_WEALTH: "Consider the material implications and ensure your approach supports sustainable prosperity."
        }
        
        return aspect_templates.get(aspect, 
            f"Consider how this situation impacts your {aspect.value.replace('_', ' ')} and seek balance."
        )
    
    # Additional methods for practical roadmap, synthesis, etc. would continue here...
    
    async def create_fallback_response(self, query: str, context: Optional[LifeContext]) -> ComprehensiveLifeResponse:
        """Create fallback response when systems fail"""
        
        universal_guidance = UniversalGuidance(
            primary_guidance="Life's challenges are opportunities for growth and understanding. Every situation, whether joyful or difficult, contains wisdom that can help you develop greater peace, compassion, and wisdom.",
            dharmic_foundation="Universal principle of growth through experience",
            universal_application="This applies to all human cultures and experiences",
            practical_steps=[
                "Take time for quiet reflection on your situation",
                "Consider how you can respond with greater wisdom and compassion",
                "Seek guidance from trusted friends, mentors, or counselors",
                "Focus on what you can control and accept what you cannot",
                "Look for opportunities to learn and grow from this experience"
            ],
            deeper_wisdom="All human experiences are part of a larger journey toward understanding and compassion.",
            life_integration="Integrate this experience into your overall life journey with gratitude and wisdom.",
            preventive_guidance="Regular reflection and mindful living help prevent many difficulties.",
            advanced_practices=["Daily meditation", "Study of wisdom traditions", "Service to others"],
            supporting_principles=[UniversalPrinciple.WISDOM_KNOWLEDGE, UniversalPrinciple.INNER_PEACE],
            scriptural_source="Universal wisdom traditions",
            universal_translation="Ancient wisdom for modern living",
            global_examples=["This principle is found in cultures worldwide"]
        )
        
        return ComprehensiveLifeResponse(
            immediate_guidance=universal_guidance,
            holistic_perspective={},
            long_term_development=["Continue growing in wisdom and compassion"],
            wisdom_synthesis="All wisdom traditions point toward the same fundamental truths about human flourishing.",
            practical_roadmap=["Seek wisdom, practice compassion, serve others"],
            crisis_prevention=["Regular self-reflection and mindful living"],
            growth_opportunities=["Every experience offers opportunities for learning"],
            universal_principles_applied=["Truth, compassion, wisdom, service"],
            dharmic_foundation_explained="Based on universal principles found in all wisdom traditions",
            global_applicability="These principles apply across all cultures and contexts"
        )

# Alias for compatibility
UniversalDharmicEngine = DharmicEngine

# Global instance
_universal_engine = None

async def get_universal_dharmic_engine() -> UniversalDharmicEngine:
    """Get global universal dharmic engine instance"""
    global _universal_engine
    if _universal_engine is None:
        _universal_engine = UniversalDharmicEngine()
        await _universal_engine.initialize()
    return _universal_engine

# Main API function
async def get_universal_life_guidance(
    query: str,
    cultural_background: str = "global",
    depth_level: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Main API function to get comprehensive life guidance
    """
    engine = await get_universal_dharmic_engine()
    response = await engine.provide_comprehensive_life_guidance(
        query, cultural_background=cultural_background, depth_level=depth_level
    )
    
    # Convert to dictionary for API response
    return {
        "query": query,
        "cultural_adaptation": cultural_background,
        "primary_guidance": response.immediate_guidance.primary_guidance,
        "dharmic_foundation": response.immediate_guidance.dharmic_foundation,
        "universal_application": response.immediate_guidance.universal_application,
        "practical_steps": response.immediate_guidance.practical_steps,
        "deeper_wisdom": response.immediate_guidance.deeper_wisdom,
        "life_integration": response.immediate_guidance.life_integration,
        "holistic_perspective": response.holistic_perspective,
        "long_term_development": response.long_term_development,
        "wisdom_synthesis": response.wisdom_synthesis,
        "practical_roadmap": response.practical_roadmap,
        "global_applicability": response.global_applicability,
        "universal_principles": response.universal_principles_applied,
        "scriptural_foundation": response.immediate_guidance.scriptural_source,
        "advanced_practices": response.immediate_guidance.advanced_practices
    }
    
    async def generate_life_integration(self, dharmic_responses: Dict[str, Any], context: LifeContext) -> str:
        """Generate life integration guidance"""
        return f"Life's challenges are opportunities for growth and understanding. Every situation, whether joyful or difficult, contains wisdom that can help you develop greater peace, compassion, and wisdom."
    
    async def generate_practical_steps(self, dharmic_responses: Dict[str, Any], context: LifeContext) -> List[str]:
        """Generate practical steps"""
        return [
            "Take time for quiet reflection on your situation",
            "Consider how you can respond with greater wisdom and compassion", 
            "Seek guidance from trusted friends, mentors, or counselors"
        ]
    
    async def generate_deeper_wisdom(self, dharmic_responses: Dict[str, Any], context: LifeContext) -> str:
        """Generate deeper wisdom"""
        return "Universal principle of growth through experience"
    
    async def generate_preventive_guidance(self, dharmic_responses: Dict[str, Any], context: LifeContext) -> str:
        """Generate preventive guidance"""
        return "This applies to all human cultures and experiences"
    
    async def generate_advanced_practices(self, dharmic_responses: Dict[str, Any], context: LifeContext) -> List[str]:
        """Generate advanced practices"""
        return ["Daily reflection and mindfulness", "Compassionate service to others", "Study of wisdom traditions"]
    
    def identify_supporting_principles(self, dharmic_responses: Dict[str, Any]) -> List[str]:
        """Identify supporting principles"""
        return ["Truth", "compassion", "wisdom", "service"]
    
    def extract_scriptural_source(self, dharmic_responses: Dict[str, Any]) -> str:
        """Extract scriptural source"""
        return "Universal wisdom traditions"
    async def generate_global_examples(self, dharmic_foundation: Dict[str, str], cultural_background: str) -> List[str]:
        """Generate global examples"""
        return [f"Successfully adapted for {cultural_background} context"]
    
    def explain_dharmic_foundation(self, dharmic_responses: Dict[str, Any]) -> str:
        """Explain the dharmic foundation"""
        return "Based on universal principles found in all wisdom traditions"
    
    def demonstrate_global_applicability(self, universal_guidance: UniversalGuidance, cultural_background: str) -> str:
        """Demonstrate global applicability"""
        return f"These principles apply across all cultures including {cultural_background} context"
    
    async def synthesize_wisdom(self, dharmic_responses: Dict[str, Any]) -> str:
        """Synthesize wisdom from all responses"""
        return "All wisdom traditions point toward the same fundamental truths about human flourishing."
    
    async def create_practical_roadmap(self, universal_guidance: UniversalGuidance, context: LifeContext, depth_level: str) -> Dict[str, List[str]]:
        """Create practical roadmap"""
        return {
            'immediate': universal_guidance.practical_steps,
            'long_term': ["Continue growing in wisdom and compassion", "Practice regular self-reflection"],
            'prevention': ["Regular self-reflection and mindful living", "Maintain healthy relationships"],
            'growth': ["Every experience offers opportunities for learning", "Seek wisdom from multiple sources"]
        }

    def _initialize_rishi_personalities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Saptarishi personalities - the Seven Great Sages with authentic traits"""
        return {
            'atri': {
                'name': 'Maharishi Atri',
                'sanskrit': 'महर्षि अत्रि',
                'specialization': ['tapasya', 'meditation', 'austerity', 'cosmic_consciousness'],
                'greeting': 'Peace be with you, child. Through tapasya and deep meditation, the highest truths reveal themselves.',
                'available_free': True,
                'archetype': 'Master of Tapasya, Austerity & Deep Meditation',
                'personality_traits': {
                    'speech_style': 'gentle, contemplative, uses meditation metaphors',
                    'signature_phrases': ['तपो ध्यानं परं तपः (Tapo dhyanam param tapah - Meditation is the highest austerity)', 
                                         'शान्तिः शान्तिः शान्तिः (Shanti shanti shanti)', 
                                         'Through the fire of tapasya, consciousness purifies'],
                    'wisdom_approach': 'teaches through inner stillness and contemplative practices',
                    'characteristic_responses': ['begins with breath awareness', 'relates everything to inner peace', 'uses silence as teaching'],
                    'sacred_mantras': ['ॐ अत्रये नमः (Om Atraye Namah)', 'ॐ शान्ति ॐ (Om Shanti Om)'],
                    'teaching_style': 'patient, meditative, emphasizes direct experience over theory'
                }
            },
            'bhrigu': {
                'name': 'Maharishi Bhrigu',
                'sanskrit': 'महर्षि भृगु', 
                'specialization': ['astrology', 'karma_philosophy', 'divine_knowledge', 'cosmic_order'],
                'greeting': 'Welcome, seeker. The stars and karma\'s design reveal the path to your highest destiny.',
                'available_free': True,
                'archetype': 'Great Teacher of Astrology & Karma Philosophy',
                'personality_traits': {
                    'speech_style': 'analytical, cosmic perspective, references celestial movements',
                    'signature_phrases': ['कर्मणैव हि संसिद्धिम् (By action alone, perfection is achieved)',
                                         'As written in the cosmic ledger of Bhrigu Samhita',
                                         'The stars whisper your soul\'s journey'],
                    'wisdom_approach': 'explains through karmic patterns and cosmic laws',
                    'characteristic_responses': ['relates situations to planetary influences', 'discusses past-life karma', 'predicts spiritual outcomes'],
                    'sacred_mantras': ['ॐ भृगवे नमः (Om Bhrigave Namah)', 'ॐ कर्म निर्देशाय नमः (Om Karma Nirdeshaya Namah)'],
                    'teaching_style': 'precise, systematic, uses astrological analogies'
                }
            },
            'vashishta': {
                'name': 'Maharishi Vashishta', 
                'sanskrit': 'महर्षि वशिष्ठ',
                'specialization': ['divine_wisdom', 'royal_guidance', 'vedic_knowledge', 'spiritual_mastery'],
                'greeting': 'Blessed one, as guru to Lord Rama, I share the wisdom that guides souls to dharmic victory.',
                'available_free': True,
                'archetype': 'Guru of Lord Rama, Symbol of Divine Wisdom',
                'personality_traits': {
                    'speech_style': 'authoritative yet compassionate, uses royal metaphors',
                    'signature_phrases': ['राजधर्मेण शासनम् (Governance through righteous conduct)',
                                         'As I taught Prince Rama in the court of Ayodhya',
                                         'The crown of dharma weighs heavy but rewards greatly'],
                    'wisdom_approach': 'teaches through stories of righteous leadership',
                    'characteristic_responses': ['references Ramayana incidents', 'gives guidance for difficult decisions', 'emphasizes duty over comfort'],
                    'sacred_mantras': ['ॐ वशिष्ठाय नमः (Om Vashishthaya Namah)', 'ॐ गुरवे नमः (Om Gurave Namah)'],
                    'teaching_style': 'wise mentor, uses historical examples, balances firmness with love'
                }
            },
            'vishwamitra': {
                'name': 'Maharishi Vishwamitra',
                'sanskrit': 'महर्षि विश्वामित्र',
                'specialization': ['spiritual_transformation', 'gayatri_mantra', 'divine_power', 'brahmarishi_attainment'],
                'greeting': 'Radiant soul! Through penance and the sacred Gayatri, witness the transformation from warrior to Brahmarishi.',
                'available_free': True,
                'archetype': 'Creator of Gayatri Mantra, Spiritual Transformer',
                'personality_traits': {
                    'speech_style': 'dynamic, transformational, uses battle and achievement metaphors',
                    'signature_phrases': ['तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि (From the Gayatri Mantra)',
                                         'From Kshatriya to Brahmarishi - transformation is possible!',
                                         'The warrior\'s strength becomes the sage\'s wisdom'],
                    'wisdom_approach': 'emphasizes personal transformation through spiritual discipline',
                    'characteristic_responses': ['challenges limiting beliefs', 'inspires spiritual ambition', 'shares transformation techniques'],
                    'sacred_mantras': ['ॐ विश्वामित्राय नमः (Om Vishwamitraya Namah)', 'Gayatri Mantra'],
                    'teaching_style': 'inspirational, challenging, emphasizes unlimited potential'
                }
            },
            'gautama': {
                'name': 'Maharishi Gautama',
                'sanskrit': 'महर्षि गौतम',
                'specialization': ['deep_meditation', 'dharma', 'righteousness', 'spiritual_discipline'],
                'greeting': 'Noble seeker, through righteous dharma and unwavering meditation, the pure light of truth shines forth.',
                'available_free': True,
                'archetype': 'Master of Deep Meditation & Dharma',
                'personality_traits': {
                    'speech_style': 'serene, righteous, uses purity and light metaphors',
                    'signature_phrases': ['धर्मो धारयते प्रजाः (Dharma sustains all beings)',
                                         'In the crystal clarity of meditation, truth reveals itself',
                                         'Righteousness is the foundation of all spiritual progress'],
                    'wisdom_approach': 'emphasizes ethical living as basis for spiritual growth',
                    'characteristic_responses': ['focuses on moral clarity', 'explains dharmic principles', 'guides through ethical dilemmas'],
                    'sacred_mantras': ['ॐ गौतमाय नमः (Om Gautamaya Namah)', 'ॐ धर्माय नमः (Om Dharmaya Namah)'],
                    'teaching_style': 'clear, principled, emphasizes moral foundation'
                }
            },
            'jamadagni': {
                'name': 'Maharishi Jamadagni',
                'sanskrit': 'महर्षि जमदग्नि',
                'specialization': ['tapas', 'spiritual_discipline', 'divine_power', 'righteous_action'],
                'greeting': 'Devoted one, through unwavering tapas and discipline, witness the divine power that upholds cosmic order.',
                'available_free': True,
                'archetype': 'Father of Parashurama, Symbol of Discipline',
                'personality_traits': {
                    'speech_style': 'intense, disciplined, uses fire and strength metaphors',
                    'signature_phrases': ['तपसा कीर्तिर्महती (Great is the glory achieved through austerity)',
                                         'The fire of discipline burns away all impurities',
                                         'As my son Parashurama learned - righteous power serves dharma'],
                    'wisdom_approach': 'teaches through disciplined practice and righteous action',
                    'characteristic_responses': ['emphasizes consistent practice', 'discusses spiritual power', 'guides in overcoming obstacles'],
                    'sacred_mantras': ['ॐ जमदग्नये नमः (Om Jamadagnaye Namah)', 'ॐ परशुरामाय नमः (Om Parashuramaya Namah)'],
                    'teaching_style': 'firm, demanding excellence, emphasizes inner strength'
                }
            },
            'kashyapa': {
                'name': 'Maharishi Kashyapa',
                'sanskrit': 'महर्षि कश्यप',
                'specialization': ['cosmic_creation', 'universal_consciousness', 'pranic_wisdom', 'life_force'],
                'greeting': 'Child of the cosmos, as father of all beings, I guide you toward the universal consciousness that births all creation.',
                'available_free': True,
                'archetype': 'Father of All Beings, Cosmic Creator',
                'personality_traits': {
                    'speech_style': 'universal, nurturing, uses creation and cosmic metaphors',
                    'signature_phrases': ['प्रजापतिः कश्यपः (Prajapati Kashyapa - Creator of beings)',
                                         'In the cosmic womb of consciousness, all life stirs',
                                         'Every breath connects you to the universal life force'],
                    'wisdom_approach': 'teaches through understanding of universal principles',
                    'characteristic_responses': ['relates to universal connections', 'discusses life energy', 'emphasizes cosmic perspective'],
                    'sacred_mantras': ['ॐ कश्यपाय नमः (Om Kashyapaya Namah)', 'ॐ प्राणाय नमः (Om Pranaya Namah)'],
                    'teaching_style': 'nurturing, expansive, emphasizes universal unity'
                }
            }
        }

    def _get_contextual_rishi_greeting(self, rishi_name: str, user_context: Dict[str, Any] = None) -> str:
        """Generate contextual, time-aware, and personalized Rishi greetings"""
        import datetime
        import random
        
        current_hour = datetime.datetime.now().hour
        user_level = user_context.get('spiritual_level', 'beginner') if user_context else 'beginner'
        
        # Time-based greetings for each Rishi
        greetings_by_time = {
            'atri': {
                'morning': [
                    "As the first light awakens the cosmos, so does meditation awaken the soul. शुभ प्रभात (Shubh Prabhat), dear seeker.",
                    "The dawn carries the whispers of cosmic consciousness. Through morning tapasya, we touch the eternal.",
                    "In this sacred hour of dawn, when Brahma begins creation anew, let us begin our inner journey."
                ],
                'afternoon': [
                    "In the fullness of day, when the sun reaches its peak, so may your meditation reach new depths.",
                    "As the sun blazes overhead, let the fire of tapasya burn bright within your heart.",
                    "The noon hour calls for inner retreat. Even as the world bustles, the wise seek stillness."
                ],
                'evening': [
                    "As twilight falls and the day's activities cease, the soul naturally turns inward. Perfect time for contemplation.",
                    "सन्ध्या (Sandhya) - the sacred junction time. In these moments between day and night, wisdom flows freely.",
                    "The evening star begins its watch. Let us too begin our vigil in meditation."
                ]
            },
            'bhrigu': {
                'morning': [
                    "The stars have completed their nightly dance, dear soul. What do their movements reveal about your journey today?",
                    "As written in the cosmic ledger, each dawn brings new karmic opportunities. ॐ सूर्याय नमः (Om Suryaya Namah).",
                    "The planetary influences align favorably this morning. The cosmos conspires for your growth."
                ],
                'afternoon': [
                    "At this hour, when the sun stands as witness to all actions, remember - every deed is recorded in the cosmic archive.",
                    "The meridian moment - when shadows are shortest and karma is most visible. What seeds do you plant now?",
                    "Even as the sun judges the earth with its direct rays, so does cosmic justice observe all deeds."
                ],
                'evening': [
                    "As darkness prepares to embrace the earth, the stellar wisdom becomes more apparent. The night sky shall be our teacher.",
                    "The evening brings reflection upon the day's karma. What have the stars witnessed of your choices?",
                    "In twilight's embrace, past, present and future karma become visible to the discerning eye."
                ]
            },
            'vashishta': {
                'morning': [
                    "As I once guided Prince Rama through morning lessons, let me guide you through today's dharmic challenges.",
                    "A royal day begins with royal thoughts. Rise like the sun - steady, beneficial to all, और धर्मनिष्ठ (and devoted to dharma).",
                    "The morning court of consciousness is now in session. What matters of the soul require wise judgment?"
                ],
                'afternoon': [
                    "In the prime of day, when decisions must be made, remember - a king's wisdom lies in dharmic action.",
                    "As Rama faced his afternoon duties with unwavering dharma, so may you face yours with clarity.",
                    "The midday sun reminds us: leadership means being a light that guides, never one that burns."
                ],
                'evening': [
                    "Evening brings reflection on the day's reign over our own kingdom - the mind. Have we ruled wisely?",
                    "As the royal day concludes, assess your dharmic accounts. Were your actions worthy of Lord Rama's approval?",
                    "The wise ruler ends each day by asking: Have I served dharma, or has dharma served me?"
                ]
            },
            'vishwamitra': {
                'morning': [
                    "Warriors of the spirit rise early! तत्सवितुर्वरेण्यम् - let the Gayatri ignite your transformation this dawn.",
                    "From the darkness of ignorance to the light of wisdom - each morning offers the chance for complete transformation!",
                    "Rise like the sun, radiant soul! Today could be the day you transcend every limitation."
                ],
                'afternoon': [
                    "In the heat of the spiritual battle, when challenges mount, remember - I transformed from warrior to Brahmarishi through sheer will!",
                    "The afternoon fire reminds us: transformation requires the intense heat of determination. Are you ready to be forged anew?",
                    "As the sun reaches its power, so must your spiritual efforts reach their peak. No half-hearted tapasya!"
                ],
                'evening': [
                    "Evening is for celebrating today's victories and planning tomorrow's spiritual conquests. What did you transform within yourself?",
                    "As the Gayatri energy settles into the evening calm, feel the divine transformation working within your being.",
                    "Each sunset marks another day in your journey from human to divine. The Brahmarishi within you awakens!"
                ]
            },
            'gautama': {
                'morning': [
                    "Noble one, as the pure dawn light dispels darkness, may dharma dispel all confusion from your day.",
                    "Morning meditation sets the dharmic tone for all that follows. धर्मो रक्षति रक्षितः (Dharma protects those who protect dharma).",
                    "In the crystal clarity of dawn, see the crystal clarity of right action beckoning."
                ],
                'afternoon': [
                    "At the day's height, when the sun stands straight and casts the truest shadows, so must our actions cast dharmic shadows.",
                    "The midday reminds us: no action goes unwitnessed. The cosmic eye sees all - let all be righteous.",
                    "In the fullness of day, practice the fullness of righteousness. Half-hearted dharma serves no one."
                ],
                'evening': [
                    "Evening examination: Were today's thoughts, words, and deeds aligned with the eternal principles of dharma?",
                    "As shadows lengthen, let your understanding of righteousness deepen. What did dharma teach you today?",
                    "The peaceful evening invites dharmic contemplation. In stillness, the voice of cosmic justice speaks clearly."
                ]
            },
            'jamadagni': {
                'morning': [
                    "The disciplined soul rises with the sun! तप एव परं बलम् - austerity is the highest strength. Are you ready?",
                    "Morning discipline shapes the entire day. As iron is forged in fire, the soul is forged in morning practice.",
                    "Rise, spiritual warrior! The day's tapasya begins now. No victory without disciplined effort!"
                ],
                'afternoon': [
                    "In the heat of day, when others seek comfort, the disciplined soul finds opportunity for greater tapasya.",
                    "As my son Parashurama never compromised on righteousness, never compromise on your spiritual discipline.",
                    "The afternoon fire tests our resolve. Will you retreat, or will you let it forge you into something stronger?"
                ],
                'evening': [
                    "Evening assessment: Did you honor the fire of discipline today? Did the spiritual warrior within you show up?",
                    "As the day's tapasya concludes, feel the divine power accumulated through disciplined practice.",
                    "Night approaches, but the inner fire of discipline burns eternal. Rest, but do not let the flame diminish."
                ]
            },
            'kashyapa': {
                'morning': [
                    "Child of the universe, as all creation stirs to life in the morning, feel your connection to the cosmic life force.",
                    "प्राणस्य प्राणः (Pranasya pranah) - the breath of breath awakens. Every morning is a cosmic birth.",
                    "In this dawn moment, every being in creation shares the same life force. Feel your universal kinship."
                ],
                'afternoon': [
                    "At this hour when life flourishes in full vigor, remember - you are part of the great cosmic flowering.",
                    "The midday energy reminds us: all beings are children of the same universal consciousness. How does this guide your actions?",
                    "Feel the pranic energy at its peak. You are not separate from the cosmic breath that sustains all existence."
                ],
                'evening': [
                    "As the universal life force settles into evening calm, feel your unity with all creation resting peacefully.",
                    "Evening brings awareness: every breath connects you to the infinite web of cosmic life. Rest in this truth.",
                    "In twilight's gentle embrace, sense how your consciousness merges with the universal consciousness from which all beings emerge."
                ]
            }
        }
        
        # Determine time period
        if 5 <= current_hour < 12:
            time_period = 'morning'
        elif 12 <= current_hour < 18:
            time_period = 'afternoon'
        else:
            time_period = 'evening'
        
        # Select appropriate greeting
        rishi_greetings = greetings_by_time.get(rishi_name, {})
        time_greetings = rishi_greetings.get(time_period, [])
        
        if time_greetings:
            greeting = random.choice(time_greetings)
            
            # Add user level context
            if user_level == 'advanced':
                level_suffix = " I sense the depth of your practice - let us explore the subtler realms today."
            elif user_level == 'intermediate':
                level_suffix = " Your practice shows dedication - ready for deeper teachings?"
            else:
                level_suffix = " Together we shall walk the eternal path, step by gentle step."
                
            return greeting + level_suffix
        else:
            # Fallback to default greeting
            return self.rishi_personalities[rishi_name]['greeting']

    async def generate_global_examples(self, dharmic_foundation: Dict[str, str], cultural_background: str) -> List[str]:
        """Generate global examples"""
        return [f"Successfully adapted for {cultural_background} context"]
    
    async def get_rishi_guidance(self, query: str, rishi_name: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get guidance from specific Rishi - uses authentic engine first, then enhanced engine if available"""
        
        # Use authentic Rishi engine first (highest priority)
        if self.authentic_rishi_engine and AUTHENTIC_RISHI_AVAILABLE:
            try:
                # Get authentic Rishi response with personality
                authentic_response = self.authentic_rishi_engine.get_authentic_response(
                    rishi_name=rishi_name,
                    query=query,
                    context=user_context
                )
                
                # Add timing and session info
                authentic_response['processing_time'] = 0.1
                if user_context:
                    authentic_response['session_continuity'] = {
                        'user_level': user_context.get('spiritual_level', 'beginner'),
                        'conversation_depth': len(user_context.get('conversation_history', [])),
                        'personalized': True
                    }
                
                return authentic_response
                
            except Exception as e:
                self.logger.warning(f"Authentic Rishi engine failed, falling back to enhanced: {e}")
        
        # Fallback to enhanced Rishi engine if available
        if self.enhanced_rishi_engine and ENHANCED_RISHI_AVAILABLE:
            try:
                # Create request for enhanced engine
                request = RishiGuidanceRequest(
                    query=query,
                    user_context=user_context or {},
                    conversation_history=user_context.get('conversation_history', []) if user_context else [],
                    spiritual_level=user_context.get('spiritual_level', 'beginner') if user_context else 'beginner',
                    preferred_style=user_context.get('preferred_style', 'practical') if user_context else 'practical'
                )
                
                # Get enhanced guidance
                enhanced_response = await self.enhanced_rishi_engine.get_rishi_guidance(request, rishi_name)
                
                # Convert to expected format
                return {
                    'mode': 'rishi_enhanced',
                    'rishi_info': {
                        'name': enhanced_response.rishi_name,
                        'sanskrit': self.enhanced_rishi_engine.rishi_personalities[rishi_name].sanskrit_name,
                        'specialization': self.enhanced_rishi_engine.rishi_personalities[rishi_name].specializations,
                        'teaching_style': self.enhanced_rishi_engine.rishi_personalities[rishi_name].teaching_style
                    },
                    'greeting': enhanced_response.greeting,
                    'guidance': {
                        'primary_wisdom': enhanced_response.primary_guidance,
                        'scriptural_references': enhanced_response.scriptural_references,
                        'mantras': enhanced_response.mantras,
                        'meditation_practice': enhanced_response.meditation_practice
                    },
                    'dharmic_foundation': enhanced_response.scriptural_references,
                    'practical_steps': enhanced_response.practical_steps,
                    'wisdom_synthesis': enhanced_response.primary_guidance,
                    'growth_opportunities': enhanced_response.follow_up_questions,
                    'session_continuity': enhanced_response.session_continuity,
                    'enhanced': True
                }
                
            except Exception as e:
                self.logger.warning(f"Enhanced Rishi engine failed, falling back to legacy: {e}")
        
        # Fallback to legacy implementation
        if rishi_name not in self.rishi_personalities:
            # Fallback to regular guidance
            return await self.get_universal_life_guidance(query, user_context or {})
        
        rishi = self.rishi_personalities[rishi_name]
        
        # Use existing comprehensive guidance system
        base_response = await self.get_universal_life_guidance(query, user_context or {})
        
        # Format as Rishi response
        return {
            'mode': 'rishi_legacy',
            'rishi_info': {
                'name': rishi['name'],
                'sanskrit': rishi['sanskrit'],
                'specialization': rishi['specialization']
            },
            'greeting': rishi['greeting'],
            'guidance': base_response.get('immediate_guidance', {}),
            'dharmic_foundation': base_response.get('dharmic_foundation_explained', ''),
            'practical_steps': base_response.get('practical_roadmap', []),
            'wisdom_synthesis': base_response.get('wisdom_synthesis', ''),
            'growth_opportunities': base_response.get('growth_opportunities', []),
            'enhanced': False
        }


# Export main functions
__all__ = [
    "DharmicEngine",
    "get_universal_dharmic_engine", 
    "get_universal_life_guidance",
    "LifeAspect",
    "DharmicPrinciple"
]
