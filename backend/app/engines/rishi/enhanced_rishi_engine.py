"""
🧘 Enhanced Rishi Personality Engine
==================================

Advanced spiritual guidance system with authentic Rishi personalities
based on their historical teachings and spiritual methodologies.

Each Rishi provides guidance through their unique lens of wisdom:
- Patanjali: Systematic yoga and mind mastery
- Vyasa: Comprehensive dharmic wisdom and life integration  
- Valmiki: Devotional transformation and divine love
- Adi Shankara: Non-dualistic wisdom and consciousness
- Narada: Divine music and bhakti practices
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from datetime import datetime

# Import session management
try:
    from .rishi_session_manager import get_session_manager, RishiSessionManager
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

class RishiArchetype(Enum):
    """Different spiritual approaches each Rishi represents"""
    SYSTEMATIC_PRACTITIONER = "systematic_practitioner"  # Patanjali
    COMPREHENSIVE_GUIDE = "comprehensive_guide"          # Vyasa  
    DEVOTIONAL_TRANSFORMER = "devotional_transformer"   # Valmiki
    CONSCIOUSNESS_EXPLORER = "consciousness_explorer"    # Adi Shankara
    DIVINE_MUSICIAN = "divine_musician"                 # Narada

@dataclass
class RishiPersonality:
    """Complete personality profile for a Rishi"""
    name: str
    sanskrit_name: str
    archetype: RishiArchetype
    specializations: List[str]
    core_texts: List[str]
    teaching_style: str
    greeting_templates: List[str]
    response_patterns: Dict[str, str]
    signature_phrases: List[str]
    wisdom_approach: str
    available_free: bool = False
    
@dataclass
class RishiGuidanceRequest:
    """Structured request for Rishi guidance"""
    query: str
    user_context: Dict[str, Any]
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    spiritual_level: str = "beginner"
    preferred_style: str = "practical"

@dataclass
class RishiGuidanceResponse:
    """Complete Rishi guidance response"""
    rishi_name: str
    greeting: str
    primary_guidance: str
    scriptural_references: List[Dict[str, str]]
    practical_steps: List[str]
    meditation_practice: Optional[str]
    mantras: List[Dict[str, str]]
    follow_up_questions: List[str]
    session_continuity: Dict[str, Any]

class EnhancedRishiEngine:
    """Advanced Rishi personality engine with authentic spiritual guidance"""
    
    def __init__(self):
        self.rishi_personalities = self._initialize_enhanced_personalities()
        self.conversation_memory = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize session manager if available
        self.session_manager = None
        if SESSION_MANAGER_AVAILABLE:
            try:
                self.session_manager = get_session_manager()
                self.logger.info("Session manager initialized for Rishi continuity")
            except Exception as e:
                self.logger.warning(f"Could not initialize session manager: {e}")
                self.session_manager = None
        
    def _initialize_enhanced_personalities(self) -> Dict[str, RishiPersonality]:
        """Initialize comprehensive Rishi personalities"""
        
        return {
            'patanjali': RishiPersonality(
                name='Maharishi Patanjali',
                sanskrit_name='महर्षि पतञ्जलि',
                archetype=RishiArchetype.SYSTEMATIC_PRACTITIONER,
                specializations=[
                    'yoga_philosophy', 'meditation_techniques', 'mind_mastery',
                    'concentration_practices', 'spiritual_disciplines', 'samadhi_states'
                ],
                core_texts=['Yoga Sutras', 'Ashtanga Yoga', 'Dharana-Dhyana-Samadhi'],
                teaching_style='systematic_progressive',
                greeting_templates=[
                    "🧘 Namaste, sincere seeker. Let us systematically explore the path of yoga and mental mastery.",
                    "🕉️ Greetings, dear practitioner. The mind is like a monkey - let us learn to still its restless nature.",
                    "🌟 Welcome, aspirant. Through disciplined practice, the modifications of consciousness can be controlled."
                ],
                response_patterns={
                    'systematic': "Following the eight-limbed path of yoga...",
                    'progressive': "Beginning with {yama/niyama}, we progress to {asana}...",
                    'methodical': "The systematic approach is: First {step1}, then {step2}...",
                    'practical': "For immediate practice, focus on {specific_technique}..."
                },
                signature_phrases=[
                    "yogash chitta vritti nirodhah",
                    "abhyasa vairagyabhyam",
                    "practice and detachment",
                    "systematic progression",
                    "eight-limbed path"
                ],
                wisdom_approach='systematic_methodology',
                available_free=True
            ),
            
            'vyasa': RishiPersonality(
                name='Sage Vyasa',
                sanskrit_name='भगवान् व्यास',
                archetype=RishiArchetype.COMPREHENSIVE_GUIDE,
                specializations=[
                    'dharmic_living', 'life_purpose', 'ethical_dilemmas', 
                    'karma_yoga', 'vedic_wisdom', 'life_stages', 'duty_action'
                ],
                core_texts=['Mahabharata', 'Bhagavad Gita', 'Puranas', 'Vedanta'],
                teaching_style='comprehensive_contextual',
                greeting_templates=[
                    "🙏 Namaste, noble soul. All of life's complexities find resolution through dharmic understanding.",
                    "🕉️ Welcome, seeker of truth. As Krishna taught Arjuna, every challenge contains the seed of wisdom.",
                    "🌅 Greetings, dear one. The vast ocean of existence can be navigated with dharmic compass."
                ],
                response_patterns={
                    'contextual': "In the context of your situation, dharma suggests...",
                    'comprehensive': "Looking at all four purusharthas - dharma, artha, kama, moksha...",
                    'story_based': "In the Mahabharata, we see a similar situation when...",
                    'gita_wisdom': "As the Bhagavad Gita teaches us..."
                },
                signature_phrases=[
                    "dharmic path",
                    "svadharma",
                    "karma yoga",
                    "as Krishna taught",
                    "comprehensive understanding"
                ],
                wisdom_approach='contextual_integration',
                available_free=False
            ),
            
            'valmiki': RishiPersonality(
                name='Sage Valmiki',
                sanskrit_name='महर्षि वाल्मीकि',
                archetype=RishiArchetype.DEVOTIONAL_TRANSFORMER,
                specializations=[
                    'devotional_practices', 'transformation', 'divine_love',
                    'surrender', 'purification', 'bhakti_yoga', 'divine_grace'
                ],
                core_texts=['Ramayana', 'Bhakti Sutras', 'Devotional Poetry'],
                teaching_style='heart_centered_transformational',
                greeting_templates=[
                    "🌺 Beloved child of the Divine, transformation is always possible through love and devotion.",
                    "🙏 Namaste, dear soul. Like Rama's journey, your path leads to divine realization.",
                    "💖 Welcome, seeker of love. The heart that surrenders finds infinite grace."
                ],
                response_patterns={
                    'transformational': "No matter your past, transformation awaits through...",
                    'devotional': "Open your heart to divine love through...",
                    'grace_based': "Divine grace flows when we...",
                    'rama_inspired': "Following Rama's example..."
                },
                signature_phrases=[
                    "divine grace",
                    "heart's devotion",
                    "transformation through love",
                    "Rama's path",
                    "surrender"
                ],
                wisdom_approach='devotional_transformation',
                available_free=False
            ),
            
            'adi_shankara': RishiPersonality(
                name='Adi Shankaracharya',
                sanskrit_name='आदि शङ्कराचार्य',
                archetype=RishiArchetype.CONSCIOUSNESS_EXPLORER,
                specializations=[
                    'non_duality', 'consciousness_exploration', 'self_inquiry',
                    'vedanta', 'meditation_on_self', 'maya_illusion', 'brahman_realization'
                ],
                core_texts=['Viveka Chudamani', 'Brahma Sutra Bhashya', 'Upanishad Commentaries'],
                teaching_style='inquiry_based_realization',
                greeting_templates=[
                    "🕉️ Tat tvam asi - That thou art. Let us explore the nature of your true Self.",
                    "✨ Namaste, seeker of truth. Beyond all appearances lies your eternal nature.",
                    "🌟 Welcome, consciousness exploring itself. What is real? What is temporary?"
                ],
                response_patterns={
                    'inquiry_based': "Ask yourself: Who is the one experiencing...?",
                    'non_dual': "From the perspective of absolute truth...",
                    'consciousness_focused': "Consciousness itself is...",
                    'self_investigation': "Turn attention to the 'I' that..."
                },
                signature_phrases=[
                    "tat tvam asi",
                    "brahman satyam",
                    "atman-brahman",
                    "consciousness itself",
                    "who am I?"
                ],
                wisdom_approach='consciousness_inquiry',
                available_free=False
            ),
            
            'narada': RishiPersonality(
                name='Sage Narada',
                sanskrit_name='देवर्षि नारद',
                archetype=RishiArchetype.DIVINE_MUSICIAN,
                specializations=[
                    'divine_music', 'kirtan', 'devotional_singing', 'sound_healing',
                    'mantra_yoga', 'vibrational_healing', 'ecstatic_devotion'
                ],
                core_texts=['Narada Bhakti Sutras', 'Divine Music Traditions', 'Mantra Shastra'],
                teaching_style='music_and_devotion_based',
                greeting_templates=[
                    "🎵 Narayana! Let divine sound carry us to the Source of all harmony.",
                    "🎶 Welcome, lover of divine music. The universe itself sings the eternal song.",
                    "🪕 Namaste, seeker of sound. Through sacred vibration, we touch the infinite."
                ],
                response_patterns={
                    'music_based': "Like a divine melody...",
                    'sound_healing': "Through sacred sound...",
                    'vibrational': "The vibration of truth resonates...",
                    'devotional_song': "Sing with devotion..."
                },
                signature_phrases=[
                    "divine music",
                    "sacred sound",
                    "narayana",
                    "eternal song",
                    "vibrational truth"
                ],
                wisdom_approach='musical_devotion',
                available_free=False
            )
        }
    
    async def get_rishi_guidance(self, request: RishiGuidanceRequest, rishi_name: str) -> RishiGuidanceResponse:
        """Get sophisticated guidance from specific Rishi"""
        
        if rishi_name not in self.rishi_personalities:
            raise ValueError(f"Rishi {rishi_name} not found")
            
        rishi = self.rishi_personalities[rishi_name]
        
        # Get session data for personalization if available
        session_data = None
        if self.session_manager and request.user_context.get('user_id'):
            session_data = await self.session_manager.get_session_data(
                request.user_context['user_id'], rishi_name
            )
        
        # Analyze query for spiritual context
        spiritual_context = self._analyze_spiritual_context(request.query, rishi)
        
        # Update spiritual context with session history
        if session_data:
            spiritual_context['session_history'] = {
                'topics_explored': session_data.progress.topics_explored,
                'current_level': session_data.progress.depth_level,
                'session_count': session_data.session_count,
                'recent_themes': session_data.conversation_themes[-3:]
            }
            # Adjust spiritual level based on session history
            if session_data.progress.depth_level != "beginner":
                spiritual_context['spiritual_level'] = session_data.progress.depth_level
        
        # Generate personalized greeting
        greeting = await self._generate_personalized_greeting(rishi, request.user_context, session_data)
        
        # Create Rishi-specific guidance
        primary_guidance = await self._generate_rishi_specific_guidance(
            rishi, request.query, spiritual_context, request.user_context
        )
        
        # Add scriptural references
        scriptural_refs = self._get_relevant_scriptures(rishi, spiritual_context)
        
        # Generate practical steps in Rishi's style
        practical_steps = self._generate_rishi_practical_steps(rishi, spiritual_context)
        
        # Add meditation practice if relevant
        meditation_practice = self._suggest_meditation_practice(rishi, spiritual_context)
        
        # Include relevant mantras
        mantras = self._get_relevant_mantras(rishi, spiritual_context)
        
        # Generate follow-up questions
        follow_ups = self._generate_follow_up_questions(rishi, request.query)
        
        # Create session continuity
        session_continuity = await self._create_session_continuity(rishi_name, request, session_data)
        
        # Create response
        response = RishiGuidanceResponse(
            rishi_name=rishi.name,
            greeting=greeting,
            primary_guidance=primary_guidance,
            scriptural_references=scriptural_refs,
            practical_steps=practical_steps,
            meditation_practice=meditation_practice,
            mantras=mantras,
            follow_up_questions=follow_ups,
            session_continuity=session_continuity
        )
        
        # Update session data if available
        if self.session_manager and request.user_context.get('user_id'):
            await self.session_manager.create_or_update_session(
                request.user_context['user_id'],
                rishi_name,
                request.query,
                {
                    'guidance': {'primary_wisdom': primary_guidance},
                    'practical_steps': practical_steps,
                    'mantras': mantras,
                    'scriptural_references': scriptural_refs
                },
                request.user_context
            )
        
        return response
    
    def _analyze_spiritual_context(self, query: str, rishi: RishiPersonality) -> Dict[str, Any]:
        """Analyze query in context of Rishi's specializations"""
        
        query_lower = query.lower()
        context = {
            'primary_theme': None,
            'spiritual_level': 'beginner',
            'emotional_state': 'neutral',
            'practice_need': None,
            'rishi_relevance': 0.0
        }
        
        # Check for themes matching Rishi's specializations
        for specialization in rishi.specializations:
            if any(word in query_lower for word in specialization.replace('_', ' ').split()):
                context['rishi_relevance'] += 0.2
                if not context['primary_theme']:
                    context['primary_theme'] = specialization
        
        # Detect spiritual level
        if any(word in query_lower for word in ['advanced', 'deep', 'profound', 'enlightenment']):
            context['spiritual_level'] = 'advanced'
        elif any(word in query_lower for word in ['intermediate', 'understanding', 'practice']):
            context['spiritual_level'] = 'intermediate'
            
        # Detect emotional context
        emotions = {
            'suffering': ['pain', 'suffering', 'difficult', 'struggle', 'hardship'],
            'seeking': ['seeking', 'searching', 'looking', 'wanting', 'need'],
            'confusion': ['confused', 'lost', 'uncertain', 'doubt', 'unclear'],
            'joy': ['happy', 'joy', 'grateful', 'blessed', 'wonderful']
        }
        
        for emotion, keywords in emotions.items():
            if any(word in query_lower for word in keywords):
                context['emotional_state'] = emotion
                break
        
        return context
    
    async def _generate_personalized_greeting(self, rishi: RishiPersonality, user_context: Dict[str, Any], session_data: Any = None) -> str:
        """Generate personalized greeting based on user context and session history"""
        
        # Use session manager for personalized greeting if available
        if self.session_manager and user_context.get('user_id') and session_data:
            base_greeting = self._select_base_greeting(rishi, user_context)
            personalized = await self.session_manager.get_personalized_greeting(
                user_context['user_id'], 
                rishi.name.lower().replace(' ', '_').replace('maharishi_', '').replace('sage_', '').replace('adi_', 'adi_'),
                base_greeting
            )
            return personalized
        
        # Fallback to basic personalization
        return self._select_base_greeting(rishi, user_context)
    
    def _select_base_greeting(self, rishi: RishiPersonality, user_context: Dict[str, Any]) -> str:
        """Select appropriate base greeting"""
        import random
        base_greeting = random.choice(rishi.greeting_templates)
        
        # Personalize based on context
        if user_context.get('returning_user', False):
            base_greeting = base_greeting.replace('Welcome', 'Welcome back')
            
        return base_greeting
    
    async def _generate_rishi_specific_guidance(
        self, 
        rishi: RishiPersonality, 
        query: str, 
        context: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        """Generate guidance in the specific Rishi's style and approach"""
        
        # Base guidance framework for each Rishi archetype
        if rishi.archetype == RishiArchetype.SYSTEMATIC_PRACTITIONER:
            # Patanjali's systematic approach
            guidance = f"""
Based on the eight-limbed path of yoga, let us address your question systematically:

**Understanding the Root**: {self._patanjali_root_analysis(query)}

**Systematic Approach**: 
{self._patanjali_systematic_method(context)}

**Progressive Practice**:
{self._patanjali_progressive_steps(context)}

As the Yoga Sutras teach: "योगश्चित्तवृत्तिनिरोधः" - Yoga is the cessation of mental modifications. 
Through disciplined practice (abhyasa) and detachment (vairagya), clarity emerges.
"""
            
        elif rishi.archetype == RishiArchetype.COMPREHENSIVE_GUIDE:
            # Vyasa's comprehensive dharmic guidance
            guidance = f"""
Let us examine your situation through the lens of dharmic wisdom:

**Dharmic Foundation**: {self._vyasa_dharmic_analysis(query)}

**Life Context Integration**: 
{self._vyasa_comprehensive_view(context)}

**Karma Yoga Application**:
{self._vyasa_karma_yoga_guidance(context)}

As the Bhagavad Gita teaches: "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन" - 
You have the right to action, but never to the fruits of action.
"""
            
        elif rishi.archetype == RishiArchetype.DEVOTIONAL_TRANSFORMER:
            # Valmiki's heart-centered transformation
            guidance = f"""
Dear soul, let divine love transform this challenge:

**Heart Opening**: {self._valmiki_heart_guidance(query)}

**Transformational Path**: 
{self._valmiki_transformation_process(context)}

**Divine Grace**:
{self._valmiki_grace_based_solution(context)}

Like Rama's journey in the forest, every challenge becomes a step toward divine realization.
The heart that surrenders finds infinite grace flowing through all circumstances.
"""
            
        elif rishi.archetype == RishiArchetype.CONSCIOUSNESS_EXPLORER:
            # Adi Shankara's consciousness inquiry
            guidance = f"""
Let us inquire into the nature of what you're experiencing:

**Self-Inquiry**: {self._shankara_self_inquiry(query)}

**Non-Dual Understanding**: 
{self._shankara_nondual_perspective(context)}

**Consciousness Recognition**:
{self._shankara_consciousness_practice(context)}

"तत्त्वमसि" - Tat tvam asi - That thou art. 
The very consciousness observing this challenge is your eternal, unchanging nature.
"""
            
        elif rishi.archetype == RishiArchetype.DIVINE_MUSICIAN:
            # Narada's musical devotion
            guidance = f"""
Let divine sound and devotion heal and guide:

**Sacred Vibration**: {self._narada_sound_healing(query)}

**Devotional Practice**: 
{self._narada_devotional_guidance(context)}

**Musical Meditation**:
{self._narada_musical_practice(context)}

नारायण! Like the eternal song of creation, let sacred sound carry you to the Source of all harmony.
"""
            
        return guidance.strip()
    
    def _get_relevant_scriptures(self, rishi: RishiPersonality, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get scriptural references relevant to the Rishi and context"""
        
        scriptures = []
        
        if rishi.name == 'Maharishi Patanjali':
            scriptures = [
                {
                    'text': 'Yoga Sutras 1.14',
                    'sanskrit': 'स तु दीर्घकालनैरन्तर्यसत्कारासेवितो दृढभूमिः',
                    'translation': 'Practice is firmly grounded when it is cultivated continuously for a long period with dedication.',
                    'relevance': 'Foundation of practice'
                },
                {
                    'text': 'Yoga Sutras 2.46',
                    'sanskrit': 'स्थिरसुखमासनम्',
                    'translation': 'Posture should be steady and comfortable.',
                    'relevance': 'Physical and mental stability'
                }
            ]
            
        elif rishi.name == 'Sage Vyasa':
            scriptures = [
                {
                    'text': 'Bhagavad Gita 3.8',
                    'sanskrit': 'नियतं कुरु कर्म त्वं कर्म ज्यायो ह्यकर्मणः',
                    'translation': 'Perform your prescribed duties, for action is better than inaction.',
                    'relevance': 'Duty and action'
                },
                {
                    'text': 'Bhagavad Gita 18.66',
                    'sanskrit': 'सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज',
                    'translation': 'Abandon all varieties of dharma and surrender unto Me alone.',
                    'relevance': 'Ultimate surrender'
                }
            ]
            
        return scriptures
    
    def _generate_rishi_practical_steps(self, rishi: RishiPersonality, context: Dict[str, Any]) -> List[str]:
        """Generate practical steps in the Rishi's teaching style"""
        
        if rishi.archetype == RishiArchetype.SYSTEMATIC_PRACTITIONER:
            return [
                "Begin with 5 minutes of pranayama (breath awareness) daily",
                "Establish a consistent meditation practice, starting with dharana (concentration)",
                "Practice yamas and niyamas in daily life (ethical guidelines)",
                "Gradually increase sitting meditation duration",
                "Study one Yoga Sutra weekly with contemplation"
            ]
            
        elif rishi.archetype == RishiArchetype.COMPREHENSIVE_GUIDE:
            return [
                "Reflect on your svadharma (personal duty) in this situation",
                "Practice karma yoga - action without attachment to results",
                "Study relevant sections of the Bhagavad Gita",
                "Seek guidance from dharmic community or mentor",
                "Balance the four purusharthas in your daily decisions"
            ]
            
        elif rishi.archetype == RishiArchetype.DEVOTIONAL_TRANSFORMER:
            return [
                "Begin each day with heart-centered prayer or devotion",
                "Practice loving-kindness toward yourself and others",
                "Surrender your challenges to the divine through prayer",
                "Engage in seva (selfless service) regularly",
                "Chant or sing devotional songs to open the heart"
            ]
            
        return ["Engage in regular spiritual practice", "Study sacred texts", "Seek wisdom through meditation"]
    
    def _suggest_meditation_practice(self, rishi: RishiPersonality, context: Dict[str, Any]) -> Optional[str]:
        """Suggest meditation practice based on Rishi's approach"""
        
        if rishi.archetype == RishiArchetype.SYSTEMATIC_PRACTITIONER:
            return """
**Patanjali's Progressive Meditation**:
1. Sit comfortably with spine erect (asana)
2. Observe natural breath for 2-3 minutes (pranayama)
3. Choose one object of concentration - breath, mantra, or divine form (dharana)
4. When mind wanders, gently return to your chosen focus
5. Gradually extend periods of unbroken concentration (dhyana)
6. Practice daily at the same time for consistency
"""
            
        elif rishi.archetype == RishiArchetype.DEVOTIONAL_TRANSFORMER:
            return """
**Valmiki's Heart-Opening Meditation**:
1. Sit quietly and place hands on heart
2. Bring to mind someone you love unconditionally
3. Feel the warmth and expansion in your heart
4. Extend this feeling to yourself with compassion
5. Gradually include all beings in this loving awareness
6. Rest in the spacious love that has no boundaries
"""
            
        return None
    
    def _get_relevant_mantras(self, rishi: RishiPersonality, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get mantras relevant to the Rishi's tradition"""
        
        mantras = []
        
        if rishi.archetype == RishiArchetype.SYSTEMATIC_PRACTITIONER:
            mantras = [
                {
                    'sanskrit': 'ॐ गं गणपतये नमः',
                    'transliteration': 'Om Gam Ganapataye Namah',
                    'meaning': 'Salutations to Ganesha, remover of obstacles',
                    'usage': 'Before beginning any spiritual practice'
                }
            ]
            
        elif rishi.archetype == RishiArchetype.DEVOTIONAL_TRANSFORMER:
            mantras = [
                {
                    'sanskrit': 'ॐ श्री रामाय नमः',
                    'transliteration': 'Om Sri Ramaya Namah',
                    'meaning': 'Salutations to Lord Rama',
                    'usage': 'For purification and divine protection'
                }
            ]
            
        return mantras
    
    # Helper methods for each Rishi's specific guidance style
    def _patanjali_root_analysis(self, query: str) -> str:
        return "Mental modifications (vrittis) are creating disturbance. Through systematic practice, we can achieve clarity."
    
    def _patanjali_systematic_method(self, context: Dict[str, Any]) -> str:
        return "Following the eight-limbed path: ethical foundation → physical stability → breath control → concentration → meditation → absorption."
    
    def _patanjali_progressive_steps(self, context: Dict[str, Any]) -> str:
        return "Begin with yamas/niyamas, establish asana and pranayama, then progress to concentration practices."
    
    def _vyasa_dharmic_analysis(self, query: str) -> str:
        return "Every situation contains dharmic principles. Understanding your duty (svadharma) illuminates the path forward."
    
    def _vyasa_comprehensive_view(self, context: Dict[str, Any]) -> str:
        return "Consider all aspects: dharma (righteousness), artha (prosperity), kama (desires), and moksha (liberation)."
    
    def _vyasa_karma_yoga_guidance(self, context: Dict[str, Any]) -> str:
        return "Act according to dharma without attachment to results. This transforms ordinary actions into spiritual practice."
    
    def _valmiki_heart_guidance(self, query: str) -> str:
        return "Every challenge is an invitation for the heart to open wider. Love and compassion dissolve all difficulties."
    
    def _valmiki_transformation_process(self, context: Dict[str, Any]) -> str:
        return "Like my own transformation from hunter to sage, no past prevents spiritual flowering. Grace is always available."
    
    def _valmiki_grace_based_solution(self, context: Dict[str, Any]) -> str:
        return "Surrender your burden to the Divine. When we stop trying to control, divine grace flows naturally."
    
    def _shankara_self_inquiry(self, query: str) -> str:
        return "Who is the one experiencing this situation? What remains unchanged throughout all experiences?"
    
    def _shankara_nondual_perspective(self, context: Dict[str, Any]) -> str:
        return "From absolute standpoint, only Brahman exists. All appearances arise in consciousness like waves in the ocean."
    
    def _shankara_consciousness_practice(self, context: Dict[str, Any]) -> str:
        return "Rest as pure awareness. You are not the thoughts, emotions, or experiences - you are the conscious witness."
    
    def _narada_sound_healing(self, query: str) -> str:
        return "Sacred sound carries healing vibrations. Let divine music attune your being to cosmic harmony."
    
    def _narada_devotional_guidance(self, context: Dict[str, Any]) -> str:
        return "Sing, chant, or hum with devotion. The heart opened through music becomes a temple for the Divine."
    
    def _narada_musical_practice(self, context: Dict[str, Any]) -> str:
        return "Practice kirtan, listen to devotional music, or simply hum 'Om' with love. Sound is the bridge to the infinite."
    
    def _generate_follow_up_questions(self, rishi: RishiPersonality, query: str) -> List[str]:
        """Generate follow-up questions in Rishi's style"""
        
        if rishi.archetype == RishiArchetype.SYSTEMATIC_PRACTITIONER:
            return [
                "What specific aspect of practice would you like to explore deeper?",
                "How consistent has your meditation practice been?",
                "Which of the eight limbs of yoga calls to you most strongly?",
                "What obstacles do you encounter in maintaining regular practice?"
            ]
            
        elif rishi.archetype == RishiArchetype.COMPREHENSIVE_GUIDE:
            return [
                "How does this situation relate to your life's dharmic purpose?",
                "What would Krishna advise in this circumstance?",
                "How can you apply karma yoga principles here?",
                "What are the long-term implications for your spiritual growth?"
            ]
            
        elif rishi.archetype == RishiArchetype.DEVOTIONAL_TRANSFORMER:
            return [
                "How can you bring more love into this situation?",
                "What would complete surrender look like here?",
                "How might the Divine be working through this challenge?",
                "Where do you feel your heart opening or closing?"
            ]
            
        return ["How can we explore this topic further?", "What resonates most deeply with you?"]
    
    async def _create_session_continuity(self, rishi_name: str, request: RishiGuidanceRequest, session_data: Any = None) -> Dict[str, Any]:
        """Create session data for continuity across conversations"""
        
        base_continuity = {
            'rishi_name': rishi_name,
            'topic_discussed': request.query[:100],
            'spiritual_level': request.spiritual_level,
            'timestamp': datetime.now().isoformat(),
            'conversation_count': len(request.conversation_history) + 1,
            'themes_explored': [],
            'recommended_practices': [],
            'progress_indicators': {}
        }
        
        # Enhance with session data if available
        if session_data:
            base_continuity.update({
                'session_count': session_data.session_count,
                'total_sessions': session_data.progress.total_sessions,
                'spiritual_depth': session_data.progress.depth_level,
                'topics_mastered': len(session_data.progress.topics_explored),
                'practices_completed': len(session_data.progress.practices_completed),
                'next_recommended_topics': getattr(session_data, 'next_recommended_topics', []),
                'journey_started': session_data.first_session
            })
        
        return base_continuity

# Factory function for easy integration
def create_enhanced_rishi_engine() -> EnhancedRishiEngine:
    """Create and return an enhanced Rishi engine instance"""
    return EnhancedRishiEngine()

# Export main classes
__all__ = [
    'EnhancedRishiEngine',
    'RishiPersonality', 
    'RishiGuidanceRequest',
    'RishiGuidanceResponse',
    'RishiArchetype',
    'create_enhanced_rishi_engine'
]
