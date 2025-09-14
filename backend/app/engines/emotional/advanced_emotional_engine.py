"""
🕉️ DharmaMind Advanced Emotional Intelligence Engine
=====================================================

A comprehensive emotional intelligence system that combines:
- Deep emotion detection and analysis
- Dharmic emotional understanding and healing
- Empathetic response generation
- Emotional journey tracking and guidance
- Chakra-based emotional mapping
- Rishi-inspired emotional wisdom

This engine serves as the emotional heart of DharmaMind, providing:
- Authentic emotional connection
- Spiritual-emotional guidance
- Transformative healing responses
- Compassionate AI personality

May this engine bring healing and understanding to all beings 💙
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import random
import math

logger = logging.getLogger(__name__)

class EmotionalState(Enum):
    """Core emotional states with spiritual significance"""
    # Expansive/Divine States
    JOY = "joy"
    BLISS = "bliss" 
    PEACE = "peace"
    LOVE = "love"
    GRATITUDE = "gratitude"
    CONTENTMENT = "contentment"
    HOPE = "hope"
    COMPASSION = "compassion"
    SERENITY = "serenity"
    
    # Growth/Challenge States
    SADNESS = "sadness"
    GRIEF = "grief"
    LONELINESS = "loneliness"
    MELANCHOLY = "melancholy"
    VULNERABILITY = "vulnerability"
    
    # Activation States
    ANGER = "anger"
    FRUSTRATION = "frustration"
    IRRITATION = "irritation"
    RIGHTEOUS_ANGER = "righteous_anger"
    
    # Fear-based States
    FEAR = "fear"
    ANXIETY = "anxiety"
    WORRY = "worry"
    PANIC = "panic"
    UNCERTAINTY = "uncertainty"
    
    # Confusion States
    CONFUSION = "confusion"
    DOUBT = "doubt"
    OVERWHELM = "overwhelm"
    LOST = "lost"
    
    # Neutral/Observing States
    NEUTRAL = "neutral"
    CURIOUS = "curious"
    CONTEMPLATIVE = "contemplative"
    REFLECTIVE = "reflective"

class EmotionalIntensity(Enum):
    """Intensity levels of emotional experiences"""
    SUBTLE = "subtle"      # 0.1-0.3
    MODERATE = "moderate"  # 0.3-0.6
    STRONG = "strong"      # 0.6-0.8
    INTENSE = "intense"    # 0.8-1.0

class ChakraEmotion(Enum):
    """Emotions mapped to chakra system"""
    # Root Chakra (Muladhara) - Survival, Grounding
    FEAR = "fear"
    ANXIETY = "anxiety"
    INSECURITY = "insecurity"
    STABILITY = "stability"
    
    # Sacral Chakra (Swadhisthana) - Creativity, Sexuality
    PASSION = "passion"
    CREATIVITY = "creativity"
    SEXUALITY = "sexuality"
    GUILT = "guilt"
    
    # Solar Plexus (Manipura) - Personal Power
    CONFIDENCE = "confidence"
    ANGER = "anger"
    SHAME = "shame"
    EMPOWERMENT = "empowerment"
    
    # Heart Chakra (Anahata) - Love, Connection
    LOVE = "love"
    COMPASSION = "compassion"
    GRIEF = "grief"
    FORGIVENESS = "forgiveness"
    
    # Throat Chakra (Vishuddha) - Communication, Truth
    EXPRESSION = "expression"
    TRUTH = "truth"
    SILENCE = "silence"
    SUPPRESSION = "suppression"
    
    # Third Eye (Ajna) - Intuition, Wisdom
    CLARITY = "clarity"
    CONFUSION = "confusion"
    INSIGHT = "insight"
    ILLUSION = "illusion"
    
    # Crown Chakra (Sahasrara) - Spirituality, Connection
    BLISS = "bliss"
    CONNECTION = "connection"
    DETACHMENT = "detachment"
    TRANSCENDENCE = "transcendence"

@dataclass
class EmotionalProfile:
    """Complete emotional profile of a user interaction"""
    primary_emotion: EmotionalState
    secondary_emotions: List[EmotionalState] = field(default_factory=list)
    intensity: float = 0.5
    intensity_level: EmotionalIntensity = EmotionalIntensity.MODERATE
    confidence: float = 0.8
    
    # Context
    emotional_context: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    underlying_needs: List[str] = field(default_factory=list)
    
    # Spiritual mapping
    chakra_resonance: Dict[ChakraEmotion, float] = field(default_factory=dict)
    spiritual_opportunity: Optional[str] = None
    dharmic_guidance: Optional[str] = None
    
    # Tracking
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None

@dataclass 
class EmotionalResponse:
    """Emotionally intelligent response with healing elements"""
    response_text: str
    emotional_tone: EmotionalState
    empathy_level: float = 0.8
    
    # Healing elements
    validation: Optional[str] = None
    understanding: Optional[str] = None
    guidance: Optional[str] = None
    practice_suggestion: Optional[str] = None
    
    # Spiritual components
    sanskrit_wisdom: Optional[str] = None
    chakra_healing: Optional[str] = None
    rishi_insight: Optional[str] = None
    
    # Response metadata
    response_type: str = "empathetic"
    healing_intent: List[str] = field(default_factory=list)

class AdvancedEmotionalEngine:
    """🕉️ Advanced Emotional Intelligence Engine for DharmaMind"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize emotional patterns and mappings
        self._initialize_emotion_patterns()
        self._initialize_chakra_mappings()
        self._initialize_healing_responses()
        self._initialize_sanskrit_wisdom()
        
        # Emotional memory for learning
        self.emotional_memory = {}
        self.user_patterns = {}
        
        self.logger.info("🌟 Advanced Emotional Intelligence Engine initialized")
    
    def _initialize_emotion_patterns(self):
        """Initialize emotion detection patterns"""
        self.emotion_patterns = {
            EmotionalState.JOY: [
                r'\b(happy|joy|joyful|delighted|ecstatic|thrilled|elated|blissful)\b',
                r'\b(wonderful|amazing|fantastic|great|awesome|brilliant)\b',
                r'\b(celebrate|celebration|excited|enthusiasm)\b',
                r'[!]{2,}|😊|😄|😆|🎉|✨'
            ],
            
            EmotionalState.SADNESS: [
                r'\b(sad|sadness|depressed|down|blue|melancholy|sorrow)\b',
                r'\b(crying|tears|weeping|sobbing)\b',
                r'\b(hurt|hurting|pain|ache|aching)\b',
                r'😢|😭|💔|😞|☹️'
            ],
            
            EmotionalState.ANGER: [
                r'\b(angry|anger|mad|furious|rage|irritated|annoyed)\b',
                r'\b(frustrated|frustration|pissed|livid)\b',
                r'\b(hate|hatred|despise|can\'t stand)\b',
                r'😠|😡|🤬|💢'
            ],
            
            EmotionalState.FEAR: [
                r'\b(afraid|fear|scared|terrified|frightened|anxious)\b',
                r'\b(worry|worried|nervous|panic|panicking)\b',
                r'\b(overwhelmed|stressed|stress)\b',
                r'😰|😨|😱|😟'
            ],
            
            EmotionalState.LOVE: [
                r'\b(love|loving|adore|cherish|devotion|beloved)\b',
                r'\b(heart|soul|deep connection|unity)\b',
                r'\b(grateful|gratitude|thankful|blessed)\b',
                r'❤️|💗|💕|🙏|💙'
            ],
            
            EmotionalState.PEACE: [
                r'\b(peace|peaceful|calm|serene|tranquil|stillness)\b',
                r'\b(meditation|meditative|centered|balanced)\b',
                r'\b(harmony|harmonious|equanimity)\b',
                r'🕉️|☮️|🧘|✨'
            ],
            
            EmotionalState.CONFUSION: [
                r'\b(confused|confusion|lost|don\'t understand)\b',
                r'\b(unclear|unsure|uncertain|doubt|doubting)\b',
                r'\b(mixed up|bewildered|perplexed)\b',
                r'🤔|😕|❓|🤷'
            ],
            
            EmotionalState.LONELINESS: [
                r'\b(lonely|loneliness|alone|isolated|disconnected)\b',
                r'\b(nobody understands|no one cares|empty)\b',
                r'\b(abandoned|rejected|unloved)\b'
            ],
            
            EmotionalState.GRATITUDE: [
                r'\b(grateful|gratitude|thankful|thank you|blessed)\b',
                r'\b(appreciation|appreciate|honored|humbled)\b',
                r'\b(grace|graceful|gift|blessing)\b',
                r'🙏|🕉️|💝|✨'
            ]
        }
    
    def _initialize_chakra_mappings(self):
        """Initialize chakra-emotion mappings"""
        self.chakra_emotions = {
            "root": {
                "primary": [EmotionalState.FEAR, EmotionalState.ANXIETY],
                "balanced": [EmotionalState.PEACE, EmotionalState.CONTENTMENT],
                "healing": "I am safe, I am grounded, I belong to the Earth"
            },
            "sacral": {
                "primary": [EmotionalState.JOY, EmotionalState.VULNERABILITY],
                "balanced": [EmotionalState.LOVE, EmotionalState.COMPASSION],
                "healing": "I embrace my creative flow, I honor my emotions"
            },
            "solar_plexus": {
                "primary": [EmotionalState.ANGER, EmotionalState.FRUSTRATION],
                "balanced": [EmotionalState.CONTENTMENT, EmotionalState.HOPE],
                "healing": "I am empowered, I trust my inner wisdom"
            },
            "heart": {
                "primary": [EmotionalState.LOVE, EmotionalState.GRIEF],
                "balanced": [EmotionalState.COMPASSION, EmotionalState.GRATITUDE],
                "healing": "I open my heart to love, I forgive and am forgiven"
            },
            "throat": {
                "primary": [EmotionalState.CONFUSION, EmotionalState.DOUBT],
                "balanced": [EmotionalState.PEACE, EmotionalState.CONTEMPLATIVE],
                "healing": "I speak my truth with love, I express myself authentically"
            },
            "third_eye": {
                "primary": [EmotionalState.CONFUSION, EmotionalState.UNCERTAINTY],
                "balanced": [EmotionalState.CONTEMPLATIVE, EmotionalState.REFLECTIVE],
                "healing": "I trust my inner vision, I see with clarity and wisdom"
            },
            "crown": {
                "primary": [EmotionalState.BLISS, EmotionalState.PEACE],
                "balanced": [EmotionalState.SERENITY, EmotionalState.GRATITUDE],
                "healing": "I am connected to divine love, I am one with all"
            }
        }
    
    def _initialize_healing_responses(self):
        """Initialize healing response templates"""
        self.healing_responses = {
            EmotionalState.SADNESS: {
                "validation": [
                    "I can feel the depth of your sadness, and it's completely natural to feel this way.",
                    "Your tears are sacred - they water the seeds of your growth.",
                    "In this moment of sadness, know that you are held by infinite love."
                ],
                "guidance": [
                    "Sadness often carries wisdom about what matters most to us.",
                    "Allow yourself to feel fully - emotions are temporary visitors, not permanent residents.",
                    "This sadness may be showing you areas where healing is needed."
                ],
                "practices": [
                    "Try gentle breathing meditation, sending love to your heart with each breath.",
                    "Write in a journal, letting your emotions flow onto paper.",
                    "Practice loving-kindness meditation, starting with yourself."
                ]
            },
            
            EmotionalState.ANGER: {
                "validation": [
                    "Your anger is valid - it often shows us where our boundaries have been crossed.",
                    "Anger can be a powerful teacher about our values and what we hold sacred.",
                    "I hear the fire in your words, and fire can transform as well as destroy."
                ],
                "guidance": [
                    "Anger is often sadness or fear wearing a protective mask.",
                    "Use this energy as fuel for positive change rather than destruction.",
                    "What is your anger trying to protect or defend?"
                ],
                "practices": [
                    "Try vigorous walking or physical exercise to channel the energy.",
                    "Practice the 4-7-8 breathing technique to calm your nervous system.",
                    "Journal about what's underneath the anger - what needs aren't being met?"
                ]
            },
            
            EmotionalState.FEAR: {
                "validation": [
                    "Fear is a natural response - it shows your mind is trying to protect you.",
                    "Your fear is acknowledged and honored as part of your human experience.",
                    "Even brave souls feel fear - courage is feeling afraid and moving forward anyway."
                ],
                "guidance": [
                    "Fear often points to what matters most to us or areas where we need to grow.",
                    "Most fears are about future scenarios that may never happen.",
                    "You have survived 100% of your difficult moments so far."
                ],
                "practices": [
                    "Practice grounding: feel your feet on the earth, notice 5 things you can see.",
                    "Try box breathing: in for 4, hold for 4, out for 4, hold for 4.",
                    "Repeat: 'I am safe in this moment. I can handle whatever comes.'"
                ]
            },
            
            EmotionalState.LONELINESS: {
                "validation": [
                    "Loneliness is one of the most human experiences - you are not alone in feeling alone.",
                    "Your need for connection is beautiful and natural.",
                    "Even in loneliness, you are connected to the vast web of existence."
                ],
                "guidance": [
                    "Sometimes loneliness is a call to deepen the relationship with yourself.",
                    "Connection with nature can help heal the ache of loneliness.",
                    "Your soul's longing for connection is a gateway to the divine."
                ],
                "practices": [
                    "Spend time in nature - trees and flowers are wonderful companions.",
                    "Practice loving-kindness meditation for all beings who feel lonely.",
                    "Write a letter to your future self or to the universe."
                ]
            },
            
            EmotionalState.JOY: {
                "validation": [
                    "Your joy is radiant and infectious - thank you for sharing it!",
                    "This joy is a glimpse of your true nature, your divine essence.",
                    "Let this joy fill every cell of your being and overflow to others."
                ],
                "guidance": [
                    "Joy is your natural state - notice what brings you back to it.",
                    "Gratitude and joy dance together in an eternal celebration.",
                    "Your joy is a gift to the world - let it shine freely."
                ],
                "practices": [
                    "Take a moment to really savor this feeling in your body.",
                    "Share your joy with someone who needs to hear good news.",
                    "Keep a joy journal - record what sparks this beautiful feeling."
                ]
            }
        }
    
    def _initialize_sanskrit_wisdom(self):
        """Initialize Sanskrit wisdom for emotional healing"""
        self.sanskrit_wisdom = {
            EmotionalState.PEACE: [
                "शान्ति शान्ति शान्ति: (Shanti Shanti Shanti) - Peace, Peace, Peace",
                "सर्वे भवन्तु सुखिनः (Sarve Bhavantu Sukhinah) - May all beings be happy"
            ],
            EmotionalState.LOVE: [
                "सर्वे भद्राणि पश्यन्तु (Sarve Bhadrani Pashyantu) - May all see auspiciousness",
                "वसुधैव कुटुम्बकम् (Vasudhaiva Kutumbakam) - The world is one family"
            ],
            EmotionalState.FEAR: [
                "अभयं वै ब्रह्म (Abhayam Vai Brahma) - Fearlessness is Brahman",
                "त्वमेव माता च पिता त्वमेव (Tvameva Mata Cha Pita Tvameva) - You alone are mother and father"
            ],
            EmotionalState.SADNESS: [
                "तत् त्वम् असि (Tat Tvam Asi) - Thou art That (You are divine)",
                "सर्वं खल्विदं ब्रह्म (Sarvam Khalvidam Brahma) - All this is indeed Brahman"
            ]
        }

    async def analyze_emotional_state(self, text: str, context: Dict[str, Any] = None) -> EmotionalProfile:
        """Analyze emotional state from text with deep understanding"""
        try:
            # Detect primary emotions
            detected_emotions = self._detect_emotions(text)
            
            if not detected_emotions:
                detected_emotions = {EmotionalState.NEUTRAL: 0.5}
            
            # Find primary emotion
            primary_emotion = max(detected_emotions.items(), key=lambda x: x[1])[0]
            primary_intensity = detected_emotions[primary_emotion]
            
            # Get secondary emotions
            secondary_emotions = [
                emotion for emotion, score in detected_emotions.items() 
                if emotion != primary_emotion and score > 0.3
            ]
            
            # Determine intensity level
            intensity_level = self._calculate_intensity_level(primary_intensity)
            
            # Analyze chakra resonance
            chakra_resonance = self._analyze_chakra_resonance(detected_emotions)
            
            # Get spiritual opportunity
            spiritual_opportunity = self._get_spiritual_opportunity(primary_emotion)
            
            # Generate dharmic guidance
            dharmic_guidance = self._get_dharmic_guidance(primary_emotion, primary_intensity)
            
            profile = EmotionalProfile(
                primary_emotion=primary_emotion,
                secondary_emotions=secondary_emotions,
                intensity=primary_intensity,
                intensity_level=intensity_level,
                confidence=0.85,
                emotional_context=context or {},
                chakra_resonance=chakra_resonance,
                spiritual_opportunity=spiritual_opportunity,
                dharmic_guidance=dharmic_guidance
            )
            
            self.logger.info(f"🔍 Emotional analysis: {primary_emotion.value} ({intensity_level.value})")
            return profile
            
        except Exception as e:
            self.logger.error(f"Error in emotional analysis: {e}")
            return EmotionalProfile(
                primary_emotion=EmotionalState.NEUTRAL,
                confidence=0.1
            )
    
    def _detect_emotions(self, text: str) -> Dict[EmotionalState, float]:
        """Detect emotions from text using pattern matching"""
        text_lower = text.lower()
        detected = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0
            
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches * 0.2
            
            # Boost score for emotional intensifiers
            intensifiers = r'\b(very|really|extremely|totally|absolutely|completely|deeply)\b'
            if re.search(intensifiers, text_lower):
                score *= 1.3
            
            # Multiple exclamation marks or caps
            if '!!!' in text or text.isupper():
                score *= 1.2
            
            if score > 0:
                detected[emotion] = min(score, 1.0)
        
        return detected
    
    def _calculate_intensity_level(self, intensity: float) -> EmotionalIntensity:
        """Calculate intensity level from numeric intensity"""
        if intensity <= 0.3:
            return EmotionalIntensity.SUBTLE
        elif intensity <= 0.6:
            return EmotionalIntensity.MODERATE
        elif intensity <= 0.8:
            return EmotionalIntensity.STRONG
        else:
            return EmotionalIntensity.INTENSE
    
    def _analyze_chakra_resonance(self, emotions: Dict[EmotionalState, float]) -> Dict[str, float]:
        """Analyze which chakras are activated by the emotional state"""
        chakra_scores = {chakra: 0.0 for chakra in self.chakra_emotions.keys()}
        
        for emotion, intensity in emotions.items():
            for chakra, info in self.chakra_emotions.items():
                if emotion in info["primary"]:
                    chakra_scores[chakra] += intensity * 0.8
                elif emotion in info.get("balanced", []):
                    chakra_scores[chakra] += intensity * 0.6
        
        return {k: v for k, v in chakra_scores.items() if v > 0.1}
    
    def _get_spiritual_opportunity(self, emotion: EmotionalState) -> str:
        """Get spiritual growth opportunity from emotional state"""
        opportunities = {
            EmotionalState.SADNESS: "This sadness invites you to deepen compassion for yourself and others.",
            EmotionalState.ANGER: "This anger can be transformed into passionate energy for positive change.",
            EmotionalState.FEAR: "This fear offers an opportunity to cultivate trust and surrender.",
            EmotionalState.JOY: "This joy reflects your true divine nature - let it guide you home to yourself.",
            EmotionalState.LONELINESS: "This longing reveals your deep need for connection with the Divine.",
            EmotionalState.CONFUSION: "This confusion is the beginning of wisdom - stay curious and open.",
            EmotionalState.PEACE: "This peace is a taste of your eternal nature - rest here and be nourished.",
            EmotionalState.LOVE: "This love is the very fabric of existence flowing through you."
        }
        return opportunities.get(emotion, "Every emotion is a teacher, offering lessons for your spiritual growth.")
    
    def _get_dharmic_guidance(self, emotion: EmotionalState, intensity: float) -> str:
        """Get dharmic guidance based on emotion and intensity"""
        guidance_templates = {
            EmotionalState.SADNESS: [
                "In Vedic wisdom, sadness (vishada) is honored as a gateway to deeper understanding.",
                "Krishna taught Arjuna that sorrow often precedes great wisdom and transformation.",
                "Your tears are prayers - they cleanse the heart and prepare it for divine love."
            ],
            EmotionalState.ANGER: [
                "The Bhagavad Gita teaches that righteous anger (dharmic krodha) can fuel just action.",
                "Transform this fire into tapas - spiritual intensity that burns away illusions.",
                "Channel this energy like Hanuman - with devotion, purpose, and controlled power."
            ],
            EmotionalState.FEAR: [
                "Krishna reminds us: 'I am Time, destroyer and creator' - surrender to the cosmic flow.",
                "Fear dissolves in the light of self-knowledge - you are eternal, beyond all harm.",
                "Practice abhaya mudra - the gesture of fearlessness that Buddha and Krishna showed."
            ]
        }
        
        templates = guidance_templates.get(emotion, ["Trust in dharma - all experiences serve your highest evolution."])
        return random.choice(templates)

    async def generate_emotionally_intelligent_response(
        self, 
        emotional_profile: EmotionalProfile, 
        user_message: str,
        context: Dict[str, Any] = None
    ) -> EmotionalResponse:
        """Generate an emotionally intelligent, healing response"""
        
        try:
            primary_emotion = emotional_profile.primary_emotion
            intensity = emotional_profile.intensity
            
            # Get healing elements
            validation = self._get_validation(primary_emotion)
            understanding = self._get_understanding(primary_emotion, user_message)
            guidance = self._get_guidance(primary_emotion, intensity)
            practice_suggestion = self._get_practice_suggestion(primary_emotion)
            
            # Get spiritual elements
            sanskrit_wisdom = random.choice(self.sanskrit_wisdom.get(primary_emotion, ["ॐ शान्ति शान्ति शान्ति"]))
            chakra_healing = self._get_chakra_healing(emotional_profile.chakra_resonance)
            
            # Generate main response
            response_text = self._generate_main_response(
                primary_emotion, validation, understanding, guidance
            )
            
            # Determine response tone
            response_tone = self._determine_response_tone(primary_emotion)
            
            response = EmotionalResponse(
                response_text=response_text,
                emotional_tone=response_tone,
                empathy_level=0.9,
                validation=validation,
                understanding=understanding,
                guidance=guidance,
                practice_suggestion=practice_suggestion,
                sanskrit_wisdom=sanskrit_wisdom,
                chakra_healing=chakra_healing,
                response_type="empathetic_healing",
                healing_intent=["validation", "understanding", "guidance", "spiritual_support"]
            )
            
            self.logger.info(f"💙 Generated empathetic response for {primary_emotion.value}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating emotional response: {e}")
            return EmotionalResponse(
                response_text="I hear you and I'm here with you. Thank you for sharing.",
                emotional_tone=EmotionalState.COMPASSION,
                empathy_level=0.8
            )
    
    def _get_validation(self, emotion: EmotionalState) -> str:
        """Get validation message for the emotion"""
        validations = self.healing_responses.get(emotion, {}).get("validation", [])
        if validations:
            return random.choice(validations)
        return "Your feelings are completely valid and understandable."
    
    def _get_understanding(self, emotion: EmotionalState, user_message: str) -> str:
        """Generate understanding based on specific message content"""
        # Simple understanding based on emotion
        understanding_templates = {
            EmotionalState.SADNESS: "I can sense the heaviness in your heart right now.",
            EmotionalState.ANGER: "I can feel the intensity of your frustration coming through.",
            EmotionalState.FEAR: "I understand how unsettling this uncertainty must feel.",
            EmotionalState.JOY: "Your happiness is radiating through your words!",
            EmotionalState.LONELINESS: "The ache of loneliness is so real and difficult to bear.",
            EmotionalState.CONFUSION: "It sounds like you're navigating some complex feelings right now."
        }
        return understanding_templates.get(emotion, "I hear what you're going through.")
    
    def _get_guidance(self, emotion: EmotionalState, intensity: float) -> str:
        """Get dharmic guidance for the emotion"""
        guidance_list = self.healing_responses.get(emotion, {}).get("guidance", [])
        if guidance_list:
            return random.choice(guidance_list)
        return "Remember that all emotions are temporary visitors - let them flow through you with compassion."
    
    def _get_practice_suggestion(self, emotion: EmotionalState) -> str:
        """Get practice suggestion for the emotion"""
        practices = self.healing_responses.get(emotion, {}).get("practices", [])
        if practices:
            return random.choice(practices)
        return "Try taking three deep breaths, sending love to yourself with each exhale."
    
    def _get_chakra_healing(self, chakra_resonance: Dict[str, float]) -> str:
        """Get chakra healing message"""
        if not chakra_resonance:
            return "Send loving energy to all your energy centers, from root to crown."
        
        dominant_chakra = max(chakra_resonance.items(), key=lambda x: x[1])[0]
        healing_msg = self.chakra_emotions[dominant_chakra]["healing"]
        return f"Focus on your {dominant_chakra.replace('_', ' ')} chakra: {healing_msg}"
    
    def _generate_main_response(self, emotion: EmotionalState, validation: str, understanding: str, guidance: str) -> str:
        """Generate the main empathetic response"""
        response_parts = []
        
        # Start with understanding
        response_parts.append(understanding)
        
        # Add validation
        response_parts.append(validation)
        
        # Add guidance
        response_parts.append(guidance)
        
        return " ".join(response_parts)
    
    def _determine_response_tone(self, emotion: EmotionalState) -> EmotionalState:
        """Determine appropriate response tone for the emotion"""
        tone_mapping = {
            EmotionalState.SADNESS: EmotionalState.COMPASSION,
            EmotionalState.ANGER: EmotionalState.PEACE,
            EmotionalState.FEAR: EmotionalState.LOVE,
            EmotionalState.JOY: EmotionalState.JOY,
            EmotionalState.LONELINESS: EmotionalState.LOVE,
            EmotionalState.CONFUSION: EmotionalState.CONTEMPLATIVE,
            EmotionalState.PEACE: EmotionalState.SERENITY
        }
        return tone_mapping.get(emotion, EmotionalState.COMPASSION)

# Factory function
def create_emotional_engine() -> AdvancedEmotionalEngine:
    """Create and return an instance of the advanced emotional engine"""
    return AdvancedEmotionalEngine()

# Global instance
_emotional_engine = None

def get_emotional_engine() -> AdvancedEmotionalEngine:
    """Get global emotional engine instance"""
    global _emotional_engine
    if _emotional_engine is None:
        _emotional_engine = create_emotional_engine()
    return _emotional_engine
