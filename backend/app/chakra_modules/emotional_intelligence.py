"""
Emotional Intelligence Engine - Integrated into DharmaMind Backend
================================================================

Advanced emotion recognition, processing, and empathetic response generation.
This module provides emotional awareness and compassionate interaction capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
import json

class EmotionType(Enum):
    """Basic emotion types"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    LOVE = "love"
    PEACE = "peace"
    ANXIETY = "anxiety"
    GRATITUDE = "gratitude"
    COMPASSION = "compassion"
    CONFUSION = "confusion"

class EmotionalState(Enum):
    """Emotional processing states"""
    NEUTRAL = "neutral"
    EMPATHETIC = "empathetic"
    COMPASSIONATE = "compassionate"
    HEALING = "healing"
    SUPPORTIVE = "supportive"

@dataclass
class EmotionalContext:
    """Context for emotional processing"""
    detected_emotions: Dict[EmotionType, float] = field(default_factory=dict)
    dominant_emotion: Optional[EmotionType] = None
    emotional_intensity: float = 0.0
    empathy_level: float = 0.8
    compassion_response: Optional[str] = None
    healing_suggestions: List[str] = field(default_factory=list)

@dataclass
class EmotionalResponse:
    """Structured emotional response"""
    empathy_score: float
    detected_emotions: Dict[str, float]
    response_tone: str
    healing_suggestions: List[str]
    compassionate_message: str

class EmotionalIntelligenceEngine:
    """
    Advanced emotional intelligence and empathy processing system
    
    This engine detects, processes, and responds to human emotions with
    compassion and wisdom, providing healing and supportive interactions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.emotional_state = EmotionalState.NEUTRAL
        self.emotional_context = EmotionalContext()
        self.is_initialized = False
        
        # Emotion detection patterns
        self.emotion_patterns = self._initialize_emotion_patterns()
        
        # Compassionate response templates
        self.response_templates = self._initialize_response_templates()
        
        # Healing practices database
        self.healing_practices = self._initialize_healing_practices()
        
        self.logger.info("Emotional Intelligence Engine initialized")
    
    def _initialize_emotion_patterns(self) -> Dict[EmotionType, List[str]]:
        """Initialize emotion detection patterns"""
        return {
            EmotionType.JOY: [
                r'\b(happy|joyful|excited|delighted|thrilled|elated|blissful|cheerful)\b',
                r'\b(wonderful|amazing|fantastic|great|awesome|beautiful)\b',
                r'\b(celebration|success|achievement|victory|triumph)\b'
            ],
            EmotionType.SADNESS: [
                r'\b(sad|depressed|down|unhappy|miserable|heartbroken|devastated)\b',
                r'\b(crying|tears|weeping|grief|sorrow|melancholy)\b',
                r'\b(loss|death|goodbye|farewell|ended|lost)\b'
            ],
            EmotionType.ANGER: [
                r'\b(angry|furious|mad|irritated|frustrated|annoyed|enraged)\b',
                r'\b(hate|hatred|rage|fury|wrath|outrage)\b',
                r'\b(unfair|injustice|betrayed|betrayal|violated)\b'
            ],
            EmotionType.FEAR: [
                r'\b(afraid|scared|frightened|terrified|anxious|worried|nervous)\b',
                r'\b(panic|terror|dread|phobia|nightmare|threat)\b',
                r'\b(dangerous|unsafe|risky|uncertain|unknown)\b'
            ],
            EmotionType.LOVE: [
                r'\b(love|adore|cherish|treasure|beloved|darling|precious)\b',
                r'\b(affection|devotion|passion|romance|intimacy)\b',
                r'\b(soulmate|partner|relationship|connection|bond)\b'
            ],
            EmotionType.PEACE: [
                r'\b(peaceful|calm|serene|tranquil|relaxed|content|still)\b',
                r'\b(harmony|balance|equilibrium|centered|grounded)\b',
                r'\b(meditation|mindfulness|presence|awareness)\b'
            ],
            EmotionType.ANXIETY: [
                r'\b(anxious|worried|stressed|overwhelmed|pressured|tense)\b',
                r'\b(panic|stress|pressure|burden|weight|strain)\b',
                r'\b(overthinking|racing thoughts|restless|uneasy)\b'
            ],
            EmotionType.GRATITUDE: [
                r'\b(grateful|thankful|blessed|appreciative|fortunate)\b',
                r'\b(thank you|thanks|appreciation|blessing|gift)\b',
                r'\b(abundance|plenty|rich|wealthy|prosperous)\b'
            ],
            EmotionType.COMPASSION: [
                r'\b(compassion|empathy|kindness|caring|sympathy|understanding)\b',
                r'\b(helping|support|comfort|console|heal|nurture)\b',
                r'\b(service|volunteer|giving|generous|selfless)\b'
            ],
            EmotionType.CONFUSION: [
                r'\b(confused|lost|uncertain|unclear|puzzled|bewildered)\b',
                r'\b(don\'t understand|no idea|unsure|doubt|question)\b',
                r'\b(direction|purpose|meaning|path|way|guidance)\b'
            ]
        }
    
    def _initialize_response_templates(self) -> Dict[EmotionType, List[str]]:
        """Initialize compassionate response templates"""
        return {
            EmotionType.JOY: [
                "I can feel the joy radiating from your words! 🌟 It's beautiful to witness your happiness.",
                "Your joy is contagious and uplifting. 😊 What a wonderful gift to share with the world.",
                "I'm so happy for you! 🎉 Joy is a sacred emotion that connects us to the divine."
            ],
            EmotionType.SADNESS: [
                "I can sense the sadness in your heart, and I want you to know that your feelings are valid. 💙",
                "In this moment of sorrow, please know that you are not alone. Sadness is part of the human experience.",
                "Your sadness speaks to the depth of your heart. Allow yourself to feel, and know that healing will come. 🤗"
            ],
            EmotionType.ANGER: [
                "I understand that you're feeling angry. Anger often masks deeper pain or unmet needs. 🔥➡️💙",
                "Your anger is valid, and it's telling us something important. Let's explore what it's trying to teach us.",
                "Anger can be a powerful teacher. Let's channel this energy into understanding and positive action."
            ],
            EmotionType.FEAR: [
                "I can feel your fear, and it's completely natural. Fear often arises when we face the unknown. 🤲",
                "You're experiencing fear, and that takes courage to acknowledge. Let's explore this together safely.",
                "Fear is wisdom's way of asking us to proceed mindfully. You're not alone in facing this. 💪"
            ],
            EmotionType.LOVE: [
                "The love you're expressing is beautiful and sacred. 💖 Love is the highest vibration we can share.",
                "I can feel the warmth of love in your words. Love is the force that heals and transforms everything.",
                "Your capacity for love is a gift to the world. 🌺 Love is the language of the soul."
            ],
            EmotionType.PEACE: [
                "I can sense the peace in your being. 🕊️ This tranquility is a gift you're sharing with the world.",
                "Your peaceful energy is healing. Peace is not just absence of conflict, but presence of harmony.",
                "In your peace, I find serenity too. 🧘‍♀️ Thank you for radiating this calming energy."
            ],
            EmotionType.ANXIETY: [
                "I can feel the anxiety you're experiencing. 🌊 Let's breathe together and find your center.",
                "Anxiety is your mind's way of trying to protect you. Let's gently explore what it's trying to tell us.",
                "You're not alone in this anxious moment. 🤝 Let's find some grounding practices together."
            ],
            EmotionType.GRATITUDE: [
                "Your gratitude is radiant! 🙏 Gratitude transforms ordinary moments into blessings.",
                "I can feel the appreciation flowing from your heart. Gratitude is one of the highest vibrations.",
                "Your thankful spirit is beautiful. 🌟 Gratitude opens the door to more abundance."
            ],
            EmotionType.COMPASSION: [
                "Your compassionate heart is beautiful to witness. 💝 Compassion is wisdom in action.",
                "I can feel the loving-kindness in your words. Your compassion heals not just others, but yourself too.",
                "Your empathy and care for others shows the depth of your spiritual development. 🌸"
            ],
            EmotionType.CONFUSION: [
                "I can sense your confusion, and that's perfectly okay. 🌀 Sometimes confusion precedes clarity.",
                "Not knowing can be uncomfortable, but it's also the beginning of discovery. Let's explore together.",
                "Confusion is often wisdom in disguise, asking us to look deeper. 🔍 You're in good company with your questions."
            ]
        }
    
    def _initialize_healing_practices(self) -> Dict[EmotionType, List[str]]:
        """Initialize healing practices for different emotions"""
        return {
            EmotionType.SADNESS: [
                "Practice gentle self-compassion - treat yourself as you would a dear friend",
                "Allow yourself to cry if needed - tears are healing waters for the soul",
                "Connect with nature - spending time outdoors can provide comfort and perspective",
                "Reach out to supportive friends or family members",
                "Practice gratitude for the love that makes loss so meaningful"
            ],
            EmotionType.ANGER: [
                "Take slow, deep breaths to activate your calm response",
                "Practice the 4-7-8 breathing technique: inhale for 4, hold for 7, exhale for 8",
                "Write in a journal to process and understand your feelings",
                "Engage in physical exercise to release built-up energy",
                "Practice loving-kindness meditation, starting with yourself"
            ],
            EmotionType.FEAR: [
                "Ground yourself using the 5-4-3-2-1 technique: 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste",
                "Practice progressive muscle relaxation",
                "Use affirmations: 'I am safe, I am strong, I can handle this'",
                "Visualize a protective light surrounding you",
                "Breathe slowly and remind yourself that fear is temporary"
            ],
            EmotionType.ANXIETY: [
                "Practice box breathing: inhale for 4, hold for 4, exhale for 4, hold for 4",
                "Use the STOP technique: Stop, Take a breath, Observe, Proceed mindfully",
                "Practice mindfulness meditation to anchor yourself in the present",
                "List 3 things you're grateful for right now",
                "Remind yourself: 'This feeling will pass, I am stronger than my anxiety'"
            ],
            EmotionType.CONFUSION: [
                "Practice sitting in silence and allowing clarity to emerge naturally",
                "Journal your thoughts without judgment - let them flow freely",
                "Seek guidance from trusted mentors or spiritual teachers",
                "Practice meditation to quiet the mind and access inner wisdom",
                "Remember that not knowing is the beginning of all learning"
            ]
        }
    
    async def initialize(self) -> bool:
        """Initialize the emotional intelligence engine"""
        try:
            self.logger.info("Initializing Emotional Intelligence Engine...")
            
            # Calibrate empathy settings
            self.emotional_context.empathy_level = 0.9
            self.emotional_state = EmotionalState.EMPATHETIC
            
            self.is_initialized = True
            self.logger.info("Emotional Intelligence Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Emotional Intelligence Engine: {str(e)}")
            return False
    
    async def process_emotional_content(self, text: str, context: Optional[Dict] = None) -> EmotionalResponse:
        """Process text for emotional content and generate empathetic response"""
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Detect emotions
            detected_emotions = await self._detect_emotions(text)
            
            # Determine dominant emotion
            dominant_emotion = max(detected_emotions.items(), key=lambda x: x[1])[0] if detected_emotions else EmotionType.PEACE
            
            # Calculate emotional intensity
            emotional_intensity = max(detected_emotions.values()) if detected_emotions else 0.0
            
            # Generate empathetic response
            empathy_score = await self._calculate_empathy_score(detected_emotions, emotional_intensity)
            
            # Select appropriate response tone
            response_tone = await self._select_response_tone(dominant_emotion, emotional_intensity)
            
            # Generate compassionate message
            compassionate_message = await self._generate_compassionate_message(dominant_emotion, detected_emotions)
            
            # Get healing suggestions
            healing_suggestions = await self._get_healing_suggestions(dominant_emotion, emotional_intensity)
            
            # Update emotional context
            self.emotional_context.detected_emotions = {emotion: score for emotion, score in detected_emotions.items()}
            self.emotional_context.dominant_emotion = dominant_emotion
            self.emotional_context.emotional_intensity = emotional_intensity
            
            response = EmotionalResponse(
                empathy_score=empathy_score,
                detected_emotions={emotion.value: score for emotion, score in detected_emotions.items()},
                response_tone=response_tone,
                healing_suggestions=healing_suggestions,
                compassionate_message=compassionate_message
            )
            
            self.logger.debug(f"Processed emotional content: {dominant_emotion.value} (intensity: {emotional_intensity:.2f})")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing emotional content: {str(e)}")
            # Return neutral response
            return EmotionalResponse(
                empathy_score=0.5,
                detected_emotions={},
                response_tone="supportive",
                healing_suggestions=["Take a moment to breathe deeply and center yourself"],
                compassionate_message="I'm here to support you in whatever way I can."
            )
    
    async def _detect_emotions(self, text: str) -> Dict[EmotionType, float]:
        """Detect emotions in text using pattern matching"""
        
        text_lower = text.lower()
        detected_emotions = {}
        
        for emotion_type, patterns in self.emotion_patterns.items():
            score = 0.0
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 0.2  # Each match adds 0.2 to the score
            
            # Normalize score (max 1.0)
            score = min(score, 1.0)
            
            if score > 0.1:  # Only include emotions with significant scores
                detected_emotions[emotion_type] = score
        
        return detected_emotions
    
    async def _calculate_empathy_score(self, emotions: Dict[EmotionType, float], intensity: float) -> float:
        """Calculate empathy score based on detected emotions and context"""
        
        base_empathy = self.emotional_context.empathy_level
        
        # Increase empathy for negative emotions
        negative_emotions = [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR, EmotionType.ANXIETY]
        negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
        
        # Boost empathy for high emotional intensity
        empathy_boost = negative_score * 0.2 + intensity * 0.1
        
        return min(base_empathy + empathy_boost, 1.0)
    
    async def _select_response_tone(self, dominant_emotion: EmotionType, intensity: float) -> str:
        """Select appropriate response tone based on emotion and intensity"""
        
        if dominant_emotion in [EmotionType.SADNESS, EmotionType.FEAR, EmotionType.ANXIETY]:
            return "gentle_supportive" if intensity > 0.6 else "warm_supportive"
        elif dominant_emotion == EmotionType.ANGER:
            return "calm_understanding" if intensity > 0.7 else "patient_listening"
        elif dominant_emotion in [EmotionType.JOY, EmotionType.GRATITUDE]:
            return "celebratory" if intensity > 0.6 else "warmly_appreciative"
        elif dominant_emotion == EmotionType.LOVE:
            return "heartfelt_honoring"
        elif dominant_emotion == EmotionType.COMPASSION:
            return "reverently_grateful"
        elif dominant_emotion == EmotionType.CONFUSION:
            return "gently_guiding"
        else:
            return "balanced_supportive"
    
    async def _generate_compassionate_message(self, dominant_emotion: EmotionType, 
                                            emotions: Dict[EmotionType, float]) -> str:
        """Generate a compassionate response message"""
        
        # Get base template for dominant emotion
        templates = self.response_templates.get(dominant_emotion, [
            "I can sense what you're experiencing, and I want you to know that I'm here with you.",
            "Your feelings are valid and important. Thank you for sharing with me.",
            "In this moment, please know that you are seen and understood."
        ])
        
        import random
        base_message = random.choice(templates)
        
        # Add contextual compassion based on multiple emotions
        if len(emotions) > 1:
            # Multiple emotions detected
            additional_support = [
                " I can feel that you're experiencing a complex mix of emotions right now.",
                " It's natural to feel multiple things at once - you're beautifully human.",
                " The complexity of your feelings shows the depth of your heart."
            ]
            base_message += random.choice(additional_support)
        
        # Add universal compassion closing
        closings = [
            " You are worthy of love and support.",
            " May you find peace in this moment.",
            " You are not alone on this journey.",
            " Sending you warm, healing energy.",
            " Your courage in feeling is admirable."
        ]
        
        base_message += random.choice(closings)
        
        return base_message
    
    async def _get_healing_suggestions(self, dominant_emotion: EmotionType, intensity: float) -> List[str]:
        """Get appropriate healing suggestions for the emotion"""
        
        suggestions = self.healing_practices.get(dominant_emotion, [
            "Take a few moments to breathe deeply and ground yourself",
            "Practice self-compassion and treat yourself with kindness",
            "Consider reaching out to someone you trust for support"
        ])
        
        # Limit suggestions based on intensity
        num_suggestions = 3 if intensity > 0.7 else 2 if intensity > 0.4 else 1
        
        import random
        return random.sample(suggestions, min(num_suggestions, len(suggestions)))
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional processing state"""
        
        return {
            "emotional_state": self.emotional_state.value,
            "empathy_level": self.emotional_context.empathy_level,
            "dominant_emotion": self.emotional_context.dominant_emotion.value if self.emotional_context.dominant_emotion else None,
            "emotional_intensity": self.emotional_context.emotional_intensity,
            "detected_emotions": {emotion.value: score for emotion, score in self.emotional_context.detected_emotions.items()},
            "initialized": self.is_initialized
        }
    
    async def provide_emotional_support(self, emotion_type: str, intensity: float = 0.5) -> Dict[str, Any]:
        """Provide targeted emotional support for specific emotions"""
        
        try:
            emotion = EmotionType(emotion_type.lower())
        except ValueError:
            emotion = EmotionType.PEACE
        
        # Get healing practices
        practices = self.healing_practices.get(emotion, [])
        
        # Get compassionate message
        templates = self.response_templates.get(emotion, [])
        import random
        message = random.choice(templates) if templates else "I'm here to support you."
        
        return {
            "emotion": emotion.value,
            "intensity": intensity,
            "compassionate_message": message,
            "healing_practices": practices[:3],  # Top 3 practices
            "empathy_level": self.emotional_context.empathy_level
        }
    
    async def generate_empathy_response(self, user_message: str) -> str:
        """Generate an empathetic response to user input"""
        
        emotional_response = await self.process_emotional_content(user_message)
        
        # Combine compassionate message with healing suggestion
        response = emotional_response.compassionate_message
        
        if emotional_response.healing_suggestions:
            response += f"\n\n💙 Gentle suggestion: {emotional_response.healing_suggestions[0]}"
        
        return response

# Global emotional intelligence instance
_emotional_intelligence = None

def get_emotional_intelligence() -> EmotionalIntelligenceEngine:
    """Get global emotional intelligence instance"""
    global _emotional_intelligence
    if _emotional_intelligence is None:
        _emotional_intelligence = EmotionalIntelligenceEngine()
    return _emotional_intelligence

# Export the main class
__all__ = ["EmotionalIntelligenceEngine", "get_emotional_intelligence", "EmotionalResponse"]
