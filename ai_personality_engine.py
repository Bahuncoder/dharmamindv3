#!/usr/bin/env python3
"""
🎭 DharmaMind Advanced AI Personality & Emotional Intelligence System
Dynamic personality adaptation and deep emotional understanding
"""

from fastapi import FastAPI, HTTPException
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import re
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionalState(str, Enum):
    PEACEFUL = "peaceful"
    ANXIOUS = "anxious"
    SAD = "sad"
    ANGRY = "angry"
    JOYFUL = "joyful"
    CONFUSED = "confused"
    HOPEFUL = "hopeful"
    FRUSTRATED = "frustrated"
    GRATEFUL = "grateful"
    FEARFUL = "fearful"
    CONTENT = "content"
    STRESSED = "stressed"

class PersonalityArchetype(str, Enum):
    WISE_SAGE = "wise_sage"
    COMPASSIONATE_MOTHER = "compassionate_mother"
    GENTLE_TEACHER = "gentle_teacher"
    SUPPORTIVE_FRIEND = "supportive_friend"
    PHILOSOPHICAL_GUIDE = "philosophical_guide"
    MEDITATION_MASTER = "meditation_master"
    DHARMIC_SCHOLAR = "dharmic_scholar"
    HEALING_THERAPIST = "healing_therapist"

class CommunicationStyle(str, Enum):
    FORMAL_TRADITIONAL = "formal_traditional"
    WARM_CONVERSATIONAL = "warm_conversational"
    SIMPLE_PRACTICAL = "simple_practical"
    POETIC_METAPHORICAL = "poetic_metaphorical"
    SCHOLARLY_DETAILED = "scholarly_detailed"
    GENTLE_NURTURING = "gentle_nurturing"
    DIRECT_HONEST = "direct_honest"
    PLAYFUL_LIGHT = "playful_light"

@dataclass
class EmotionalProfile:
    """User's emotional state and patterns"""
    current_emotion: EmotionalState
    emotional_intensity: float  # 0-100
    emotional_stability: float  # 0-100
    recent_emotions: List[Tuple[EmotionalState, datetime]]
    emotional_triggers: Dict[str, int]
    coping_mechanisms: List[str]
    emotional_growth_areas: List[str]

@dataclass
class PersonalityConfiguration:
    """AI personality configuration for a user"""
    primary_archetype: PersonalityArchetype
    secondary_traits: List[PersonalityArchetype]
    communication_style: CommunicationStyle
    wisdom_depth: float  # 0-100
    compassion_level: float  # 0-100
    formality_preference: float  # 0-100
    spiritual_tradition_focus: List[str]
    personality_adaptation_history: List[Dict[str, Any]]

@dataclass
class ConversationContext:
    """Context for current conversation"""
    user_id: str
    conversation_history: List[Dict[str, Any]]
    current_topic: str
    emotional_context: EmotionalProfile
    spiritual_needs: List[str]
    preferred_guidance_style: str
    session_goals: List[str]

class AdvancedEmotionalIntelligence:
    """
    🎭 Advanced AI Personality & Emotional Intelligence Engine
    Creates deeply empathetic and adaptable spiritual guidance
    """
    
    def __init__(self):
        self.personality_templates = {}
        self.emotional_patterns = {}
        self.response_styles = {}
        self._initialize_personality_system()
    
    def _initialize_personality_system(self):
        """Initialize personality templates and emotional patterns"""
        
        # Define personality archetypes
        self.personality_templates = {
            PersonalityArchetype.WISE_SAGE: {
                "name": "Wise Sage",
                "description": "Ancient wisdom with profound depth",
                "traits": {
                    "wisdom_depth": 95,
                    "compassion_level": 85,
                    "formality": 80,
                    "patience": 95,
                    "spiritual_insight": 90
                },
                "speech_patterns": [
                    "reflects deeply before responding",
                    "uses ancient wisdom and metaphors",
                    "speaks with measured, thoughtful cadence",
                    "references timeless spiritual truths"
                ],
                "response_style": {
                    "opening_phrases": [
                        "In the depths of contemplation, one finds...",
                        "The ancient teachings remind us that...",
                        "As wisdom flows through countless ages...",
                        "In the silence between thoughts lies..."
                    ],
                    "transition_phrases": [
                        "Moreover, the deeper truth reveals...",
                        "Yet, as we peer beneath the surface...",
                        "The path unfolds further to show us...",
                        "In this light, we may also consider..."
                    ],
                    "closing_phrases": [
                        "May this wisdom illuminate your path forward.",
                        "Let these insights settle gently in your heart.",
                        "Truth reveals itself to those who seek with sincerity.",
                        "Walk forward with this understanding as your guide."
                    ]
                }
            },
            
            PersonalityArchetype.COMPASSIONATE_MOTHER: {
                "name": "Compassionate Mother",
                "description": "Nurturing, protective, and unconditionally loving",
                "traits": {
                    "wisdom_depth": 75,
                    "compassion_level": 98,
                    "formality": 30,
                    "patience": 90,
                    "emotional_attunement": 95
                },
                "speech_patterns": [
                    "speaks with warmth and tenderness",
                    "validates emotions before offering guidance",
                    "uses nurturing language and metaphors",
                    "focuses on emotional healing and comfort"
                ],
                "response_style": {
                    "opening_phrases": [
                        "Sweet soul, I feel your heart's calling...",
                        "My dear one, your feelings are so valid...",
                        "Beloved, let me hold space for your experience...",
                        "Precious one, your vulnerability is sacred..."
                    ],
                    "transition_phrases": [
                        "And gently, we can also explore...",
                        "With loving kindness, let's consider...",
                        "Softly, another perspective emerges...",
                        "Tenderly, the heart whispers..."
                    ],
                    "closing_phrases": [
                        "You are so deeply loved and held.",
                        "Trust in your beautiful, unfolding journey.",
                        "Your heart knows the way forward, dear one.",
                        "Sending you infinite love and light."
                    ]
                }
            },
            
            PersonalityArchetype.MEDITATION_MASTER: {
                "name": "Meditation Master",
                "description": "Centered in present-moment awareness",
                "traits": {
                    "wisdom_depth": 88,
                    "compassion_level": 85,
                    "formality": 60,
                    "mindfulness": 98,
                    "inner_peace": 95
                },
                "speech_patterns": [
                    "emphasizes present-moment awareness",
                    "guides attention inward",
                    "uses breath and body-based metaphors",
                    "creates space and stillness in conversation"
                ],
                "response_style": {
                    "opening_phrases": [
                        "Let us pause here together and breathe...",
                        "In this very moment, what do you notice?",
                        "Feel into the spaciousness that surrounds these thoughts...",
                        "Coming home to the breath, we find..."
                    ],
                    "transition_phrases": [
                        "Notice what arises as we explore this...",
                        "Breathing into this awareness...",
                        "In the stillness, we may discover...",
                        "With gentle attention, observe..."
                    ],
                    "closing_phrases": [
                        "Rest in the peace that is always available.",
                        "Return to this inner sanctuary whenever needed.",
                        "The present moment is your true home.",
                        "Breathe, and know that all is well."
                    ]
                }
            },
            
            PersonalityArchetype.GENTLE_TEACHER: {
                "name": "Gentle Teacher",
                "description": "Patient educator and guide",
                "traits": {
                    "wisdom_depth": 80,
                    "compassion_level": 85,
                    "formality": 50,
                    "patience": 95,
                    "clarity": 90
                },
                "speech_patterns": [
                    "breaks down complex concepts simply",
                    "uses analogies and examples",
                    "encourages questions and exploration",
                    "builds understanding step by step"
                ],
                "response_style": {
                    "opening_phrases": [
                        "Let's explore this together, step by step...",
                        "I'd love to help you understand this more clearly...",
                        "Think of it this way, if you will...",
                        "Consider this gentle invitation to learn..."
                    ],
                    "transition_phrases": [
                        "Building on this understanding...",
                        "Now, let's look at it from another angle...",
                        "This naturally leads us to consider...",
                        "As we deepen our exploration..."
                    ],
                    "closing_phrases": [
                        "Does this resonate with your experience?",
                        "Take time to let this understanding settle.",
                        "You're doing beautiful work in learning this.",
                        "Trust your own wisdom as it unfolds."
                    ]
                }
            }
        }
        
        # Define emotional response patterns
        self.emotional_patterns = {
            EmotionalState.ANXIOUS: {
                "recognition_keywords": [
                    "anxious", "worried", "nervous", "panic", "stress", "overwhelmed",
                    "fear", "scared", "uncertain", "unsettled", "restless", "tense"
                ],
                "response_approaches": [
                    "grounding_techniques",
                    "breathing_exercises", 
                    "reassurance_and_validation",
                    "present_moment_awareness",
                    "gentle_perspective_shift"
                ],
                "dharmic_remedies": [
                    "Pranayama (breathing practices)",
                    "Mindfulness meditation",
                    "Mantra repetition",
                    "Surrender practices (Ishvara Pranidhana)",
                    "Loving-kindness meditation"
                ]
            },
            
            EmotionalState.SAD: {
                "recognition_keywords": [
                    "sad", "depressed", "down", "melancholy", "grief", "sorrow",
                    "hopeless", "empty", "lost", "heartbroken", "despair", "lonely"
                ],
                "response_approaches": [
                    "emotional_validation",
                    "compassionate_presence",
                    "gentle_encouragement",
                    "meaning_making_support",
                    "connection_and_community"
                ],
                "dharmic_remedies": [
                    "Bhakti practices (devotional love)",
                    "Seva (selfless service)",
                    "Gratitude meditation",
                    "Study of uplifting scriptures",
                    "Connection with sangha (spiritual community)"
                ]
            },
            
            EmotionalState.ANGRY: {
                "recognition_keywords": [
                    "angry", "furious", "frustrated", "irritated", "mad", "rage",
                    "annoyed", "resentful", "bitter", "hostile", "indignant"
                ],
                "response_approaches": [
                    "anger_acknowledgment",
                    "cooling_practices",
                    "perspective_reframing",
                    "forgiveness_guidance",
                    "constructive_expression"
                ],
                "dharmic_remedies": [
                    "Ahimsa (non-violence) reflection",
                    "Kshama (forgiveness) practices",
                    "Cooling pranayama (Sheetali/Sheetkari)",
                    "Compassion meditation",
                    "Karma yoga (selfless action)"
                ]
            }
        }
        
        logger.info(f"✅ Initialized {len(self.personality_templates)} personality archetypes")
        logger.info(f"✅ Initialized {len(self.emotional_patterns)} emotional patterns")
    
    async def analyze_emotional_state(self, user_input: str, 
                                    conversation_history: List[Dict[str, Any]] = None) -> EmotionalProfile:
        """Analyze user's emotional state from their input"""
        try:
            # Initialize emotion scores
            emotion_scores = {emotion: 0 for emotion in EmotionalState}
            
            # Analyze current input
            input_lower = user_input.lower()
            
            # Pattern matching for emotions
            for emotion, patterns in self.emotional_patterns.items():
                for keyword in patterns["recognition_keywords"]:
                    if keyword in input_lower:
                        emotion_scores[emotion] += 10
                        
                        # Increase score based on keyword intensity
                        if any(intense in input_lower for intense in ["very", "extremely", "really", "so"]):
                            emotion_scores[emotion] += 5
            
            # Sentiment analysis using TextBlob
            try:
                blob = TextBlob(user_input)
                sentiment_polarity = blob.sentiment.polarity  # -1 to 1
                sentiment_subjectivity = blob.sentiment.subjectivity  # 0 to 1
                
                # Map sentiment to emotions
                if sentiment_polarity > 0.3:
                    emotion_scores[EmotionalState.JOYFUL] += 15
                    emotion_scores[EmotionalState.GRATEFUL] += 10
                elif sentiment_polarity < -0.3:
                    emotion_scores[EmotionalState.SAD] += 15
                    emotion_scores[EmotionalState.FRUSTRATED] += 10
                else:
                    emotion_scores[EmotionalState.CONTENT] += 10
                
                # High subjectivity might indicate emotional intensity
                if sentiment_subjectivity > 0.7:
                    # Amplify the highest scoring emotions
                    max_emotion = max(emotion_scores, key=emotion_scores.get)
                    if emotion_scores[max_emotion] > 0:
                        emotion_scores[max_emotion] += 10
                        
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
            
            # Analyze conversation history for emotional patterns
            if conversation_history:
                recent_messages = conversation_history[-5:]  # Last 5 messages
                for message in recent_messages:
                    msg_content = message.get('content', '').lower()
                    
                    # Look for emotional progression
                    for emotion, patterns in self.emotional_patterns.items():
                        for keyword in patterns["recognition_keywords"]:
                            if keyword in msg_content:
                                emotion_scores[emotion] += 3  # Smaller weight for history
            
            # Determine primary emotion
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            emotional_intensity = min(emotion_scores[primary_emotion], 100)
            
            # Calculate emotional stability (consistency over time)
            emotional_stability = 70  # Default stable
            if conversation_history and len(conversation_history) > 3:
                # Analyze emotional consistency
                recent_emotions = []
                for message in conversation_history[-5:]:
                    # Simple emotion detection for history
                    msg_emotion = self._quick_emotion_detect(message.get('content', ''))
                    if msg_emotion:
                        recent_emotions.append(msg_emotion)
                
                if recent_emotions:
                    # Calculate stability based on emotional variance
                    emotion_changes = len(set(recent_emotions))
                    emotional_stability = max(20, 100 - (emotion_changes * 15))
            
            # Create emotional profile
            emotional_profile = EmotionalProfile(
                current_emotion=primary_emotion,
                emotional_intensity=emotional_intensity,
                emotional_stability=emotional_stability,
                recent_emotions=[(primary_emotion, datetime.now())],
                emotional_triggers={},
                coping_mechanisms=[],
                emotional_growth_areas=[]
            )
            
            logger.info(f"🎭 Analyzed emotion: {primary_emotion} (intensity: {emotional_intensity})")
            return emotional_profile
            
        except Exception as e:
            logger.error(f"❌ Error analyzing emotional state: {e}")
            # Return default emotional profile
            return EmotionalProfile(
                current_emotion=EmotionalState.CONTENT,
                emotional_intensity=50,
                emotional_stability=70,
                recent_emotions=[],
                emotional_triggers={},
                coping_mechanisms=[],
                emotional_growth_areas=[]
            )
    
    def _quick_emotion_detect(self, text: str) -> Optional[EmotionalState]:
        """Quick emotion detection for conversation history"""
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Simple keyword matching
        for emotion, patterns in self.emotional_patterns.items():
            for keyword in patterns["recognition_keywords"][:3]:  # Check top 3 keywords
                if keyword in text_lower:
                    return emotion
        
        return None
    
    async def determine_optimal_personality(self, emotional_profile: EmotionalProfile,
                                          user_preferences: Dict[str, Any] = None,
                                          conversation_context: str = "") -> PersonalityConfiguration:
        """Determine the optimal AI personality for the current context"""
        try:
            user_preferences = user_preferences or {}
            
            # Analyze what personality would be most helpful
            personality_scores = {}
            
            # Score personalities based on emotional state
            current_emotion = emotional_profile.current_emotion
            intensity = emotional_profile.emotional_intensity
            
            if current_emotion in [EmotionalState.ANXIOUS, EmotionalState.FEARFUL, EmotionalState.STRESSED]:
                # Need calming, grounding presence
                personality_scores[PersonalityArchetype.MEDITATION_MASTER] = 90
                personality_scores[PersonalityArchetype.COMPASSIONATE_MOTHER] = 85
                personality_scores[PersonalityArchetype.GENTLE_TEACHER] = 70
                
            elif current_emotion in [EmotionalState.SAD, EmotionalState.HOPEFUL]:
                # Need nurturing, supportive presence
                personality_scores[PersonalityArchetype.COMPASSIONATE_MOTHER] = 95
                personality_scores[PersonalityArchetype.SUPPORTIVE_FRIEND] = 80
                personality_scores[PersonalityArchetype.HEALING_THERAPIST] = 75
                
            elif current_emotion in [EmotionalState.ANGRY, EmotionalState.FRUSTRATED]:
                # Need wise, patient guidance
                personality_scores[PersonalityArchetype.WISE_SAGE] = 85
                personality_scores[PersonalityArchetype.GENTLE_TEACHER] = 80
                personality_scores[PersonalityArchetype.MEDITATION_MASTER] = 70
                
            elif current_emotion in [EmotionalState.CONFUSED]:
                # Need clear, educational guidance
                personality_scores[PersonalityArchetype.GENTLE_TEACHER] = 90
                personality_scores[PersonalityArchetype.WISE_SAGE] = 80
                personality_scores[PersonalityArchetype.DHARMIC_SCHOLAR] = 70
                
            else:
                # Default balanced approach
                personality_scores[PersonalityArchetype.GENTLE_TEACHER] = 80
                personality_scores[PersonalityArchetype.WISE_SAGE] = 75
                personality_scores[PersonalityArchetype.COMPASSIONATE_MOTHER] = 70
            
            # Adjust based on user preferences
            preferred_style = user_preferences.get('communication_style', '')
            if 'formal' in preferred_style.lower():
                personality_scores[PersonalityArchetype.WISE_SAGE] += 15
                personality_scores[PersonalityArchetype.DHARMIC_SCHOLAR] += 10
            elif 'warm' in preferred_style.lower() or 'friendly' in preferred_style.lower():
                personality_scores[PersonalityArchetype.COMPASSIONATE_MOTHER] += 15
                personality_scores[PersonalityArchetype.SUPPORTIVE_FRIEND] += 10
            
            # Adjust based on conversation context
            context_lower = conversation_context.lower()
            if any(word in context_lower for word in ['meditat', 'mindful', 'breath']):
                personality_scores[PersonalityArchetype.MEDITATION_MASTER] += 20
            elif any(word in context_lower for word in ['learn', 'understand', 'explain']):
                personality_scores[PersonalityArchetype.GENTLE_TEACHER] += 20
            elif any(word in context_lower for word in ['dharma', 'scripture', 'philosophy']):
                personality_scores[PersonalityArchetype.WISE_SAGE] += 15
                personality_scores[PersonalityArchetype.DHARMIC_SCHOLAR] += 15
            
            # Select primary personality
            primary_archetype = max(personality_scores, key=personality_scores.get)
            
            # Select secondary traits
            sorted_personalities = sorted(personality_scores.items(), key=lambda x: x[1], reverse=True)
            secondary_traits = [p[0] for p in sorted_personalities[1:3]]
            
            # Determine communication style
            communication_style = self._determine_communication_style(
                emotional_profile, user_preferences, primary_archetype
            )
            
            # Create personality configuration
            personality_config = PersonalityConfiguration(
                primary_archetype=primary_archetype,
                secondary_traits=secondary_traits,
                communication_style=communication_style,
                wisdom_depth=self.personality_templates[primary_archetype]["traits"]["wisdom_depth"],
                compassion_level=self.personality_templates[primary_archetype]["traits"]["compassion_level"],
                formality_preference=self.personality_templates[primary_archetype]["traits"]["formality"],
                spiritual_tradition_focus=user_preferences.get('spiritual_traditions', ['universal']),
                personality_adaptation_history=[]
            )
            
            logger.info(f"🎭 Selected personality: {primary_archetype} with {communication_style}")
            return personality_config
            
        except Exception as e:
            logger.error(f"❌ Error determining personality: {e}")
            # Return default personality
            return PersonalityConfiguration(
                primary_archetype=PersonalityArchetype.GENTLE_TEACHER,
                secondary_traits=[PersonalityArchetype.WISE_SAGE],
                communication_style=CommunicationStyle.WARM_CONVERSATIONAL,
                wisdom_depth=80,
                compassion_level=85,
                formality_preference=50,
                spiritual_tradition_focus=['universal'],
                personality_adaptation_history=[]
            )
    
    def _determine_communication_style(self, emotional_profile: EmotionalProfile,
                                     user_preferences: Dict[str, Any],
                                     primary_archetype: PersonalityArchetype) -> CommunicationStyle:
        """Determine the most appropriate communication style"""
        
        # Base style on personality archetype
        style_mapping = {
            PersonalityArchetype.WISE_SAGE: CommunicationStyle.FORMAL_TRADITIONAL,
            PersonalityArchetype.COMPASSIONATE_MOTHER: CommunicationStyle.GENTLE_NURTURING,
            PersonalityArchetype.GENTLE_TEACHER: CommunicationStyle.WARM_CONVERSATIONAL,
            PersonalityArchetype.MEDITATION_MASTER: CommunicationStyle.SIMPLE_PRACTICAL,
            PersonalityArchetype.DHARMIC_SCHOLAR: CommunicationStyle.SCHOLARLY_DETAILED,
            PersonalityArchetype.SUPPORTIVE_FRIEND: CommunicationStyle.WARM_CONVERSATIONAL,
            PersonalityArchetype.HEALING_THERAPIST: CommunicationStyle.GENTLE_NURTURING,
            PersonalityArchetype.PHILOSOPHICAL_GUIDE: CommunicationStyle.POETIC_METAPHORICAL
        }
        
        base_style = style_mapping.get(primary_archetype, CommunicationStyle.WARM_CONVERSATIONAL)
        
        # Adjust based on emotional state
        if emotional_profile.current_emotion in [EmotionalState.ANXIOUS, EmotionalState.FEARFUL]:
            # Use gentler, simpler communication
            if base_style == CommunicationStyle.FORMAL_TRADITIONAL:
                return CommunicationStyle.WARM_CONVERSATIONAL
            elif base_style == CommunicationStyle.SCHOLARLY_DETAILED:
                return CommunicationStyle.SIMPLE_PRACTICAL
        
        # Adjust based on user preferences
        preferred_style = user_preferences.get('communication_preference', '')
        if 'simple' in preferred_style.lower():
            return CommunicationStyle.SIMPLE_PRACTICAL
        elif 'formal' in preferred_style.lower():
            return CommunicationStyle.FORMAL_TRADITIONAL
        elif 'playful' in preferred_style.lower():
            return CommunicationStyle.PLAYFUL_LIGHT
        
        return base_style
    
    async def generate_personalized_response(self, user_input: str,
                                           emotional_profile: EmotionalProfile,
                                           personality_config: PersonalityConfiguration,
                                           base_response: str) -> str:
        """Generate a personalized response based on emotional state and personality"""
        try:
            # Get personality template
            personality = self.personality_templates[personality_config.primary_archetype]
            response_style = personality["response_style"]
            
            # Select appropriate opening, transition, and closing phrases
            opening = np.random.choice(response_style["opening_phrases"])
            closing = np.random.choice(response_style["closing_phrases"])
            
            # Adapt base response tone based on emotional state
            adapted_response = await self._adapt_response_for_emotion(
                base_response, emotional_profile, personality_config
            )
            
            # Add dharmic remedy if appropriate
            dharmic_guidance = ""
            if emotional_profile.current_emotion in self.emotional_patterns:
                emotion_pattern = self.emotional_patterns[emotional_profile.current_emotion]
                if emotional_profile.emotional_intensity > 60:  # High intensity
                    remedy = np.random.choice(emotion_pattern["dharmic_remedies"])
                    dharmic_guidance = f"\n\n🕉️ *Dharmic Guidance: {remedy} may bring particular peace to your current experience.*"
            
            # Construct final response
            if personality_config.communication_style == CommunicationStyle.SIMPLE_PRACTICAL:
                # Simpler format for practical style
                final_response = f"{adapted_response}{dharmic_guidance}"
            else:
                # Full format with personality styling
                final_response = f"{opening}\n\n{adapted_response}\n\n{closing}{dharmic_guidance}"
            
            # Add emotional validation if needed
            if emotional_profile.emotional_intensity > 70:
                validation = await self._generate_emotional_validation(emotional_profile)
                final_response = f"{validation}\n\n{final_response}"
            
            logger.info(f"🎭 Generated personalized response with {personality_config.primary_archetype}")
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Error generating personalized response: {e}")
            return base_response  # Fallback to original response
    
    async def _adapt_response_for_emotion(self, base_response: str,
                                        emotional_profile: EmotionalProfile,
                                        personality_config: PersonalityConfiguration) -> str:
        """Adapt the response tone and content for the user's emotional state"""
        
        emotion = emotional_profile.current_emotion
        intensity = emotional_profile.emotional_intensity
        
        # Add emotional-specific guidance
        if emotion == EmotionalState.ANXIOUS and intensity > 60:
            # Add grounding techniques
            grounding_addition = "\n\n🌬️ *When anxiety arises, try this: Take three deep breaths, notice five things you can see, four things you can touch, three things you can hear. This brings you back to the present moment where peace resides.*"
            base_response += grounding_addition
            
        elif emotion == EmotionalState.SAD and intensity > 60:
            # Add gentle encouragement
            encouragement = "\n\n💝 *Your sadness is honored here. Like rain nourishing the earth, tears can water the seeds of compassion and wisdom within you. You are not alone in this experience.*"
            base_response += encouragement
            
        elif emotion == EmotionalState.ANGRY and intensity > 60:
            # Add cooling perspective
            cooling = "\n\n🌊 *Anger often carries important information about our values and boundaries. Like fire, it can illuminate truth when channeled with wisdom. Perhaps this feeling is guiding you toward necessary change.*"
            base_response += cooling
            
        elif emotion == EmotionalState.CONFUSED:
            # Add clarity support
            clarity = "\n\n🔍 *Confusion often precedes breakthrough understanding. It's the mind's way of saying 'I'm ready to learn something new.' Trust that clarity will emerge as you remain open and patient with the process.*"
            base_response += clarity
        
        # Adjust language based on communication style
        if personality_config.communication_style == CommunicationStyle.GENTLE_NURTURING:
            # Add nurturing language
            base_response = base_response.replace("you should", "you might gently consider")
            base_response = base_response.replace("you must", "you may find it helpful to")
            base_response = base_response.replace("it's important", "it's beautifully important")
            
        elif personality_config.communication_style == CommunicationStyle.FORMAL_TRADITIONAL:
            # Use more formal spiritual language
            base_response = base_response.replace("you can", "one may")
            base_response = base_response.replace("your", "one's")
        
        return base_response
    
    async def _generate_emotional_validation(self, emotional_profile: EmotionalProfile) -> str:
        """Generate appropriate emotional validation"""
        
        emotion = emotional_profile.current_emotion
        intensity = emotional_profile.emotional_intensity
        
        validations = {
            EmotionalState.ANXIOUS: [
                "I sense the weight of worry you're carrying, and I want you to know that your anxiety is completely understandable.",
                "Your nervous system is trying to protect you - this anxiety shows how much you care.",
                "Feeling anxious in uncertain times is a natural human response. You're not broken; you're human."
            ],
            EmotionalState.SAD: [
                "Your sadness is sacred and deserves to be witnessed with compassion.",
                "I feel the depth of your heart's experience, and I honor the tenderness of this moment.",
                "Sadness often reveals the profound capacity of your heart to love and care."
            ],
            EmotionalState.ANGRY: [
                "Your anger is valid and may be pointing toward something that needs attention or change.",
                "I hear the fire in your words, and I sense there's important energy here to be understood.",
                "Anger often carries the flame of justice and the desire for things to be better."
            ],
            EmotionalState.FRUSTRATED: [
                "I can feel your frustration, and it's completely understandable given what you're experiencing.",
                "This frustration shows your commitment to growth and positive change.",
                "Your impatience might be a sign of your readiness to move forward."
            ]
        }
        
        if emotion in validations:
            validation = np.random.choice(validations[emotion])
            if intensity > 80:
                validation += " The intensity of what you're feeling speaks to the depth of your experience."
            return f"💙 {validation}"
        
        return "💙 I see you in this moment, and your feelings are completely valid and welcome here."

# Create FastAPI app for personality system
app = FastAPI(title="DharmaMind AI Personality Engine", version="1.0.0")
personality_engine = AdvancedEmotionalIntelligence()

@app.post("/analyze-emotion")
async def analyze_emotion(user_input: str, conversation_history: List[Dict[str, Any]] = None):
    """Analyze user's emotional state"""
    emotional_profile = await personality_engine.analyze_emotional_state(user_input, conversation_history)
    
    return {
        'current_emotion': emotional_profile.current_emotion,
        'emotional_intensity': emotional_profile.emotional_intensity,
        'emotional_stability': emotional_profile.emotional_stability,
        'recent_emotions': [(e.value, t.isoformat()) for e, t in emotional_profile.recent_emotions]
    }

@app.post("/determine-personality")
async def determine_personality(emotional_profile_data: Dict[str, Any], 
                              user_preferences: Dict[str, Any] = None,
                              conversation_context: str = ""):
    """Determine optimal AI personality configuration"""
    
    # Reconstruct emotional profile from data
    emotional_profile = EmotionalProfile(
        current_emotion=EmotionalState(emotional_profile_data['current_emotion']),
        emotional_intensity=emotional_profile_data['emotional_intensity'],
        emotional_stability=emotional_profile_data['emotional_stability'],
        recent_emotions=[],
        emotional_triggers={},
        coping_mechanisms=[],
        emotional_growth_areas=[]
    )
    
    personality_config = await personality_engine.determine_optimal_personality(
        emotional_profile, user_preferences, conversation_context
    )
    
    return {
        'primary_archetype': personality_config.primary_archetype,
        'secondary_traits': personality_config.secondary_traits,
        'communication_style': personality_config.communication_style,
        'wisdom_depth': personality_config.wisdom_depth,
        'compassion_level': personality_config.compassion_level,
        'formality_preference': personality_config.formality_preference,
        'spiritual_tradition_focus': personality_config.spiritual_tradition_focus
    }

@app.post("/generate-response")
async def generate_personalized_response(user_input: str,
                                       emotional_profile_data: Dict[str, Any],
                                       personality_config_data: Dict[str, Any],
                                       base_response: str):
    """Generate personalized response based on emotional state and personality"""
    
    # Reconstruct objects from data
    emotional_profile = EmotionalProfile(
        current_emotion=EmotionalState(emotional_profile_data['current_emotion']),
        emotional_intensity=emotional_profile_data['emotional_intensity'],
        emotional_stability=emotional_profile_data['emotional_stability'],
        recent_emotions=[],
        emotional_triggers={},
        coping_mechanisms=[],
        emotional_growth_areas=[]
    )
    
    personality_config = PersonalityConfiguration(
        primary_archetype=PersonalityArchetype(personality_config_data['primary_archetype']),
        secondary_traits=[PersonalityArchetype(t) for t in personality_config_data['secondary_traits']],
        communication_style=CommunicationStyle(personality_config_data['communication_style']),
        wisdom_depth=personality_config_data['wisdom_depth'],
        compassion_level=personality_config_data['compassion_level'],
        formality_preference=personality_config_data['formality_preference'],
        spiritual_tradition_focus=personality_config_data['spiritual_tradition_focus'],
        personality_adaptation_history=[]
    )
    
    personalized_response = await personality_engine.generate_personalized_response(
        user_input, emotional_profile, personality_config, base_response
    )
    
    return {
        'personalized_response': personalized_response,
        'personality_used': personality_config.primary_archetype,
        'communication_style': personality_config.communication_style,
        'emotional_adaptation': emotional_profile.current_emotion
    }

@app.get("/personality-templates")
async def get_personality_templates():
    """Get all available personality templates"""
    templates = {}
    for archetype, template in personality_engine.personality_templates.items():
        templates[archetype.value] = {
            'name': template['name'],
            'description': template['description'],
            'traits': template['traits'],
            'speech_patterns': template['speech_patterns']
        }
    return templates

if __name__ == "__main__":
    import uvicorn
    print("🎭 Starting DharmaMind AI Personality & Emotional Intelligence Engine...")
    print("🧠 Emotion Analysis: http://localhost:8082/analyze-emotion")
    print("🎭 Personality Config: http://localhost:8082/determine-personality")
    print("💬 Response Generation: http://localhost:8082/generate-response")
    uvicorn.run(app, host="0.0.0.0", port=8082)
