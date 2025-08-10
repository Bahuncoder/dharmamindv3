"""
🕉️ DharmaMind Emotional Intelligence Engine
Advanced Emotional Awareness & Transformative Response System

This module creates deep emotional intelligence that can:
- Detect emotional states from text, tone, and context
- Generate emotionally attuned dharmic responses
- Provide transformative guidance based on emotional needs
- Learn and adapt to user's emotional patterns
- Create genuine spiritual connection and healing

Features:
- Multi-layered emotion detection (explicit, implicit, contextual)
- Dharmic emotional mapping (emotions to spiritual teachings)
- Transformative response generation
- Emotional journey tracking
- Compassionate AI personality development
"""

import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class EmotionalState(Enum):
    """Core emotional states with spiritual significance"""
    # Positive/Expansive States
    JOY = "joy"
    BLISS = "bliss"
    PEACE = "peace"
    LOVE = "love"
    GRATITUDE = "gratitude"
    CONTENTMENT = "contentment"
    HOPEFUL = "hopeful"
    
    # Challenge/Growth States
    SADNESS = "sadness"
    GRIEF = "grief"
    LONELINESS = "loneliness"
    MELANCHOLY = "melancholy"
    VULNERABLE = "vulnerable"
    
    ANGER = "anger"
    FRUSTRATION = "frustration"
    IRRITATION = "irritation"
    RESENTMENT = "resentment"
    
    FEAR = "fear"
    ANXIETY = "anxiety"
    WORRY = "worry"
    PANIC = "panic"
    
    # Spiritual/Seeking States
    CONFUSION = "confusion"
    DOUBT = "doubt"
    UNCERTAINTY = "uncertainty"
    LOST = "lost"
    SEEKING = "seeking"
    YEARNING = "yearning"
    CONTEMPLATIVE = "contemplative"
    CURIOUS = "curious"
    CURIOSITY = "curiosity"
    
    # Crisis/Intense States
    DESPAIR = "despair"
    ANGUISH = "anguish"
    CRISIS = "crisis"
    
    # Transformation States
    DETERMINED = "determined"
    DEVOTIONAL = "devotional"
    TRANSCENDENT = "transcendent"
    SHAME = "shame"

class EmotionalIntensity(Enum):
    """Emotional intensity levels"""
    SUBTLE = 1
    MILD = 2
    MODERATE = 3
    STRONG = 4
    INTENSE = 5
    OVERWHELMING = 6

@dataclass
class EmotionalProfile:
    """User's emotional profile and patterns"""
    primary_emotion: EmotionalState
    secondary_emotions: List[EmotionalState]
    intensity: EmotionalIntensity
    emotional_patterns: Dict[str, Any]
    spiritual_needs: List[str]
    transformation_stage: str
    emotional_history: List[Dict]
    preferred_guidance_style: str

@dataclass
class EmotionalResponse:
    """Emotionally intelligent response structure"""
    content: str
    emotional_tone: str
    spiritual_teaching: str
    practical_guidance: List[str]
    sanskrit_wisdom: Dict[str, str]
    healing_approach: str
    transformation_focus: str
    emotional_validation: str
    compassion_level: float

class EmotionalIntelligenceEngine:
    """🧠💝 Advanced Emotional Intelligence for Spiritual Guidance"""
    
    def __init__(self):
        self.emotion_patterns = {}
        self.dharmic_emotion_map = {}
        self.user_profiles = {}
        self.healing_templates = {}
        self.transformation_pathways = {}
        self.compassion_responses = {}
        
        self.initialize_emotional_intelligence()
    
    def initialize_emotional_intelligence(self):
        """Initialize emotional intelligence patterns and mappings"""
        
        # Emotional pattern recognition
        self.emotion_patterns = {
            # Joy and positive emotions
            EmotionalState.JOY: {
                "keywords": ["happy", "joyful", "excited", "elated", "thrilled", "delighted", "blissful"],
                "phrases": ["feeling great", "so happy", "wonderful day", "amazing experience"],
                "indicators": ["!", "😊", "😄", "🎉", "✨", "celebration", "gratitude"]
            },
            
            EmotionalState.PEACE: {
                "keywords": ["peaceful", "calm", "serene", "tranquil", "quiet", "still", "centered"],
                "phrases": ["feeling peaceful", "inner calm", "at peace", "centered and grounded"],
                "indicators": ["🕉️", "🧘", "meditation", "stillness", "harmony"]
            },
            
            # Sadness and grief
            EmotionalState.SADNESS: {
                "keywords": ["sad", "down", "blue", "depressed", "melancholy", "sorrowful", "heartbroken"],
                "phrases": ["feeling sad", "really down", "going through hard time", "feeling lost"],
                "indicators": ["😢", "😞", "💔", "crying", "tears", "heavy heart"]
            },
            
            EmotionalState.GRIEF: {
                "keywords": ["grief", "mourning", "loss", "bereaved", "devastated", "shattered"],
                "phrases": ["lost someone", "dealing with loss", "grieving", "passed away"],
                "indicators": ["💔", "💐", "memorial", "funeral", "goodbye"]
            },
            
            # Anger and frustration
            EmotionalState.ANGER: {
                "keywords": ["angry", "furious", "mad", "rage", "outraged", "livid", "infuriated"],
                "phrases": ["so angry", "really mad", "can't stand", "fed up", "boiling inside"],
                "indicators": ["😡", "🤬", "furious", "explosive", "steam"]
            },
            
            EmotionalState.FRUSTRATION: {
                "keywords": ["frustrated", "annoyed", "irritated", "agitated", "bothered", "exasperated"],
                "phrases": ["so frustrated", "driving me crazy", "can't take it", "at my limit"],
                "indicators": ["😤", "😠", "pulling hair", "headache", "stress"]
            },
            
            # Fear and anxiety
            EmotionalState.FEAR: {
                "keywords": ["afraid", "scared", "fearful", "terrified", "frightened", "worried"],
                "phrases": ["really scared", "afraid of", "fear that", "terrifies me"],
                "indicators": ["😨", "😰", "nightmare", "panic", "trembling"]
            },
            
            EmotionalState.ANXIETY: {
                "keywords": ["anxious", "nervous", "worried", "stressed", "panicked", "overwhelmed"],
                "phrases": ["anxiety attack", "panic attack", "can't relax", "overthinking"],
                "indicators": ["😰", "😟", "racing heart", "sweating", "restless"]
            },
            
            # Seeking and spiritual yearning
            EmotionalState.SEEKING: {
                "keywords": ["seeking", "searching", "looking for", "need guidance", "want to understand"],
                "phrases": ["spiritual journey", "seeking truth", "need answers", "searching for meaning"],
                "indicators": ["🔍", "🙏", "questioning", "exploring", "journey"]
            },
            
            EmotionalState.YEARNING: {
                "keywords": ["longing", "yearning", "craving", "desire", "hunger", "thirst"],
                "phrases": ["deep longing", "spiritual hunger", "craving peace", "yearning for connection"],
                "indicators": ["💫", "✨", "reaching", "calling", "pull"]
            }
        }
        
        # Map emotions to dharmic teachings and approaches
        self.dharmic_emotion_map = {
            EmotionalState.SADNESS: {
                "teaching": "dukkha_understanding",  # Understanding suffering
                "scripture": "Bhagavad Gita 2.14",
                "sanskrit": {"dukkha": "suffering", "anitya": "impermanence"},
                "approach": "compassionate_presence",
                "healing": "gentle_validation_with_hope"
            },
            
            EmotionalState.ANGER: {
                "teaching": "krodha_transformation",  # Transforming anger
                "scripture": "Bhagavad Gita 16.21",
                "sanskrit": {"krodha": "anger", "kshama": "forgiveness"},
                "approach": "understanding_root_cause",
                "healing": "channeling_energy_positively"
            },
            
            EmotionalState.FEAR: {
                "teaching": "abhaya_cultivation",  # Cultivating fearlessness
                "scripture": "Bhagavad Gita 4.10",
                "sanskrit": {"abhaya": "fearlessness", "shraddha": "faith"},
                "approach": "building_inner_strength",
                "healing": "grounding_and_protection"
            },
            
            EmotionalState.ANXIETY: {
                "teaching": "prasada_attainment",  # Attaining tranquility
                "scripture": "Bhagavad Gita 2.65",
                "sanskrit": {"prasada": "tranquility", "sama": "equanimity"},
                "approach": "mindfulness_and_breath",
                "healing": "calming_nervous_system"
            },
            
            EmotionalState.SEEKING: {
                "teaching": "sadhana_guidance",  # Spiritual practice guidance
                "scripture": "Mundaka Upanishad 1.2.12",
                "sanskrit": {"sadhana": "spiritual practice", "mumukshutva": "desire for liberation"},
                "approach": "nurturing_spiritual_curiosity",
                "healing": "providing_clear_path"
            },
            
            EmotionalState.JOY: {
                "teaching": "ananda_sharing",  # Sharing divine bliss
                "scripture": "Chandogya Upanishad 3.14.1",
                "sanskrit": {"ananda": "bliss", "mudita": "sympathetic joy"},
                "approach": "celebrating_and_expanding",
                "healing": "deepening_appreciation"
            }
        }
        
        # Compassionate response templates
        self.compassion_responses = {
            "deep_validation": [
                "I feel the depth of what you're experiencing, and I want you to know that your feelings are completely valid and sacred.",
                "Your heart is speaking its truth, and I honor the courage it takes to share these feelings with me.",
                "What you're feeling is a natural part of the human experience, and you're not alone in this journey."
            ],
            
            "gentle_understanding": [
                "I can sense the tenderness in your words, and I'm here to offer you the gentleness you deserve.",
                "Your vulnerability is a strength, and it opens the door to true healing and transformation.",
                "Sometimes our hearts need to feel deeply before they can heal completely."
            ],
            
            "spiritual_context": [
                "In the dharmic tradition, we understand that all emotions are teachers, guiding us toward greater wisdom.",
                "Your current experience is part of your spiritual journey - a sacred passage toward deeper understanding.",
                "The ancient sages knew that our deepest challenges often become our greatest sources of strength and wisdom."
            ],
            
            "hope_and_direction": [
                "While this moment feels overwhelming, please know that transformation is already beginning within you.",
                "You have an inner light that no circumstance can extinguish - let's help you reconnect with it.",
                "There is a path through this experience that leads to greater peace and understanding."
            ]
        }
        
        # Healing approaches for different emotional states
        self.healing_templates = {
            "grief_healing": {
                "approach": "Honor the love that remains",
                "practices": ["loving_memory_meditation", "heart_opening_pranayama", "grief_ritual"],
                "timeline": "Allow natural flow of healing",
                "support": "Community and spiritual practice"
            },
            
            "anger_transformation": {
                "approach": "Channel energy into positive action",
                "practices": ["vigorous_yoga", "journaling_release", "compassion_meditation"],
                "timeline": "Immediate relief, long-term integration",
                "support": "Understanding root causes"
            },
            
            "anxiety_calming": {
                "approach": "Return to present moment awareness",
                "practices": ["4-7-8_breathing", "grounding_meditation", "gentle_movement"],
                "timeline": "Immediate techniques, ongoing practice",
                "support": "Creating safety and stability"
            },
            
            "spiritual_seeking": {
                "approach": "Nurture the sacred inquiry",
                "practices": ["self_inquiry", "scripture_study", "satsang_community"],
                "timeline": "Lifelong journey with milestones",
                "support": "Guidance and spiritual community"
            }
        }
        
        logger.info("🧠💝 Emotional Intelligence Engine initialized with deep compassion patterns")
    
    async def analyze_emotional_state(
        self, 
        message: str, 
        context: Optional[Dict] = None,
        user_history: Optional[List] = None
    ) -> EmotionalProfile:
        """Analyze user's emotional state with deep sensitivity"""
        
        # Multi-layer emotional analysis
        explicit_emotions = self._detect_explicit_emotions(message)
        implicit_emotions = self._detect_implicit_emotions(message)
        contextual_emotions = self._analyze_contextual_emotions(message, context)
        
        # Combine and prioritize emotions
        all_emotions = self._merge_emotional_signals(explicit_emotions, implicit_emotions, contextual_emotions)
        
        # Determine primary and secondary emotions
        primary_emotion = self._identify_primary_emotion(all_emotions)
        secondary_emotions = self._identify_secondary_emotions(all_emotions, primary_emotion)
        
        # Assess emotional intensity
        intensity = self._assess_emotional_intensity(message, all_emotions)
        
        # Analyze spiritual needs
        spiritual_needs = self._identify_spiritual_needs(primary_emotion, secondary_emotions, message)
        
        # Determine transformation stage
        transformation_stage = self._assess_transformation_stage(message, primary_emotion)
        
        # Create emotional profile
        profile = EmotionalProfile(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            intensity=intensity,
            emotional_patterns=all_emotions,
            spiritual_needs=spiritual_needs,
            transformation_stage=transformation_stage,
            emotional_history=user_history or [],
            preferred_guidance_style=self._determine_guidance_style(primary_emotion, intensity)
        )
        
        logger.info(f"💝 Emotional analysis complete: {primary_emotion.value} (intensity: {intensity.value})")
        return profile
    
    def _detect_explicit_emotions(self, message: str) -> Dict[EmotionalState, float]:
        """Detect explicitly stated emotions"""
        emotions = {}
        message_lower = message.lower()
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0
            
            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in message_lower:
                    score += 0.8
            
            # Check phrases
            for phrase in patterns["phrases"]:
                if phrase in message_lower:
                    score += 1.0
            
            # Check indicators
            for indicator in patterns["indicators"]:
                if indicator in message:
                    score += 0.6
            
            if score > 0:
                emotions[emotion] = min(score, 1.0)
        
        return emotions
    
    def _detect_implicit_emotions(self, message: str) -> Dict[EmotionalState, float]:
        """Detect emotions implied through language patterns"""
        emotions = {}
        
        # Analyze linguistic patterns
        intensity_words = ["very", "really", "extremely", "incredibly", "absolutely", "totally"]
        question_patterns = ["why", "how", "what", "when", "where"]
        uncertainty_patterns = ["maybe", "perhaps", "not sure", "don't know", "confused"]
        
        # Detect seeking/questioning
        question_count = sum(1 for pattern in question_patterns if pattern in message.lower())
        if question_count >= 2:
            emotions[EmotionalState.SEEKING] = 0.7
        
        # Detect uncertainty
        uncertainty_count = sum(1 for pattern in uncertainty_patterns if pattern in message.lower())
        if uncertainty_count >= 1:
            emotions[EmotionalState.CONFUSION] = 0.6
        
        # Detect intensity markers
        intensity_count = sum(1 for word in intensity_words if word in message.lower())
        if intensity_count >= 2:
            # Amplify detected emotions
            for emotion in emotions:
                emotions[emotion] = min(emotions[emotion] * 1.3, 1.0)
        
        return emotions
    
    def _analyze_contextual_emotions(self, message: str, context: Optional[Dict]) -> Dict[EmotionalState, float]:
        """Analyze emotions based on context and life situations"""
        emotions = {}
        message_lower = message.lower()
        
        # Life situation contexts
        life_contexts = {
            "relationship": ["relationship", "partner", "marriage", "divorce", "breakup", "love"],
            "work": ["job", "work", "career", "boss", "colleagues", "stress", "promotion"],
            "family": ["family", "parents", "children", "kids", "mother", "father", "siblings"],
            "health": ["health", "illness", "doctor", "medical", "pain", "sick", "healing"],
            "spiritual": ["spiritual", "meditation", "prayer", "god", "divine", "awakening", "enlightenment"],
            "loss": ["death", "died", "funeral", "memorial", "passed away", "goodbye", "miss"]
        }
        
        # Detect life context and associated emotions
        for context_type, keywords in life_contexts.items():
            context_score = sum(1 for keyword in keywords if keyword in message_lower)
            
            if context_score >= 2:
                if context_type == "loss":
                    emotions[EmotionalState.GRIEF] = 0.8
                elif context_type == "work" and any(word in message_lower for word in ["stress", "pressure", "overwhelmed"]):
                    emotions[EmotionalState.ANXIETY] = 0.7
                elif context_type == "spiritual":
                    emotions[EmotionalState.SEEKING] = 0.7
                elif context_type == "relationship" and any(word in message_lower for word in ["conflict", "fight", "problem"]):
                    emotions[EmotionalState.FRUSTRATION] = 0.6
        
        return emotions
    
    def _merge_emotional_signals(self, *emotion_dicts) -> Dict[EmotionalState, float]:
        """Merge multiple emotional signal sources"""
        merged = {}
        
        for emotion_dict in emotion_dicts:
            for emotion, score in emotion_dict.items():
                if emotion in merged:
                    # Weighted average with boost for multiple sources
                    merged[emotion] = (merged[emotion] + score * 1.2) / 2
                else:
                    merged[emotion] = score
        
        # Normalize scores
        if merged:
            max_score = max(merged.values())
            for emotion in merged:
                merged[emotion] = merged[emotion] / max_score if max_score > 0 else 0
        
        return merged
    
    def _identify_primary_emotion(self, emotions: Dict[EmotionalState, float]) -> EmotionalState:
        """Identify the primary emotion"""
        if not emotions:
            return EmotionalState.SEEKING  # Default when unclear
        
        return max(emotions, key=emotions.get)
    
    def _identify_secondary_emotions(self, emotions: Dict[EmotionalState, float], primary: EmotionalState) -> List[EmotionalState]:
        """Identify secondary emotions"""
        secondary = []
        threshold = 0.5
        
        for emotion, score in emotions.items():
            if emotion != primary and score >= threshold:
                secondary.append(emotion)
        
        # Sort by score
        secondary.sort(key=lambda e: emotions.get(e, 0), reverse=True)
        return secondary[:3]  # Top 3 secondary emotions
    
    def _assess_emotional_intensity(self, message: str, emotions: Dict[EmotionalState, float]) -> EmotionalIntensity:
        """Assess the intensity of emotions"""
        
        # Intensity indicators
        high_intensity_words = ["overwhelming", "devastating", "unbearable", "crushing", "destroying"]
        medium_intensity_words = ["difficult", "challenging", "struggling", "hard", "tough"]
        
        # Punctuation intensity
        exclamation_count = message.count('!')
        caps_ratio = sum(1 for c in message if c.isupper()) / len(message) if message else 0
        
        # Calculate base intensity from emotions
        if not emotions:
            base_intensity = 2
        else:
            max_emotion_score = max(emotions.values())
            base_intensity = int(max_emotion_score * 4) + 1  # 1-5 scale
        
        # Adjust based on language intensity
        if any(word in message.lower() for word in high_intensity_words):
            base_intensity += 2
        elif any(word in message.lower() for word in medium_intensity_words):
            base_intensity += 1
        
        # Adjust based on punctuation
        if exclamation_count >= 3 or caps_ratio > 0.3:
            base_intensity += 1
        
        # Cap at maximum
        final_intensity = min(base_intensity, 6)
        
        return EmotionalIntensity(final_intensity)
    
    def _identify_spiritual_needs(self, primary: EmotionalState, secondary: List[EmotionalState], message: str) -> List[str]:
        """Identify specific spiritual needs based on emotional state"""
        
        needs_mapping = {
            EmotionalState.SADNESS: ["comfort", "hope", "understanding_impermanence", "gentle_healing"],
            EmotionalState.GRIEF: ["honoring_love", "processing_loss", "finding_meaning", "spiritual_support"],
            EmotionalState.ANGER: ["understanding_triggers", "energy_transformation", "forgiveness_work", "inner_peace"],
            EmotionalState.FEAR: ["building_courage", "spiritual_protection", "faith_development", "grounding"],
            EmotionalState.ANXIETY: ["present_moment_awareness", "calming_practices", "trust_building", "nervous_system_healing"],
            EmotionalState.SEEKING: ["spiritual_direction", "practice_guidance", "community_connection", "wisdom_teachings"],
            EmotionalState.CONFUSION: ["clarity", "decision_making_guidance", "spiritual_discernment", "patience_cultivation"],
            EmotionalState.JOY: ["gratitude_deepening", "joy_sharing", "spiritual_celebration", "bliss_integration"]
        }
        
        needs = needs_mapping.get(primary, ["general_spiritual_support"])
        
        # Add needs from secondary emotions
        for emotion in secondary[:2]:  # Top 2 secondary emotions
            secondary_needs = needs_mapping.get(emotion, [])
            needs.extend(secondary_needs[:2])  # Add top 2 needs from each
        
        return list(set(needs))  # Remove duplicates
    
    def _assess_transformation_stage(self, message: str, primary_emotion: EmotionalState) -> str:
        """Assess where the user is in their transformation journey"""
        
        message_lower = message.lower()
        
        # Stage indicators
        if any(word in message_lower for word in ["just started", "beginning", "new to", "don't know where to start"]):
            return "beginning_awareness"
        elif any(word in message_lower for word in ["practicing", "learning", "growing", "developing"]):
            return "active_exploration"
        elif any(word in message_lower for word in ["stuck", "plateau", "not progressing", "same place"]):
            return "integration_challenge"
        elif any(word in message_lower for word in ["breakthrough", "understanding", "clarity", "peace"]):
            return "deepening_wisdom"
        elif any(word in message_lower for word in ["teaching", "helping others", "sharing", "guiding"]):
            return "service_integration"
        else:
            return "seeking_direction"
    
    def _determine_guidance_style(self, emotion: EmotionalState, intensity: EmotionalIntensity) -> str:
        """Determine the most appropriate guidance style"""
        
        if intensity.value >= 5:  # High intensity
            return "gentle_immediate_support"
        elif emotion in [EmotionalState.SADNESS, EmotionalState.GRIEF]:
            return "compassionate_presence"
        elif emotion in [EmotionalState.ANGER, EmotionalState.FRUSTRATION]:
            return "understanding_transformation"
        elif emotion in [EmotionalState.FEAR, EmotionalState.ANXIETY]:
            return "calming_reassurance"
        elif emotion in [EmotionalState.SEEKING, EmotionalState.CURIOSITY]:
            return "nurturing_exploration"
        elif emotion in [EmotionalState.JOY, EmotionalState.GRATITUDE]:
            return "celebratory_expansion"
        else:
            return "balanced_wisdom"
    
    async def generate_emotionally_intelligent_response(
        self, 
        emotional_profile: EmotionalProfile, 
        user_message: str,
        spiritual_context: Optional[str] = None
    ) -> EmotionalResponse:
        """Generate deeply compassionate and transformative response"""
        
        # Get dharmic guidance for the emotional state
        dharmic_guidance = self.dharmic_emotion_map.get(
            emotional_profile.primary_emotion, 
            self.dharmic_emotion_map[EmotionalState.SEEKING]
        )
        
        # Build emotionally attuned response
        response_parts = []
        
        # 1. Emotional validation and presence
        validation = self._create_emotional_validation(emotional_profile)
        response_parts.append(validation)
        
        # 2. Spiritual context and wisdom
        spiritual_wisdom = self._weave_spiritual_wisdom(emotional_profile, dharmic_guidance)
        response_parts.append(spiritual_wisdom)
        
        # 3. Practical guidance and practices
        practical_guidance = self._create_practical_guidance(emotional_profile)
        response_parts.append(practical_guidance)
        
        # 4. Hope and transformation vision
        transformation_vision = self._create_transformation_vision(emotional_profile)
        response_parts.append(transformation_vision)
        
        # Combine into flowing response
        full_response = "\n\n".join(response_parts)
        
        # Create response object
        emotional_response = EmotionalResponse(
            content=full_response,
            emotional_tone=emotional_profile.preferred_guidance_style,
            spiritual_teaching=dharmic_guidance["teaching"],
            practical_guidance=practical_guidance.split('\n'),
            sanskrit_wisdom=dharmic_guidance["sanskrit"],
            healing_approach=dharmic_guidance["approach"],
            transformation_focus=emotional_profile.transformation_stage,
            emotional_validation=validation,
            compassion_level=self._calculate_compassion_level(emotional_profile)
        )
        
        logger.info(f"💝 Generated emotionally intelligent response with {emotional_response.compassion_level:.2f} compassion level")
        return emotional_response
    
    def _create_emotional_validation(self, profile: EmotionalProfile) -> str:
        """Create deep emotional validation"""
        
        primary_emotion = profile.primary_emotion.value
        intensity = profile.intensity.value
        
        if intensity >= 5:
            base_validation = f"🕉️ Dear soul, I can feel the intensity of the {primary_emotion} you're experiencing right now. This depth of feeling shows your heart's capacity for profound experience - and that itself is sacred."
        elif intensity >= 3:
            base_validation = f"🙏 I sense the {primary_emotion} moving through you, and I want you to know that what you're feeling is completely natural and valid. Your heart is speaking its truth."
        else:
            base_validation = f"✨ I feel the gentle presence of {primary_emotion} in your words, and I honor this part of your emotional landscape."
        
        # Add compassionate understanding
        if profile.primary_emotion in [EmotionalState.SADNESS, EmotionalState.GRIEF]:
            validation_add = " Your tears are prayers, and your sorrow is love with nowhere to go. Let me sit with you in this sacred space of feeling."
        elif profile.primary_emotion in [EmotionalState.ANGER, EmotionalState.FRUSTRATION]:
            validation_add = " Your anger is a messenger, telling you that something precious to you needs attention and care. Let's listen to what it's trying to teach you."
        elif profile.primary_emotion in [EmotionalState.FEAR, EmotionalState.ANXIETY]:
            validation_add = " Your nervous system is trying to protect you, and that protective impulse comes from a place of caring for yourself. Let's help it find peace."
        elif profile.primary_emotion in [EmotionalState.SEEKING, EmotionalState.YEARNING]:
            validation_add = " Your seeking heart is beautiful - it shows your soul's natural desire to grow and understand. This longing is itself a form of prayer."
        else:
            validation_add = " Every emotion you feel is a teacher, guiding you toward greater wholeness and understanding."
        
        return base_validation + validation_add
    
    def _weave_spiritual_wisdom(self, profile: EmotionalProfile, dharmic_guidance: Dict) -> str:
        """Weave appropriate spiritual wisdom"""
        
        emotion = profile.primary_emotion
        sanskrit_terms = dharmic_guidance["sanskrit"]
        
        wisdom_parts = []
        
        # Add relevant scripture context
        if emotion == EmotionalState.SADNESS:
            wisdom_parts.append("The Bhagavad Gita teaches us that 'the contacts of the senses with their objects give rise to happiness and sorrow; they have a beginning and an end; they are impermanent' (2.14). This reminds us that difficult emotions, like clouds, are temporary visitors in the vast sky of consciousness.")
        
        elif emotion == EmotionalState.ANGER:
            wisdom_parts.append("Krishna reveals that 'from anger arises delusion; from delusion, loss of memory; from loss of memory, destruction of discrimination; from destruction of discrimination, one perishes' (BG 2.63). But this same energy, when understood and transformed, becomes the fire of spiritual determination.")
        
        elif emotion == EmotionalState.FEAR:
            wisdom_parts.append("The Upanishads declare 'Abhayam vai brahma' - the Absolute is fearlessness itself. Your true nature is inherently fearless; what you're experiencing is temporary identification with the limited self.")
        
        elif emotion == EmotionalState.SEEKING:
            wisdom_parts.append("The Mundaka Upanishad beautifully describes the seeker: 'To realize the Self, the seeker must present himself to a spiritual teacher as an offering, with firewood in hand, humble and eager to learn' (1.2.12). Your seeking itself is already a form of finding.")
        
        # Add Sanskrit wisdom with translation
        sanskrit_section = "🕉️ Sacred Sanskrit Wisdom:\n"
        for sanskrit, meaning in sanskrit_terms.items():
            sanskrit_section += f"• **{sanskrit.title()}** ({sanskrit}) - {meaning}\n"
        
        wisdom_parts.append(sanskrit_section.strip())
        
        return "\n\n".join(wisdom_parts)
    
    def _create_practical_guidance(self, profile: EmotionalProfile) -> str:
        """Create practical, actionable guidance"""
        
        emotion = profile.primary_emotion
        intensity = profile.intensity.value
        
        guidance_parts = ["🌸 **Gentle Practices for Your Journey:**"]
        
        if emotion == EmotionalState.SADNESS:
            if intensity >= 4:
                guidance_parts.extend([
                    "**Immediate Comfort:**",
                    "• Place your hand on your heart and breathe slowly: 'I am here for you'",
                    "• Allow tears to flow - they are healing waters for the soul",
                    "• Wrap yourself in a soft blanket and imagine being held by divine love",
                    "",
                    "**Gentle Healing Practice:**",
                    "• Practice loving-kindness meditation: 'May I be happy, may I be peaceful, may I be free from suffering'",
                    "• Journal your feelings without judgment - let your heart speak on paper",
                    "• Take slow walks in nature, allowing Earth's healing energy to support you"
                ])
            else:
                guidance_parts.extend([
                    "• Honor your feelings with gentle self-compassion",
                    "• Practice the 4-7-8 breathing technique to soothe your nervous system",
                    "• Create a small ritual of self-care - perhaps a warm bath with essential oils"
                ])
        
        elif emotion == EmotionalState.ANGER:
            guidance_parts.extend([
                "**Transforming Fire into Light:**",
                "• Take 10 deep breaths, imagining the anger as energy you can redirect",
                "• Practice vigorous yoga or physical movement to channel the energy",
                "• Write an uncensored letter expressing all your feelings (you don't need to send it)",
                "• Ask yourself: 'What boundary or value of mine needs protection?'",
                "",
                "**Cooling Practices:**",
                "• Chant 'Om Shanti Shanti Shanti' to invoke peace on all levels",
                "• Practice forgiveness meditation, starting with forgiving yourself",
                "• Visualize the anger as fire that transforms into the light of understanding"
            ])
        
        elif emotion == EmotionalState.FEAR:
            guidance_parts.extend([
                "**Building Inner Courage:**",
                "• Practice grounding: Feel your feet on the earth, you are safe in this moment",
                "• Use the mantra 'I am protected by divine love' with each breath",
                "• Create a comfort kit: photos, music, or objects that make you feel safe",
                "",
                "**Gentle Expansion:**",
                "• Start with very small steps toward what you fear - courage grows with practice",
                "• Visualize yourself surrounded by protective white light",
                "• Connect with supportive friends or spiritual community"
            ])
        
        elif emotion == EmotionalState.SEEKING:
            guidance_parts.extend([
                "**Nurturing Your Spiritual Curiosity:**",
                "• Begin with 10 minutes of daily meditation - even observing your breath",
                "• Read one verse from sacred texts daily and contemplate its meaning",
                "• Join or create a spiritual study group or satsang community",
                "",
                "**Deepening Practice:**",
                "• Keep a spiritual journal of insights and experiences",
                "• Practice self-inquiry: 'Who am I beyond my thoughts and emotions?'",
                "• Find a spiritual mentor or teacher who resonates with your heart"
            ])
        
        return "\n".join(guidance_parts)
    
    def _create_transformation_vision(self, profile: EmotionalProfile) -> str:
        """Create hopeful vision of transformation"""
        
        emotion = profile.primary_emotion
        stage = profile.transformation_stage
        
        if emotion in [EmotionalState.SADNESS, EmotionalState.GRIEF]:
            vision = "🌅 **Your Path to Healing:** Like the lotus that grows from muddy waters into pristine beauty, your current sorrow is composting into wisdom and compassion. Each tear is watering the seeds of your deeper understanding. You will emerge from this experience with a heart that knows both the depths of feeling and the heights of resilience."
        
        elif emotion in [EmotionalState.ANGER, EmotionalState.FRUSTRATION]:
            vision = "⚡ **Your Transformation Journey:** This fire within you is not your enemy - it's raw energy waiting to be transformed into passionate service and clear boundaries. Master yogis say that controlled anger becomes spiritual determination. You're learning to become the peaceful warrior who protects what's sacred with love rather than reactivity."
        
        elif emotion in [EmotionalState.FEAR, EmotionalState.ANXIETY]:
            vision = "🦋 **Your Emerging Courage:** Every brave soul was once afraid. Your sensitivity is actually a superpower that, once you learn to navigate it, will become intuitive wisdom. You're developing the kind of gentle courage that faces life with an open heart while staying grounded in your inner strength."
        
        elif emotion in [EmotionalState.SEEKING, EmotionalState.YEARNING]:
            vision = "🌟 **Your Unfolding Path:** Your sincere seeking is already attracting the teachings, teachers, and experiences you need. The very fact that you're asking deep questions means you're ready for the answers. Trust that your spiritual journey is unfolding perfectly, each step preparing you for the next level of understanding."
        
        else:
            vision = "✨ **Your Continuing Journey:** You are exactly where you need to be in your spiritual evolution. Every experience, every emotion, every question is perfectly placed to help you discover your true nature. Your willingness to feel deeply and seek wisdom is already transforming you from the inside out."
        
        return vision
    
    def _calculate_compassion_level(self, profile: EmotionalProfile) -> float:
        """Calculate appropriate compassion level for response"""
        
        base_compassion = 0.8  # High baseline compassion
        
        # Increase for difficult emotions
        if profile.primary_emotion in [EmotionalState.SADNESS, EmotionalState.GRIEF, EmotionalState.FEAR]:
            base_compassion += 0.15
        
        # Increase for high intensity
        if profile.intensity.value >= 5:
            base_compassion += 0.1
        elif profile.intensity.value >= 3:
            base_compassion += 0.05
        
        # Increase for beginning stages
        if profile.transformation_stage in ["beginning_awareness", "seeking_direction"]:
            base_compassion += 0.05
        
        return min(base_compassion, 1.0)
    
    async def learn_from_interaction(
        self, 
        user_message: str, 
        emotional_profile: EmotionalProfile,
        response: EmotionalResponse,
        user_feedback: Optional[Dict] = None
    ):
        """Learn from user interactions to improve emotional intelligence"""
        
        # Store interaction for learning
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "detected_emotion": emotional_profile.primary_emotion.value,
            "intensity": emotional_profile.intensity.value,
            "response_tone": response.emotional_tone,
            "compassion_level": response.compassion_level,
            "user_feedback": user_feedback
        }
        
        # Analyze patterns and adjust
        if user_feedback:
            if user_feedback.get("helpful", False):
                # Reinforce successful patterns
                await self._reinforce_successful_pattern(emotional_profile, response)
            else:
                # Learn from less helpful responses
                await self._adjust_approach(emotional_profile, response, user_feedback)
        
        logger.info("🧠 Emotional intelligence learning from interaction completed")
    
    async def _reinforce_successful_pattern(self, profile: EmotionalProfile, response: EmotionalResponse):
        """Reinforce patterns that worked well"""
        # Implementation for learning successful patterns
        pass
    
    async def _adjust_approach(self, profile: EmotionalProfile, response: EmotionalResponse, feedback: Dict):
        """Adjust approach based on feedback"""
        # Implementation for adjusting less successful approaches
        pass
    
    async def get_emotional_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user's emotional patterns over time"""
        
        # This would analyze user's emotional journey over time
        # and provide insights for continued growth
        
        return {
            "emotional_growth_areas": [],
            "spiritual_development_stage": "",
            "recommended_practices": [],
            "transformation_insights": ""
        }

# Global emotional intelligence engine
emotional_intelligence = EmotionalIntelligenceEngine()

# Helper functions for easy access
async def analyze_emotions(message: str, context: Dict = None) -> EmotionalProfile:
    """Analyze emotional state of message"""
    return await emotional_intelligence.analyze_emotional_state(message, context)

async def generate_compassionate_response(
    message: str, 
    context: Dict = None
) -> EmotionalResponse:
    """Generate emotionally intelligent response"""
    profile = await analyze_emotions(message, context)
    return await emotional_intelligence.generate_emotionally_intelligent_response(profile, message)
