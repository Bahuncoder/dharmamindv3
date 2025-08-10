"""
🎵 DharmaMind Advanced Tone Analysis & Response Modulation System

This module analyzes the subtle emotional tones and energy patterns in user messages
to generate deeply attuned, transformative responses that can genuinely help users
transform their emotional states and spiritual growth.

Features:
- Multi-dimensional tone analysis (emotional, energetic, spiritual)
- Dharmic resonance matching
- Transformative response crafting
- Emotional calibration for healing
- Adaptive communication style
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class ToneSignature(Enum):
    """Different tone signatures in communication"""
    VULNERABLE = "vulnerable"          # Open, seeking help, raw
    CONTEMPLATIVE = "contemplative"    # Thoughtful, philosophical, deep
    URGENT = "urgent"                  # Immediate need, crisis, intense
    HOPEFUL = "hopeful"               # Optimistic, looking forward
    PEACEFUL = "peaceful"             # Calm, centered, balanced
    CONFUSED = "confused"             # Lost, uncertain, scattered
    GRATEFUL = "grateful"             # Appreciative, thankful
    SEEKING = "seeking"               # Questioning, exploring, curious
    SUFFERING = "suffering"           # Pain, distress, anguish
    CELEBRATORY = "celebratory"       # Joy, celebration, achievement


class EnergeticResonance(Enum):
    """Energetic patterns in communication"""
    HIGH_VIBRATION = "high_vibration"     # Joy, love, peace, gratitude
    NEUTRAL = "neutral"                   # Balanced, stable, clear
    LOW_VIBRATION = "low_vibration"       # Fear, anger, sadness, despair
    ASCENDING = "ascending"               # Moving from low to high
    DESCENDING = "descending"             # Moving from high to low
    TURBULENT = "turbulent"               # Chaotic, unstable, mixed


@dataclass
class ToneAnalysis:
    """Complete tone analysis of user communication"""
    primary_tone: ToneSignature
    secondary_tones: List[ToneSignature]
    energetic_resonance: EnergeticResonance
    spiritual_openness: float  # 0.0 to 1.0
    emotional_intensity: float  # 0.0 to 1.0
    transformation_readiness: float  # 0.0 to 1.0
    communication_style: str
    healing_needs: List[str]
    dharmic_alignment: float  # 0.0 to 1.0


@dataclass
class ResponseModulation:
    """How to modulate the response based on tone analysis"""
    energy_adjustment: str  # "lift", "match", "ground", "balance"
    compassion_level: float  # 0.0 to 1.0
    directness_level: float  # 0.0 to 1.0 (direct vs gentle)
    spiritual_depth: float  # 0.0 to 1.0
    practical_focus: float  # 0.0 to 1.0
    response_length: str  # "brief", "moderate", "comprehensive"
    tone_to_match: str  # The tone quality to embody in response


class AdvancedToneAnalyzer:
    """🎵 Advanced tone analysis for deeply attuned spiritual responses"""
    
    def __init__(self):
        self.tone_patterns = {}
        self.energetic_markers = {}
        self.dharmic_resonance_map = {}
        self.healing_response_templates = {}
        
        self.initialize_tone_analysis_system()
    
    def initialize_tone_analysis_system(self):
        """Initialize comprehensive tone analysis patterns"""
        
        # Tone signature patterns
        self.tone_patterns = {
            ToneSignature.VULNERABLE: {
                "keywords": ["vulnerable", "scared", "don't know", "help me", "lost", "broken"],
                "phrases": ["I don't know what to do", "feeling lost", "need help", "scared", "vulnerable"],
                "linguistic_markers": ["question_heavy", "uncertainty_words", "emotional_words"],
                "energy_markers": ["soft", "open", "seeking_safety"]
            },
            
            ToneSignature.CONTEMPLATIVE: {
                "keywords": ["wondering", "thinking", "pondering", "consider", "reflect", "contemplate"],
                "phrases": ["I've been thinking", "wondering about", "what does it mean", "how do we"],
                "linguistic_markers": ["philosophical_words", "abstract_concepts", "questioning"],
                "energy_markers": ["thoughtful", "deep", "introspective"]
            },
            
            ToneSignature.URGENT: {
                "keywords": ["urgent", "now", "immediately", "crisis", "emergency", "desperate"],
                "phrases": ["I need help now", "urgent situation", "can't wait", "crisis mode"],
                "linguistic_markers": ["time_pressure", "intensity_words", "crisis_language"],
                "energy_markers": ["intense", "pressing", "activated"]
            },
            
            ToneSignature.HOPEFUL: {
                "keywords": ["hope", "optimistic", "positive", "better", "improve", "bright"],
                "phrases": ["things are getting better", "feeling hopeful", "looking forward"],
                "linguistic_markers": ["future_oriented", "positive_words", "improvement_language"],
                "energy_markers": ["uplifting", "forward_moving", "light"]
            },
            
            ToneSignature.PEACEFUL: {
                "keywords": ["peaceful", "calm", "serene", "balanced", "centered", "tranquil"],
                "phrases": ["feeling peaceful", "at peace", "calm and centered", "inner peace"],
                "linguistic_markers": ["calm_language", "balanced_expression", "centered_perspective"],
                "energy_markers": ["stable", "grounded", "harmonious"]
            },
            
            ToneSignature.CONFUSED: {
                "keywords": ["confused", "unclear", "don't understand", "mixed up", "bewildered"],
                "phrases": ["I'm confused", "don't understand", "unclear about", "mixed feelings"],
                "linguistic_markers": ["confusion_words", "contradiction", "uncertainty"],
                "energy_markers": ["scattered", "unclear", "seeking_clarity"]
            },
            
            ToneSignature.SUFFERING: {
                "keywords": ["suffering", "pain", "agony", "torment", "anguish", "devastated"],
                "phrases": ["in pain", "suffering deeply", "can't bear", "overwhelming pain"],
                "linguistic_markers": ["pain_language", "intensity_markers", "desperation"],
                "energy_markers": ["heavy", "dense", "contracted"]
            },
            
            ToneSignature.SEEKING: {
                "keywords": ["seeking", "searching", "looking for", "want to learn", "guide me"],
                "phrases": ["seeking guidance", "looking for answers", "want to understand"],
                "linguistic_markers": ["inquiry_words", "learning_language", "guidance_seeking"],
                "energy_markers": ["curious", "open", "expanding"]
            }
        }
        
        # Energetic resonance markers
        self.energetic_markers = {
            EnergeticResonance.HIGH_VIBRATION: {
                "words": ["love", "joy", "peace", "bliss", "gratitude", "light", "divine", "beautiful"],
                "energy_indicators": ["exclamation_moderate", "positive_imagery", "uplifting_language"],
                "feeling_tone": "expansive"
            },
            
            EnergeticResonance.LOW_VIBRATION: {
                "words": ["hate", "despair", "darkness", "hopeless", "terrible", "awful", "nightmare"],
                "energy_indicators": ["heavy_language", "contraction_words", "dark_imagery"],
                "feeling_tone": "contracted"
            },
            
            EnergeticResonance.NEUTRAL: {
                "words": ["okay", "fine", "normal", "regular", "usual", "standard", "typical"],
                "energy_indicators": ["balanced_language", "matter_of_fact", "even_tone"],
                "feeling_tone": "balanced"
            },
            
            EnergeticResonance.TURBULENT: {
                "words": ["chaotic", "overwhelming", "conflicted", "torn", "mixed", "unstable"],
                "energy_indicators": ["contradictions", "emotional_swings", "instability_markers"],
                "feeling_tone": "unstable"
            }
        }
        
        # Response modulation templates
        self.healing_response_templates = {
            "vulnerable_support": {
                "energy_adjustment": "gentle_lifting",
                "opening": "🕉️ Dear soul, I feel the tender vulnerability in your words, and I want you to know that this openness is actually a form of courage...",
                "approach": "extra_gentle_with_safety",
                "practices": ["grounding", "self_compassion", "gentle_breathing"]
            },
            
            "urgent_calming": {
                "energy_adjustment": "immediate_grounding",
                "opening": "🙏 I sense the urgency of what you're experiencing. Let's take a deep breath together right now...",
                "approach": "calm_authority_with_immediate_help",
                "practices": ["emergency_breathing", "grounding", "crisis_support"]
            },
            
            "contemplative_deepening": {
                "energy_adjustment": "wisdom_matching",
                "opening": "✨ Your thoughtful inquiry touches upon profound spiritual territory. Let's explore this together with both depth and clarity...",
                "approach": "philosophical_with_practical_wisdom",
                "practices": ["contemplation", "self_inquiry", "wisdom_study"]
            },
            
            "suffering_healing": {
                "energy_adjustment": "compassionate_presence",
                "opening": "💝 I feel the depth of pain you're carrying, and I want to sit with you in this sacred space of suffering that seeks healing...",
                "approach": "deep_compassion_with_hope",
                "practices": ["grief_healing", "heart_opening", "gentle_movement"]
            }
        }
        
        logger.info("🎵 Advanced Tone Analysis System initialized")
    
    async def analyze_communication_tone(
        self, 
        message: str, 
        context: Optional[Dict] = None,
        user_history: Optional[List] = None
    ) -> ToneAnalysis:
        """Perform comprehensive tone analysis"""
        
        # Detect primary tone signature
        primary_tone = self._detect_primary_tone(message)
        
        # Detect secondary tones
        secondary_tones = self._detect_secondary_tones(message, primary_tone)
        
        # Analyze energetic resonance
        energetic_resonance = self._analyze_energetic_resonance(message)
        
        # Assess spiritual openness
        spiritual_openness = self._assess_spiritual_openness(message, context)
        
        # Measure emotional intensity
        emotional_intensity = self._measure_emotional_intensity(message)
        
        # Assess transformation readiness
        transformation_readiness = self._assess_transformation_readiness(message, context)
        
        # Determine communication style
        communication_style = self._determine_communication_style(message)
        
        # Identify healing needs
        healing_needs = self._identify_healing_needs(primary_tone, secondary_tones, message)
        
        # Calculate dharmic alignment
        dharmic_alignment = self._calculate_dharmic_alignment(message, spiritual_openness)
        
        tone_analysis = ToneAnalysis(
            primary_tone=primary_tone,
            secondary_tones=secondary_tones,
            energetic_resonance=energetic_resonance,
            spiritual_openness=spiritual_openness,
            emotional_intensity=emotional_intensity,
            transformation_readiness=transformation_readiness,
            communication_style=communication_style,
            healing_needs=healing_needs,
            dharmic_alignment=dharmic_alignment
        )
        
        logger.info(f"🎵 Tone analysis complete: {primary_tone.value} with {energetic_resonance.value} energy")
        return tone_analysis
    
    def _detect_primary_tone(self, message: str) -> ToneSignature:
        """Detect the primary tone signature"""
        
        message_lower = message.lower()
        tone_scores = {}
        
        # Score each tone based on patterns
        for tone, patterns in self.tone_patterns.items():
            score = 0.0
            
            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in message_lower:
                    score += 1.0
            
            # Check phrases
            for phrase in patterns["phrases"]:
                if phrase in message_lower:
                    score += 1.5
            
            # Analyze linguistic markers
            score += self._analyze_linguistic_markers(message, patterns["linguistic_markers"])
            
            if score > 0:
                tone_scores[tone] = score
        
        # Return highest scoring tone, or SEEKING as default
        if tone_scores:
            return max(tone_scores, key=tone_scores.get)
        else:
            return ToneSignature.SEEKING
    
    def _detect_secondary_tones(self, message: str, primary_tone: ToneSignature) -> List[ToneSignature]:
        """Detect secondary tone signatures"""
        
        message_lower = message.lower()
        secondary_scores = {}
        
        for tone, patterns in self.tone_patterns.items():
            if tone == primary_tone:
                continue
                
            score = 0.0
            
            # Check for secondary presence
            for keyword in patterns["keywords"]:
                if keyword in message_lower:
                    score += 0.5
            
            for phrase in patterns["phrases"]:
                if phrase in message_lower:
                    score += 0.8
            
            if score >= 0.5:  # Threshold for secondary tone
                secondary_scores[tone] = score
        
        # Return top 3 secondary tones
        sorted_tones = sorted(secondary_scores.items(), key=lambda x: x[1], reverse=True)
        return [tone for tone, score in sorted_tones[:3]]
    
    def _analyze_linguistic_markers(self, message: str, markers: List[str]) -> float:
        """Analyze linguistic markers for tone detection"""
        
        score = 0.0
        message_lower = message.lower()
        
        for marker in markers:
            if marker == "question_heavy":
                question_count = message.count('?')
                if question_count >= 2:
                    score += 0.5
            
            elif marker == "uncertainty_words":
                uncertainty_words = ["maybe", "perhaps", "not sure", "don't know", "unclear"]
                uncertainty_count = sum(1 for word in uncertainty_words if word in message_lower)
                score += uncertainty_count * 0.3
            
            elif marker == "emotional_words":
                emotional_words = ["feel", "feeling", "emotion", "heart", "soul", "deeply"]
                emotional_count = sum(1 for word in emotional_words if word in message_lower)
                score += emotional_count * 0.2
            
            elif marker == "time_pressure":
                time_words = ["now", "urgent", "immediately", "asap", "quickly", "soon"]
                time_count = sum(1 for word in time_words if word in message_lower)
                score += time_count * 0.4
            
            elif marker == "positive_words":
                positive_words = ["good", "great", "wonderful", "amazing", "beautiful", "love"]
                positive_count = sum(1 for word in positive_words if word in message_lower)
                score += positive_count * 0.3
        
        return score
    
    def _analyze_energetic_resonance(self, message: str) -> EnergeticResonance:
        """Analyze the energetic resonance of the message"""
        
        message_lower = message.lower()
        resonance_scores = {}
        
        for resonance, markers in self.energetic_markers.items():
            score = 0.0
            
            # Check resonance words
            for word in markers["words"]:
                if word in message_lower:
                    score += 1.0
            
            # Check energy indicators
            for indicator in markers["energy_indicators"]:
                if indicator == "exclamation_moderate":
                    exclamation_count = message.count('!')
                    if 1 <= exclamation_count <= 3:
                        score += 0.5
                elif indicator == "heavy_language":
                    heavy_words = ["heavy", "weight", "burden", "crushing", "overwhelming"]
                    score += sum(0.3 for word in heavy_words if word in message_lower)
                elif indicator == "uplifting_language":
                    uplifting_words = ["lift", "rise", "soar", "bright", "light", "elevate"]
                    score += sum(0.3 for word in uplifting_words if word in message_lower)
            
            if score > 0:
                resonance_scores[resonance] = score
        
        # Determine resonance or default to neutral
        if resonance_scores:
            return max(resonance_scores, key=resonance_scores.get)
        else:
            return EnergeticResonance.NEUTRAL
    
    def _assess_spiritual_openness(self, message: str, context: Optional[Dict]) -> float:
        """Assess how open the user is to spiritual guidance"""
        
        message_lower = message.lower()
        openness_score = 0.5  # Baseline
        
        # Spiritual vocabulary
        spiritual_words = [
            "spiritual", "divine", "sacred", "holy", "meditation", "prayer", "dharma",
            "consciousness", "enlightenment", "awakening", "soul", "spirit", "god",
            "universe", "cosmic", "transcendent", "mystical", "mindfulness"
        ]
        
        spiritual_count = sum(1 for word in spiritual_words if word in message_lower)
        openness_score += spiritual_count * 0.1
        
        # Questions about meaning
        meaning_phrases = [
            "meaning of life", "purpose", "why am I here", "spiritual path",
            "deeper understanding", "higher power", "divine guidance"
        ]
        
        meaning_count = sum(1 for phrase in meaning_phrases if phrase in message_lower)
        openness_score += meaning_count * 0.15
        
        # Resistance indicators (reduce openness)
        resistance_words = ["don't believe", "not religious", "skeptical", "doubt", "nonsense"]
        resistance_count = sum(1 for word in resistance_words if word in message_lower)
        openness_score -= resistance_count * 0.2
        
        return max(0.0, min(1.0, openness_score))
    
    def _measure_emotional_intensity(self, message: str) -> float:
        """Measure the emotional intensity of the message"""
        
        intensity_score = 0.0
        
        # Intensity words
        high_intensity = ["extremely", "incredibly", "overwhelming", "devastating", "crushing"]
        medium_intensity = ["very", "really", "quite", "deeply", "strongly"]
        
        high_count = sum(1 for word in high_intensity if word in message.lower())
        medium_count = sum(1 for word in medium_intensity if word in message.lower())
        
        intensity_score += high_count * 0.3 + medium_count * 0.2
        
        # Punctuation intensity
        exclamation_count = message.count('!')
        caps_ratio = sum(1 for c in message if c.isupper()) / len(message) if message else 0
        
        intensity_score += min(exclamation_count * 0.1, 0.3)
        intensity_score += caps_ratio * 0.5
        
        # Repetition (indicates intensity)
        words = message.lower().split()
        if words:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values())
            if max_repetition > 2:
                intensity_score += (max_repetition - 1) * 0.1
        
        return min(1.0, intensity_score)
    
    def _assess_transformation_readiness(self, message: str, context: Optional[Dict]) -> float:
        """Assess how ready the user is for transformation"""
        
        message_lower = message.lower()
        readiness_score = 0.3  # Baseline
        
        # Readiness indicators
        ready_phrases = [
            "ready to change", "want to grow", "need to transform", "ready for something new",
            "willing to try", "open to change", "ready to heal", "want to be different"
        ]
        
        ready_count = sum(1 for phrase in ready_phrases if phrase in message_lower)
        readiness_score += ready_count * 0.2
        
        # Action-oriented language
        action_words = ["will", "going to", "plan to", "commit", "practice", "start", "begin"]
        action_count = sum(1 for word in action_words if word in message_lower)
        readiness_score += action_count * 0.1
        
        # Resistance indicators
        resistance_phrases = [
            "can't change", "impossible", "too hard", "not ready", "don't want to",
            "won't work", "tried everything", "nothing helps"
        ]
        
        resistance_count = sum(1 for phrase in resistance_phrases if phrase in message_lower)
        readiness_score -= resistance_count * 0.15
        
        return max(0.0, min(1.0, readiness_score))
    
    def _determine_communication_style(self, message: str) -> str:
        """Determine the user's preferred communication style"""
        
        message_lower = message.lower()
        
        # Formal vs informal
        formal_indicators = ["please", "thank you", "would you", "could you", "appreciate"]
        informal_indicators = ["hey", "yeah", "gonna", "wanna", "kinda"]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in message_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in message_lower)
        
        # Direct vs indirect
        direct_indicators = ["tell me", "I need", "what should", "give me"]
        indirect_indicators = ["wondering if", "perhaps", "maybe you could", "if possible"]
        
        direct_count = sum(1 for indicator in direct_indicators if indicator in message_lower)
        indirect_count = sum(1 for indicator in indirect_indicators if indicator in message_lower)
        
        # Determine style
        if formal_count > informal_count and indirect_count > direct_count:
            return "formal_gentle"
        elif informal_count > formal_count and direct_count > indirect_count:
            return "casual_direct"
        elif formal_count > informal_count and direct_count > indirect_count:
            return "formal_direct"
        elif informal_count > formal_count and indirect_count > direct_count:
            return "casual_gentle"
        else:
            return "balanced"
    
    def _identify_healing_needs(self, primary_tone: ToneSignature, secondary_tones: List[ToneSignature], message: str) -> List[str]:
        """Identify specific healing needs based on tone analysis"""
        
        healing_needs = []
        
        tone_healing_map = {
            ToneSignature.VULNERABLE: ["safety", "gentle_support", "emotional_validation"],
            ToneSignature.URGENT: ["immediate_relief", "crisis_support", "grounding"],
            ToneSignature.SUFFERING: ["deep_healing", "compassionate_presence", "pain_relief"],
            ToneSignature.CONFUSED: ["clarity", "direction", "understanding"],
            ToneSignature.SEEKING: ["guidance", "spiritual_direction", "learning_support"],
            ToneSignature.CONTEMPLATIVE: ["wisdom", "deeper_understanding", "philosophical_guidance"],
            ToneSignature.HOPEFUL: ["encouragement", "vision_support", "momentum_building"],
            ToneSignature.PEACEFUL: ["celebration", "deepening", "integration_support"]
        }
        
        # Add needs from primary tone
        healing_needs.extend(tone_healing_map.get(primary_tone, ["general_support"]))
        
        # Add needs from secondary tones
        for tone in secondary_tones[:2]:  # Top 2 secondary tones
            secondary_needs = tone_healing_map.get(tone, [])
            healing_needs.extend(secondary_needs[:2])  # Add top 2 needs from each
        
        return list(set(healing_needs))  # Remove duplicates
    
    def _calculate_dharmic_alignment(self, message: str, spiritual_openness: float) -> float:
        """Calculate how aligned the message is with dharmic principles"""
        
        message_lower = message.lower()
        dharmic_score = spiritual_openness * 0.5  # Base from spiritual openness
        
        # Dharmic values mentioned
        dharmic_values = [
            "compassion", "wisdom", "truth", "non-violence", "peace", "love",
            "service", "selflessness", "humility", "patience", "forgiveness",
            "detachment", "surrender", "devotion", "righteousness"
        ]
        
        dharmic_count = sum(1 for value in dharmic_values if value in message_lower)
        dharmic_score += dharmic_count * 0.1
        
        # Dharmic concepts
        dharmic_concepts = [
            "karma", "dharma", "moksha", "liberation", "enlightenment",
            "self-realization", "divine", "sacred", "holy", "spiritual growth"
        ]
        
        concept_count = sum(1 for concept in dharmic_concepts if concept in message_lower)
        dharmic_score += concept_count * 0.15
        
        return min(1.0, dharmic_score)
    
    async def create_response_modulation(self, tone_analysis: ToneAnalysis) -> ResponseModulation:
        """Create response modulation strategy based on tone analysis"""
        
        # Determine energy adjustment needed
        energy_adjustment = self._determine_energy_adjustment(tone_analysis)
        
        # Calculate compassion level
        compassion_level = self._calculate_compassion_level(tone_analysis)
        
        # Determine directness level
        directness_level = self._calculate_directness_level(tone_analysis)
        
        # Set spiritual depth
        spiritual_depth = tone_analysis.spiritual_openness * tone_analysis.dharmic_alignment
        
        # Set practical focus
        practical_focus = self._calculate_practical_focus(tone_analysis)
        
        # Determine response length
        response_length = self._determine_response_length(tone_analysis)
        
        # Determine tone to match
        tone_to_match = self._determine_response_tone(tone_analysis)
        
        return ResponseModulation(
            energy_adjustment=energy_adjustment,
            compassion_level=compassion_level,
            directness_level=directness_level,
            spiritual_depth=spiritual_depth,
            practical_focus=practical_focus,
            response_length=response_length,
            tone_to_match=tone_to_match
        )
    
    def _determine_energy_adjustment(self, analysis: ToneAnalysis) -> str:
        """Determine how to adjust energy in response"""
        
        if analysis.energetic_resonance == EnergeticResonance.LOW_VIBRATION:
            if analysis.emotional_intensity > 0.7:
                return "gentle_lifting"  # Lift energy slowly and gently
            else:
                return "gradual_elevation"  # Gradually elevate energy
        
        elif analysis.energetic_resonance == EnergeticResonance.HIGH_VIBRATION:
            return "harmonious_matching"  # Match the high energy
        
        elif analysis.energetic_resonance == EnergeticResonance.TURBULENT:
            return "stabilizing_grounding"  # Ground and stabilize
        
        elif analysis.primary_tone == ToneSignature.URGENT:
            return "immediate_calming"  # Calm urgency
        
        else:
            return "balanced_presence"  # Maintain balanced energy
    
    def _calculate_compassion_level(self, analysis: ToneAnalysis) -> float:
        """Calculate appropriate compassion level"""
        
        base_compassion = 0.8  # High baseline
        
        # Increase for vulnerable or suffering states
        if analysis.primary_tone in [ToneSignature.VULNERABLE, ToneSignature.SUFFERING]:
            base_compassion += 0.15
        
        # Increase for high emotional intensity
        base_compassion += analysis.emotional_intensity * 0.1
        
        # Increase for low vibration energy
        if analysis.energetic_resonance == EnergeticResonance.LOW_VIBRATION:
            base_compassion += 0.1
        
        return min(1.0, base_compassion)
    
    def _calculate_directness_level(self, analysis: ToneAnalysis) -> float:
        """Calculate appropriate directness level"""
        
        # Start with moderate directness
        directness = 0.6
        
        # More direct for urgent situations
        if analysis.primary_tone == ToneSignature.URGENT:
            directness += 0.3
        
        # Less direct for vulnerable states
        if analysis.primary_tone == ToneSignature.VULNERABLE:
            directness -= 0.2
        
        # Adjust based on communication style
        if analysis.communication_style == "casual_direct":
            directness += 0.2
        elif analysis.communication_style in ["formal_gentle", "casual_gentle"]:
            directness -= 0.2
        
        return max(0.2, min(1.0, directness))
    
    def _calculate_practical_focus(self, analysis: ToneAnalysis) -> float:
        """Calculate how practical vs philosophical the response should be"""
        
        practical_focus = 0.7  # Default practical focus
        
        # More practical for urgent or suffering states
        if analysis.primary_tone in [ToneSignature.URGENT, ToneSignature.SUFFERING]:
            practical_focus += 0.2
        
        # Less practical for contemplative states
        if analysis.primary_tone == ToneSignature.CONTEMPLATIVE:
            practical_focus -= 0.3
        
        # Adjust based on transformation readiness
        practical_focus += analysis.transformation_readiness * 0.2
        
        return max(0.2, min(1.0, practical_focus))
    
    def _determine_response_length(self, analysis: ToneAnalysis) -> str:
        """Determine appropriate response length"""
        
        if analysis.primary_tone == ToneSignature.URGENT:
            return "brief"  # Quick, focused help
        elif analysis.primary_tone == ToneSignature.CONTEMPLATIVE:
            return "comprehensive"  # Deep, thorough exploration
        elif analysis.emotional_intensity > 0.8:
            return "moderate"  # Substantial but not overwhelming
        else:
            return "moderate"
    
    def _determine_response_tone(self, analysis: ToneAnalysis) -> str:
        """Determine what tone the response should embody"""
        
        tone_map = {
            ToneSignature.VULNERABLE: "gentle_protective",
            ToneSignature.URGENT: "calm_authoritative",
            ToneSignature.SUFFERING: "deeply_compassionate",
            ToneSignature.CONTEMPLATIVE: "wise_philosophical",
            ToneSignature.SEEKING: "nurturing_guidance",
            ToneSignature.HOPEFUL: "encouraging_celebratory",
            ToneSignature.PEACEFUL: "harmonious_deepening",
            ToneSignature.CONFUSED: "clarifying_patient"
        }
        
        return tone_map.get(analysis.primary_tone, "balanced_wisdom")


# Global tone analyzer instance
tone_analyzer = AdvancedToneAnalyzer()

# Helper functions
async def analyze_tone(message: str, context: Dict = None) -> ToneAnalysis:
    """Analyze tone of message"""
    return await tone_analyzer.analyze_communication_tone(message, context)

async def create_modulation(tone_analysis: ToneAnalysis) -> ResponseModulation:
    """Create response modulation strategy"""
    return await tone_analyzer.create_response_modulation(tone_analysis)
