"""
🕉️ Dharmic Emotional Intelligence Engine
==========================================
Pure Hindu scripture-based emotional healing and guidance.
NO Western psychology - ONLY authentic dharmic wisdom.

Sources:
- Bhagavad Gita (primary)
- Upanishads (Taittiriya, Katha, Mundaka, etc.)
- Yoga Sutras of Patanjali
- Brahma Samhita
- Bhagavata Purana
- Traditional Vedantic teachings

Philosophy:
- Emotions are temporary modifications (vrittis) of mind
- True Self (Atman) is beyond all emotions
- Healing through Self-knowledge, not emotional management
- Goal: Witness emotions without identification (Titiksha)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DharmicEmotion(Enum):
    """Emotions covered by dharmic wisdom"""
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    ANXIETY = "anxiety"
    CONFUSION = "confusion"
    JOY = "joy"
    LOVE = "love"
    PEACE = "peace"
    GRATITUDE = "gratitude"
    COMPASSION = "compassion"


class DharmicEmotionalIntelligence:
    """
    Provides emotional healing guidance using ONLY Hindu scriptures.
    
    Features:
    - Sanskrit verse quotation with translations
    - Traditional dharmic practices (Japa, Pranayama, Atma-Vichara)
    - Vedantic concepts (Atman, Brahman, Samatva, Titiksha)
    - NO Western psychology (no CBT, no affirmations, no generic breathing)
    """
    
    def __init__(self, wisdom_path: Optional[str] = None):
        """
        Initialize with dharmic emotional wisdom database.
        
        Args:
            wisdom_path: Path to dharmic_emotional_wisdom.json
                        If None, auto-detects from project structure
        """
        if wisdom_path is None:
            # Auto-detect path
            current_dir = Path(__file__).parent.parent
            wisdom_path = current_dir / "data" / "emotional" / "dharmic_emotional_wisdom.json"
        
        self.wisdom_path = Path(wisdom_path)
        self.wisdom_data = self._load_wisdom()
        
        # Emotion detection patterns (dharmic context)
        self.emotion_patterns = {
            DharmicEmotion.SADNESS: [
                r'\b(sad|grief|sorrow|mourning|loss|hurt|pain|depressed|melancholy)\b',
                r'\b(i feel (bad|down|low|heavy|heartbroken))\b',
                r'\b(i am (grieving|hurting|in pain))\b'
            ],
            DharmicEmotion.ANGER: [
                r'\b(angry|rage|fury|mad|frustrated|irritated|annoyed|resentful)\b',
                r'\b(i (hate|despise|can\'t stand))\b',
                r'\b(makes me (angry|mad|furious))\b'
            ],
            DharmicEmotion.FEAR: [
                r'\b(afraid|scared|fear|terrified|frightened|anxious|worried|panic)\b',
                r'\b(i am (scared|afraid|fearful|terrified))\b',
                r'\b(i fear|scares me)\b'
            ],
            DharmicEmotion.ANXIETY: [
                r'\b(anxious|anxiety|worry|worried|stress|stressed|nervous|uneasy|restless)\b',
                r'\b(i am (anxious|worried|stressed|nervous))\b',
                r'\b(can\'t (relax|calm down|stop worrying))\b'
            ],
            DharmicEmotion.CONFUSION: [
                r'\b(confused|lost|uncertain|doubt|doubtful|unclear|bewildered|puzzled)\b',
                r'\b(i (don\'t know|am not sure|am confused|am lost))\b',
                r'\b(what should i do)\b'
            ],
            DharmicEmotion.JOY: [
                r'\b(happy|joy|joyful|delighted|pleased|glad|cheerful|elated|bliss)\b',
                r'\b(i (feel|am) (happy|joyful|great|wonderful))\b',
                r'\b(i love (this|it))\b'
            ],
            DharmicEmotion.LOVE: [
                r'\b(love|loving|affection|devotion|adore|care deeply|cherish)\b',
                r'\b(i love|i adore|i cherish)\b',
                r'\b(feel (love|loving|devoted))\b'
            ],
            DharmicEmotion.PEACE: [
                r'\b(peace|peaceful|calm|serene|tranquil|quiet|stillness|shanti)\b',
                r'\b(i feel (peace|peaceful|calm|serene))\b',
                r'\b(at peace)\b'
            ],
            DharmicEmotion.GRATITUDE: [
                r'\b(grateful|gratitude|thankful|blessed|appreciate|appreciation)\b',
                r'\b(i am (grateful|thankful|blessed))\b',
                r'\b(thank you|thanks)\b'
            ],
            DharmicEmotion.COMPASSION: [
                r'\b(compassion|compassionate|empathy|sympathy|care for|kindness)\b',
                r'\b(i feel (compassion|empathy|sympathy) for)\b',
                r'\b(want to help|feel for)\b'
            ]
        }
    
    def _load_wisdom(self) -> Dict:
        """Load dharmic emotional wisdom from JSON"""
        if not self.wisdom_path.exists():
            raise FileNotFoundError(
                f"Dharmic wisdom file not found: {self.wisdom_path}\n"
                f"Run: python3 data/scripts/build_dharmic_emotional_wisdom.py"
            )
        
        with open(self.wisdom_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def detect_emotions(self, text: str) -> List[Tuple[DharmicEmotion, float]]:
        """
        Detect emotions in text using dharmic context.
        
        Args:
            text: User's message
        
        Returns:
            List of (emotion, confidence) tuples, sorted by confidence
        """
        text_lower = text.lower()
        emotions_detected = []
        
        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    # Simple confidence: number of matches
                    confidence = min(1.0, len(matches) * 0.3)
                    emotions_detected.append((emotion, confidence))
                    break  # One pattern match is enough per emotion
        
        # Sort by confidence (highest first)
        emotions_detected.sort(key=lambda x: x[1], reverse=True)
        
        return emotions_detected
    
    def get_primary_emotion(self, text: str) -> Optional[DharmicEmotion]:
        """
        Get the most prominent emotion in text.
        
        Args:
            text: User's message
        
        Returns:
            Primary emotion or None if no emotion detected
        """
        emotions = self.detect_emotions(text)
        if emotions:
            return emotions[0][0]
        return None
    
    def get_dharmic_wisdom(
        self, 
        emotion: DharmicEmotion,
        verse_index: Optional[int] = None
    ) -> Dict:
        """
        Get authentic dharmic wisdom for an emotion.
        
        Args:
            emotion: The emotion to get wisdom for
            verse_index: Specific verse index (0-based), or None for first verse
        
        Returns:
            Dict with 'verse', 'translation', 'source', 'context', 'practice'
        """
        emotion_key = emotion.value
        emotion_data = self.wisdom_data['emotions'].get(emotion_key, {})
        
        teachings = emotion_data.get('teachings', [])
        if not teachings:
            return {
                'verse': '',
                'translation': 'Wisdom not yet compiled for this emotion.',
                'source': 'DharmaMind',
                'context': emotion_key,
                'practice': 'Practice meditation and self-inquiry.'
            }
        
        # Get specific verse or first one
        if verse_index is not None and 0 <= verse_index < len(teachings):
            return teachings[verse_index]
        return teachings[0]
    
    def get_all_teachings(self, emotion: DharmicEmotion) -> List[Dict]:
        """
        Get all dharmic teachings for an emotion.
        
        Args:
            emotion: The emotion to get teachings for
        
        Returns:
            List of teaching dicts
        """
        emotion_key = emotion.value
        emotion_data = self.wisdom_data['emotions'].get(emotion_key, {})
        return emotion_data.get('teachings', [])
    
    def get_dharmic_practices(self, emotion: DharmicEmotion) -> List[str]:
        """
        Get traditional dharmic practices for an emotion.
        
        Args:
            emotion: The emotion to get practices for
        
        Returns:
            List of practice descriptions
        """
        emotion_key = emotion.value
        emotion_data = self.wisdom_data['emotions'].get(emotion_key, {})
        return emotion_data.get('practices', [])
    
    def suggest_practice(
        self, 
        emotion: DharmicEmotion,
        intensity: float = 0.5
    ) -> str:
        """
        Suggest an appropriate dharmic practice based on emotion intensity.
        
        Args:
            emotion: The emotion to address
            intensity: Emotion intensity (0.0-1.0)
        
        Returns:
            Practice description string
        """
        practices = self.get_dharmic_practices(emotion)
        if not practices:
            return "Practice meditation on the eternal Self (Atman)."
        
        # Higher intensity → more active practices (earlier in list)
        # Lower intensity → gentler practices (later in list)
        if intensity > 0.7:
            return practices[0]  # Most active/direct
        elif intensity > 0.4:
            return practices[min(2, len(practices) - 1)]  # Moderate
        else:
            return practices[-1]  # Gentle/contemplative
    
    def generate_dharmic_response(
        self,
        text: str,
        include_all_verses: bool = False,
        max_practices: int = 3
    ) -> Dict:
        """
        Generate complete dharmic emotional guidance.
        
        Args:
            text: User's message expressing emotion
            include_all_verses: If True, include all relevant verses
            max_practices: Maximum number of practices to suggest
        
        Returns:
            Dict with:
                - emotion: Detected emotion
                - sanskrit_verse: Devanagari verse
                - translation: English translation
                - source: Scripture source (e.g., "Bhagavad Gita 2.13")
                - context: Emotional context
                - immediate_practice: Direct instruction from verse
                - suggested_practices: List of traditional practices
                - dharmic_concept: Key Vedantic concept explanation
                - all_verses: (Optional) All relevant verses if requested
        """
        # Detect emotion
        primary_emotion = self.get_primary_emotion(text)
        if not primary_emotion:
            return self._generate_general_guidance()
        
        # Get primary wisdom
        wisdom = self.get_dharmic_wisdom(primary_emotion)
        
        # Get practices
        practices = self.get_dharmic_practices(primary_emotion)[:max_practices]
        
        # Get dharmic concept explanation
        concept = self._get_dharmic_concept(primary_emotion)
        
        response = {
            'emotion': primary_emotion.value,
            'sanskrit_verse': wisdom.get('verse', ''),
            'translation': wisdom.get('translation', ''),
            'source': wisdom.get('source', ''),
            'context': wisdom.get('context', ''),
            'immediate_practice': wisdom.get('practice', ''),
            'suggested_practices': practices,
            'dharmic_concept': concept
        }
        
        # Include all verses if requested
        if include_all_verses:
            response['all_verses'] = self.get_all_teachings(primary_emotion)
        
        return response
    
    def _get_dharmic_concept(self, emotion: DharmicEmotion) -> str:
        """Get key Vedantic concept for emotion"""
        concepts = {
            DharmicEmotion.SADNESS: (
                "Remember the eternal nature of Atman (Self). Sadness arises from "
                "identification with the temporary body-mind. The wise witness sadness "
                "without attachment, knowing 'I am not this emotion, I am the eternal Self.'"
            ),
            DharmicEmotion.ANGER: (
                "Anger is a modification (vritti) arising from Rajas (passion). The Bhagavad "
                "Gita teaches that anger destroys wisdom and leads to ruin. Practice Shama "
                "(calmness) and Kshama (forgiveness) to return to Sattva (purity)."
            ),
            DharmicEmotion.FEAR: (
                "Fear arises from ignorance of the immortal Self. The Atman is never born, "
                "never dies, and is eternally free. Abhaya (fearlessness) comes from "
                "Self-knowledge: 'I am That eternal Brahman, beyond birth and death.'"
            ),
            DharmicEmotion.ANXIETY: (
                "Anxiety comes from attachment to results (Sakama Karma). Practice Nishkama "
                "Karma - action without attachment to outcomes. Surrender to Ishvara (Divine) "
                "and cultivate Samatva (equanimity): 'Yogasthah kuru karmani' - established "
                "in yoga, perform actions."
            ),
            DharmicEmotion.CONFUSION: (
                "Confusion arises when Buddhi (intellect) is clouded. Practice Viveka "
                "(discrimination) between Real and unreal, eternal and temporary. Seek "
                "guidance from scripture (Shastra) and realized teachers (Guru)."
            ),
            DharmicEmotion.JOY: (
                "True joy (Ananda) is the nature of Brahman. Temporary joys come and go, but "
                "the bliss of Self-realization is eternal. Cultivate Santosha (contentment) "
                "and recognize joy as a glimpse of your true nature."
            ),
            DharmicEmotion.LOVE: (
                "Divine love (Bhakti/Prema) transcends ordinary attachment. Love all beings as "
                "manifestations of the one Brahman. Practice Ishvara Pranidhana (devotion to "
                "the Divine) and see God in all."
            ),
            DharmicEmotion.PEACE: (
                "Peace (Shanti) is the natural state of the Self when mind is still. Through "
                "meditation and dispassion (Vairagya), the turbulence of thoughts settles, "
                "revealing the inherent peace of Atman."
            ),
            DharmicEmotion.GRATITUDE: (
                "Gratitude (Kritajnata) recognizes the Divine in all blessings. All is given "
                "by Ishvara. Offer gratitude through service, devotion, and remembrance that "
                "all belongs to God."
            ),
            DharmicEmotion.COMPASSION: (
                "Compassion (Karuna/Daya) flows from seeing the one Self in all beings. "
                "Practice Ahimsa (non-violence) and serve all as service to the Divine. "
                "Compassion is the mark of spiritual maturity."
            )
        }
        return concepts.get(emotion, "Meditate on the eternal Self beyond all emotions.")
    
    def _generate_general_guidance(self) -> Dict:
        """Generate general dharmic guidance when no specific emotion detected"""
        return {
            'emotion': 'general',
            'sanskrit_verse': 'समत्वं योग उच्यते',
            'translation': 'Evenness of mind is called Yoga.',
            'source': 'Bhagavad Gita 2.48',
            'context': 'general spiritual guidance',
            'immediate_practice': (
                'Cultivate equanimity (Samatva) in all situations. Practice witnessing '
                'thoughts and emotions without attachment.'
            ),
            'suggested_practices': [
                'Daily meditation on the Self (Atma-Vichara)',
                'Japa (mantra repetition) - "ॐ" (Om) 108 times',
                'Study of scripture (Swadhyaya) - Bhagavad Gita daily'
            ],
            'dharmic_concept': (
                'The goal of dharmic practice is Self-realization - knowing your true nature '
                'as Atman/Brahman, beyond body, mind, and emotions. All practices lead to this.'
            )
        }
    
    def format_response_for_user(self, response: Dict) -> str:
        """
        Format dharmic response for beautiful display.
        
        Args:
            response: Dict from generate_dharmic_response()
        
        Returns:
            Formatted string for display
        """
        lines = []
        lines.append("🕉️ " + "=" * 70)
        lines.append(f"   DHARMIC WISDOM FOR {response['emotion'].upper()}")
        lines.append("=" * 72)
        lines.append("")
        
        # Sanskrit verse
        if response.get('sanskrit_verse'):
            lines.append("📖 SACRED VERSE:")
            lines.append(f"   {response['sanskrit_verse']}")
            lines.append("")
        
        # Translation
        if response.get('translation'):
            lines.append("💬 TRANSLATION:")
            lines.append(f"   {response['translation']}")
            lines.append("")
        
        # Source
        if response.get('source'):
            lines.append(f"📚 SOURCE: {response['source']}")
            lines.append("")
        
        # Immediate practice from verse
        if response.get('immediate_practice'):
            lines.append("🧘 IMMEDIATE PRACTICE:")
            lines.append(f"   {response['immediate_practice']}")
            lines.append("")
        
        # Dharmic concept
        if response.get('dharmic_concept'):
            lines.append("🌟 DHARMIC UNDERSTANDING:")
            lines.append(f"   {response['dharmic_concept']}")
            lines.append("")
        
        # Traditional practices
        if response.get('suggested_practices'):
            lines.append("🙏 TRADITIONAL PRACTICES:")
            for i, practice in enumerate(response['suggested_practices'], 1):
                lines.append(f"   {i}. {practice}")
            lines.append("")
        
        lines.append("🕉️ " + "=" * 70)
        lines.append("   May you realize the eternal Self beyond all emotions")
        lines.append("=" * 72)
        
        return "\n".join(lines)


# Convenience function for quick usage
def get_dharmic_emotional_guidance(text: str) -> str:
    """
    Quick function to get dharmic emotional guidance.
    
    Args:
        text: User's emotional expression
    
    Returns:
        Formatted dharmic guidance string
    """
    engine = DharmicEmotionalIntelligence()
    response = engine.generate_dharmic_response(text)
    return engine.format_response_for_user(response)


if __name__ == "__main__":
    # Test the dharmic emotional intelligence engine
    print("🕉️ Testing Dharmic Emotional Intelligence Engine\n")
    
    test_messages = [
        "I feel so sad and lost",
        "I am angry at what happened",
        "I'm afraid of the future",
        "I feel anxious and can't relax",
        "I'm confused about what to do",
        "I feel so much joy and gratitude"
    ]
    
    engine = DharmicEmotionalIntelligence()
    
    for message in test_messages:
        print(f"\n{'=' * 80}")
        print(f"USER: {message}")
        print('=' * 80)
        
        response = engine.generate_dharmic_response(message)
        formatted = engine.format_response_for_user(response)
        print(formatted)
        print()
