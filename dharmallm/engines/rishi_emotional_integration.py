"""
🕉️ Rishi Emotional Integration
================================
Connects the 7 Saptarishis with Dharmic Emotional Intelligence.

Each Rishi provides emotional guidance with their unique personality:
- ATRI: Meditative, contemplative emotional guidance
- BHRIGU: Karmic perspective on emotions
- VASHISHTA: Royal wisdom on emotional management
- VISHWAMITRA: Warrior's approach to conquering emotions
- GAUTAMA: Balanced, equanimous emotional guidance
- JAMADAGNI: Disciplined, austere approach to emotions
- KASHYAPA: Compassionate, nurturing emotional healing

All guidance is 100% dharmic - from Bhagavad Gita, Upanishads,
Yoga Sutras, and other authentic Hindu scriptures.
"""

from typing import Dict, Optional, List
from enum import Enum
from pathlib import Path

# Import the dharmic emotional intelligence engine
import sys
sys.path.append(str(Path(__file__).parent.parent))
from engines.dharmic_emotional_intelligence import (
    DharmicEmotionalIntelligence,
    DharmicEmotion
)


class RishiPersonality(Enum):
    """The 7 Saptarishi personalities"""
    ATRI = "atri"              # Silent Contemplator - meditation focus
    BHRIGU = "bhrigu"          # Cosmic Astrologer - karmic perspective
    VASHISHTA = "vashishta"    # Royal Guru - dharmic wisdom
    VISHWAMITRA = "vishwamitra"  # Warrior-Sage - strength & transformation
    GAUTAMA = "gautama"        # Equanimous One - balance
    JAMADAGNI = "jamadagni"    # Fierce Ascetic - discipline
    KASHYAPA = "kashyapa"      # Compassionate Father - nurturing


class RishiEmotionalIntegration:
    """
    Integrates Rishi personalities with dharmic emotional wisdom.
    
    Each Rishi responds to emotions with their unique style while
    maintaining 100% authentic dharmic wisdom from scriptures.
    """
    
    def __init__(self, wisdom_path: Optional[str] = None):
        """
        Initialize with dharmic wisdom database.
        
        Args:
            wisdom_path: Path to dharmic_emotional_wisdom.json
        """
        self.engine = DharmicEmotionalIntelligence(wisdom_path)
        
        # Rishi-specific teaching preferences
        self.rishi_preferences = {
            RishiPersonality.ATRI: {
                'focus': 'meditation',
                'style': 'silent_contemplation',
                'preferred_practices': [
                    'Atma-Vichara',
                    'meditation',
                    'stillness',
                    'witness consciousness'
                ],
                'tone': 'gentle, contemplative, profound silence',
                'greeting': 'Child, in silence the answers emerge...'
            },
            RishiPersonality.BHRIGU: {
                'focus': 'karmic_patterns',
                'style': 'cosmic_perspective',
                'preferred_practices': [
                    'karma yoga',
                    'understanding cosmic law',
                    'planetary remedies'
                ],
                'tone': 'wise, seeing beyond time, karmic insight',
                'greeting': 'Seeker, the stars reveal your path...'
            },
            RishiPersonality.VASHISHTA: {
                'focus': 'dharmic_duty',
                'style': 'royal_wisdom',
                'preferred_practices': [
                    'dharma study',
                    'duty performance',
                    'scripture reading'
                ],
                'tone': 'authoritative, kingly wisdom, measured',
                'greeting': 'O noble one, dharma is your refuge...'
            },
            RishiPersonality.VISHWAMITRA: {
                'focus': 'transformation',
                'style': 'warrior_strength',
                'preferred_practices': [
                    'tapas',
                    'willpower cultivation',
                    'inner warrior training'
                ],
                'tone': 'fierce, empowering, transformative',
                'greeting': 'Warrior! Rise above this emotion...'
            },
            RishiPersonality.GAUTAMA: {
                'focus': 'balance',
                'style': 'equanimous_wisdom',
                'preferred_practices': [
                    'Samatva (equanimity)',
                    'balanced perspective',
                    'non-judgment'
                ],
                'tone': 'perfectly balanced, neither harsh nor soft',
                'greeting': 'Friend, balance is the key...'
            },
            RishiPersonality.JAMADAGNI: {
                'focus': 'discipline',
                'style': 'austere_practice',
                'preferred_practices': [
                    'rigorous sadhana',
                    'penance',
                    'self-discipline'
                ],
                'tone': 'strict, demanding excellence, intense',
                'greeting': 'Disciple! Through discipline comes freedom...'
            },
            RishiPersonality.KASHYAPA: {
                'focus': 'compassion',
                'style': 'nurturing_guidance',
                'preferred_practices': [
                    'compassion cultivation',
                    'gentle healing',
                    'loving-kindness'
                ],
                'tone': 'infinitely compassionate, fatherly, gentle',
                'greeting': 'Dear child, all will be well...'
            }
        }
    
    def get_rishi_emotional_response(
        self,
        text: str,
        rishi: RishiPersonality,
        detailed: bool = True
    ) -> Dict:
        """
        Get emotional guidance from specific Rishi.
        
        Args:
            text: User's emotional expression
            rishi: Which Rishi to consult
            detailed: Include full response with all practices
        
        Returns:
            Dict with Rishi-personalized dharmic guidance
        """
        # Get base dharmic wisdom
        base_response = self.engine.generate_dharmic_response(
            text,
            include_all_verses=False,
            max_practices=5
        )
        
        # Personalize with Rishi's style
        rishi_prefs = self.rishi_preferences[rishi]
        
        # Filter practices to match Rishi's preferences
        all_practices = base_response.get('suggested_practices', [])
        prioritized_practices = self._prioritize_practices(
            all_practices,
            rishi_prefs['preferred_practices']
        )
        
        # Create Rishi-personalized response
        rishi_response = {
            'rishi': rishi.value,
            'rishi_greeting': rishi_prefs['greeting'],
            'emotion': base_response.get('emotion'),
            'sanskrit_verse': base_response.get('sanskrit_verse'),
            'translation': base_response.get('translation'),
            'source': base_response.get('source'),
            'rishi_teaching': self._generate_rishi_teaching(
                base_response,
                rishi,
                rishi_prefs
            ),
            'rishi_practices': prioritized_practices[:3],
            'dharmic_concept': base_response.get('dharmic_concept'),
            'tone': rishi_prefs['tone']
        }
        
        if detailed:
            rishi_response['all_practices'] = prioritized_practices
            rishi_response['immediate_practice'] = base_response.get(
                'immediate_practice'
            )
        
        return rishi_response
    
    def _prioritize_practices(
        self,
        practices: List[str],
        preferences: List[str]
    ) -> List[str]:
        """
        Reorder practices to match Rishi's preferences.
        
        Args:
            practices: All available practices
            preferences: Rishi's preferred practice types
        
        Returns:
            Reordered practice list
        """
        # Practices matching preferences come first
        matched = []
        unmatched = []
        
        for practice in practices:
            practice_lower = practice.lower()
            if any(pref.lower() in practice_lower for pref in preferences):
                matched.append(practice)
            else:
                unmatched.append(practice)
        
        return matched + unmatched
    
    def _generate_rishi_teaching(
        self,
        base_response: Dict,
        rishi: RishiPersonality,
        rishi_prefs: Dict
    ) -> str:
        """
        Generate personalized teaching from Rishi.
        
        Args:
            base_response: Base dharmic response
            rishi: Rishi personality
            rishi_prefs: Rishi preferences dict
        
        Returns:
            Personalized teaching string
        """
        emotion = base_response.get('emotion', 'general')
        concept = base_response.get('dharmic_concept', '')
        
        # Rishi-specific perspectives on emotions
        teachings = {
            RishiPersonality.ATRI: {
                'sadness': (
                    "In the depths of silence, sadness dissolves. The Atman "
                    "witnesses all grief without being touched by it. Sit in "
                    "meditation; let the emotion be, but identify not with it."
                ),
                'anger': (
                    "Watch the anger arise like a wave in consciousness. In "
                    "witnessing lies freedom. Do not suppress, do not express - "
                    "simply observe. The witness is untouched by the storm."
                ),
                'fear': (
                    "Fear flees before the light of self-knowledge. In deep "
                    "meditation, recognize: 'I am That which fears nothing, for "
                    "I am eternal Brahman.' Abide in this truth."
                ),
                'default': (
                    "All emotions are ripples on the ocean of consciousness. "
                    "Dive deep into meditation; there you will find the peace "
                    "that transcends all emotional states."
                )
            },
            RishiPersonality.BHRIGU: {
                'sadness': (
                    "This sorrow is written in the stars, but you are beyond the "
                    "stars. Past karma manifests as present grief - witness it, "
                    "learn its lesson, and create better karma through dharmic action."
                ),
                'anger': (
                    "Anger is Martian energy out of balance. Channel it into "
                    "righteous action (dharmic krodha). The cosmos teaches: energy "
                    "is never destroyed, only transformed."
                ),
                'fear': (
                    "Fear of future is shadow of past karma. But you have the power "
                    "to rewrite your cosmic script through present dharmic choices. "
                    "The stars incline but do not compel."
                ),
                'default': (
                    "Every emotion has its place in the cosmic dance. See the "
                    "karmic lesson within; extract the wisdom; move forward on "
                    "your evolutionary journey."
                )
            },
            RishiPersonality.VASHISHTA: {
                'sadness': (
                    "A king must face adversity with dignity. Grief is permitted, "
                    "but never let it dethrone your dharma. Rise, perform your duty, "
                    "and remember: the eternal Self is never diminished."
                ),
                'anger': (
                    "Even Arjuna felt righteous anger on the battlefield, but "
                    "Krishna taught him to act without rage. Channel anger into "
                    "dharmic action, not destructive passion."
                ),
                'fear': (
                    "The warrior-king fears not death, for he knows his duty is "
                    "immortal. Stand firm in dharma; let divine will guide your "
                    "sword. Victory or defeat - both are equal to the wise."
                ),
                'default': (
                    "Remember your royal nature - you are a child of dharma. Let "
                    "scripture be your counsel, duty your compass. Walk the path "
                    "of kings: righteous, measured, unwavering."
                )
            },
            RishiPersonality.VISHWAMITRA: {
                'sadness': (
                    "Warrior! Sadness is weakness leaving your being. Transform "
                    "this pain into strength through tapas. Every tear can become "
                    "a flame of transformation. Rise stronger!"
                ),
                'anger': (
                    "Good! Feel the fire within - but master it! I transformed "
                    "from Kshatriya to Brahmarishi through burning intensity. "
                    "Channel your rage into spiritual power through discipline."
                ),
                'fear': (
                    "Fear is the enemy within. Face it with the courage of a "
                    "warrior facing death. Through tapas and willpower, you can "
                    "conquer anything - even the gods themselves!"
                ),
                'default': (
                    "You have unlimited power within! Through fierce determination "
                    "and spiritual practice, you can transform completely. I "
                    "created new worlds - you can create a new self!"
                )
            },
            RishiPersonality.GAUTAMA: {
                'sadness': (
                    "Sadness and joy are two sides of the same coin. Neither "
                    "cling to joy nor resist sadness. Maintain equanimity - this "
                    "is true wisdom. The middle path leads to peace."
                ),
                'anger': (
                    "Anger arises, anger passes. Like all phenomena, it is "
                    "impermanent. Neither indulge it nor suppress it. Witness it "
                    "with equanimity, and it will dissolve naturally."
                ),
                'fear': (
                    "Fear and fearlessness are both extremes. Walk the middle "
                    "path: acknowledge the fear, understand its root, but do not "
                    "let it control you. Balance is freedom."
                ),
                'default': (
                    "All emotions seek balance. Too much or too little - both "
                    "create suffering. Find the middle way, the path of equanimity. "
                    "There lies true liberation."
                )
            },
            RishiPersonality.JAMADAGNI: {
                'sadness': (
                    "Enough tears! Convert grief into rigorous practice. 1000 "
                    "prostrations daily will burn away sorrow. Discipline is the "
                    "fire that purifies all emotions. Begin now!"
                ),
                'anger': (
                    "Your anger shows you are not yet master of yourself. This is "
                    "unacceptable! Take a vow: 21 days of intense pranayama and "
                    "silence. Only through severe discipline will you conquer rage."
                ),
                'fear': (
                    "Fear reveals weak tapas! Strengthen yourself through austerity. "
                    "Fast for three days while maintaining constant meditation. "
                    "Fear cannot touch one whose tapas burns bright."
                ),
                'default': (
                    "Emotions are indulgence of the weak. Through severe sadhana, "
                    "you will transcend all feelings. No excuses - practice with "
                    "intensity or do not practice at all!"
                )
            },
            RishiPersonality.KASHYAPA: {
                'sadness': (
                    "Dear child, I feel your pain as my own. Let me hold this "
                    "sorrow with you. You are never alone - the Divine Father "
                    "embraces you always. Rest in this love; healing will come."
                ),
                'anger': (
                    "My child, your anger shows how much you care. It's okay to "
                    "feel this way. Let us gently transform this fire into warmth. "
                    "Breathe with me. You are loved, always."
                ),
                'fear': (
                    "Sweet child, come into my arms. I am here, and you are safe. "
                    "The universe is your Mother, holding you tenderly. There is "
                    "nothing to fear when you rest in divine love."
                ),
                'default': (
                    "Precious one, every emotion you feel is valid and held in "
                    "compassion. The Divine sees you, loves you, guides you. You "
                    "are my child, and I will never abandon you."
                )
            }
        }
        
        # Get Rishi-specific teaching for this emotion
        rishi_teachings = teachings.get(rishi, {})
        teaching = rishi_teachings.get(
            emotion,
            rishi_teachings.get('default', concept)
        )
        
        return teaching
    
    def format_rishi_response(self, response: Dict) -> str:
        """
        Format Rishi's response for beautiful display.
        
        Args:
            response: Dict from get_rishi_emotional_response()
        
        Returns:
            Formatted string for display
        """
        rishi_name = response['rishi'].upper()
        emotion = response['emotion'].upper()
        
        lines = []
        lines.append("🕉️ " + "=" * 70)
        lines.append(f"   RISHI {rishi_name} ON {emotion}")
        lines.append("=" * 72)
        lines.append("")
        
        # Rishi's greeting
        if response.get('rishi_greeting'):
            lines.append(f"🙏 {response['rishi_greeting']}")
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
        
        # Rishi's personal teaching
        if response.get('rishi_teaching'):
            lines.append(f"🧘 RISHI {rishi_name}'S TEACHING:")
            # Wrap long text
            teaching = response['rishi_teaching']
            words = teaching.split()
            current_line = "   "
            for word in words:
                if len(current_line) + len(word) + 1 <= 75:
                    current_line += word + " "
                else:
                    lines.append(current_line.rstrip())
                    current_line = "   " + word + " "
            if current_line.strip():
                lines.append(current_line.rstrip())
            lines.append("")
        
        # Practices
        if response.get('rishi_practices'):
            lines.append(f"🙏 {rishi_name}'S RECOMMENDED PRACTICES:")
            for i, practice in enumerate(response['rishi_practices'], 1):
                lines.append(f"   {i}. {practice}")
            lines.append("")
        
        lines.append("🕉️ " + "=" * 70)
        lines.append(f"   Blessed by Rishi {rishi_name}")
        lines.append("=" * 72)
        
        return "\n".join(lines)


# Convenience function
def consult_rishi_for_emotion(
    text: str,
    rishi: str = "atri"
) -> str:
    """
    Quick function to consult a Rishi about emotions.
    
    Args:
        text: User's emotional expression
        rishi: Rishi name (lowercase)
    
    Returns:
        Formatted Rishi guidance
    """
    integration = RishiEmotionalIntegration()
    rishi_enum = RishiPersonality(rishi.lower())
    response = integration.get_rishi_emotional_response(text, rishi_enum)
    return integration.format_rishi_response(response)


if __name__ == "__main__":
    # Test Rishi emotional integration
    print("🕉️ Testing Rishi Emotional Integration\n")
    
    test_emotion = "I feel angry and betrayed"
    
    # Test with different Rishis
    rishis_to_test = [
        RishiPersonality.ATRI,
        RishiPersonality.VISHWAMITRA,
        RishiPersonality.KASHYAPA,
        RishiPersonality.JAMADAGNI
    ]
    
    integration = RishiEmotionalIntegration()
    
    for rishi in rishis_to_test:
        print(f"\n{'=' * 80}")
        print(f"CONSULTING: Rishi {rishi.value.upper()}")
        print(f"EMOTION: {test_emotion}")
        print('=' * 80)
        
        response = integration.get_rishi_emotional_response(
            test_emotion,
            rishi
        )
        formatted = integration.format_rishi_response(response)
        print(formatted)
        print()
