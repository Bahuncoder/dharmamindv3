"""
🕉️ Emotional Intelligence Module - Compatibility Layer
======================================================

This module provides backward compatibility by exposing EmotionalIntelligence
from the dharmic_emotional_intelligence module.

The DharmicEmotionalIntelligence class provides:
- Dharmic emotion analysis with Sanskrit terminology
- Compassionate response generation
- Emotional regulation guidance based on Vedantic principles
- Crisis support with spiritual wisdom
"""

from .dharmic_emotional_intelligence import (
    DharmicEmotionalIntelligence,
    DharmicEmotion,
)

# Create alias for backward compatibility
EmotionalIntelligence = DharmicEmotionalIntelligence
EmotionState = DharmicEmotion  # Alias

# Factory function for getting emotional intelligence instance
_instance = None


def get_emotional_intelligence() -> EmotionalIntelligence:
    """Get or create the global emotional intelligence instance."""
    global _instance
    if _instance is None:
        _instance = EmotionalIntelligence()
    return _instance


# Export all for convenience
__all__ = [
    "EmotionalIntelligence",
    "DharmicEmotionalIntelligence",
    "EmotionState",
    "DharmicEmotion",
    "get_emotional_intelligence",
]

