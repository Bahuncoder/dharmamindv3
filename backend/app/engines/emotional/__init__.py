"""
🕉️ Emotional Intelligence Engines
==================================

Advanced emotional intelligence and healing engines for DharmaMind:

- advanced_emotional_engine.py - Comprehensive emotional intelligence with dharmic healing
- empathy_engine.py           - Deep empathy and understanding engine  
- healing_response_engine.py  - Transformative healing response generation
- emotional_memory_engine.py  - Emotional pattern learning and memory

These engines work together to provide:
- Deep emotional understanding and validation
- Dharmic emotional healing and guidance
- Chakra-based emotional mapping
- Sanskrit wisdom integration
- Compassionate AI personality
- Transformative healing responses

May these engines bring emotional healing and understanding to all beings 💙
"""

# Import main classes with fallback handling
try:
    from .advanced_emotional_engine import (
        AdvancedEmotionalEngine,
        EmotionalProfile,
        EmotionalResponse,
        EmotionalState,
        EmotionalIntensity,
        ChakraEmotion,
        create_emotional_engine,
        get_emotional_engine
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import AdvancedEmotionalEngine: {e}")
    
    class AdvancedEmotionalEngine:
        pass
    class EmotionalProfile:
        pass
    class EmotionalResponse:
        pass
    class EmotionalState:
        pass
    class EmotionalIntensity:
        pass
    class ChakraEmotion:
        pass
    
    def create_emotional_engine():
        return AdvancedEmotionalEngine()
    
    def get_emotional_engine():
        return AdvancedEmotionalEngine()

__all__ = [
    'AdvancedEmotionalEngine',
    'EmotionalProfile', 
    'EmotionalResponse',
    'EmotionalState',
    'EmotionalIntensity',
    'ChakraEmotion',
    'create_emotional_engine',
    'get_emotional_engine'
]
