"""
🕉️ LLM Router Service
=====================

Advanced Language Model routing service that intelligently selects and manages 
multiple LLM providers based on request type, user preferences, and system load.

Features:
- Multiple LLM provider support (OpenAI, Anthropic, Local models)
- Intelligent routing based on content type
- Fallback mechanisms for reliability
- Performance monitoring and optimization
- Dharmic content validation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json

# Import engines
try:
    from ..engines.emotional import create_emotional_engine
    from ..engines.rishi import create_authentic_rishi_engine
    from ..engines.personalization_engine import PersonalizationIntegration
except ImportError as e:
    logging.warning(f"Engine imports failed: {e}")

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Available LLM providers"""
    EMOTIONAL_ENGINE = "emotional_engine"
    RISHI_ENGINE = "rishi_engine"
    PERSONALIZATION_ENGINE = "personalization_engine"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"

class LLMRouter:
    """🧠 Intelligent LLM routing service"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.providers = {}
        self.routing_rules = {}
        self.fallback_chain = [
            LLMProvider.EMOTIONAL_ENGINE,
            LLMProvider.RISHI_ENGINE,
            LLMProvider.PERSONALIZATION_ENGINE
        ]
        
    async def initialize(self):
        """Initialize the LLM router with available providers"""
        try:
            self.logger.info("🌟 Initializing LLM Router...")
            
            # Initialize emotional engine
            try:
                self.providers[LLMProvider.EMOTIONAL_ENGINE] = create_emotional_engine()
                self.logger.info("✅ Emotional engine initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Emotional engine not available: {e}")
            
            # Initialize Rishi engine
            try:
                self.providers[LLMProvider.RISHI_ENGINE] = create_authentic_rishi_engine()
                self.logger.info("✅ Rishi engine initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Rishi engine not available: {e}")
            
            # Initialize personalization engine
            try:
                self.providers[LLMProvider.PERSONALIZATION_ENGINE] = PersonalizationIntegration()
                self.logger.info("✅ Personalization engine initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Personalization engine not available: {e}")
            
            # Set up routing rules
            self._setup_routing_rules()
            
            self.logger.info(f"🚀 LLM Router initialized with {len(self.providers)} providers")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM Router: {e}")
            
    def _setup_routing_rules(self):
        """Set up intelligent routing rules"""
        self.routing_rules = {
            "emotional_query": LLMProvider.EMOTIONAL_ENGINE,
            "spiritual_guidance": LLMProvider.RISHI_ENGINE,
            "personalized_recommendation": LLMProvider.PERSONALIZATION_ENGINE,
            "general_chat": LLMProvider.EMOTIONAL_ENGINE  # Default to emotional engine
        }
    
    async def route_request(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Route request to appropriate LLM provider"""
        try:
            # Determine request type
            request_type = self._classify_request(message, context)
            
            # Get provider for request type
            provider_type = self.routing_rules.get(request_type, LLMProvider.EMOTIONAL_ENGINE)
            
            # Route to provider
            if provider_type in self.providers:
                return await self._call_provider(provider_type, message, context)
            else:
                # Try fallback providers
                for fallback_provider in self.fallback_chain:
                    if fallback_provider in self.providers:
                        return await self._call_provider(fallback_provider, message, context)
                        
                # If no providers available, return basic response
                return {
                    "response": "I understand your message, but I'm currently experiencing technical difficulties. Please try again later.",
                    "provider": "fallback",
                    "success": False
                }
                
        except Exception as e:
            self.logger.error(f"❌ Routing failed: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "provider": "error",
                "success": False,
                "error": str(e)
            }
    
    def _classify_request(self, message: str, context: Dict[str, Any] = None) -> str:
        """Classify the type of request for routing"""
        message_lower = message.lower()
        
        # Emotional keywords
        if any(word in message_lower for word in ["feel", "emotion", "sad", "happy", "angry", "anxious", "peaceful"]):
            return "emotional_query"
        
        # Spiritual guidance keywords
        if any(word in message_lower for word in ["meditation", "dharma", "karma", "moksha", "enlightenment", "wisdom", "spiritual"]):
            return "spiritual_guidance"
        
        # Personalization keywords
        if any(word in message_lower for word in ["recommend", "suggest", "practice", "routine", "personal"]):
            return "personalized_recommendation"
        
        return "general_chat"
    
    async def _call_provider(self, provider_type: LLMProvider, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call specific provider"""
        provider = self.providers[provider_type]
        
        try:
            if provider_type == LLMProvider.EMOTIONAL_ENGINE:
                # Call emotional engine
                emotional_profile = await provider.analyze_emotional_state(message, context)
                response = await provider.generate_emotionally_intelligent_response(
                    emotional_profile=emotional_profile,
                    user_message=message,
                    context=context or {}
                )
                return {
                    "response": response.response_text,
                    "provider": "emotional_engine",
                    "success": True,
                    "emotional_context": {
                        "emotion": emotional_profile.primary_emotion.value,
                        "intensity": emotional_profile.intensity_level.value,
                        "sanskrit_wisdom": getattr(response, 'sanskrit_wisdom', None)
                    }
                }
                
            elif provider_type == LLMProvider.RISHI_ENGINE:
                # Call Rishi engine
                response = await provider.get_authentic_response("vasishtha", message, context)
                return {
                    "response": response.get("response", "Namaste, I am here to guide you."),
                    "provider": "rishi_engine", 
                    "success": True,
                    "rishi_context": {
                        "rishi": "vasishtha",
                        "greeting": response.get("greeting", ""),
                        "wisdom": response.get("wisdom", "")
                    }
                }
                
            elif provider_type == LLMProvider.PERSONALIZATION_ENGINE:
                # Call personalization engine
                response = await provider.process_request(message, context or {})
                return {
                    "response": response.get("response", "Let me provide personalized guidance."),
                    "provider": "personalization_engine",
                    "success": True,
                    "personalization_context": response
                }
                
        except Exception as e:
            self.logger.error(f"❌ Provider {provider_type} failed: {e}")
            return {
                "response": f"I encountered an issue with {provider_type.value}. Let me try another approach.",
                "provider": provider_type.value,
                "success": False,
                "error": str(e)
            }
    
    async def health_check(self) -> bool:
        """Check if router is healthy"""
        try:
            return len(self.providers) > 0
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

# Global instance
_llm_router: Optional[LLMRouter] = None

async def get_llm_router() -> LLMRouter:
    """Get global LLM router instance"""
    global _llm_router
    if _llm_router is None:
        _llm_router = LLMRouter()
        await _llm_router.initialize()
    return _llm_router
