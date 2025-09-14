"""
🌐 LLM Gateway Client
====================

Client for connecting to external LLM services and APIs.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
import asyncio
import warnings

try:
    import httpx
except ImportError as e:
    warnings.warn(f"httpx not available: {e}")
    httpx = None

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class LLMGatewayClient:
    """🌐 LLM Gateway Client for external LLM services"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Provider configurations
        self.providers = {
            LLMProvider.OPENAI: {
                "base_url": "https://api.openai.com/v1",
                "default_model": "gpt-3.5-turbo"
            },
            LLMProvider.ANTHROPIC: {
                "base_url": "https://api.anthropic.com/v1",
                "default_model": "claude-3-sonnet"
            },
            LLMProvider.GOOGLE: {
                "base_url": "https://generativelanguage.googleapis.com/v1",
                "default_model": "gemini-pro"
            },
            LLMProvider.LOCAL: {
                "base_url": "http://localhost:8080/v1",
                "default_model": "local-llm"
            }
        }
        
        # Mock responses for development
        self.mock_responses = [
            "As the ancient wisdom teaches, the path to understanding begins with mindful awareness. Consider how this moment offers an opportunity for growth and reflection.",
            "In the Vedic tradition, we learn that every challenge contains the seeds of wisdom. Trust in the process and remain open to the lessons being offered.",
            "The Rishis remind us that true knowledge comes not from external sources alone, but from the integration of learning with inner knowing. What does your heart tell you?",
            "Like a river flowing toward the ocean, all paths eventually lead to the same truth. Stay present with your journey and honor each step along the way.",
            "The ancient scriptures speak of dharma as our righteous path. Reflect on how your actions align with your deepest values and highest purpose."
        ]
        
        self.logger.info("🌐 LLM Gateway Client initialized")
    
    async def send_request(
        self,
        prompt: str,
        provider: LLMProvider = LLMProvider.LOCAL,
        model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send request to LLM provider"""
        
        try:
            # For development, return mock responses
            if not httpx or provider == LLMProvider.LOCAL:
                return await self._get_mock_response(prompt, context)
            
            # Real provider implementation would go here
            provider_config = self.providers.get(provider)
            if not provider_config:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # This would be the actual API call
            response = await self._call_provider_api(
                provider_config, model or provider_config["default_model"], prompt, context
            )
            
            return {
                "response": response,
                "provider": provider.value,
                "model": model or provider_config["default_model"],
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            # Fallback to mock response
            return await self._get_mock_response(prompt, context)
    
    async def _get_mock_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate mock response for development"""
        
        # Simple keyword-based response selection
        prompt_lower = prompt.lower()
        
        if "sad" in prompt_lower or "depression" in prompt_lower:
            response = "The path through sadness often leads to deeper compassion. As Sage Vasishtha teaches, honor your feelings while remembering your eternal, unshakeable nature. This too shall pass, and wisdom will emerge from this experience."
        elif "angry" in prompt_lower or "anger" in prompt_lower:
            response = "Channel this fiery energy wisely, as Sage Jamadagni demonstrates. Righteous anger can fuel positive change when directed with dharmic purpose. Breathe deeply and ask: what is this emotion teaching me?"
        elif "anxious" in prompt_lower or "worry" in prompt_lower:
            response = "Anxiety often arises when we project into an uncertain future. Return to this present moment, where peace resides. As the Rishis teach, surrender what you cannot control and act skillfully on what you can."
        elif "meditation" in prompt_lower or "spiritual" in prompt_lower:
            response = "Meditation is the doorway to inner knowing. Begin with simple breath awareness, allowing thoughts to come and go like clouds in the vast sky of consciousness. Consistency matters more than duration."
        elif "purpose" in prompt_lower or "meaning" in prompt_lower:
            response = "Your dharma - your life's purpose - is discovered through self-inquiry and service to others. Ask yourself: What unique gifts do you bring to the world? How can you serve the greater good while honoring your authentic nature?"
        else:
            # Default spiritual guidance
            import random
            response = random.choice(self.mock_responses)
        
        return {
            "response": response,
            "provider": "mock",
            "model": "dharmic-wisdom-engine",
            "success": True,
            "is_mock": True
        }
    
    async def _call_provider_api(
        self, 
        provider_config: Dict[str, Any], 
        model: str, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Call actual provider API (placeholder for real implementation)"""
        
        # This would contain the actual API calls to providers
        # For now, return mock response
        await asyncio.sleep(0.1)  # Simulate API delay
        return "This would be a real LLM response from the external provider."
    
    async def get_available_models(self, provider: LLMProvider) -> List[str]:
        """Get available models for a provider"""
        
        model_lists = {
            LLMProvider.OPENAI: ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            LLMProvider.ANTHROPIC: ["claude-3-sonnet", "claude-3-opus", "claude-3-haiku"],
            LLMProvider.GOOGLE: ["gemini-pro", "gemini-pro-vision"],
            LLMProvider.LOCAL: ["local-llm", "dharma-llm"],
            LLMProvider.HUGGINGFACE: ["mistral-7b", "llama2-chat"]
        }
        
        return model_lists.get(provider, ["default"])
    
    def get_provider_status(self, provider: LLMProvider) -> Dict[str, Any]:
        """Get provider status and health"""
        
        # Mock status for development
        return {
            "provider": provider.value,
            "status": "available",
            "latency_ms": 150,
            "rate_limit_remaining": 1000,
            "last_updated": "2024-01-01T12:00:00Z"
        }

# Global LLM gateway client instance
_llm_gateway_client: Optional[LLMGatewayClient] = None

def get_llm_gateway_client() -> LLMGatewayClient:
    """Get global LLM gateway client instance"""
    global _llm_gateway_client
    if _llm_gateway_client is None:
        _llm_gateway_client = LLMGatewayClient()
    return _llm_gateway_client

def create_llm_gateway_client() -> LLMGatewayClient:
    """Create new LLM gateway client instance"""
    return LLMGatewayClient()

# Export commonly used classes and functions
__all__ = [
    'LLMGatewayClient',
    'LLMProvider',
    'get_llm_gateway_client',
    'create_llm_gateway_client'
]
