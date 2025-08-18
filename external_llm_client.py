#!/usr/bin/env python3
"""
External LLM Gateway Client - Integration Module
Connects main backend to external LLM gateway service
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExternalLLMRequest:
    """Request to external LLM gateway"""
    prompt: str
    provider: str = "dharma_quantum"
    model: str = "quantum_consciousness"
    max_tokens: int = 1000
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class ExternalLLMResponse:
    """Response from external LLM gateway"""
    content: str
    provider: str
    model: str
    usage: Dict[str, int]
    latency_ms: float
    timestamp: str
    metadata: Dict[str, Any]
    error: Optional[str] = None
    dharma_enhanced: bool = False

class ExternalLLMGatewayClient:
    """Client for communicating with external LLM gateway"""
    
    def __init__(self, gateway_url: str = "http://localhost:8003", api_key: str = "llm-gateway-secure-key-123"):
        self.gateway_url = gateway_url
        self.api_key = api_key
        self.session = None
        
        # Import our AI optimizer for dharma enhancement
        try:
            from ai_ml_optimizer import get_ai_optimizer
            self.ai_optimizer = get_ai_optimizer()
            self.dharma_available = True
        except ImportError:
            self.ai_optimizer = None
            self.dharma_available = False
        
        logger.info(f"🌉 External LLM Gateway Client initialized - Gateway: {gateway_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def process_request(self, request: ExternalLLMRequest) -> ExternalLLMResponse:
        """Process LLM request through external gateway with dharma enhancement"""
        start_time = time.time()
        
        try:
            # For dharma_quantum provider, use our internal system
            if request.provider == "dharma_quantum":
                return await self._process_dharma_internal(request, start_time)
            
            # For external providers, route to external gateway
            return await self._process_external_gateway(request, start_time)
            
        except Exception as e:
            logger.error(f"External LLM Gateway error: {e}")
            return ExternalLLMResponse(
                content="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                provider=request.provider,
                model=request.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
                metadata={},
                error=str(e)
            )
    
    async def _process_dharma_internal(self, request: ExternalLLMRequest, start_time: float) -> ExternalLLMResponse:
        """Process dharma requests internally for spiritual enhancement"""
        if self.dharma_available and self.ai_optimizer:
            # Use our quantum AI optimizer for dharma-enhanced responses
            result = await self.ai_optimizer.optimize_inference(
                request.prompt,
                model_name="dharma_quantum",
                system_prompt=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            # Enhance with spiritual context
            enhanced_content = await self._enhance_dharma_content(result.get("response", ""), request.prompt)
            
            return ExternalLLMResponse(
                content=enhanced_content,
                provider="dharma_quantum",
                model="quantum_consciousness",
                usage={
                    "prompt_tokens": len(request.prompt.split()),
                    "completion_tokens": result.get("tokens_generated", 0),
                    "total_tokens": len(request.prompt.split()) + result.get("tokens_generated", 0)
                },
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "quantum_coherence": result.get("quantum_coherence", 0),
                    "dharma_relevance": result.get("dharma_relevance", 0),
                    "optimizations_applied": result.get("optimizations_applied", [])
                },
                dharma_enhanced=True
            )
        else:
            # Fallback dharma response
            return await self._generate_fallback_dharma_response(request, start_time)
    
    async def _process_external_gateway(self, request: ExternalLLMRequest, start_time: float) -> ExternalLLMResponse:
        """Process request through external LLM gateway"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "provider": request.provider,
            "model": request.model,
            "query": request.prompt,
            "system_prompt": request.system_prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "user_id": request.user_id
        }
        
        try:
            async with self.session.post(
                f"{self.gateway_url}/generate",
                json=payload,
                headers=headers,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return ExternalLLMResponse(
                        content=result.get("content", ""),
                        provider=request.provider,
                        model=request.model,
                        usage=result.get("usage", {}),
                        latency_ms=(time.time() - start_time) * 1000,
                        timestamp=datetime.now().isoformat(),
                        metadata={"external_gateway": True, "gateway_response": result}
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"External gateway error {response.status}: {error_text}")
                    return await self._generate_error_response(request, f"Gateway error: {response.status}", start_time)
        
        except asyncio.TimeoutError:
            return await self._generate_error_response(request, "Request timeout", start_time)
        except Exception as e:
            return await self._generate_error_response(request, str(e), start_time)
    
    async def _enhance_dharma_content(self, content: str, original_prompt: str) -> str:
        """Enhance content with dharma wisdom"""
        # Detect spiritual context
        dharma_keywords = [
            "meditation", "mindfulness", "dharma", "buddha", "sanskrit", "yoga",
            "consciousness", "enlightenment", "suffering", "compassion", "wisdom"
        ]
        
        prompt_lower = original_prompt.lower()
        is_dharma_related = any(keyword in prompt_lower for keyword in dharma_keywords)
        
        if is_dharma_related:
            # Add appropriate spiritual emoji if not present
            if not any(emoji in content for emoji in ["🧘", "🙏", "🌸", "📿", "🌅"]):
                if "meditation" in content.lower() or "breath" in content.lower():
                    content += " 🧘‍♂️"
                elif "wisdom" in content.lower() or "teaching" in content.lower():
                    content += " 🙏"
                elif "peace" in content.lower() or "compassion" in content.lower():
                    content += " 🌸"
                elif "practice" in content.lower():
                    content += " 📿"
        
        return content
    
    async def _generate_fallback_dharma_response(self, request: ExternalLLMRequest, start_time: float) -> ExternalLLMResponse:
        """Generate fallback dharma response when quantum engine unavailable"""
        prompt_lower = request.prompt.lower()
        
        if "meditation" in prompt_lower:
            content = """Begin by finding a comfortable seated position with your spine naturally erect. Close your eyes gently and take three deep breaths, allowing each exhale to release any tension.

Now, simply observe your natural breath without trying to change it. When your mind wanders—and it will—gently return your attention to the breath. This is not a failure but the practice itself.

Start with just 5-10 minutes daily. As the Buddha taught, "Better than a thousand hollow words is one word that brings peace." Your consistent practice, however brief, is more valuable than sporadic long sessions.

May your practice bring you peace and insight. 🧘‍♂️"""
        
        elif "dharma" in prompt_lower:
            content = """Dharma refers to the natural order of existence and the path of righteous living. In Buddhist context, it encompasses the Buddha's teachings and the universal principles that govern liberation from suffering.

The Dharma guides us to understand the Four Noble Truths:
1. Life contains suffering (dukkha)
2. Suffering arises from attachment and craving
3. Suffering can cease
4. The Eightfold Path leads to the cessation of suffering

Living according to Dharma means aligning our actions, speech, and thoughts with wisdom and compassion. It's not mere philosophy but a practical guide for reducing suffering and cultivating joy.

How might you apply these teachings to your current situation? 🙏"""
        
        elif "sanskrit" in prompt_lower:
            content = """Sanskrit, meaning "perfectly formed" (संस्कृत), is the sacred language of many Hindu and Buddhist texts. Each syllable carries vibrational significance, believed to resonate with cosmic frequencies.

Common terms you might encounter:
• Om (ॐ) - The primordial sound, representing universal consciousness
• Namaste (नमस्ते) - "I bow to you," honoring the divine in another
• Dharma (धर्म) - Natural law, righteous path
• Karma (कर्म) - Action and its consequences
• Moksha (मोक्ष) - Liberation, release from cyclic existence

Would you like help with a specific Sanskrit term or phrase? I can provide pronunciation and deeper meanings. 📿"""
        
        else:
            content = """Thank you for reaching out. I'm here to support your spiritual journey with wisdom from various contemplative traditions.

Whether you're seeking guidance on meditation, understanding dharma teachings, exploring Sanskrit concepts, or simply need a moment of peace, I'm here to help.

Remember: The path of awakening is walked one step at a time. Be patient and compassionate with yourself as you grow in wisdom and understanding.

How may I assist you today? 🌸"""
        
        return ExternalLLMResponse(
            content=content,
            provider="dharma_quantum",
            model="fallback_dharma",
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(request.prompt.split()) + len(content.split())
            },
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=datetime.now().isoformat(),
            metadata={"fallback_mode": True},
            dharma_enhanced=True
        )
    
    async def _generate_error_response(self, request: ExternalLLMRequest, error: str, start_time: float) -> ExternalLLMResponse:
        """Generate error response"""
        return ExternalLLMResponse(
            content="I apologize, but I'm having trouble connecting to the AI service right now. Please try again in a moment.",
            provider=request.provider,
            model=request.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=datetime.now().isoformat(),
            metadata={},
            error=error
        )
    
    async def get_available_providers(self) -> Dict[str, Any]:
        """Get available providers from external gateway + dharma"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with self.session.get(
                f"{self.gateway_url}/providers",
                headers=headers,
                timeout=10
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Add dharma quantum provider
                    result["providers"]["dharma_quantum"] = {
                        "available": True,
                        "models": ["quantum_consciousness", "dharma_enhanced"]
                    }
                    
                    return result
                else:
                    # Fallback to dharma only
                    return {
                        "providers": {
                            "dharma_quantum": {
                                "available": True,
                                "models": ["quantum_consciousness", "dharma_enhanced"]
                            }
                        }
                    }
        except Exception as e:
            logger.error(f"Error getting providers: {e}")
            return {
                "providers": {
                    "dharma_quantum": {
                        "available": True,
                        "models": ["quantum_consciousness", "dharma_enhanced"]
                    }
                }
            }
    
    async def get_gateway_stats(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with self.session.get(
                f"{self.gateway_url}/metrics",
                headers=headers,
                timeout=10
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Gateway returned {response.status}"}
        except Exception as e:
            return {"error": str(e)}

# Global client instance
_external_llm_client = None

async def get_external_llm_client() -> ExternalLLMGatewayClient:
    """Get global external LLM client instance"""
    global _external_llm_client
    if _external_llm_client is None:
        _external_llm_client = ExternalLLMGatewayClient()
    return _external_llm_client

async def demo_external_gateway():
    """Demo external LLM gateway functionality"""
    print("🌉 External LLM Gateway Client Demo")
    print("=" * 50)
    
    async with ExternalLLMGatewayClient() as client:
        # Test requests
        test_requests = [
            ExternalLLMRequest(
                prompt="How do I start a meditation practice?",
                provider="dharma_quantum",
                model="quantum_consciousness"
            ),
            ExternalLLMRequest(
                prompt="What does 'namaste' mean in Sanskrit?",
                provider="dharma_quantum",
                model="dharma_enhanced"
            ),
            ExternalLLMRequest(
                prompt="I'm feeling anxious. Can you help?",
                provider="dharma_quantum",
                model="quantum_consciousness"
            )
        ]
        
        print("🔄 Testing External LLM Gateway Client...")
        for i, request in enumerate(test_requests, 1):
            print(f"\n📝 Test {i}: {request.prompt[:40]}...")
            
            response = await client.process_request(request)
            
            print(f"✅ Response ({response.latency_ms:.1f}ms):")
            print(f"   {response.content[:100]}...")
            print(f"🎯 Dharma Enhanced: {response.dharma_enhanced}")
            print(f"📊 Tokens: {response.usage['total_tokens']}")
        
        # Show available providers
        print("\n🌟 Available Providers:")
        providers = await client.get_available_providers()
        for provider, details in providers.get("providers", {}).items():
            status = "✅" if details.get("available") else "❌"
            print(f"  {status} {provider}: {len(details.get('models', []))} models")
        
        print("\n✅ External LLM Gateway Client Demo Complete!")

if __name__ == "__main__":
    asyncio.run(demo_external_gateway())
