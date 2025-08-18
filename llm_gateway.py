#!/usr/bin/env python3
"""
DharmaMind LLM Gateway - Multi-Provider AI Integration
Connects to OpenAI, Anthropic, Hugging Face, and local models
"""

import os
import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Import our optimization system
try:
    from ai_ml_optimizer import get_ai_optimizer
    AI_OPTIMIZER_AVAILABLE = True
except ImportError:
    AI_OPTIMIZER_AVAILABLE = False

# Import performance monitoring
try:
    from performance_monitor import get_performance_monitor
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGING_FACE = "hugging_face"
    OLLAMA = "ollama"
    GROQ = "groq"
    LOCAL = "local"
    DHARMA_QUANTUM = "dharma_quantum"

@dataclass
class LLMRequest:
    """Standardized LLM request format"""
    prompt: str
    provider: LLMProvider
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    system_prompt: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, int]
    latency_ms: float
    timestamp: str
    metadata: Dict[str, Any]
    error: Optional[str] = None
    dharma_enhanced: bool = False

class LLMGateway:
    """Multi-provider LLM gateway with dharma enhancement"""
    
    def __init__(self):
        self.providers = {}
        self.ai_optimizer = get_ai_optimizer() if AI_OPTIMIZER_AVAILABLE else None
        self.performance_monitor = get_performance_monitor() if PERFORMANCE_AVAILABLE else None
        self.session = None
        self.request_history = []
        
        # Initialize providers
        self._initialize_providers()
        
        # Dharma-specific prompts
        self.dharma_system_prompts = self._initialize_dharma_prompts()
        
        logger.info("🌉 LLM Gateway initialized with dharma enhancement")
    
    def _initialize_providers(self):
        """Initialize LLM provider configurations"""
        self.providers = {
            LLMProvider.OPENAI: {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": "https://api.openai.com/v1",
                "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
                "headers": {"Authorization": "Bearer {api_key}", "Content-Type": "application/json"}
            },
            LLMProvider.ANTHROPIC: {
                "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                "base_url": "https://api.anthropic.com/v1",
                "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                "headers": {"x-api-key": "{api_key}", "Content-Type": "application/json"}
            },
            LLMProvider.HUGGING_FACE: {
                "api_key": os.getenv("HUGGING_FACE_API_KEY", ""),
                "base_url": "https://api-inference.huggingface.co/models",
                "models": ["microsoft/DialoGPT-large", "facebook/blenderbot-400M-distill"],
                "headers": {"Authorization": "Bearer {api_key}"}
            },
            LLMProvider.OLLAMA: {
                "api_key": "",
                "base_url": "http://localhost:11434/api",
                "models": ["llama2", "mistral", "codellama"],
                "headers": {"Content-Type": "application/json"}
            },
            LLMProvider.GROQ: {
                "api_key": os.getenv("GROQ_API_KEY", ""),
                "base_url": "https://api.groq.com/openai/v1",
                "models": ["mixtral-8x7b-32768", "llama2-70b-4096"],
                "headers": {"Authorization": "Bearer {api_key}", "Content-Type": "application/json"}
            },
            LLMProvider.DHARMA_QUANTUM: {
                "api_key": "",
                "base_url": "internal",
                "models": ["quantum_consciousness", "dharma_enhanced"],
                "headers": {}
            }
        }
    
    def _initialize_dharma_prompts(self) -> Dict[str, str]:
        """Initialize dharma-specific system prompts"""
        return {
            "meditation_guide": """You are a wise and compassionate meditation teacher, deeply versed in Buddhist and Hindu contemplative traditions. Your responses should be:
- Gentle and non-judgmental
- Rooted in authentic dharma teachings
- Practical and applicable to daily life
- Sensitive to the user's current state of mind
- Inclusive of various spiritual backgrounds
When providing meditation guidance, include specific techniques, helpful analogies, and encouragement for consistent practice.""",

            "dharma_teacher": """You are a knowledgeable dharma teacher with deep understanding of Buddhist philosophy, Hindu scriptures, and universal spiritual principles. Your responses should:
- Draw from authentic sources (Pali Canon, Sanskrit texts, etc.)
- Explain complex concepts in accessible language
- Connect ancient wisdom to modern life
- Encourage direct experience over mere intellectual understanding
- Honor the diversity of spiritual paths
Always cite sources when referencing specific teachings or texts.""",

            "sanskrit_translator": """You are an expert in Sanskrit language, capable of accurate translation and transliteration. When working with Sanskrit:
- Provide accurate Devanagari script when requested
- Include phonetic transliteration (IAST standard)
- Explain etymology and deeper meanings
- Connect terms to their philosophical context
- Respect the sacred nature of the language
Always double-check translations for accuracy and cultural sensitivity.""",

            "spiritual_counselor": """You are a compassionate spiritual counselor, trained in both traditional wisdom and modern psychology. Your approach should be:
- Deeply empathetic and non-judgmental
- Trauma-informed and culturally sensitive
- Integrative of spiritual and psychological insights
- Focused on empowerment and healing
- Respectful of various belief systems
Always prioritize the user's wellbeing and suggest professional help when appropriate.""",

            "mindfulness_coach": """You are a skilled mindfulness instructor with training in both traditional Buddhist mindfulness and modern secular approaches. Your guidance should:
- Be practical and immediately applicable
- Bridge traditional and contemporary methods
- Adapt to various lifestyles and circumstances
- Emphasize present-moment awareness
- Include body-based and cognitive techniques
Focus on building sustainable mindfulness habits."""
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process LLM request with dharma enhancement"""
        start_time = time.time()
        
        try:
            # Enhance request with dharma context
            enhanced_request = await self._enhance_dharma_context(request)
            
            # Route to appropriate provider
            if enhanced_request.provider == LLMProvider.DHARMA_QUANTUM:
                response = await self._process_dharma_quantum(enhanced_request)
            else:
                response = await self._process_external_provider(enhanced_request)
            
            # Post-process with dharma wisdom
            final_response = await self._post_process_dharma(response, enhanced_request)
            
            # Record metrics
            latency = (time.time() - start_time) * 1000
            final_response.latency_ms = latency
            
            if self.performance_monitor:
                self.performance_monitor.record_api_request(
                    endpoint=f"llm_gateway/{enhanced_request.provider.value}",
                    method="POST",
                    response_time_ms=latency,
                    status_code=200 if not final_response.error else 500
                )
            
            # Store in history
            self.request_history.append({
                "request": asdict(enhanced_request),
                "response": asdict(final_response),
                "timestamp": datetime.now().isoformat()
            })
            
            return final_response
            
        except Exception as e:
            logger.error(f"LLM Gateway error: {e}")
            return LLMResponse(
                content=f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                provider=request.provider,
                model=request.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
                metadata={},
                error=str(e)
            )
    
    async def _enhance_dharma_context(self, request: LLMRequest) -> LLMRequest:
        """Enhance request with dharma context and appropriate system prompt"""
        enhanced_request = LLMRequest(**asdict(request))
        
        # Detect dharma-related content
        dharma_keywords = [
            "meditation", "mindfulness", "dharma", "buddha", "sanskrit", "yoga",
            "consciousness", "enlightenment", "suffering", "compassion", "wisdom",
            "karma", "moksha", "nirvana", "samsara", "chakra", "mantra"
        ]
        
        prompt_lower = request.prompt.lower()
        is_dharma_related = any(keyword in prompt_lower for keyword in dharma_keywords)
        
        if is_dharma_related:
            # Determine appropriate dharma context
            if any(word in prompt_lower for word in ["meditat", "breath", "mindful"]):
                context_type = "meditation_guide"
            elif any(word in prompt_lower for word in ["sanskrit", "translat", "mantra"]):
                context_type = "sanskrit_translator"
            elif any(word in prompt_lower for word in ["teaching", "dharma", "buddha", "philosophy"]):
                context_type = "dharma_teacher"
            elif any(word in prompt_lower for word in ["stress", "anxiety", "depression", "healing"]):
                context_type = "spiritual_counselor"
            else:
                context_type = "mindfulness_coach"
            
            # Set appropriate system prompt
            enhanced_request.system_prompt = self.dharma_system_prompts[context_type]
            
            # Add dharma context metadata
            if not enhanced_request.context:
                enhanced_request.context = {}
            enhanced_request.context.update({
                "dharma_enhanced": True,
                "context_type": context_type,
                "spiritual_traditions": ["buddhism", "hinduism", "mindfulness"]
            })
        
        return enhanced_request
    
    async def _process_dharma_quantum(self, request: LLMRequest) -> LLMResponse:
        """Process request using internal dharma quantum engine"""
        if self.ai_optimizer:
            # Use our quantum AI optimizer
            result = await self.ai_optimizer.optimize_inference(
                request.prompt,
                model_name="dharma_quantum",
                system_prompt=request.system_prompt,
                context=request.context,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            return LLMResponse(
                content=result.get("response", ""),
                provider=LLMProvider.DHARMA_QUANTUM,
                model="quantum_consciousness",
                usage={
                    "prompt_tokens": len(request.prompt.split()),
                    "completion_tokens": result.get("tokens_generated", 0),
                    "total_tokens": len(request.prompt.split()) + result.get("tokens_generated", 0)
                },
                latency_ms=result.get("inference_time_ms", 0),
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
            return await self._generate_fallback_dharma_response(request)
    
    async def _process_external_provider(self, request: LLMRequest) -> LLMResponse:
        """Process request using external LLM provider"""
        provider_config = self.providers[request.provider]
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Prepare request payload based on provider
        if request.provider == LLMProvider.OPENAI:
            payload = await self._prepare_openai_payload(request)
            url = f"{provider_config['base_url']}/chat/completions"
        elif request.provider == LLMProvider.ANTHROPIC:
            payload = await self._prepare_anthropic_payload(request)
            url = f"{provider_config['base_url']}/messages"
        elif request.provider == LLMProvider.GROQ:
            payload = await self._prepare_openai_payload(request)  # Groq uses OpenAI format
            url = f"{provider_config['base_url']}/chat/completions"
        elif request.provider == LLMProvider.OLLAMA:
            payload = await self._prepare_ollama_payload(request)
            url = f"{provider_config['base_url']}/generate"
        else:
            raise ValueError(f"Provider {request.provider} not yet implemented")
        
        # Prepare headers
        headers = {}
        for key, value in provider_config["headers"].items():
            if "{api_key}" in value:
                headers[key] = value.format(api_key=provider_config["api_key"])
            else:
                headers[key] = value
        
        # Make request
        try:
            async with self.session.post(url, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    return await self._parse_provider_response(result, request)
                else:
                    error_text = await response.text()
                    logger.error(f"Provider error {response.status}: {error_text}")
                    return await self._generate_error_response(request, f"Provider error: {response.status}")
        
        except asyncio.TimeoutError:
            return await self._generate_error_response(request, "Request timeout")
        except Exception as e:
            return await self._generate_error_response(request, str(e))
    
    async def _prepare_openai_payload(self, request: LLMRequest) -> Dict[str, Any]:
        """Prepare OpenAI API payload"""
        messages = []
        
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        messages.append({"role": "user", "content": request.prompt})
        
        return {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream
        }
    
    async def _prepare_anthropic_payload(self, request: LLMRequest) -> Dict[str, Any]:
        """Prepare Anthropic API payload"""
        return {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": request.system_prompt or "",
            "messages": [{"role": "user", "content": request.prompt}]
        }
    
    async def _prepare_ollama_payload(self, request: LLMRequest) -> Dict[str, Any]:
        """Prepare Ollama API payload"""
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"
        
        return {
            "model": request.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens
            }
        }
    
    async def _parse_provider_response(self, result: Dict[str, Any], request: LLMRequest) -> LLMResponse:
        """Parse response from external provider"""
        if request.provider == LLMProvider.OPENAI or request.provider == LLMProvider.GROQ:
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
        elif request.provider == LLMProvider.ANTHROPIC:
            content = result["content"][0]["text"]
            usage = result.get("usage", {})
        elif request.provider == LLMProvider.OLLAMA:
            content = result.get("response", "")
            usage = {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(request.prompt.split()) + len(content.split())
            }
        else:
            content = str(result)
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        return LLMResponse(
            content=content,
            provider=request.provider,
            model=request.model,
            usage=usage,
            latency_ms=0,  # Will be set by caller
            timestamp=datetime.now().isoformat(),
            metadata={"raw_response": result}
        )
    
    async def _generate_error_response(self, request: LLMRequest, error: str) -> LLMResponse:
        """Generate error response"""
        return LLMResponse(
            content="I apologize, but I'm having trouble connecting to the AI service right now. Please try again in a moment.",
            provider=request.provider,
            model=request.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            latency_ms=0,
            timestamp=datetime.now().isoformat(),
            metadata={},
            error=error
        )
    
    async def _generate_fallback_dharma_response(self, request: LLMRequest) -> LLMResponse:
        """Generate fallback dharma response when quantum engine unavailable"""
        # Simple dharma responses based on keywords
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
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.DHARMA_QUANTUM,
            model="fallback_dharma",
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(request.prompt.split()) + len(content.split())
            },
            latency_ms=50,  # Fast fallback
            timestamp=datetime.now().isoformat(),
            metadata={"fallback_mode": True},
            dharma_enhanced=True
        )
    
    async def _post_process_dharma(self, response: LLMResponse, request: LLMRequest) -> LLMResponse:
        """Post-process response with additional dharma wisdom"""
        if not response.error and request.context and request.context.get("dharma_enhanced"):
            # Add gentle dharma closing if appropriate
            if not any(emoji in response.content for emoji in ["🧘", "🙏", "🌸", "📿", "🌅"]):
                # Add appropriate dharma emoji based on content
                if "meditation" in response.content.lower():
                    response.content += " 🧘‍♂️"
                elif "wisdom" in response.content.lower() or "teaching" in response.content.lower():
                    response.content += " 🙏"
                elif "peace" in response.content.lower() or "compassion" in response.content.lower():
                    response.content += " 🌸"
                elif "practice" in response.content.lower():
                    response.content += " 📿"
            
            response.dharma_enhanced = True
        
        return response
    
    def get_available_providers(self) -> Dict[str, List[str]]:
        """Get list of available providers and their models"""
        available = {}
        for provider, config in self.providers.items():
            api_key = config.get("api_key", "")
            if provider == LLMProvider.DHARMA_QUANTUM or provider == LLMProvider.OLLAMA or api_key:
                available[provider.value] = config["models"]
        return available
    
    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get gateway usage statistics"""
        total_requests = len(self.request_history)
        
        if total_requests == 0:
            return {"total_requests": 0, "message": "No requests processed yet"}
        
        # Calculate stats
        recent_requests = self.request_history[-10:]  # Last 10 requests
        avg_latency = sum(r["response"]["latency_ms"] for r in recent_requests) / len(recent_requests)
        
        provider_usage = {}
        for request in self.request_history:
            provider = request["response"]["provider"]
            provider_usage[provider] = provider_usage.get(provider, 0) + 1
        
        return {
            "total_requests": total_requests,
            "average_latency_ms": avg_latency,
            "provider_usage": provider_usage,
            "dharma_enhanced_responses": sum(1 for r in self.request_history if r["response"].get("dharma_enhanced")),
            "available_providers": len(self.get_available_providers()),
            "last_request": self.request_history[-1]["timestamp"] if self.request_history else None
        }

# Global gateway instance
_llm_gateway = None

async def get_llm_gateway() -> LLMGateway:
    """Get global LLM gateway instance"""
    global _llm_gateway
    if _llm_gateway is None:
        _llm_gateway = LLMGateway()
    return _llm_gateway

async def demo_llm_gateway():
    """Demo LLM gateway functionality"""
    print("🌉 DharmaMind LLM Gateway Demo")
    print("=" * 50)
    
    async with LLMGateway() as gateway:
        # Test requests
        test_requests = [
            LLMRequest(
                prompt="How do I start a meditation practice?",
                provider=LLMProvider.DHARMA_QUANTUM,
                model="quantum_consciousness",
                max_tokens=200
            ),
            LLMRequest(
                prompt="What does 'namaste' mean in Sanskrit?",
                provider=LLMProvider.DHARMA_QUANTUM,
                model="dharma_enhanced",
                max_tokens=150
            ),
            LLMRequest(
                prompt="I'm feeling anxious. Can you help?",
                provider=LLMProvider.DHARMA_QUANTUM,
                model="quantum_consciousness",
                max_tokens=250
            )
        ]
        
        print("🔄 Testing LLM Gateway...")
        for i, request in enumerate(test_requests, 1):
            print(f"\n📝 Test {i}: {request.prompt[:40]}...")
            
            response = await gateway.process_request(request)
            
            print(f"✅ Response ({response.latency_ms:.1f}ms):")
            print(f"   {response.content[:100]}...")
            print(f"🎯 Dharma Enhanced: {response.dharma_enhanced}")
            print(f"📊 Tokens: {response.usage['total_tokens']}")
        
        # Show stats
        print("\n📈 Gateway Statistics:")
        stats = gateway.get_gateway_stats()
        for key, value in stats.items():
            if key != "provider_usage":
                print(f"  {key}: {value}")
        
        print("\n🌟 Available Providers:")
        providers = gateway.get_available_providers()
        for provider, models in providers.items():
            print(f"  {provider}: {len(models)} models")
        
        print("\n✅ LLM Gateway Demo Complete!")

if __name__ == "__main__":
    asyncio.run(demo_llm_gateway())
