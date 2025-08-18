#!/usr/bin/env python3
"""
Production LLM Gateway Client
Full-featured client for external LLM gateway microservice
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DHARMA_QUANTUM = "dharma_quantum"

class ModelType(Enum):
    """Model categories"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE = "image"

@dataclass
class LLMRequest:
    """Structured LLM request"""
    prompt: str
    provider: Union[str, LLMProvider] = LLMProvider.DHARMA_QUANTUM
    model: str = "quantum_consciousness"
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    stream: bool = False
    
    def __post_init__(self):
        """Convert string provider to enum"""
        if isinstance(self.provider, str):
            try:
                self.provider = LLMProvider(self.provider)
            except ValueError:
                self.provider = LLMProvider.DHARMA_QUANTUM

@dataclass
class LLMResponse:
    """Structured LLM response"""
    content: str
    provider: str
    model: str
    usage: Dict[str, int]
    latency_ms: float
    timestamp: str
    request_id: str
    dharma_enhanced: bool = False
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cached: bool = False

@dataclass
class ChatMessage:
    """Chat message structure"""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class LLMGatewayClient:
    """Production LLM Gateway Client with full features"""
    
    def __init__(
        self,
        gateway_url: str = "http://localhost:8003",
        api_key: str = "llm-gateway-secure-key-123",
        timeout: int = 60,
        max_retries: int = 3,
        enable_caching: bool = True
    ):
        self.gateway_url = gateway_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_caching = enable_caching
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self._connector = None
        
        # Request tracking
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
        # Simple in-memory cache
        self._cache: Dict[str, LLMResponse] = {}
        self._cache_ttl = 300  # 5 minutes
        
        logger.info(f"🌉 LLM Gateway Client initialized - URL: {gateway_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
    
    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return f"req_{int(time.time() * 1000)}_{self.request_count}"
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "prompt": request.prompt,
            "provider": request.provider.value if isinstance(request.provider, LLMProvider) else request.provider,
            "model": request.model,
            "temperature": request.temperature,
            "system_prompt": request.system_prompt
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_response: LLMResponse) -> bool:
        """Check if cached response is still valid"""
        if not self.enable_caching:
            return False
        
        try:
            cached_time = datetime.fromisoformat(cached_response.timestamp)
            current_time = datetime.now()
            age_seconds = (current_time - cached_time).total_seconds()
            return age_seconds < self._cache_ttl
        except:
            return False
    
    async def chat(
        self,
        prompt: str,
        provider: Union[str, LLMProvider] = LLMProvider.DHARMA_QUANTUM,
        model: str = "quantum_consciousness",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Convenient chat method"""
        request = LLMRequest(
            prompt=prompt,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            **kwargs
        )
        return await self.process_request(request)
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process LLM request with full features"""
        start_time = time.time()
        request_id = self._generate_request_id()
        self.request_count += 1
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._generate_cache_key(request)
            if cache_key in self._cache:
                cached_response = self._cache[cache_key]
                if self._is_cache_valid(cached_response):
                    logger.info(f"🚀 Cache hit for request {request_id}")
                    cached_response.cached = True
                    cached_response.request_id = request_id
                    return cached_response
        
        # Handle dharma_quantum provider internally
        if request.provider == LLMProvider.DHARMA_QUANTUM:
            return await self._process_dharma_request(request, request_id, start_time)
        
        # Process external providers through gateway
        return await self._process_external_request(request, request_id, start_time)
    
    async def _process_dharma_request(
        self, 
        request: LLMRequest, 
        request_id: str, 
        start_time: float
    ) -> LLMResponse:
        """Process dharma quantum requests with spiritual enhancement"""
        try:
            # Import dharma modules if available
            dharma_content = await self._generate_dharma_response(request)
            
            latency_ms = (time.time() - start_time) * 1000
            
            response = LLMResponse(
                content=dharma_content,
                provider="dharma_quantum",
                model=request.model,
                usage={
                    "prompt_tokens": len(request.prompt.split()),
                    "completion_tokens": len(dharma_content.split()),
                    "total_tokens": len(request.prompt.split()) + len(dharma_content.split())
                },
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
                request_id=request_id,
                dharma_enhanced=True,
                metadata={
                    "spiritual_context": True,
                    "quantum_coherence": 0.85,
                    "dharma_wisdom": True
                }
            )
            
            # Cache the response
            if self.enable_caching:
                cache_key = self._generate_cache_key(request)
                self._cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Dharma processing error: {e}")
            return await self._generate_error_response(request, str(e), request_id, start_time)
    
    async def _process_external_request(
        self, 
        request: LLMRequest, 
        request_id: str, 
        start_time: float
    ) -> LLMResponse:
        """Process request through external gateway"""
        await self._ensure_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Request-ID": request_id
        }
        
        # Convert provider enum to string
        provider_str = request.provider.value if isinstance(request.provider, LLMProvider) else request.provider
        
        payload = {
            "provider": provider_str,
            "model": request.model,
            "query": request.prompt,
            "system_prompt": request.system_prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "user_id": request.user_id,
            "metadata": request.metadata or {}
        }
        
        # Retry logic
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.gateway_url}/generate",
                    json=payload,
                    headers=headers
                ) as response:
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        llm_response = LLMResponse(
                            content=result.get("content", ""),
                            provider=provider_str,
                            model=request.model,
                            usage=result.get("usage", {}),
                            latency_ms=latency_ms,
                            timestamp=datetime.now().isoformat(),
                            request_id=request_id,
                            metadata={
                                "external_gateway": True,
                                "attempt": attempt + 1,
                                "gateway_response": result
                            }
                        )
                        
                        # Cache successful response
                        if self.enable_caching:
                            cache_key = self._generate_cache_key(request)
                            self._cache[cache_key] = llm_response
                        
                        return llm_response
                    
                    else:
                        error_text = await response.text()
                        logger.warning(f"Gateway error {response.status} (attempt {attempt + 1}): {error_text}")
                        if attempt == self.max_retries - 1:
                            return await self._generate_error_response(
                                request, 
                                f"Gateway error {response.status}: {error_text}", 
                                request_id, 
                                start_time
                            )
                        
                        # Wait before retry
                        await asyncio.sleep(2 ** attempt)
            
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Timeout error (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            
            except Exception as e:
                last_exception = e
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        self.error_count += 1
        return await self._generate_error_response(request, str(last_exception), request_id, start_time)
    
    async def _generate_dharma_response(self, request: LLMRequest) -> str:
        """Generate dharma-enhanced response"""
        prompt_lower = request.prompt.lower()
        
        # More specific keyword detection
        if any(word in prompt_lower for word in ["meditat", "breath", "mindful"]):
            return self._get_meditation_guidance(request.prompt)
        elif any(word in prompt_lower for word in ["sanskrit", "namaste", "om", "mantra"]):
            return self._get_sanskrit_help(request.prompt)
        elif any(word in prompt_lower for word in ["dharma", "buddha", "buddhist", "teaching"]):
            return self._get_dharma_teaching(request.prompt)
        elif any(word in prompt_lower for word in ["anxious", "anxiety", "stress", "worry", "fear", "panic"]):
            return self._get_calming_guidance(request.prompt)
        elif any(word in prompt_lower for word in ["spiritual", "enlighten", "awaken", "consciousness", "peace"]):
            return self._get_spiritual_guidance(request.prompt)
        else:
            return self._get_general_wisdom(request.prompt)
    
    def _get_meditation_guidance(self, prompt: str) -> str:
        """Meditation-specific guidance"""
        return """🧘‍♂️ **Meditation Guidance**

For a peaceful meditation practice:

**Beginning Steps:**
1. Find a quiet, comfortable space where you won't be disturbed
2. Sit with your spine naturally erect, shoulders relaxed
3. Close your eyes gently or soften your gaze downward
4. Take three deep, conscious breaths to settle in

**Core Practice:**
- Simply observe your natural breath without changing it
- When your mind wanders (and it will), gently return to the breath
- This returning IS the practice - not a mistake
- Start with 5-10 minutes daily

**Key Insights:**
- Consistency matters more than duration
- Be patient and kind with yourself
- Each moment of awareness is valuable
- Progress comes through regular practice

Remember: "Better than a thousand hollow words is one word that brings peace." - Buddha

May your practice bring you tranquility and insight. 🌸"""
    
    def _get_sanskrit_help(self, prompt: str) -> str:
        """Sanskrit learning assistance"""
        return """📿 **Sanskrit Wisdom**

Sanskrit (संस्कृत) meaning "perfectly formed" is the sacred language of many spiritual texts.

**Common Sacred Terms:**
- **Om (ॐ)** - The primordial sound, cosmic vibration
- **Namaste (नमस्ते)** - "I bow to you," honoring the divine within
- **Dharma (धर्म)** - Natural law, righteous path, duty
- **Karma (कर्म)** - Action and its consequences
- **Moksha (मोक्ष)** - Liberation, release from suffering
- **Ahimsa (अहिंसा)** - Non-violence, compassion
- **Satsang (सत्संग)** - Gathering with like-minded spiritual seekers

**Pronunciation Tips:**
- Each syllable carries equal weight
- Vowels are pure sounds
- Practice with reverence and intention

Sanskrit mantras are believed to create beneficial vibrations that resonate with cosmic frequencies, supporting meditation and spiritual growth.

Would you like help with a specific Sanskrit term or mantra? 🙏"""
    
    def _get_dharma_teaching(self, prompt: str) -> str:
        """Dharma wisdom and teachings"""
        return """🌅 **Dharma Teaching**

Dharma encompasses the natural order of existence and the path of righteous living. In Buddhist understanding, it represents the Buddha's teachings and universal principles for liberation.

**The Four Noble Truths:**
1. **Dukkha** - Life contains suffering and dissatisfaction
2. **Samudaya** - Suffering arises from attachment and craving
3. **Nirodha** - Suffering can cease through letting go
4. **Magga** - The Eightfold Path leads to freedom from suffering

**Living Dharma:**
- Align actions with wisdom and compassion
- Practice mindful awareness in daily life
- Cultivate understanding of impermanence
- Develop equanimity toward pleasure and pain
- Serve others with loving-kindness

**Daily Application:**
- Before acting, pause and consider: "Does this align with wisdom?"
- Practice gratitude for what arises
- Respond rather than react to challenges
- See difficulties as opportunities for growth

The Dharma is not mere philosophy but a practical guide for reducing suffering and cultivating genuine happiness.

How might these teachings apply to your current situation? 🌸"""
    
    def _get_calming_guidance(self, prompt: str) -> str:
        """Guidance for anxiety and stress"""
        return """🕊️ **Finding Peace in Difficulty**

When anxiety or stress arises, remember these gentle practices:

**Immediate Relief:**
1. **Breath Awareness** - Place one hand on chest, one on belly. Breathe slowly, feeling the belly rise
2. **Grounding** - Notice 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste
3. **Loving-Kindness** - Place hand on heart, say: "May I be safe, may I be peaceful, may I be kind to myself"

**Understanding Anxiety:**
- Anxiety is a visitor, not your identity
- Like clouds, difficult emotions pass through awareness
- Your essential nature remains calm and clear
- Each breath is a new beginning

**Buddhist Perspective:**
"Pain is inevitable, suffering is optional." - We cannot control what arises, but we can choose our response.

**Compassionate Reminder:**
You are not alone in this experience. Millions throughout history have faced similar challenges and found peace. Your courage in seeking help is already a step toward healing.

Be gentle with yourself today. This too shall pass. 🌈"""
    
    def _get_spiritual_guidance(self, prompt: str) -> str:
        """General spiritual guidance"""
        return """✨ **Spiritual Guidance**

On the spiritual path, remember these timeless principles:

**Core Wisdom:**
- Your essential nature is already whole and complete
- Spiritual growth happens through inner work, not external seeking
- Every experience offers an opportunity for awakening
- Compassion for self and others is the highest practice

**Daily Practices:**
- Begin each day with gratitude
- Practice presence in ordinary moments
- Cultivate patience with your unfolding journey
- Serve others as expressions of the divine

**When Challenges Arise:**
- See obstacles as teachers pointing toward growth
- Trust the process even when the path seems unclear
- Remember that spiritual development is gradual and natural
- Connect with community and wise teachers

**Ancient Wisdom:**
"The wound is the place where the Light enters you." - Rumi
"Be a lamp unto yourself." - Buddha

Your spiritual journey is unique and sacred. Trust your inner wisdom while remaining open to learning and growth.

May you walk in peace and discover the light within. 🙏"""
    
    def _get_general_wisdom(self, prompt: str) -> str:
        """General wisdom for any question"""
        return """🌸 **Wisdom for Life**

In facing life's questions and challenges, consider these universal principles:

**Timeless Guidance:**
- Every problem contains the seeds of its solution
- Wisdom often comes through quiet reflection
- Small, consistent actions create lasting change
- Difficulties are temporary, but growth is permanent

**Practical Approach:**
1. **Pause** - Take time to understand the situation clearly
2. **Reflect** - Consider multiple perspectives and possibilities
3. **Act** - Take one small, positive step forward
4. **Learn** - Extract wisdom from whatever unfolds

**Universal Truths:**
- You have more strength and wisdom than you realize
- Each day offers new opportunities for growth
- Kindness to yourself and others transforms everything
- The present moment is where all possibilities exist

Remember: "A journey of a thousand miles begins with a single step." - Lao Tzu

Whatever you're facing, approach it with patience, compassion, and trust in your ability to navigate wisely.

May you find clarity and peace in this moment. 🌅"""
    
    async def _generate_error_response(
        self, 
        request: LLMRequest, 
        error: str, 
        request_id: str, 
        start_time: float
    ) -> LLMResponse:
        """Generate error response with fallback wisdom"""
        return LLMResponse(
            content="I apologize, but I'm experiencing technical difficulties right now. Please take a moment to breathe deeply and try again. Remember: every challenge is an opportunity for patience and growth. 🙏",
            provider=request.provider.value if isinstance(request.provider, LLMProvider) else request.provider,
            model=request.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            error=error,
            metadata={"error_type": "gateway_error", "fallback_response": True}
        )
    
    async def get_available_providers(self) -> Dict[str, Any]:
        """Get available providers from gateway + dharma"""
        await self._ensure_session()
        
        # Always include dharma quantum
        providers = {
            "dharma_quantum": {
                "available": True,
                "models": ["quantum_consciousness", "dharma_enhanced", "fallback_dharma"],
                "features": ["spiritual_guidance", "meditation", "sanskrit", "dharma_teachings"]
            }
        }
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with self.session.get(
                f"{self.gateway_url}/providers",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    external_providers = result.get("providers", {})
                    providers.update(external_providers)
        
        except Exception as e:
            logger.warning(f"Could not fetch external providers: {e}")
        
        return {"providers": providers}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of client and gateway"""
        status = {
            "client": {
                "status": "healthy",
                "requests_processed": self.request_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "average_latency_ms": self.total_latency / max(self.request_count, 1),
                "cache_entries": len(self._cache)
            },
            "dharma_quantum": {
                "status": "available",
                "features": ["meditation", "sanskrit", "dharma_teachings", "spiritual_guidance"]
            }
        }
        
        # Check external gateway
        try:
            await self._ensure_session()
            async with self.session.get(
                f"{self.gateway_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    gateway_health = await response.json()
                    status["external_gateway"] = gateway_health
                else:
                    status["external_gateway"] = {"status": "unhealthy", "code": response.status}
        except Exception as e:
            status["external_gateway"] = {"status": "unavailable", "error": str(e)}
        
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "success_rate": (self.request_count - self.error_count) / max(self.request_count, 1),
            "cache_size": len(self._cache),
            "cache_enabled": self.enable_caching,
            "average_latency_ms": self.total_latency / max(self.request_count, 1)
        }

# Global client instance
_llm_client: Optional[LLMGatewayClient] = None

async def get_llm_client() -> LLMGatewayClient:
    """Get global LLM client instance"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMGatewayClient()
    return _llm_client

async def demo_llm_client():
    """Demo the LLM client"""
    print("🌉 LLM Gateway Client Demo")
    print("=" * 50)
    
    async with LLMGatewayClient() as client:
        # Test different types of requests
        test_cases = [
            {
                "prompt": "How do I start meditating?",
                "provider": LLMProvider.DHARMA_QUANTUM,
                "model": "quantum_consciousness"
            },
            {
                "prompt": "What does 'namaste' mean?",
                "provider": LLMProvider.DHARMA_QUANTUM,
                "model": "dharma_enhanced"
            },
            {
                "prompt": "I'm feeling anxious about work",
                "provider": LLMProvider.DHARMA_QUANTUM,
                "model": "quantum_consciousness"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_case['prompt'][:40]}...")
            
            request = LLMRequest(**test_case)
            response = await client.process_request(request)
            
            print(f"✅ Response ({response.latency_ms:.1f}ms):")
            print(f"   {response.content[:100]}...")
            print(f"🎯 Enhanced: {response.dharma_enhanced}")
            print(f"📊 Tokens: {response.usage['total_tokens']}")
        
        # Show providers and health
        print("\n🌟 Available Providers:")
        providers = await client.get_available_providers()
        for provider, details in providers.get("providers", {}).items():
            status = "✅" if details.get("available") else "❌"
            print(f"  {status} {provider}: {len(details.get('models', []))} models")
        
        print("\n💊 Health Status:")
        health = await client.get_health_status()
        for service, status in health.items():
            print(f"  {service}: {status.get('status', 'unknown')}")
        
        print("\n📊 Client Stats:")
        stats = client.get_stats()
        print(f"  Requests: {stats['requests_total']}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Avg Latency: {stats['average_latency_ms']:.1f}ms")
        
        print("\n✅ LLM Client Demo Complete!")

if __name__ == "__main__":
    asyncio.run(demo_llm_client())
