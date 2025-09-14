"""
🤖 Advanced Multi-Model LLM Router

Intelligent routing system with fallbacks, load balancing, and performance optimization
for enterprise-grade AI response generation.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
import hashlib
from collections import defaultdict, deque
import redis
from pydantic import BaseModel
import httpx

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    GOOGLE_GEMINI = "google_gemini"
    LOCAL_LLAMA = "local_llama"
    DHARMA_SPECIALIZED = "dharma_specialized"

class ResponseQuality(str, Enum):
    """Response quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

@dataclass
class LLMEndpoint:
    """LLM endpoint configuration"""
    provider: LLMProvider
    base_url: str
    api_key: Optional[str]
    model_name: str
    max_tokens: int
    temperature: float
    priority: int  # 1 = highest, 10 = lowest
    cost_per_token: float
    average_latency: float = 0.0
    success_rate: float = 1.0
    current_load: int = 0
    max_concurrent: int = 10
    enabled: bool = True

@dataclass
class RoutingRequest:
    """Request for LLM routing"""
    prompt: str
    user_id: str
    session_id: str
    context_type: str  # chat, spiritual, technical, etc.
    quality_requirement: str  # fast, balanced, high_quality
    max_cost: Optional[float] = None
    timeout: int = 30
    metadata: Dict[str, Any] = None

@dataclass
class RoutingResponse:
    """Response from LLM routing"""
    response_text: str
    provider_used: LLMProvider
    tokens_used: int
    cost: float
    latency: float
    quality_score: float
    cached: bool = False
    fallback_used: bool = False

class ResponseCache:
    """Intelligent response caching system"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.cache_prefix = "llm_cache:"
        self.default_ttl = 3600  # 1 hour
        
    def _generate_cache_key(self, request: RoutingRequest) -> str:
        """Generate cache key for request"""
        cache_data = {
            "prompt": request.prompt,
            "context_type": request.context_type,
            "quality_requirement": request.quality_requirement
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.sha256(cache_string.encode()).hexdigest()
        return f"{self.cache_prefix}{cache_hash}"
    
    async def get_cached_response(self, request: RoutingRequest) -> Optional[RoutingResponse]:
        """Get cached response if available"""
        try:
            cache_key = self._generate_cache_key(request)
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                response_data = json.loads(cached_data)
                response = RoutingResponse(**response_data)
                response.cached = True
                logger.info(f"Cache hit for request: {cache_key[:16]}...")
                return response
            
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    async def cache_response(self, request: RoutingRequest, response: RoutingResponse, ttl: int = None):
        """Cache response for future use"""
        try:
            cache_key = self._generate_cache_key(request)
            cache_ttl = ttl or self.default_ttl
            
            # Don't cache poor quality responses
            if response.quality_score < 0.6:
                return
            
            response_data = asdict(response)
            response_data["cached"] = False  # Reset for storage
            
            self.redis.setex(cache_key, cache_ttl, json.dumps(response_data))
            logger.info(f"Response cached: {cache_key[:16]}...")
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

class LoadBalancer:
    """Intelligent load balancer for LLM endpoints"""
    
    def __init__(self):
        self.endpoint_stats = defaultdict(lambda: {
            "request_count": 0,
            "success_count": 0,
            "total_latency": 0.0,
            "error_count": 0,
            "last_error": None
        })
        
    def select_best_endpoint(
        self, 
        endpoints: List[LLMEndpoint], 
        request: RoutingRequest
    ) -> Optional[LLMEndpoint]:
        """Select the best endpoint based on multiple factors"""
        
        available_endpoints = [
            ep for ep in endpoints 
            if ep.enabled and ep.current_load < ep.max_concurrent
        ]
        
        if not available_endpoints:
            return None
        
        # Score each endpoint
        scored_endpoints = []
        for endpoint in available_endpoints:
            score = self._calculate_endpoint_score(endpoint, request)
            scored_endpoints.append((score, endpoint))
        
        # Sort by score (higher is better)
        scored_endpoints.sort(key=lambda x: x[0], reverse=True)
        
        return scored_endpoints[0][1] if scored_endpoints else None
    
    def _calculate_endpoint_score(self, endpoint: LLMEndpoint, request: RoutingRequest) -> float:
        """Calculate endpoint suitability score"""
        stats = self.endpoint_stats[endpoint.provider.value]
        
        # Base score from priority (inverted - lower priority number = higher score)
        priority_score = (11 - endpoint.priority) / 10.0
        
        # Success rate score
        success_rate = (stats["success_count"] / max(stats["request_count"], 1))
        success_score = success_rate * 0.3
        
        # Latency score (lower latency = higher score)
        avg_latency = stats["total_latency"] / max(stats["success_count"], 1)
        latency_score = max(0, (5.0 - avg_latency) / 5.0) * 0.3
        
        # Load score (lower load = higher score)
        load_ratio = endpoint.current_load / endpoint.max_concurrent
        load_score = (1.0 - load_ratio) * 0.2
        
        # Cost score (if budget specified)
        cost_score = 0.1
        if request.max_cost:
            estimated_cost = endpoint.cost_per_token * 1000  # Estimate for 1k tokens
            if estimated_cost <= request.max_cost:
                cost_score = (request.max_cost - estimated_cost) / request.max_cost * 0.1
            else:
                cost_score = 0
        
        total_score = priority_score + success_score + latency_score + load_score + cost_score
        
        logger.debug(f"Endpoint {endpoint.provider.value} score: {total_score:.3f}")
        return total_score
    
    def update_endpoint_stats(self, endpoint: LLMEndpoint, latency: float, success: bool, error: str = None):
        """Update endpoint performance statistics"""
        stats = self.endpoint_stats[endpoint.provider.value]
        
        stats["request_count"] += 1
        if success:
            stats["success_count"] += 1
            stats["total_latency"] += latency
        else:
            stats["error_count"] += 1
            stats["last_error"] = error or "Unknown error"

class QualityScorer:
    """AI response quality assessment"""
    
    def __init__(self):
        self.quality_metrics = {
            "coherence": 0.3,
            "relevance": 0.3,
            "completeness": 0.2,
            "accuracy": 0.2
        }
    
    def score_response(self, request: RoutingRequest, response: str, metadata: Dict[str, Any] = None) -> float:
        """Score response quality (0.0 to 1.0)"""
        
        # Basic quality metrics
        scores = {
            "coherence": self._score_coherence(response),
            "relevance": self._score_relevance(request.prompt, response),
            "completeness": self._score_completeness(request.prompt, response),
            "accuracy": self._score_accuracy(response, request.context_type)
        }
        
        # Weighted average
        total_score = sum(
            scores[metric] * weight 
            for metric, weight in self.quality_metrics.items()
        )
        
        return min(1.0, max(0.0, total_score))
    
    def _score_coherence(self, response: str) -> float:
        """Score response coherence"""
        if not response or len(response.strip()) < 10:
            return 0.0
        
        # Basic coherence indicators
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.6
        
        # Check for repeated words/phrases
        words = response.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        
        return min(1.0, unique_ratio * 1.2)
    
    def _score_relevance(self, prompt: str, response: str) -> float:
        """Score response relevance to prompt"""
        if not prompt or not response:
            return 0.0
        
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Simple keyword overlap
        overlap = len(prompt_words.intersection(response_words))
        relevance = overlap / len(prompt_words) if prompt_words else 0
        
        return min(1.0, relevance * 2.0)
    
    def _score_completeness(self, prompt: str, response: str) -> float:
        """Score response completeness"""
        if not response:
            return 0.0
        
        # Basic length and structure checks
        response_length = len(response.strip())
        
        if response_length < 50:
            return 0.3
        elif response_length < 200:
            return 0.7
        else:
            return 1.0
    
    def _score_accuracy(self, response: str, context_type: str) -> float:
        """Score response accuracy for context"""
        # Context-specific accuracy checks
        if context_type == "spiritual":
            # Check for appropriate spiritual concepts
            spiritual_terms = ["dharma", "wisdom", "compassion", "meditation", "mindfulness"]
            term_count = sum(1 for term in spiritual_terms if term in response.lower())
            return min(1.0, term_count / 3.0)
        elif context_type == "technical":
            # Check for technical coherence
            return 0.8  # Placeholder
        else:
            return 0.7  # Default

class AdvancedLLMRouter:
    """Advanced multi-model LLM router with intelligent routing"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.cache = ResponseCache(redis_client)
        self.load_balancer = LoadBalancer()
        self.quality_scorer = QualityScorer()
        self.endpoints = self._initialize_endpoints()
        self.fallback_chain = self._create_fallback_chain()
        
    def _initialize_endpoints(self) -> List[LLMEndpoint]:
        """Initialize LLM endpoints configuration"""
        return [
            LLMEndpoint(
                provider=LLMProvider.OPENAI_GPT4,
                base_url="https://api.openai.com/v1",
                api_key=None,  # Set from environment
                model_name="gpt-4",
                max_tokens=4000,
                temperature=0.7,
                priority=1,
                cost_per_token=0.00003,
                max_concurrent=5
            ),
            LLMEndpoint(
                provider=LLMProvider.ANTHROPIC_CLAUDE,
                base_url="https://api.anthropic.com",
                api_key=None,  # Set from environment
                model_name="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                priority=2,
                cost_per_token=0.000015,
                max_concurrent=5
            ),
            LLMEndpoint(
                provider=LLMProvider.OPENAI_GPT35,
                base_url="https://api.openai.com/v1",
                api_key=None,  # Set from environment
                model_name="gpt-3.5-turbo",
                max_tokens=4000,
                temperature=0.7,
                priority=3,
                cost_per_token=0.000001,
                max_concurrent=10
            ),
            LLMEndpoint(
                provider=LLMProvider.LOCAL_LLAMA,
                base_url="http://localhost:11434",
                api_key=None,
                model_name="llama2",
                max_tokens=2000,
                temperature=0.7,
                priority=4,
                cost_per_token=0.0,  # Local model
                max_concurrent=3
            )
        ]
    
    def _create_fallback_chain(self) -> Dict[str, List[LLMProvider]]:
        """Create fallback chains for different scenarios"""
        return {
            "high_quality": [
                LLMProvider.OPENAI_GPT4,
                LLMProvider.ANTHROPIC_CLAUDE,
                LLMProvider.OPENAI_GPT35
            ],
            "balanced": [
                LLMProvider.ANTHROPIC_CLAUDE,
                LLMProvider.OPENAI_GPT35,
                LLMProvider.LOCAL_LLAMA
            ],
            "fast": [
                LLMProvider.OPENAI_GPT35,
                LLMProvider.LOCAL_LLAMA,
                LLMProvider.ANTHROPIC_CLAUDE
            ]
        }
    
    async def route_request(self, request: RoutingRequest) -> RoutingResponse:
        """Route request to best available LLM with fallbacks"""
        
        # 1. Check cache first
        cached_response = await self.cache.get_cached_response(request)
        if cached_response:
            return cached_response
        
        # 2. Select fallback chain based on quality requirement
        fallback_providers = self.fallback_chain.get(
            request.quality_requirement, 
            self.fallback_chain["balanced"]
        )
        
        # 3. Try each provider in fallback chain
        last_error = None
        for provider in fallback_providers:
            try:
                endpoint = self._get_endpoint_by_provider(provider)
                if not endpoint or not endpoint.enabled:
                    continue
                
                # Check if endpoint is available
                if endpoint.current_load >= endpoint.max_concurrent:
                    logger.warning(f"Endpoint {provider.value} at capacity")
                    continue
                
                # Attempt request
                start_time = time.time()
                endpoint.current_load += 1
                
                try:
                    response = await self._make_llm_request(endpoint, request)
                    latency = time.time() - start_time
                    
                    # Score response quality
                    quality_score = self.quality_scorer.score_response(request, response)
                    
                    # Create routing response
                    routing_response = RoutingResponse(
                        response_text=response,
                        provider_used=provider,
                        tokens_used=len(response.split()) * 1.3,  # Rough estimate
                        cost=endpoint.cost_per_token * len(response.split()) * 1.3,
                        latency=latency,
                        quality_score=quality_score,
                        fallback_used=(provider != fallback_providers[0])
                    )
                    
                    # Update stats
                    self.load_balancer.update_endpoint_stats(endpoint, latency, True)
                    
                    # Cache successful response
                    await self.cache.cache_response(request, routing_response)
                    
                    logger.info(f"Successful response from {provider.value} (quality: {quality_score:.2f})")
                    return routing_response
                    
                except Exception as e:
                    last_error = str(e)
                    self.load_balancer.update_endpoint_stats(endpoint, 0, False, str(e))
                    logger.error(f"Request failed for {provider.value}: {e}")
                    
                finally:
                    endpoint.current_load = max(0, endpoint.current_load - 1)
                    
            except Exception as e:
                logger.error(f"Endpoint selection error for {provider.value}: {e}")
                continue
        
        # 4. All providers failed
        raise Exception(f"All LLM providers failed. Last error: {last_error}")
    
    def _get_endpoint_by_provider(self, provider: LLMProvider) -> Optional[LLMEndpoint]:
        """Get endpoint configuration by provider"""
        for endpoint in self.endpoints:
            if endpoint.provider == provider:
                return endpoint
        return None
    
    async def _make_llm_request(self, endpoint: LLMEndpoint, request: RoutingRequest) -> str:
        """Make actual LLM API request"""
        
        if endpoint.provider in [LLMProvider.OPENAI_GPT4, LLMProvider.OPENAI_GPT35]:
            return await self._make_openai_request(endpoint, request)
        elif endpoint.provider == LLMProvider.ANTHROPIC_CLAUDE:
            return await self._make_anthropic_request(endpoint, request)
        elif endpoint.provider == LLMProvider.LOCAL_LLAMA:
            return await self._make_local_request(endpoint, request)
        else:
            raise ValueError(f"Unsupported provider: {endpoint.provider}")
    
    async def _make_openai_request(self, endpoint: LLMEndpoint, request: RoutingRequest) -> str:
        """Make OpenAI API request"""
        
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": endpoint.max_tokens,
            "temperature": endpoint.temperature
        }
        
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            response = await client.post(
                f"{endpoint.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    async def _make_anthropic_request(self, endpoint: LLMEndpoint, request: RoutingRequest) -> str:
        """Make Anthropic Claude API request"""
        
        headers = {
            "x-api-key": endpoint.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": endpoint.model_name,
            "max_tokens": endpoint.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}]
        }
        
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            response = await client.post(
                f"{endpoint.base_url}/v1/messages",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data["content"][0]["text"]
    
    async def _make_local_request(self, endpoint: LLMEndpoint, request: RoutingRequest) -> str:
        """Make local Ollama API request"""
        
        payload = {
            "model": endpoint.model_name,
            "prompt": request.prompt,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            response = await client.post(
                f"{endpoint.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data["response"]
    
    async def get_router_stats(self) -> Dict[str, Any]:
        """Get router performance statistics"""
        
        stats = {
            "endpoints": [],
            "total_requests": 0,
            "cache_hit_rate": 0.0,
            "average_latency": 0.0,
            "cost_savings": 0.0
        }
        
        for endpoint in self.endpoints:
            endpoint_stats = self.load_balancer.endpoint_stats[endpoint.provider.value]
            stats["endpoints"].append({
                "provider": endpoint.provider.value,
                "enabled": endpoint.enabled,
                "current_load": endpoint.current_load,
                "requests": endpoint_stats["request_count"],
                "success_rate": endpoint_stats["success_count"] / max(endpoint_stats["request_count"], 1),
                "average_latency": endpoint_stats["total_latency"] / max(endpoint_stats["success_count"], 1),
                "errors": endpoint_stats["error_count"]
            })
            
            stats["total_requests"] += endpoint_stats["request_count"]
        
        return stats

# Global router instance
advanced_llm_router: Optional[AdvancedLLMRouter] = None

def get_advanced_llm_router() -> AdvancedLLMRouter:
    """Get the global advanced LLM router instance"""
    if advanced_llm_router is None:
        raise RuntimeError("Advanced LLM router not initialized")
    return advanced_llm_router

def init_advanced_llm_router(redis_client: redis.Redis) -> AdvancedLLMRouter:
    """Initialize the global advanced LLM router"""
    global advanced_llm_router
    advanced_llm_router = AdvancedLLMRouter(redis_client)
    return advanced_llm_router
