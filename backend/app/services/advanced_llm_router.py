"""
🚀 Advanced LLM Router
=====================

Advanced routing system for multiple LLM providers with intelligent load balancing,
failover, and performance optimization.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum
import random
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Available LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class RoutingStrategy(str, Enum):
    """LLM routing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    PERFORMANCE_BASED = "performance_based"
    COST_OPTIMIZED = "cost_optimized"
    FAILOVER = "failover"

class AdvancedLLMRouter:
    """🚀 Advanced LLM Router with intelligent routing"""
    
    def __init__(self, redis_client=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.redis_client = redis_client
        
        # Provider configurations
        self.providers = {
            LLMProvider.OPENAI: {
                "weight": 0.4,
                "cost_per_token": 0.002,
                "avg_response_time": 1.2,
                "reliability": 0.99,
                "enabled": True
            },
            LLMProvider.ANTHROPIC: {
                "weight": 0.3,
                "cost_per_token": 0.0015,
                "avg_response_time": 1.0,
                "reliability": 0.98,
                "enabled": True
            },
            LLMProvider.GOOGLE: {
                "weight": 0.2,
                "cost_per_token": 0.001,
                "avg_response_time": 1.5,
                "reliability": 0.97,
                "enabled": True
            },
            LLMProvider.LOCAL: {
                "weight": 0.1,
                "cost_per_token": 0.0,
                "avg_response_time": 2.0,
                "reliability": 0.95,
                "enabled": False  # Disabled by default
            }
        }
        
        # Routing state
        self.current_index = 0
        self.performance_metrics = {}
        self.circuit_breakers = {}
        
        self.logger.info("🚀 Advanced LLM Router initialized")
    
    async def route_request(
        self, 
        prompt: str, 
        strategy: RoutingStrategy = RoutingStrategy.PERFORMANCE_BASED,
        preferred_provider: Optional[LLMProvider] = None
    ) -> LLMProvider:
        """Route request to optimal LLM provider"""
        
        if preferred_provider and self.providers[preferred_provider]["enabled"]:
            return preferred_provider
        
        available_providers = [
            provider for provider, config in self.providers.items() 
            if config["enabled"] and not self._is_circuit_broken(provider)
        ]
        
        if not available_providers:
            self.logger.warning("No available LLM providers, falling back to OpenAI")
            return LLMProvider.OPENAI
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_routing(available_providers)
        elif strategy == RoutingStrategy.WEIGHTED:
            return self._weighted_routing(available_providers)
        elif strategy == RoutingStrategy.PERFORMANCE_BASED:
            return self._performance_based_routing(available_providers)
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_routing(available_providers)
        else:
            return available_providers[0]
    
    def _round_robin_routing(self, providers: List[LLMProvider]) -> LLMProvider:
        """Round-robin routing strategy"""
        provider = providers[self.current_index % len(providers)]
        self.current_index += 1
        return provider
    
    def _weighted_routing(self, providers: List[LLMProvider]) -> LLMProvider:
        """Weighted random routing based on provider weights"""
        weights = [self.providers[p]["weight"] for p in providers]
        return random.choices(providers, weights=weights)[0]
    
    def _performance_based_routing(self, providers: List[LLMProvider]) -> LLMProvider:
        """Route based on performance metrics"""
        scored_providers = []
        
        for provider in providers:
            config = self.providers[provider]
            metrics = self.performance_metrics.get(provider, {})
            
            # Calculate performance score
            response_time_score = 1.0 / max(config["avg_response_time"], 0.1)
            reliability_score = config["reliability"]
            recent_success_rate = metrics.get("success_rate", 1.0)
            
            total_score = (response_time_score * 0.4 + 
                          reliability_score * 0.4 + 
                          recent_success_rate * 0.2)
            
            scored_providers.append((provider, total_score))
        
        # Return provider with highest score
        return max(scored_providers, key=lambda x: x[1])[0]
    
    def _cost_optimized_routing(self, providers: List[LLMProvider]) -> LLMProvider:
        """Route to lowest cost provider"""
        return min(providers, key=lambda p: self.providers[p]["cost_per_token"])
    
    def _is_circuit_broken(self, provider: LLMProvider) -> bool:
        """Check if circuit breaker is open for provider"""
        breaker = self.circuit_breakers.get(provider)
        if not breaker:
            return False
        
        # Circuit breaker logic
        if breaker["state"] == "open":
            if datetime.now() > breaker["next_attempt"]:
                breaker["state"] = "half_open"
                return False
            return True
        
        return False
    
    async def record_response_metrics(
        self, 
        provider: LLMProvider, 
        response_time: float, 
        success: bool,
        token_count: Optional[int] = None
    ):
        """Record response metrics for provider"""
        
        if provider not in self.performance_metrics:
            self.performance_metrics[provider] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_response_time": 0.0,
                "success_rate": 1.0,
                "avg_response_time": 0.0
            }
        
        metrics = self.performance_metrics[provider]
        metrics["total_requests"] += 1
        metrics["total_response_time"] += response_time
        
        if success:
            metrics["successful_requests"] += 1
        
        # Update derived metrics
        metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
        metrics["avg_response_time"] = metrics["total_response_time"] / metrics["total_requests"]
        
        # Update provider configuration
        self.providers[provider]["avg_response_time"] = metrics["avg_response_time"]
        
        # Circuit breaker logic
        if not success:
            self._handle_failure(provider)
        else:
            self._handle_success(provider)
    
    def _handle_failure(self, provider: LLMProvider):
        """Handle provider failure for circuit breaker"""
        if provider not in self.circuit_breakers:
            self.circuit_breakers[provider] = {
                "failures": 0,
                "state": "closed",
                "next_attempt": None
            }
        
        breaker = self.circuit_breakers[provider]
        breaker["failures"] += 1
        
        # Open circuit breaker after 3 failures
        if breaker["failures"] >= 3 and breaker["state"] == "closed":
            breaker["state"] = "open"
            breaker["next_attempt"] = datetime.now() + timedelta(minutes=5)
            self.logger.warning(f"Circuit breaker opened for {provider}")
    
    def _handle_success(self, provider: LLMProvider):
        """Handle provider success for circuit breaker"""
        if provider in self.circuit_breakers:
            breaker = self.circuit_breakers[provider]
            if breaker["state"] == "half_open":
                breaker["state"] = "closed"
                breaker["failures"] = 0
                self.logger.info(f"Circuit breaker closed for {provider}")
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {}
        
        for provider, config in self.providers.items():
            metrics = self.performance_metrics.get(provider, {})
            breaker = self.circuit_breakers.get(provider, {})
            
            status[provider.value] = {
                "enabled": config["enabled"],
                "weight": config["weight"],
                "avg_response_time": config["avg_response_time"],
                "reliability": config["reliability"],
                "circuit_breaker_state": breaker.get("state", "closed"),
                "total_requests": metrics.get("total_requests", 0),
                "success_rate": metrics.get("success_rate", 1.0)
            }
        
        return status

# Global router instance
_advanced_router: Optional[AdvancedLLMRouter] = None

def init_advanced_llm_router(redis_client=None) -> AdvancedLLMRouter:
    """Initialize advanced LLM router"""
    global _advanced_router
    if _advanced_router is None:
        _advanced_router = AdvancedLLMRouter(redis_client)
    return _advanced_router

def get_advanced_llm_router() -> AdvancedLLMRouter:
    """Get advanced LLM router instance"""
    global _advanced_router
    if _advanced_router is None:
        _advanced_router = AdvancedLLMRouter()
    return _advanced_router

# Export commonly used classes and functions
__all__ = [
    'AdvancedLLMRouter',
    'LLMProvider',
    'RoutingStrategy',
    'init_advanced_llm_router',
    'get_advanced_llm_router'
]
