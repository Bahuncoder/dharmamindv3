"""
Advanced LLM Router for DharmaMind platform

Intelligent routing of requests to optimal LLM providers based on content,
user preferences, and system performance.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from .llm_gateway_client import LLMProvider, LLMModel, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

class RoutingStrategy(str, Enum):
    """LLM routing strategies"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COST = "cost"
    DHARMIC_ALIGNMENT = "dharmic_alignment"
    LOAD_BALANCED = "load_balanced"
    FAILOVER = "failover"

class RequestType(str, Enum):
    """Types of requests for routing decisions"""
    SPIRITUAL_GUIDANCE = "spiritual_guidance"
    MEDITATION_INSTRUCTION = "meditation_instruction"
    PHILOSOPHICAL_INQUIRY = "philosophical_inquiry"
    GENERAL_CHAT = "general_chat"
    KNOWLEDGE_QUERY = "knowledge_query"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"

class LLMCapability(BaseModel):
    """Capabilities of an LLM provider/model"""
    provider: LLMProvider = Field(..., description="LLM provider")
    model: LLMModel = Field(..., description="LLM model")
    
    # Performance metrics
    avg_response_time_ms: float = Field(default=1000.0, description="Average response time")
    success_rate: float = Field(default=0.95, ge=0.0, le=1.0, description="Success rate")
    uptime_percentage: float = Field(default=99.0, ge=0.0, le=100.0, description="Uptime percentage")
    
    # Quality metrics
    dharmic_alignment_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Dharmic alignment")
    creativity_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Creativity capability")
    knowledge_depth: float = Field(default=0.8, ge=0.0, le=1.0, description="Knowledge depth")
    
    # Cost and limits
    cost_per_1k_tokens: float = Field(default=0.002, description="Cost per 1k tokens")
    max_tokens: int = Field(default=4096, description="Maximum tokens")
    rate_limit_rpm: int = Field(default=60, description="Rate limit requests per minute")
    
    # Specializations
    strong_areas: List[RequestType] = Field(default_factory=list, description="Strong capability areas")
    weak_areas: List[RequestType] = Field(default_factory=list, description="Weak capability areas")

class RoutingDecision(BaseModel):
    """Decision made by the router"""
    chosen_provider: LLMProvider = Field(..., description="Selected provider")
    chosen_model: LLMModel = Field(..., description="Selected model")
    reasoning: str = Field(..., description="Reasoning for the choice")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in decision")
    
    # Alternative options
    fallback_options: List[Dict[str, str]] = Field(default_factory=list, description="Fallback options")
    
    # Metadata
    routing_strategy: RoutingStrategy = Field(..., description="Strategy used")
    decision_time_ms: float = Field(..., description="Time taken to make decision")
    timestamp: datetime = Field(default_factory=datetime.now, description="Decision timestamp")

class AdvancedLLMRouter:
    """Advanced router for LLM requests"""
    
    def __init__(self):
        self.provider_capabilities = self._initialize_capabilities()
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.routing_history: List[RoutingDecision] = []
        
    def _initialize_capabilities(self) -> Dict[str, LLMCapability]:
        """Initialize provider capabilities"""
        return {
            "openai_gpt4": LLMCapability(
                provider=LLMProvider.OPENAI,
                model=LLMModel.GPT_4,
                avg_response_time_ms=2000.0,
                success_rate=0.98,
                uptime_percentage=99.5,
                dharmic_alignment_score=0.85,
                creativity_score=0.9,
                knowledge_depth=0.95,
                cost_per_1k_tokens=0.03,
                max_tokens=8192,
                strong_areas=[RequestType.PHILOSOPHICAL_INQUIRY, RequestType.KNOWLEDGE_QUERY, RequestType.ANALYSIS],
                weak_areas=[]
            ),
            "openai_gpt35": LLMCapability(
                provider=LLMProvider.OPENAI,
                model=LLMModel.GPT_3_5,
                avg_response_time_ms=800.0,
                success_rate=0.96,
                uptime_percentage=99.0,
                dharmic_alignment_score=0.8,
                creativity_score=0.8,
                knowledge_depth=0.85,
                cost_per_1k_tokens=0.002,
                max_tokens=4096,
                strong_areas=[RequestType.GENERAL_CHAT, RequestType.MEDITATION_INSTRUCTION],
                weak_areas=[RequestType.CREATIVE_WRITING]
            ),
            "anthropic_claude3": LLMCapability(
                provider=LLMProvider.ANTHROPIC,
                model=LLMModel.CLAUDE_3,
                avg_response_time_ms=1500.0,
                success_rate=0.97,
                uptime_percentage=98.5,
                dharmic_alignment_score=0.9,
                creativity_score=0.85,
                knowledge_depth=0.9,
                cost_per_1k_tokens=0.015,
                max_tokens=8192,
                strong_areas=[RequestType.SPIRITUAL_GUIDANCE, RequestType.PHILOSOPHICAL_INQUIRY],
                weak_areas=[]
            )
        }
    
    async def initialize(self) -> bool:
        """Initialize the advanced LLM router"""
        try:
            logger.info("Advanced LLM Router initialized with {} providers".format(
                len(self.provider_capabilities)
            ))
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Advanced LLM Router: {e}")
            return False
    
    async def route_request(
        self,
        request: LLMRequest,
        request_type: RequestType = RequestType.GENERAL_CHAT,
        strategy: RoutingStrategy = RoutingStrategy.DHARMIC_ALIGNMENT,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """Route a request to the optimal LLM provider"""
        try:
            start_time = datetime.now()
            
            # Analyze request for routing hints
            request_analysis = await self._analyze_request(request, request_type)
            
            # Score all available providers
            provider_scores = await self._score_providers(
                request_analysis,
                strategy,
                user_preferences
            )
            
            # Select best provider
            best_provider = max(provider_scores.items(), key=lambda x: x[1]["total_score"])
            provider_key, scores = best_provider
            
            capability = self.provider_capabilities[provider_key]
            
            # Generate fallback options
            fallback_options = []
            sorted_providers = sorted(
                provider_scores.items(),
                key=lambda x: x[1]["total_score"],
                reverse=True
            )[1:3]  # Get top 2 alternatives
            
            for provider_key, score_data in sorted_providers:
                cap = self.provider_capabilities[provider_key]
                fallback_options.append({
                    "provider": cap.provider.value,
                    "model": cap.model.value,
                    "score": score_data["total_score"]
                })
            
            # Create routing decision
            end_time = datetime.now()
            decision_time = (end_time - start_time).total_seconds() * 1000
            
            decision = RoutingDecision(
                chosen_provider=capability.provider,
                chosen_model=capability.model,
                reasoning=scores.get("reasoning", "Best overall match for request"),
                confidence=scores.get("confidence", 0.8),
                fallback_options=fallback_options,
                routing_strategy=strategy,
                decision_time_ms=decision_time
            )
            
            # Record decision
            self.routing_history.append(decision)
            
            # Limit history size
            if len(self.routing_history) > 1000:
                self.routing_history = self.routing_history[-500:]
            
            return decision
            
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            # Return default routing
            return RoutingDecision(
                chosen_provider=LLMProvider.OPENAI,
                chosen_model=LLMModel.GPT_3_5,
                reasoning="Default fallback due to routing error",
                confidence=0.5,
                routing_strategy=strategy,
                decision_time_ms=10.0
            )
    
    async def _analyze_request(
        self,
        request: LLMRequest,
        request_type: RequestType
    ) -> Dict[str, Any]:
        """Analyze request for routing hints"""
        analysis = {
            "request_type": request_type,
            "estimated_complexity": "medium",
            "requires_dharmic_alignment": request.dharmic_mode,
            "message_count": len(request.messages),
            "estimated_tokens": sum(len(msg.get("content", "").split()) for msg in request.messages) * 1.3,
            "creativity_needed": False,
            "knowledge_intensive": False
        }
        
        # Analyze content for complexity and requirements
        content = " ".join(msg.get("content", "") for msg in request.messages).lower()
        
        # Check for philosophical/spiritual content
        spiritual_keywords = ["dharma", "meditation", "enlightenment", "consciousness", "soul", "spirit"]
        if any(keyword in content for keyword in spiritual_keywords):
            analysis["requires_dharmic_alignment"] = True
            analysis["estimated_complexity"] = "high"
        
        # Check for creative requests
        creative_keywords = ["create", "write", "compose", "imagine", "story"]
        if any(keyword in content for keyword in creative_keywords):
            analysis["creativity_needed"] = True
        
        # Check for knowledge-intensive requests
        knowledge_keywords = ["explain", "what is", "how does", "define", "analyze"]
        if any(keyword in content for keyword in knowledge_keywords):
            analysis["knowledge_intensive"] = True
        
        return analysis
    
    async def _score_providers(
        self,
        request_analysis: Dict[str, Any],
        strategy: RoutingStrategy,
        user_preferences: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Score all providers for the request"""
        scores = {}
        
        for provider_key, capability in self.provider_capabilities.items():
            score_data = {
                "performance_score": 0.0,
                "quality_score": 0.0,
                "cost_score": 0.0,
                "dharmic_score": 0.0,
                "specialization_score": 0.0,
                "total_score": 0.0,
                "reasoning": "",
                "confidence": 0.8
            }
            
            # Performance scoring
            score_data["performance_score"] = (
                (capability.success_rate * 0.4) +
                (min(2000.0 / capability.avg_response_time_ms, 1.0) * 0.3) +
                (capability.uptime_percentage / 100.0 * 0.3)
            )
            
            # Quality scoring
            score_data["quality_score"] = (
                capability.knowledge_depth * 0.5 +
                capability.creativity_score * 0.3 +
                capability.dharmic_alignment_score * 0.2
            )
            
            # Cost scoring (inverse - lower cost = higher score)
            max_cost = 0.05  # Assumed max cost per 1k tokens
            score_data["cost_score"] = 1.0 - (capability.cost_per_1k_tokens / max_cost)
            
            # Dharmic alignment scoring
            score_data["dharmic_score"] = capability.dharmic_alignment_score
            if request_analysis.get("requires_dharmic_alignment", False):
                score_data["dharmic_score"] *= 1.5  # Boost if dharmic alignment needed
            
            # Specialization scoring
            request_type = request_analysis.get("request_type", RequestType.GENERAL_CHAT)
            if request_type in capability.strong_areas:
                score_data["specialization_score"] = 1.0
            elif request_type in capability.weak_areas:
                score_data["specialization_score"] = 0.3
            else:
                score_data["specialization_score"] = 0.7
            
            # Calculate total score based on strategy
            if strategy == RoutingStrategy.PERFORMANCE:
                score_data["total_score"] = (
                    score_data["performance_score"] * 0.6 +
                    score_data["specialization_score"] * 0.4
                )
            elif strategy == RoutingStrategy.QUALITY:
                score_data["total_score"] = (
                    score_data["quality_score"] * 0.5 +
                    score_data["specialization_score"] * 0.3 +
                    score_data["performance_score"] * 0.2
                )
            elif strategy == RoutingStrategy.COST:
                score_data["total_score"] = (
                    score_data["cost_score"] * 0.6 +
                    score_data["performance_score"] * 0.4
                )
            elif strategy == RoutingStrategy.DHARMIC_ALIGNMENT:
                score_data["total_score"] = (
                    score_data["dharmic_score"] * 0.4 +
                    score_data["quality_score"] * 0.3 +
                    score_data["specialization_score"] * 0.3
                )
            else:  # Load balanced or default
                score_data["total_score"] = (
                    score_data["performance_score"] * 0.25 +
                    score_data["quality_score"] * 0.25 +
                    score_data["dharmic_score"] * 0.25 +
                    score_data["specialization_score"] * 0.25
                )
            
            # Apply user preferences if provided
            if user_preferences:
                preferred_provider = user_preferences.get("preferred_provider")
                if preferred_provider and preferred_provider == capability.provider.value:
                    score_data["total_score"] *= 1.2  # 20% boost for user preference
            
            score_data["reasoning"] = f"Score: {score_data['total_score']:.2f} (Strategy: {strategy.value})"
            scores[provider_key] = score_data
        
        return scores
    
    async def update_performance_metrics(
        self,
        provider: LLMProvider,
        model: LLMModel,
        response_time_ms: float,
        success: bool,
        dharmic_score: Optional[float] = None
    ):
        """Update performance metrics for a provider"""
        try:
            provider_key = f"{provider.value}_{model.value.replace('-', '')}"
            
            if provider_key in self.provider_capabilities:
                capability = self.provider_capabilities[provider_key]
                
                # Update response time (moving average)
                current_avg = capability.avg_response_time_ms
                capability.avg_response_time_ms = (current_avg * 0.9) + (response_time_ms * 0.1)
                
                # Update success rate (moving average)
                current_success = capability.success_rate
                success_value = 1.0 if success else 0.0
                capability.success_rate = (current_success * 0.95) + (success_value * 0.05)
                
                # Update dharmic score if provided
                if dharmic_score is not None:
                    current_dharmic = capability.dharmic_alignment_score
                    capability.dharmic_alignment_score = (current_dharmic * 0.9) + (dharmic_score * 0.1)
                
                logger.debug(f"Updated metrics for {provider_key}")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def get_router_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        try:
            recent_decisions = self.routing_history[-100:] if self.routing_history else []
            
            # Provider usage distribution
            provider_usage = {}
            for decision in recent_decisions:
                provider_key = f"{decision.chosen_provider.value}_{decision.chosen_model.value}"
                provider_usage[provider_key] = provider_usage.get(provider_key, 0) + 1
            
            # Strategy usage
            strategy_usage = {}
            for decision in recent_decisions:
                strategy = decision.routing_strategy.value
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            # Average confidence
            avg_confidence = sum(d.confidence for d in recent_decisions) / len(recent_decisions) if recent_decisions else 0.0
            
            return {
                "total_routing_decisions": len(self.routing_history),
                "recent_decisions": len(recent_decisions),
                "provider_usage_distribution": provider_usage,
                "strategy_usage_distribution": strategy_usage,
                "average_confidence": avg_confidence,
                "average_decision_time_ms": sum(d.decision_time_ms for d in recent_decisions) / len(recent_decisions) if recent_decisions else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting router stats: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check router health"""
        return {
            "status": "healthy",
            "router": "advanced_llm",
            "providers_configured": len(self.provider_capabilities),
            "routing_decisions_made": len(self.routing_history),
            "performance_tracking_active": True
        }

# Global router instance
_advanced_llm_router: Optional[AdvancedLLMRouter] = None

async def get_advanced_llm_router() -> AdvancedLLMRouter:
    """Get the global advanced LLM router instance"""
    global _advanced_llm_router
    
    if _advanced_llm_router is None:
        _advanced_llm_router = AdvancedLLMRouter()
        await _advanced_llm_router.initialize()
    
    return _advanced_llm_router

async def init_advanced_llm_router() -> AdvancedLLMRouter:
    """Initialize and return the advanced LLM router"""
    return await get_advanced_llm_router()