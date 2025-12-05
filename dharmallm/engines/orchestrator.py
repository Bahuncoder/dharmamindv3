"""
Enterprise LLM Orchestrator / Router
===================================

Professional orchestrator implementing model selection, prompt templating,
retries, fallbacks, A/B testing, and cost management.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ModelSize(Enum):
    """Model size categories for intelligent routing"""

    SMALL = "small"  # Fast, cheap models
    MEDIUM = "medium"  # Balanced models
    LARGE = "large"  # High-quality models
    XLARGE = "xlarge"  # Premium models


class RoutingStrategy(Enum):
    """Model routing strategies"""

    COST_OPTIMIZED = "cost_optimized"  # Cheapest first
    QUALITY_OPTIMIZED = "quality_optimized"  # Best quality first
    SPEED_OPTIMIZED = "speed_optimized"  # Fastest first
    BALANCED = "balanced"  # Balance cost/quality/speed
    DHARMIC_FIRST = "dharmic_first"  # Dharmic models prioritized


@dataclass
class ModelDefinition:
    """Model definition with capabilities and costs"""

    model_id: str
    provider: str
    size: ModelSize
    cost_per_1k_tokens: float
    avg_latency_ms: float
    quality_score: float
    context_window: int
    dharmic_score: float
    capabilities: List[str]
    max_concurrent: int


@dataclass
class RoutingRequest:
    """Request for model routing"""

    prompt: str
    tenant_id: str
    user_preferences: Dict[str, Any]
    cost_budget: Optional[float]
    max_latency_ms: Optional[float]
    quality_threshold: float
    dharmic_required: bool
    a_b_test_group: Optional[str]


@dataclass
class RoutingResult:
    """Result of model routing"""

    selected_model: str
    fallback_models: List[str]
    routing_reason: str
    estimated_cost: float
    estimated_latency: float
    a_b_test_variant: Optional[str]


class PromptTemplate:
    """Professional prompt templating system"""

    def __init__(self, template_id: str, template: str, variables: List[str]):
        self.template_id = template_id
        self.template = template
        self.variables = variables

    def render(self, **kwargs) -> str:
        """Render template with variables"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")


class ABTestManager:
    """A/B testing for model selection"""

    def __init__(self):
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, List[Dict]] = {}

    def create_test(
        self,
        test_id: str,
        variants: Dict[str, Any],
        traffic_split: Dict[str, float],
    ):
        """Create new A/B test"""
        self.active_tests[test_id] = {
            "variants": variants,
            "traffic_split": traffic_split,
            "created_at": datetime.utcnow(),
            "total_requests": 0,
        }

    def assign_variant(self, test_id: str, user_id: str) -> Optional[str]:
        """Assign user to test variant"""
        if test_id not in self.active_tests:
            return None

        test = self.active_tests[test_id]

        # Simple hash-based assignment for consistency
        user_hash = hash(f"{test_id}:{user_id}") % 100

        cumulative = 0
        for variant, percentage in test["traffic_split"].items():
            cumulative += percentage * 100
            if user_hash < cumulative:
                return variant

        return list(test["variants"].keys())[0]  # Default to first variant

    def record_result(
        self, test_id: str, variant: str, metrics: Dict[str, Any]
    ):
        """Record test result"""
        if test_id not in self.test_results:
            self.test_results[test_id] = []

        self.test_results[test_id].append(
            {
                "variant": variant,
                "timestamp": datetime.utcnow(),
                "metrics": metrics,
            }
        )


class FallbackManager:
    """Intelligent fallback and retry management"""

    def __init__(self):
        self.model_health: Dict[str, Dict[str, Any]] = {}
        self.failure_counts: Dict[str, int] = {}

    def record_failure(self, model_id: str, error_type: str):
        """Record model failure"""
        if model_id not in self.failure_counts:
            self.failure_counts[model_id] = 0

        self.failure_counts[model_id] += 1

        self.model_health[model_id] = {
            "status": (
                "degraded"
                if self.failure_counts[model_id] < 5
                else "unhealthy"
            ),
            "failure_count": self.failure_counts[model_id],
            "last_failure": datetime.utcnow(),
        }

    def record_success(self, model_id: str):
        """Record successful response"""
        if model_id in self.failure_counts:
            self.failure_counts[model_id] = max(
                0, self.failure_counts[model_id] - 1
            )

        if self.failure_counts.get(model_id, 0) == 0:
            self.model_health[model_id] = {
                "status": "healthy",
                "failure_count": 0,
                "last_success": datetime.utcnow(),
            }

    def is_model_available(self, model_id: str) -> bool:
        """Check if model is available for use"""
        health = self.model_health.get(model_id, {"status": "healthy"})
        return health["status"] in ["healthy", "degraded"]

    def get_fallback_order(
        self, primary_model: str, available_models: List[str]
    ) -> List[str]:
        """Get ordered list of fallback models"""
        # Filter out unhealthy models
        healthy_models = [
            m for m in available_models if self.is_model_available(m)
        ]

        # Sort by health and failure count

        def health_score(model_id: str) -> int:
            health = self.model_health.get(model_id, {"failure_count": 0})
            return health["failure_count"]

        return sorted(healthy_models, key=health_score)


class CostManager:
    """Intelligent cost management and optimization"""

    def __init__(self):
        self.cost_tracking: Dict[str, List[Dict]] = {}
        self.budget_alerts: Dict[str, Dict] = {}

    def track_cost(
        self, tenant_id: str, model_id: str, cost: float, tokens: int
    ):
        """Track cost for tenant"""
        if tenant_id not in self.cost_tracking:
            self.cost_tracking[tenant_id] = []

        self.cost_tracking[tenant_id].append(
            {
                "timestamp": datetime.utcnow(),
                "model_id": model_id,
                "cost": cost,
                "tokens": tokens,
            }
        )

    def get_daily_cost(self, tenant_id: str) -> float:
        """Get daily cost for tenant"""
        if tenant_id not in self.cost_tracking:
            return 0.0

        today = datetime.utcnow().date()
        daily_costs = [
            record["cost"]
            for record in self.cost_tracking[tenant_id]
            if record["timestamp"].date() == today
        ]

        return sum(daily_costs)

    def check_budget(
        self, tenant_id: str, budget_limit: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if tenant is within budget"""
        daily_cost = self.get_daily_cost(tenant_id)

        if daily_cost >= budget_limit:
            return False, {
                "error": "budget_exceeded",
                "daily_cost": daily_cost,
                "budget_limit": budget_limit,
            }

        # Warning at 80% of budget
        if daily_cost >= budget_limit * 0.8:
            return True, {
                "warning": "budget_warning",
                "daily_cost": daily_cost,
                "budget_limit": budget_limit,
                "remaining": budget_limit - daily_cost,
            }

        return True, {
            "daily_cost": daily_cost,
            "budget_limit": budget_limit,
            "remaining": budget_limit - daily_cost,
        }


class LLMOrchestrator:
    """
    Enterprise LLM Orchestrator / Router

    Implements professional patterns for:
    - Intelligent model selection (small→large)
    - Prompt templating and tools
    - Retries, fallbacks, A/B testing
    - Cost caps and optimization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}

        # Core components
        self.models: Dict[str, ModelDefinition] = {}
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        self.ab_test_manager = ABTestManager()
        self.fallback_manager = FallbackManager()
        self.cost_manager = CostManager()

        # Initialize default models and templates
        self._initialize_models()
        self._initialize_templates()
        self._initialize_ab_tests()

        self.logger.info("🎯 Enterprise LLM Orchestrator initialized")

    def _initialize_models(self):
        """Initialize model registry with dharmic and standard models"""

        models_config = [
            # Small/Efficient Models
            ModelDefinition(
                model_id="local-dharma-small",
                provider="local",
                size=ModelSize.SMALL,
                cost_per_1k_tokens=0.0,
                avg_latency_ms=200,
                quality_score=0.7,
                context_window=2048,
                dharmic_score=0.95,
                capabilities=["dharmic_guidance", "fast_response"],
                max_concurrent=10,
            ),
            # Medium Models
            ModelDefinition(
                model_id="dharma-llm-v1",
                provider="dharma",
                size=ModelSize.MEDIUM,
                cost_per_1k_tokens=0.005,
                avg_latency_ms=800,
                quality_score=0.85,
                context_window=4096,
                dharmic_score=0.98,
                capabilities=[
                    "dharmic_guidance",
                    "spiritual_insights",
                    "ethical_reasoning",
                ],
                max_concurrent=5,
            ),
            # Large Models
            ModelDefinition(
                model_id="gpt-4",
                provider="openai",
                size=ModelSize.LARGE,
                cost_per_1k_tokens=0.03,
                avg_latency_ms=2000,
                quality_score=0.95,
                context_window=8192,
                dharmic_score=0.6,
                capabilities=["general_knowledge", "reasoning", "creativity"],
                max_concurrent=3,
            ),
            # Premium Models
            ModelDefinition(
                model_id="dharma-sage-premium",
                provider="dharma",
                size=ModelSize.XLARGE,
                cost_per_1k_tokens=0.1,
                avg_latency_ms=3000,
                quality_score=0.98,
                context_window=16384,
                dharmic_score=0.99,
                capabilities=[
                    "deep_dharmic_insights",
                    "scriptural_analysis",
                    "life_guidance",
                ],
                max_concurrent=1,
            ),
        ]

        for model in models_config:
            self.models[model.model_id] = model

    def _initialize_templates(self):
        """Initialize prompt templates"""

        templates = [
            PromptTemplate(
                "dharmic_guidance",
                "As a wise dharmic guide, please provide guidance on: {query}\n\nConsider these principles: {principles}\n\nResponse:",
                ["query", "principles"],
            ),
            PromptTemplate(
                "ethical_dilemma",
                "You are asked to help with an ethical dilemma. " +
                "Context: {context}\n\nDilemma: {dilemma}\n\nPlease provide dharmic perspective considering ahimsa, satya, and dharma:",
                ["context", "dilemma"],
            ),
            PromptTemplate(
                "spiritual_reflection",
                "Guide me in spiritual reflection on: {topic}\n\nMy current situation: {situation}\n\nWhat wisdom can you share?",
                ["topic", "situation"],
            ),
        ]

        for template in templates:
            self.prompt_templates[template.template_id] = template

    def _initialize_ab_tests(self):
        """Initialize A/B tests"""

        # Test dharmic models vs standard models
        self.ab_test_manager.create_test(
            "dharmic_vs_standard",
            {
                "dharmic": {"prefer_dharmic": True},
                "standard": {"prefer_dharmic": False},
            },
            {"dharmic": 0.8, "standard": 0.2},
        )

        # Test model sizes
        self.ab_test_manager.create_test(
            "model_size_optimization",
            {
                "small_first": {"size_preference": "small"},
                "medium_first": {"size_preference": "medium"},
            },
            {"small_first": 0.6, "medium_first": 0.4},
        )

    async def route_request(self, request: RoutingRequest) -> RoutingResult:
        """Route request to optimal model"""

        start_time = time.time()

        try:
            # Get A/B test variant
            ab_variant = None
            if request.tenant_id:
                ab_variant = self.ab_test_manager.assign_variant(
                    "dharmic_vs_standard", request.tenant_id
                )

            # Apply routing strategy
            strategy = self._determine_routing_strategy(request, ab_variant)

            # Get candidate models
            candidates = self._get_candidate_models(request, strategy)

            # Select primary model
            primary_model = self._select_primary_model(
                candidates, request, strategy
            )

            # Get fallback models
            fallback_models = self.fallback_manager.get_fallback_order(
                primary_model,
                [
                    m.model_id
                    for m in candidates
                    if m.model_id != primary_model
                ],
            )

            # Estimate costs and latency
            model_def = self.models[primary_model]
            estimated_tokens = (
                len(request.prompt.split()) * 1.3
            )  # Rough estimate
            estimated_cost = (
                estimated_tokens / 1000
            ) * model_def.cost_per_1k_tokens

            routing_time = (time.time() - start_time) * 1000

            return RoutingResult(
                selected_model=primary_model,
                fallback_models=fallback_models[:3],  # Top 3 fallbacks
                routing_reason=f"Strategy: {strategy}, Quality: {
                    model_def.quality_score}, Cost: ${
                    estimated_cost:.4f}",
                estimated_cost=estimated_cost,
                estimated_latency=model_def.avg_latency_ms + routing_time,
                a_b_test_variant=ab_variant,
            )

        except Exception as e:
            self.logger.error(f"Routing failed: {e}")

            # Emergency fallback
            return RoutingResult(
                selected_model="local-dharma-small",
                fallback_models=[],
                routing_reason=f"Emergency fallback due to error: {e}",
                estimated_cost=0.0,
                estimated_latency=500,
                a_b_test_variant=None,
            )

    def _determine_routing_strategy(
        self, request: RoutingRequest, ab_variant: Optional[str]
    ) -> RoutingStrategy:
        """Determine routing strategy based on request and A/B test"""

        # A/B test override
        if ab_variant == "dharmic":
            return RoutingStrategy.DHARMIC_FIRST

        # User preferences
        if request.user_preferences.get("strategy"):
            return RoutingStrategy(request.user_preferences["strategy"])

        # Budget constraints
        if request.cost_budget and request.cost_budget < 0.01:
            return RoutingStrategy.COST_OPTIMIZED

        # Latency requirements
        if request.max_latency_ms and request.max_latency_ms < 1000:
            return RoutingStrategy.SPEED_OPTIMIZED

        # Dharmic requirements
        if request.dharmic_required:
            return RoutingStrategy.DHARMIC_FIRST

        # Default
        return RoutingStrategy.BALANCED

    def _get_candidate_models(
        self, request: RoutingRequest, strategy: RoutingStrategy
    ) -> List[ModelDefinition]:
        """Get candidate models based on strategy"""

        candidates = list(self.models.values())

        # Filter by availability
        candidates = [
            m
            for m in candidates
            if self.fallback_manager.is_model_available(m.model_id)
        ]

        # Filter by quality threshold
        candidates = [
            m
            for m in candidates
            if m.quality_score >= request.quality_threshold
        ]

        # Filter by dharmic requirement
        if request.dharmic_required:
            candidates = [m for m in candidates if m.dharmic_score >= 0.8]

        # Budget filtering
        if request.cost_budget:
            estimated_tokens = len(request.prompt.split()) * 1.3
            candidates = [
                m
                for m in candidates
                if (estimated_tokens / 1000) * m.cost_per_1k_tokens
                <= request.cost_budget
            ]

        return candidates

    def _select_primary_model(
        self,
        candidates: List[ModelDefinition],
        request: RoutingRequest,
        strategy: RoutingStrategy,
    ) -> str:
        """Select primary model from candidates"""

        if not candidates:
            return "local-dharma-small"  # Emergency fallback

        if strategy == RoutingStrategy.COST_OPTIMIZED:
            return min(candidates, key=lambda m: m.cost_per_1k_tokens).model_id

        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            return max(candidates, key=lambda m: m.quality_score).model_id

        elif strategy == RoutingStrategy.SPEED_OPTIMIZED:
            return min(candidates, key=lambda m: m.avg_latency_ms).model_id

        elif strategy == RoutingStrategy.DHARMIC_FIRST:
            dharmic_models = [m for m in candidates if m.dharmic_score >= 0.9]
            if dharmic_models:
                return max(
                    dharmic_models, key=lambda m: m.dharmic_score
                ).model_id
            return max(candidates, key=lambda m: m.dharmic_score).model_id

        else:  # BALANCED
            # Score based on quality, cost, and speed


            def balance_score(model: ModelDefinition) -> float:
                quality_norm = model.quality_score  # Already 0-1
                # Normalize to 0-1
                cost_norm = 1 - min(model.cost_per_1k_tokens / 0.1, 1)
                # Normalize to 0-1
                speed_norm = 1 - min(model.avg_latency_ms / 5000, 1)
                dharmic_bonus = (
                    model.dharmic_score * 0.2
                    if request.dharmic_required
                    else 0
                )

                return (
                    quality_norm * 0.4
                    + cost_norm * 0.3
                    + speed_norm * 0.3
                    + dharmic_bonus
                )

            return max(candidates, key=balance_score).model_id

    def render_prompt(self, template_id: str, **kwargs) -> str:
        """Render prompt template"""
        if template_id not in self.prompt_templates:
            raise ValueError(f"Template {template_id} not found")

        return self.prompt_templates[template_id].render(**kwargs)

    async def execute_with_fallback(
        self, request: RoutingRequest, executor_func, max_retries: int = 3
    ) -> Dict[str, Any]:
        """Execute request with intelligent fallback"""

        routing_result = await self.route_request(request)
        models_to_try = [
            routing_result.selected_model
        ] + routing_result.fallback_models

        last_error = None

        for attempt, model_id in enumerate(models_to_try[:max_retries]):
            try:
                # Execute with current model
                result = await executor_func(model_id, request)

                # Record success
                self.fallback_manager.record_success(model_id)

                # Track cost
                cost = result.get("cost", 0.0)
                tokens = result.get("usage", {}).get("total_tokens", 0)
                self.cost_manager.track_cost(
                    request.tenant_id, model_id, cost, tokens
                )

                # Record A/B test result
                if routing_result.a_b_test_variant:
                    self.ab_test_manager.record_result(
                        "dharmic_vs_standard",
                        routing_result.a_b_test_variant,
                        {
                            "success": True,
                            "model_used": model_id,
                            "cost": cost,
                            "quality": result.get("quality_score", 0.8),
                        },
                    )

                # Add routing metadata
                result["routing_info"] = {
                    "selected_model": model_id,
                    "attempt": attempt + 1,
                    "routing_strategy": routing_result.routing_reason,
                    "a_b_variant": routing_result.a_b_test_variant,
                }

                return result

            except Exception as e:
                self.logger.warning(
                    f"Model {model_id} failed (attempt {
                        attempt + 1}): {e}"
                )

                # Record failure
                self.fallback_manager.record_failure(
                    model_id, str(type(e).__name__)
                )

                last_error = e

                # Wait before retry (exponential backoff)
                if attempt < len(models_to_try) - 1:
                    await asyncio.sleep(2**attempt)

        # All models failed
        raise Exception(f"All models failed. Last error: {last_error}")

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "registered_models": len(self.models),
            "active_ab_tests": len(self.ab_test_manager.active_tests),
            "prompt_templates": len(self.prompt_templates),
            "model_health": self.fallback_manager.model_health,
            "cost_tracking_tenants": len(self.cost_manager.cost_tracking),
        }


# Example usage
async def demo_orchestrator():
    """Demonstrate orchestrator functionality"""

    orchestrator = LLMOrchestrator()

    # Create routing request
    request = RoutingRequest(
        prompt="How can I balance work and spiritual practice?",
        tenant_id="demo_tenant",
        user_preferences={"strategy": "balanced"},
        cost_budget=0.05,
        max_latency_ms=3000,
        quality_threshold=0.8,
        dharmic_required=True,
        a_b_test_group="test_group_1",
    )

    # Route request
    routing_result = await orchestrator.route_request(request)

    print(f"🎯 Orchestrator Routing Result:")
    print(f"Selected Model: {routing_result.selected_model}")
    print(f"Fallback Models: {routing_result.fallback_models}")
    print(f"Routing Reason: {routing_result.routing_reason}")
    print(f"Estimated Cost: ${routing_result.estimated_cost:.4f}")
    print(f"A/B Test Variant: {routing_result.a_b_test_variant}")

    # Render template
    prompt = orchestrator.render_prompt(
        "dharmic_guidance",
        query="work-life balance",
        principles="dharma, karma, moksha",
    )
    print(f"\n📝 Rendered Prompt: {prompt}")


if __name__ == "__main__":
    asyncio.run(demo_orchestrator())
