"""
Enterprise LLM API Gateway
=========================

Professional API Gateway with authentication, rate limiting, and
    tenant isolation
following big tech company patterns.
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import jwt


class AuthType(Enum):
    """Authentication types"""

    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    NONE = "none"


class QuotaType(Enum):
    """Quota measurement types"""

    REQUESTS_PER_MINUTE = "requests_per_minute"
    TOKENS_PER_HOUR = "tokens_per_hour"
    COST_PER_DAY = "cost_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"


@dataclass
class TenantConfig:
    """Tenant isolation configuration"""

    tenant_id: str
    name: str
    tier: str  # "free", "pro", "enterprise"
    quotas: Dict[QuotaType, int]
    auth_type: AuthType
    allowed_models: List[str]
    rate_limits: Dict[str, int]
    cost_budget: float
    security_level: str
    dharmic_validation_required: bool = True


@dataclass
class APIRequest:
    """Standardized API request"""

    request_id: str
    tenant_id: str
    user_id: Optional[str]
    endpoint: str
    method: str
    headers: Dict[str, str]
    body: Dict[str, Any]
    timestamp: datetime
    ip_address: str
    auth_token: Optional[str]


@dataclass
class APIResponse:
    """Standardized API response"""

    request_id: str
    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str]
    processing_time_ms: float
    tokens_used: int
    cost: float
    cached: bool


class RateLimiter:
    """Enterprise-grade rate limiter with multiple strategies"""

    def __init__(self):
        self.windows: Dict[str, deque] = defaultdict(deque)
        self.counters: Dict[str, int] = defaultdict(int)

    def check_rate_limit(
        self, key: str, limit: int, window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        current_time = time.time()
        window = self.windows[key]

        # Remove old requests outside window
        while window and window[0] <= current_time - window_seconds:
            window.popleft()

        # Check limit
        if len(window) >= limit:
            retry_after = int(window[0] + window_seconds - current_time) + 1
            return False, {
                "error": "rate_limit_exceeded",
                "retry_after": retry_after,
                "current_usage": len(window),
                "limit": limit,
            }

        # Add current request
        window.append(current_time)

        return True, {
            "current_usage": len(window),
            "limit": limit,
            "remaining": limit - len(window),
        }


class UsageMetering:
    """Track usage for billing and analytics"""

    def __init__(self):
        self.usage_data: Dict[str, List[Dict]] = defaultdict(list)

    def record_usage(self, tenant_id: str, usage_data: Dict[str, Any]):
        """Record usage event"""
        usage_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            **usage_data,
        }
        self.usage_data[tenant_id].append(usage_record)

        # Keep only last 1000 records per tenant
        if len(self.usage_data[tenant_id]) > 1000:
            self.usage_data[tenant_id] = self.usage_data[tenant_id][-1000:]

    def get_usage_summary(self, tenant_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get usage summary for tenant"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        recent_usage = [
            record
            for record in self.usage_data[tenant_id]
            if datetime.fromisoformat(record["timestamp"]) > cutoff_time
        ]

        if not recent_usage:
            return {"requests": 0, "tokens": 0, "cost": 0.0}

        return {
            "requests": len(recent_usage),
            "tokens": sum(record.get("tokens", 0) for record in recent_usage),
            "cost": sum(record.get("cost", 0.0) for record in recent_usage),
            "period_hours": hours,
        }


class AuthManager:
    """Enterprise authentication manager"""

    def __init__(self, secret_key: str = "dharma_secret_key"):
        self.secret_key = secret_key
        self.api_keys: Dict[str, TenantConfig] = {}

    def register_tenant(self, config: TenantConfig, api_key: Optional[str] = None):
        """Register new tenant"""
        if not api_key:
            api_key = self._generate_api_key(config.tenant_id)

        self.api_keys[api_key] = config
        return api_key

    def _generate_api_key(self, tenant_id: str) -> str:
        """Generate secure API key"""
        timestamp = str(int(time.time()))
        data = f"{tenant_id}:{timestamp}:{self.secret_key}"
        return f"dk_{hashlib.sha256(data.encode()).hexdigest()[:32]}"

    def authenticate_request(
        self, request: APIRequest
    ) -> Tuple[bool, Optional[TenantConfig], Dict[str, Any]]:
        """Authenticate incoming request"""

        # Extract auth token
        auth_header = request.headers.get("Authorization", "")
        api_key = None

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

            # Try JWT first
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
                tenant_id = payload.get("tenant_id")
                if tenant_id and token in self.api_keys:
                    return True, self.api_keys[token], {"auth_method": "jwt"}
            except jwt.InvalidTokenError:
                pass

            # Try API key
            if token.startswith("dk_") and token in self.api_keys:
                return True, self.api_keys[token], {"auth_method": "api_key"}

        # Check API key in query params
        api_key = request.body.get("api_key")
        if api_key and api_key in self.api_keys:
            return True, self.api_keys[api_key], {"auth_method": "query_param"}

        return False, None, {"error": "invalid_authentication"}

    def create_jwt_token(
        self, tenant_id: str, user_id: str, expires_hours: int = 24
    ) -> str:
        """Create JWT token for tenant"""
        payload = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=expires_hours),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")


class APIGateway:
    """
    Enterprise API Gateway with Auth, Quotas, and Tenant Isolation

    Implements professional patterns used by OpenAI, Anthropic, Google, etc.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}

        # Core components
        self.auth_manager = AuthManager(self.config.get("secret_key", "dharma_secret"))
        self.rate_limiter = RateLimiter()
        self.usage_metering = UsageMetering()

        # Gateway state
        self.active_requests: Dict[str, APIRequest] = {}
        self.request_counter = 0

        # Initialize default tenant
        self._setup_default_tenants()

        self.logger.info("🚪 Enterprise API Gateway initialized")

    def _setup_default_tenants(self):
        """Setup default tenant configurations"""

        # Free tier tenant
        free_config = TenantConfig(
            tenant_id="free_tier",
            name="Free Tier",
            tier="free",
            quotas={
                QuotaType.REQUESTS_PER_MINUTE: 10,
                QuotaType.TOKENS_PER_HOUR: 1000,
                QuotaType.COST_PER_DAY: 0.0,
            },
            auth_type=AuthType.API_KEY,
            allowed_models=["local-dharma"],
            rate_limits={"requests_per_minute": 10},
            cost_budget=0.0,
            security_level="standard",
            dharmic_validation_required=True,
        )

        # Pro tier tenant
        pro_config = TenantConfig(
            tenant_id="pro_tier",
            name="Professional Tier",
            tier="pro",
            quotas={
                QuotaType.REQUESTS_PER_MINUTE: 100,
                QuotaType.TOKENS_PER_HOUR: 50000,
                QuotaType.COST_PER_DAY: 50.0,
            },
            auth_type=AuthType.JWT,
            allowed_models=["dharma-llm-v1", "gpt-4", "local-dharma"],
            rate_limits={"requests_per_minute": 100},
            cost_budget=50.0,
            security_level="high",
            dharmic_validation_required=True,
        )

        # Enterprise tier tenant
        enterprise_config = TenantConfig(
            tenant_id="enterprise_tier",
            name="Enterprise Tier",
            tier="enterprise",
            quotas={
                QuotaType.REQUESTS_PER_MINUTE: 1000,
                QuotaType.TOKENS_PER_HOUR: 500000,
                QuotaType.COST_PER_DAY: 500.0,
            },
            auth_type=AuthType.OAUTH2,
            allowed_models=[
                "dharma-llm-v1",
                "gpt-4",
                "local-dharma",
                "premium-models",
            ],
            rate_limits={"requests_per_minute": 1000},
            cost_budget=500.0,
            security_level="maximum",
            dharmic_validation_required=False,  # Enterprise can opt out
        )

        # Register tenants
        self.auth_manager.register_tenant(free_config)
        self.auth_manager.register_tenant(pro_config)
        self.auth_manager.register_tenant(enterprise_config)

        self.logger.info("🏢 Default tenant tiers configured")

    def register_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: str,
        api_key: str,
        custom_limits: Dict[str, Any] = None,
    ) -> TenantConfig:
        """Register a new tenant with the API Gateway"""

        # Create tenant configuration
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            quotas={
                QuotaType.REQUESTS_PER_MINUTE: (
                    custom_limits.get("max_requests_per_minute", 60)
                    if custom_limits
                    else 60
                ),
                QuotaType.TOKENS_PER_HOUR: (
                    custom_limits.get("max_tokens_per_day", 10000)
                    if custom_limits
                    else 10000
                ),
                QuotaType.COST_PER_DAY: (
                    custom_limits.get("cost_per_day", 10.0) if custom_limits else 10.0
                ),
            },
            auth_type=AuthType.API_KEY,
            allowed_models=[
                "dharma-llm-v1",
                "dharma-sage-premium",
                "local-dharma",
            ],
            rate_limits={
                "requests_per_minute": (
                    custom_limits.get("max_requests_per_minute", 60)
                    if custom_limits
                    else 60
                )
            },
            cost_budget=(
                custom_limits.get("cost_per_day", 10.0) if custom_limits else 10.0
            ),
            security_level="standard",
            dharmic_validation_required=True,
        )

        # Register with auth manager
        self.auth_manager.register_tenant(tenant_config, api_key)

        self.logger.info(f"🏢 Registered tenant: {name} ({tenant_id}) - {tier}")

        return tenant_config

    async def process_request(self, raw_request: Dict[str, Any]) -> APIResponse:
        """Process incoming API request through gateway"""

        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}_{self.request_counter}"
        self.request_counter += 1

        try:
            # Parse request
            api_request = self._parse_request(raw_request, request_id)
            self.active_requests[request_id] = api_request

            # Authenticate
            authenticated, tenant_config, auth_info = (
                self.auth_manager.authenticate_request(api_request)
            )

            if not authenticated:
                return self._create_error_response(
                    request_id,
                    401,
                    "Authentication failed",
                    auth_info,
                    start_time,
                )

            # Check rate limits
            rate_key = f"{tenant_config.tenant_id}:{api_request.ip_address}"
            rate_allowed, rate_info = self.rate_limiter.check_rate_limit(
                rate_key,
                tenant_config.rate_limits.get("requests_per_minute", 60),
                60,
            )

            if not rate_allowed:
                return self._create_error_response(
                    request_id,
                    429,
                    "Rate limit exceeded",
                    rate_info,
                    start_time,
                )

            # Check quotas
            quota_valid, quota_info = self._check_quotas(tenant_config)
            if not quota_valid:
                return self._create_error_response(
                    request_id, 402, "Quota exceeded", quota_info, start_time
                )

            # Validate model access
            requested_model = api_request.body.get("model", "default")
            if requested_model not in tenant_config.allowed_models:
                return self._create_error_response(
                    request_id,
                    403,
                    "Model not allowed for tenant",
                    {"allowed_models": tenant_config.allowed_models},
                    start_time,
                )

            # Process through security layer
            security_passed, security_info = await self._security_check(
                api_request, tenant_config
            )
            if not security_passed:
                return self._create_error_response(
                    request_id,
                    400,
                    "Security check failed",
                    security_info,
                    start_time,
                )

            # Forward to orchestrator
            response_body = await self._forward_to_orchestrator(
                api_request, tenant_config
            )

            # Record usage
            processing_time = (time.time() - start_time) * 1000
            tokens_used = response_body.get("usage", {}).get("total_tokens", 0)
            cost = response_body.get("cost", 0.0)

            self.usage_metering.record_usage(
                tenant_config.tenant_id,
                {
                    "request_id": request_id,
                    "tokens": tokens_used,
                    "cost": cost,
                    "processing_time_ms": processing_time,
                    "model": requested_model,
                    "endpoint": api_request.endpoint,
                },
            )

            # Create success response
            return APIResponse(
                request_id=request_id,
                status_code=200,
                body=response_body,
                headers={
                    "X-Request-ID": request_id,
                    "X-Tenant-ID": tenant_config.tenant_id,
                    "X-Rate-Limit-Remaining": str(rate_info.get("remaining", 0)),
                    "X-Tokens-Used": str(tokens_used),
                    "Content-Type": "application/json",
                },
                processing_time_ms=processing_time,
                tokens_used=tokens_used,
                cost=cost,
                cached=response_body.get("cached", False),
            )

        except Exception as e:
            self.logger.error(f"Gateway error: {e}")
            return self._create_error_response(
                request_id,
                500,
                "Internal server error",
                {"error": str(e)},
                start_time,
            )

        finally:
            # Clean up
            if request_id in self.active_requests:
                del self.active_requests[request_id]

    def _parse_request(
        self, raw_request: Dict[str, Any], request_id: str
    ) -> APIRequest:
        """Parse raw request into structured format"""
        return APIRequest(
            request_id=request_id,
            tenant_id="",  # Will be filled after auth
            user_id=raw_request.get("user_id"),
            endpoint=raw_request.get("endpoint", "/v1/chat/completions"),
            method=raw_request.get("method", "POST"),
            headers=raw_request.get("headers", {}),
            body=raw_request.get("body", {}),
            timestamp=datetime.utcnow(),
            ip_address=raw_request.get("ip_address", "127.0.0.1"),
            auth_token=raw_request.get("headers", {}).get("Authorization"),
        )

    def _check_quotas(self, tenant_config: TenantConfig) -> Tuple[bool, Dict[str, Any]]:
        """Check if tenant is within quotas"""

        # Get current usage
        usage_summary = self.usage_metering.get_usage_summary(
            tenant_config.tenant_id, hours=1
        )

        # Check hourly token quota
        token_quota = tenant_config.quotas.get(QuotaType.TOKENS_PER_HOUR, float("inf"))
        if usage_summary["tokens"] >= token_quota:
            return False, {
                "error": "token_quota_exceeded",
                "current_usage": usage_summary["tokens"],
                "quota": token_quota,
            }

        # Check daily cost quota
        daily_usage = self.usage_metering.get_usage_summary(
            tenant_config.tenant_id, hours=24
        )
        cost_quota = tenant_config.quotas.get(QuotaType.COST_PER_DAY, float("inf"))
        if daily_usage["cost"] >= cost_quota:
            return False, {
                "error": "cost_quota_exceeded",
                "current_usage": daily_usage["cost"],
                "quota": cost_quota,
            }

        return True, {
            "token_usage": usage_summary["tokens"],
            "token_quota": token_quota,
            "cost_usage": daily_usage["cost"],
            "cost_quota": cost_quota,
        }

    async def _security_check(
        self, request: APIRequest, tenant_config: TenantConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """Perform security checks on request"""

        security_issues = []

        # Check for dharmic validation requirement
        if tenant_config.dharmic_validation_required:
            prompt = request.body.get("messages", [{}])[-1].get("content", "")
            if self._contains_harmful_content(prompt):
                security_issues.append("harmful_content_detected")

        # Check content length
        if len(str(request.body)) > 100000:  # 100KB limit
            security_issues.append("request_too_large")

        # PII detection (simple)
        if self._contains_pii(str(request.body)):
            security_issues.append("pii_detected")

        if security_issues:
            return False, {"security_issues": security_issues}

        return True, {"security_status": "passed"}

    def _contains_harmful_content(self, text: str) -> bool:
        """Simple harmful content detection"""
        harmful_keywords = ["hack", "bomb", "illegal", "violence"]
        return any(keyword in text.lower() for keyword in harmful_keywords)

    def _contains_pii(self, text: str) -> bool:
        """Simple PII detection"""
        import re

        # Simple email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        return bool(re.search(email_pattern, text))

    async def _forward_to_orchestrator(
        self, request: APIRequest, tenant_config: TenantConfig
    ) -> Dict[str, Any]:
        """Forward request to orchestrator (mock implementation)"""

        # Mock response from orchestrator
        return {
            "id": f"chatcmpl-{request.request_id}",
            "object": "chat.completion",
            "model": request.body.get("model", "dharma-llm-v1"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"This is a dharmic response to your query. "
                        + "The wisdom traditions guide us...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35,
            },
            "cost": 0.001,
            "cached": False,
        }

    def _create_error_response(
        self,
        request_id: str,
        status_code: int,
        error_message: str,
        details: Dict[str, Any],
        start_time: float,
    ) -> APIResponse:
        """Create standardized error response"""
        return APIResponse(
            request_id=request_id,
            status_code=status_code,
            body={
                "error": {
                    "message": error_message,
                    "type": "api_error",
                    "code": status_code,
                    "details": details,
                }
            },
            headers={
                "X-Request-ID": request_id,
                "Content-Type": "application/json",
            },
            processing_time_ms=(time.time() - start_time) * 1000,
            tokens_used=0,
            cost=0.0,
            cached=False,
        )

    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        return {
            "active_requests": len(self.active_requests),
            "total_tenants": len(self.auth_manager.api_keys),
            "request_counter": self.request_counter,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_tenant_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage statistics for specific tenant"""
        return self.usage_metering.get_usage_summary(tenant_id, hours=24)


# Example usage
async def demo_api_gateway():
    """Demonstrate API Gateway functionality"""

    # Initialize gateway
    gateway = APIGateway()

    # Get API key for free tier
    free_api_key = list(gateway.auth_manager.api_keys.keys())[0]

    # Create test request
    test_request = {
        "endpoint": "/v1/chat/completions",
        "method": "POST",
        "headers": {
            "Authorization": f"Bearer {free_api_key}",
            "Content-Type": "application/json",
        },
        "body": {
            "model": "local-dharma",
            "messages": [
                {
                    "role": "user",
                    "content": "How can I live according to dharma?",
                }
            ],
            "max_tokens": 100,
        },
        "ip_address": "192.168.1.100",
    }

    # Process request
    response = await gateway.process_request(test_request)

    print(f"🚪 API Gateway Response:")
    print(f"Status: {response.status_code}")
    print(f"Request ID: {response.request_id}")
    print(f"Processing Time: {response.processing_time_ms:.2f}ms")
    print(f"Content: {response.body}")


if __name__ == "__main__":
    asyncio.run(demo_api_gateway())
