"""
🔐 LLM Gateway Service - Separate Microservice

Independent service for handling external LLM APIs (ChatGPT, Claude, etc.)
completely isolated from the main DharmaMind backend.

Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Frontend      │ ── │  LLM Gateway     │ ── │  DharmaMind Backend │
│   (React/Next)  │    │  (Port 8003)     │    │  (Port 8000)        │
└─────────────────┘    └──────────────────┘    └─────────────────────┘

Security Benefits:
- 🔒 External API keys isolated from main backend
- 🚀 Independent scaling and deployment
- 🛡️ No direct access to spiritual modules or user data
- 🔧 Easier maintenance and LLM provider updates
- 💳 Centralized usage tracking per provider
- 🌐 Rate limiting per external service

Usage Flow:
1. Frontend sends request to LLM Gateway
2. Gateway calls external LLM API (ChatGPT/Claude)
3. Gateway returns raw response + usage data
4. Frontend sends to DharmaMind Backend for dharmic processing
5. Backend processes through spiritual modules
6. Enhanced dharmic response returned to frontend

May this gateway serve with security and wisdom 🕉️
"""

import asyncio
import logging
import time
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import httpx
import uvicorn

# Try to import redis, fallback if not available
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LLMGateway")

# ===============================
# CONFIGURATION
# ===============================

class Config:
    """LLM Gateway Configuration"""
    
    # Server
    HOST: str = os.getenv("LLM_GATEWAY_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("LLM_GATEWAY_PORT", "8003"))
    
    # Security
    API_KEY: str = os.getenv("LLM_GATEWAY_API_KEY", "llm-gateway-secure-key-123")
    
    # Backend Communication
    BACKEND_URL: str = os.getenv("DHARMAMIND_BACKEND_URL", "http://localhost:8000")
    
    # External LLM APIs
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:8000",
        "https://yourdomain.com"
    ]

config = Config()

# ===============================
# MODELS
# ===============================

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class LLMRequest(BaseModel):
    """Request to LLM Gateway"""
    provider: LLMProvider = Field(..., description="LLM provider")
    model: str = Field(..., description="Model to use")
    query: str = Field(..., description="User query")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    max_tokens: Optional[int] = Field(1000, description="Maximum tokens")
    temperature: Optional[float] = Field(0.7, description="Response temperature")
    user_id: Optional[str] = Field(None, description="User identifier")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError('Temperature must be between 0 and 2')
        return v

class LLMResponse(BaseModel):
    """Response from LLM Gateway"""
    success: bool
    request_id: str
    provider: LLMProvider
    model: str
    content: str
    usage: Dict[str, Any]
    response_time: float
    timestamp: str
    cached: bool = False

class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    providers: Dict[str, bool]
    uptime: float

# ===============================
# EXTERNAL LLM CLIENTS
# ===============================

class OpenAIClient:
    """OpenAI API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        
    async def generate(
        self,
        model: str,
        query: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate response from OpenAI"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenAI API error: {response.text}"
                )
            
            result = response.json()
            return {
                "content": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
                "model": model
            }

class AnthropicClient:
    """Anthropic Claude API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        
    async def generate(
        self,
        model: str,
        query: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate response from Anthropic Claude"""
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": query}]
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=data,
                timeout=60.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Anthropic API error: {response.text}"
                )
            
            result = response.json()
            return {
                "content": result["content"][0]["text"],
                "usage": result.get("usage", {}),
                "model": model
            }

# ===============================
# GATEWAY SERVICE
# ===============================

class LLMGatewayService:
    """Main LLM Gateway service"""
    
    def __init__(self):
        self.openai_client: Optional[OpenAIClient] = None
        self.anthropic_client: Optional[AnthropicClient] = None
        self.start_time = time.time()
        self.request_cache = {}  # Simple in-memory cache
        self.rate_limiter = {}   # Simple rate limiting
        
    async def initialize(self):
        """Initialize the gateway service"""
        logger.info("🔐 Initializing LLM Gateway Service...")
        
        # Initialize LLM clients
        if config.OPENAI_API_KEY:
            self.openai_client = OpenAIClient(config.OPENAI_API_KEY)
            logger.info("✅ OpenAI client initialized")
        else:
            logger.warning("⚠️ OpenAI API key not provided")
            
        if config.ANTHROPIC_API_KEY:
            self.anthropic_client = AnthropicClient(config.ANTHROPIC_API_KEY)
            logger.info("✅ Anthropic client initialized")
        else:
            logger.warning("⚠️ Anthropic API key not provided")
        
        if not self.openai_client and not self.anthropic_client:
            logger.error("❌ No LLM providers configured!")
            
        logger.info("🚀 LLM Gateway Service ready")
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process LLM request"""
        request_id = self._generate_request_id(request)
        start_time = time.time()
        
        try:
            # Check rate limits
            self._check_rate_limits(request.user_id or "anonymous")
            
            # Check cache
            cached_response = self._get_cached_response(request_id)
            if cached_response:
                logger.info(f"📋 Cache hit for request {request_id}")
                return cached_response
            
            # Route to appropriate provider
            if request.provider == LLMProvider.OPENAI:
                result = await self._process_openai_request(request)
            elif request.provider == LLMProvider.ANTHROPIC:
                result = await self._process_anthropic_request(request)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported provider: {request.provider}"
                )
            
            # Create response
            response_time = time.time() - start_time
            gateway_response = LLMResponse(
                success=True,
                request_id=request_id,
                provider=request.provider,
                model=request.model,
                content=result["content"],
                usage=result["usage"],
                response_time=response_time,
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Cache response
            self._cache_response(request_id, gateway_response)
            
            # Log usage
            self._log_usage(request, gateway_response)
            
            return gateway_response
            
        except Exception as e:
            logger.error(f"❌ Error processing request {request_id}: {e}")
            raise
    
    async def _process_openai_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Process OpenAI request"""
        if not self.openai_client:
            raise HTTPException(status_code=503, detail="OpenAI client not available")
        
        return await self.openai_client.generate(
            model=request.model,
            query=request.query,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
    
    async def _process_anthropic_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Process Anthropic request"""
        if not self.anthropic_client:
            raise HTTPException(status_code=503, detail="Anthropic client not available")
        
        return await self.anthropic_client.generate(
            model=request.model,
            query=request.query,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens
        )
    
    def _generate_request_id(self, request: LLMRequest) -> str:
        """Generate unique request ID"""
        content = f"{request.provider}{request.model}{request.query}{request.system_prompt}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_cached_response(self, request_id: str) -> Optional[LLMResponse]:
        """Get cached response if available"""
        cached = self.request_cache.get(request_id)
        if cached and time.time() - cached["timestamp"] < 3600:  # 1 hour cache
            response_data = cached["response"]
            response_data.cached = True
            return response_data
        return None
    
    def _cache_response(self, request_id: str, response: LLMResponse):
        """Cache response for future use"""
        self.request_cache[request_id] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Simple cache cleanup - keep last 1000 entries
        if len(self.request_cache) > 1000:
            oldest_keys = sorted(
                self.request_cache.keys(),
                key=lambda k: self.request_cache[k]["timestamp"]
            )[:100]
            for key in oldest_keys:
                del self.request_cache[key]
    
    def _check_rate_limits(self, user_id: str):
        """Simple rate limiting"""
        current_time = int(time.time() / 60)  # Current minute
        key = f"{user_id}:{current_time}"
        
        if key not in self.rate_limiter:
            self.rate_limiter[key] = 0
        
        if self.rate_limiter[key] >= config.RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        self.rate_limiter[key] += 1
        
        # Cleanup old entries
        old_keys = [k for k in self.rate_limiter.keys() 
                   if int(k.split(":")[1]) < current_time - 5]
        for key in old_keys:
            del self.rate_limiter[key]
    
    def _log_usage(self, request: LLMRequest, response: LLMResponse):
        """Log usage for analytics"""
        logger.info(
            f"📊 Usage: {request.provider}/{request.model} "
            f"- User: {request.user_id} "
            f"- Tokens: {response.usage.get('total_tokens', 0)} "
            f"- Time: {response.response_time:.2f}s"
        )
    
    async def get_health_status(self) -> HealthStatus:
        """Get service health status"""
        providers = {
            "openai": bool(self.openai_client and config.OPENAI_API_KEY),
            "anthropic": bool(self.anthropic_client and config.ANTHROPIC_API_KEY)
        }
        
        return HealthStatus(
            status="healthy" if any(providers.values()) else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            providers=providers,
            uptime=time.time() - self.start_time
        )

# Global service instance
gateway_service = LLMGatewayService()

# ===============================
# SECURITY
# ===============================

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key"""
    if credentials.credentials != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# ===============================
# FASTAPI APPLICATION
# ===============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    # Startup
    await gateway_service.initialize()
    yield
    # Shutdown - cleanup if needed

app = FastAPI(
    title="LLM Gateway Service",
    description="""
    🔐 Secure External LLM Gateway
    
    **Independent microservice for external LLM API access**
    
    This service provides secure, rate-limited access to external LLM providers
    while maintaining complete isolation from the main DharmaMind backend.
    
    **Security Features:**
    - 🔒 API key isolation (external keys never touch main backend)
    - 🚀 Independent deployment and scaling  
    - 🛡️ Rate limiting per user
    - 📊 Usage tracking and analytics
    - ⚡ Response caching
    - 🌐 CORS configuration
    
    **Supported Providers:**
    - OpenAI (GPT-4, GPT-3.5-turbo)
    - Anthropic (Claude-3 models)
    
    **Architecture:**
    Frontend ↔ LLM Gateway (Port 8003) ↔ External APIs
                    ↕
    Frontend ↔ DharmaMind Backend (Port 8000) ↔ Spiritual Modules
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ===============================
# API ROUTES
# ===============================

@app.post("/generate", response_model=LLMResponse)
async def generate_response(
    request: LLMRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    🤖 Generate response from external LLM
    
    **Usage Flow:**
    1. Frontend sends request to this endpoint
    2. Gateway calls external LLM API (ChatGPT/Claude)
    3. Raw response returned with usage metrics
    4. Frontend then sends to DharmaMind backend for dharmic processing
    
    **Security:** External API keys are isolated in this service only.
    **Performance:** Responses are cached for identical requests.
    """
    try:
        response = await gateway_service.process_request(request)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """🔍 Service health check"""
    return await gateway_service.get_health_status()

@app.get("/providers")
async def get_providers(api_key: str = Depends(verify_api_key)):
    """📋 Get available LLM providers and models"""
    
    openai_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"] if config.OPENAI_API_KEY else []
    anthropic_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229"] if config.ANTHROPIC_API_KEY else []
    
    return {
        "providers": {
            "openai": {
                "available": bool(config.OPENAI_API_KEY),
                "models": openai_models
            },
            "anthropic": {
                "available": bool(config.ANTHROPIC_API_KEY),
                "models": anthropic_models
            }
        },
        "rate_limits": {
            "per_minute": config.RATE_LIMIT_PER_MINUTE
        }
    }

@app.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)):
    """📊 Get usage statistics"""
    return {
        "cache_size": len(gateway_service.request_cache),
        "rate_limiter_entries": len(gateway_service.rate_limiter),
        "uptime": time.time() - gateway_service.start_time,
        "providers_configured": {
            "openai": bool(config.OPENAI_API_KEY),
            "anthropic": bool(config.ANTHROPIC_API_KEY)
        }
    }

@app.delete("/cache")
async def clear_cache(api_key: str = Depends(verify_api_key)):
    """🧹 Clear response cache"""
    cache_size = len(gateway_service.request_cache)
    gateway_service.request_cache.clear()
    return {"message": f"Cleared {cache_size} cached responses"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "LLM Gateway",
        "version": "1.0.0",
        "status": "running",
        "description": "Secure gateway for external LLM APIs",
        "endpoints": {
            "generate": "/generate",
            "health": "/health", 
            "providers": "/providers",
            "stats": "/stats"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
