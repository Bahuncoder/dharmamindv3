"""
LLM Gateway Client for DharmaMind platform

Client for interacting with external LLM gateway services
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"

class LLMModel(str, Enum):
    """Available LLM models"""
    GPT_4 = "gpt-4"
    GPT_3_5 = "gpt-3.5-turbo"
    CLAUDE_3 = "claude-3-opus"
    CLAUDE_2 = "claude-2"
    GEMINI_PRO = "gemini-pro"
    COMMAND_R = "command-r"

class LLMRequest(BaseModel):
    """Request to LLM gateway"""
    model: LLMModel = Field(..., description="Model to use")
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature setting")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens")
    stream: bool = Field(default=False, description="Stream response")
    
    # DharmaMind specific
    dharmic_mode: bool = Field(default=True, description="Apply dharmic filtering")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="User context")

class LLMResponse(BaseModel):
    """Response from LLM gateway"""
    model: str = Field(..., description="Model used")
    content: str = Field(..., description="Response content")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage")
    
    # Metadata
    provider: str = Field(..., description="Provider used")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(default=None, description="Response confidence")
    dharmic_score: Optional[float] = Field(default=None, description="Dharmic alignment score")

class LLMGatewayClient:
    """Client for LLM Gateway service"""
    
    def __init__(self, gateway_url: str = "http://localhost:8001"):
        self.gateway_url = gateway_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.provider_models = {
            LLMProvider.OPENAI: [LLMModel.GPT_4, LLMModel.GPT_3_5],
            LLMProvider.ANTHROPIC: [LLMModel.CLAUDE_3, LLMModel.CLAUDE_2],
            LLMProvider.GOOGLE: [LLMModel.GEMINI_PRO],
            LLMProvider.COHERE: [LLMModel.COMMAND_R],
        }
        
    async def initialize(self) -> bool:
        """Initialize the LLM gateway client"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            )
            
            # Test connection
            if await self._health_check():
                logger.info("LLM Gateway client initialized successfully")
                return True
            else:
                logger.warning("LLM Gateway not available, using fallback mode")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM Gateway client: {e}")
            return False
    
    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        """Send chat completion request to LLM gateway"""
        try:
            if not self.session:
                await self.initialize()
            
            start_time = datetime.now()
            
            # Prepare request data
            request_data = {
                "model": request.model.value,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": request.stream,
                "dharmic_mode": request.dharmic_mode,
                "user_context": request.user_context
            }
            
            if self.session:
                try:
                    async with self.session.post(
                        f"{self.gateway_url}/chat/completions",
                        json=request_data
                    ) as response:
                        response_data = await response.json()
                        
                        if response.status == 200:
                            end_time = datetime.now()
                            response_time = (end_time - start_time).total_seconds() * 1000
                            
                            return LLMResponse(
                                model=response_data.get("model", request.model.value),
                                content=response_data.get("content", ""),
                                usage=response_data.get("usage", {}),
                                provider=response_data.get("provider", "unknown"),
                                response_time_ms=response_time,
                                confidence_score=response_data.get("confidence_score"),
                                dharmic_score=response_data.get("dharmic_score")
                            )
                        else:
                            logger.error(f"LLM Gateway error: {response.status}")
                            return await self._fallback_response(request)
                            
                except aiohttp.ClientError as e:
                    logger.error(f"Client error calling LLM Gateway: {e}")
                    return await self._fallback_response(request)
            else:
                return await self._fallback_response(request)
                
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return await self._fallback_response(request)
    
    async def _fallback_response(self, request: LLMRequest) -> LLMResponse:
        """Generate fallback response when gateway is unavailable"""
        fallback_responses = [
            "I understand you're seeking spiritual guidance. While our advanced AI is temporarily unavailable, I encourage you to take a moment for quiet reflection and mindfulness.",
            "In this moment of seeking, remember that wisdom often comes from within. Consider spending a few minutes in meditation or contemplative thought.",
            "The dharmic path teaches us patience. While I work to provide you with the best guidance, perhaps this is an opportunity for inner reflection and peace.",
            "Sometimes the most profound teachings come in moments of stillness. I invite you to find a quiet space and connect with your inner wisdom while our systems restore."
        ]
        
        import random
        content = random.choice(fallback_responses)
        
        return LLMResponse(
            model="fallback",
            content=content,
            usage={"total_tokens": len(content.split())},
            provider="dharmamind_fallback",
            response_time_ms=50,
            dharmic_score=0.8
        )
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models from gateway"""
        try:
            if not self.session:
                await self.initialize()
            
            if self.session:
                async with self.session.get(f"{self.gateway_url}/models") as response:
                    if response.status == 200:
                        return await response.json()
            
            # Fallback to default models
            return {
                provider.value: [model.value for model in models]
                for provider, models in self.provider_models.items()
            }
            
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return {}
    
    async def _health_check(self) -> bool:
        """Check if LLM gateway is healthy"""
        try:
            if not self.session:
                return False
                
            async with self.session.get(
                f"{self.gateway_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
                
        except Exception:
            return False
    
    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def streaming_chat(self, request: LLMRequest):
        """Stream chat completion response"""
        try:
            if not self.session:
                await self.initialize()
            
            request_data = {
                "model": request.model.value,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True,
                "dharmic_mode": request.dharmic_mode,
                "user_context": request.user_context
            }
            
            if self.session:
                async with self.session.post(
                    f"{self.gateway_url}/chat/stream",
                    json=request_data
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                yield line.decode('utf-8').strip()
                    else:
                        # Fallback to non-streaming
                        fallback = await self._fallback_response(request)
                        yield fallback.content
            else:
                # Fallback mode
                fallback = await self._fallback_response(request)
                yield fallback.content
                
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            fallback = await self._fallback_response(request)
            yield fallback.content
    
    async def health_check(self) -> Dict[str, Any]:
        """Get client health status"""
        gateway_healthy = await self._health_check()
        
        return {
            "status": "healthy" if gateway_healthy else "degraded",
            "client": "llm_gateway",
            "gateway_url": self.gateway_url,
            "gateway_healthy": gateway_healthy,
            "session_active": self.session is not None
        }

# Global client instance
_llm_gateway_client: Optional[LLMGatewayClient] = None

async def get_llm_gateway_client() -> LLMGatewayClient:
    """Get the global LLM gateway client instance"""
    global _llm_gateway_client
    
    if _llm_gateway_client is None:
        _llm_gateway_client = LLMGatewayClient()
        await _llm_gateway_client.initialize()
    
    return _llm_gateway_client

def create_llm_gateway_client(gateway_url: str = "http://localhost:8001") -> LLMGatewayClient:
    """Create a new LLM gateway client instance"""
    return LLMGatewayClient(gateway_url)