"""
<<<<<<< HEAD
LLM Gateway Client Service
=========================

Client for communicating with external LLM gateway services.
Temporary implementation for backward compatibility.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMGatewayClient:
    """Client for LLM gateway services"""
    
    def __init__(self):
        self.initialized = True
        self.gateway_url = "http://localhost:8003"  # Default gateway URL
        logger.info("LLM Gateway Client initialized")
    
    async def send_request(
        self,
        prompt: str,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send request to LLM gateway"""
        try:
            # Simulate gateway response
            response = {
                "response": f"Gateway processed: {prompt}",
                "model_used": model or "default_model",
                "processing_time": 0.5,
                "gateway_status": "simulated",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.debug(f"Processed gateway request for model: {model}")
            return response
            
        except Exception as e:
            logger.error(f"Error in LLM gateway request: {e}")
            return {
                "response": "Gateway service temporarily unavailable.",
                "error": str(e),
                "gateway_status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }

# Global client instance
_llm_gateway_client = None

def get_llm_gateway_client() -> LLMGatewayClient:
    """Get or create LLM gateway client instance"""
    global _llm_gateway_client
    
    if _llm_gateway_client is None:
        _llm_gateway_client = LLMGatewayClient()
    
    return _llm_gateway_client
=======
🔗 LLM Gateway Client

Client for communicating with the separate LLM Gateway service.
This allows the main DharmaMind backend to request external LLM responses
while keeping external API keys isolated in the gateway service.

Architecture:
Backend (this client) → LLM Gateway → External APIs (ChatGPT/Claude)

Benefits:
- 🔒 External API keys isolated from main backend
- 🚀 Independent gateway service scaling
- 🛡️ Clear security boundaries
- 🔧 Easy gateway maintenance without backend changes

May this bridge connect wisdom across all boundaries 🕉️
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx
from dataclasses import dataclass

from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class LLMGatewayResponse:
    """Response from LLM Gateway"""
    success: bool
    request_id: str
    provider: str
    model: str
    content: str
    usage: Dict[str, Any]
    response_time: float
    timestamp: str
    cached: bool = False

class LLMGatewayClient:
    """Client for communicating with LLM Gateway service"""
    
    def __init__(self):
        self.gateway_url = getattr(settings, 'LLM_GATEWAY_URL', 'http://localhost:8003')
        self.api_key = getattr(settings, 'LLM_GATEWAY_API_KEY', 'llm-gateway-secure-key-change-this-in-production')
        self.timeout = 120.0  # 2 minutes timeout for LLM responses
        
    async def generate_response(
        self,
        provider: str,
        model: str,
        query: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 0.7,
        user_id: Optional[str] = None
    ) -> LLMGatewayResponse:
        """
        Generate response from external LLM via gateway
        
        Args:
            provider: LLM provider ('openai' or 'anthropic')
            model: Model name (e.g., 'gpt-4', 'claude-3-opus-20240229')
            query: User query
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-2.0)
            user_id: User identifier for tracking
            
        Returns:
            LLMGatewayResponse with the generated content
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "provider": provider,
                "model": model,
                "query": query,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            if system_prompt:
                payload["system_prompt"] = system_prompt
            if user_id:
                payload["user_id"] = user_id
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.gateway_url}/generate",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 401:
                    raise Exception("Invalid LLM Gateway API key")
                elif response.status_code == 429:
                    raise Exception("LLM Gateway rate limit exceeded")
                elif response.status_code != 200:
                    raise Exception(f"LLM Gateway error: {response.status_code} - {response.text}")
                
                data = response.json()
                
                return LLMGatewayResponse(
                    success=data.get("success", False),
                    request_id=data.get("request_id", ""),
                    provider=data.get("provider", provider),
                    model=data.get("model", model),
                    content=data.get("content", ""),
                    usage=data.get("usage", {}),
                    response_time=data.get("response_time", 0.0),
                    timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
                    cached=data.get("cached", False)
                )
                
        except httpx.TimeoutException:
            logger.error("LLM Gateway request timed out")
            raise Exception("LLM Gateway request timed out")
        except httpx.ConnectError:
            logger.error("Failed to connect to LLM Gateway")
            raise Exception("LLM Gateway service unavailable")
        except Exception as e:
            logger.error(f"LLM Gateway client error: {e}")
            raise
    
    async def get_chatgpt_response(
        self,
        query: str,
        model: str = "gpt-4",
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        user_id: Optional[str] = None
    ) -> LLMGatewayResponse:
        """Get response from ChatGPT via gateway"""
        
        return await self.generate_response(
            provider="openai",
            model=model,
            query=query,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            user_id=user_id
        )
    
    async def get_claude_response(
        self,
        query: str,
        model: str = "claude-3-opus-20240229",
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        user_id: Optional[str] = None
    ) -> LLMGatewayResponse:
        """Get response from Claude via gateway"""
        
        return await self.generate_response(
            provider="anthropic",
            model=model,
            query=query,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.7,  # Claude doesn't use temperature the same way
            user_id=user_id
        )
    
    async def get_available_providers(self) -> Dict[str, Any]:
        """Get available LLM providers and models from gateway"""
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.gateway_url}/providers",
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Failed to get providers: {response.status_code}")
                    return {"providers": {}}
                    
        except Exception as e:
            logger.error(f"Error getting providers: {e}")
            return {"providers": {}}
    
    async def health_check(self) -> bool:
        """Check if LLM Gateway is healthy"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.gateway_url}/health",
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status") == "healthy"
                    
        except Exception as e:
            logger.warning(f"LLM Gateway health check failed: {e}")
            
        return False
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics from gateway"""
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.gateway_url}/stats",
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"Failed to get stats: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {"error": str(e)}

# Singleton instance
_llm_gateway_client = None

def get_llm_gateway_client() -> LLMGatewayClient:
    """Get singleton LLM Gateway client"""
    global _llm_gateway_client
    if _llm_gateway_client is None:
        _llm_gateway_client = LLMGatewayClient()
    return _llm_gateway_client
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
