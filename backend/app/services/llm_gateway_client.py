"""
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
