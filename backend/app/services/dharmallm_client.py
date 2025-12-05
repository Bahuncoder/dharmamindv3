"""
🕉️ DharmaLLM Client Service
===========================

HTTP client for communicating with DharmaLLM AI service.
Handles all communication between backend and DharmaLLM microservice.

Features:
- Direct DharmaLLM API communication
- Rishi persona routing  
- Session management
- Error handling with fallbacks
- Response caching
"""

import logging
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..config import settings

logger = logging.getLogger(__name__)

class DharmaLLMClient:
    """HTTP client for DharmaLLM AI service communication"""
    
    def __init__(self):
        self.base_url = settings.DHARMALLM_SERVICE_URL or "http://localhost:8001"
        self.timeout = 30.0
        self.max_retries = 2
        
        # Rishi persona mappings
        self.rishi_personas = {
            "valmiki": {
                "name": "Sage Valmiki",
                "specialty": "Epic storytelling, Ramayana wisdom",
                "style": "narrative_wisdom",
                "context": "ancient_epics"
            },
            "vyasa": {
                "name": "Sage Vyasa", 
                "specialty": "Mahabharata, Vedic compilation",
                "style": "comprehensive_knowledge",
                "context": "vedic_wisdom"
            },
            "narada": {
                "name": "Sage Narada",
                "specialty": "Divine devotion, cosmic travel",
                "style": "devotional_guidance", 
                "context": "bhakti_path"
            },
            "vasishta": {
                "name": "Sage Vasishta",
                "specialty": "Yoga Vasishta, consciousness",
                "style": "philosophical_inquiry",
                "context": "advaita_vedanta"
            },
            "patanjali": {
                "name": "Sage Patanjali",
                "specialty": "Yoga philosophy, meditation",
                "style": "systematic_practice",
                "context": "raja_yoga"
            }
        }
        
        logger.info(f"✅ DharmaLLM client initialized - URL: {self.base_url}")

    async def chat_with_rishi(
        self, 
        message: str, 
        rishi: str, 
        user_id: str, 
        session_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Chat with specific Rishi persona"""
        
        try:
            # Get rishi configuration
            rishi_config = self.rishi_personas.get(rishi.lower())
            if not rishi_config:
                raise ValueError(f"Unknown rishi: {rishi}")
            
            # Prepare enhanced request
            request_data = {
                "message": message,
                "session_id": session_id or f"{user_id}_{rishi}_{datetime.now().timestamp()}",
                "user_id": user_id,
                "persona": rishi.lower(),
                "rishi_config": rishi_config,
                "user_context": user_context or {},
                "temperature": 0.8,
                "max_tokens": 1024,
                "dharmic_guidance_level": "high",
                "cultural_sensitivity": True
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/chat/rishi",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Enhance response with rishi metadata
                    result.update({
                        "rishi_name": rishi_config["name"],
                        "rishi_specialty": rishi_config["specialty"],
                        "guidance_style": rishi_config["style"],
                        "spiritual_context": rishi_config["context"],
                        "service_source": "dharmallm_ai",
                        "backend_processed": True
                    })
                    
                    logger.info(f"✅ Rishi {rishi} responded for user {user_id}")
                    return result
                    
                else:
                    raise httpx.HTTPStatusError(f"DharmaLLM returned {response.status_code}", request=response.request, response=response)
                    
        except Exception as e:
            logger.error(f"❌ Error communicating with Rishi {rishi}: {e}")
            
            # Fallback wisdom response
            return await self._get_fallback_rishi_response(rishi, message, user_id)

    async def general_chat(
        self, 
        message: str, 
        user_id: str, 
        session_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """General spiritual chat without specific rishi"""
        
        try:
            request_data = {
                "message": message,
                "session_id": session_id or f"{user_id}_general_{datetime.now().timestamp()}",
                "user_id": user_id,
                "user_context": user_context or {},
                "temperature": 0.7,
                "max_tokens": 1024,
                "dharmic_guidance_level": "balanced"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/chat",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    result.update({
                        "service_source": "dharmallm_ai",
                        "backend_processed": True,
                        "guidance_type": "general_spiritual"
                    })
                    
                    logger.info(f"✅ General chat response for user {user_id}")
                    return result
                    
                else:
                    raise httpx.HTTPStatusError(f"DharmaLLM returned {response.status_code}", request=response.request, response=response)
                    
        except Exception as e:
            logger.error(f"❌ Error in general chat: {e}")
            return await self._get_fallback_general_response(message, user_id)

    async def _get_fallback_rishi_response(self, rishi: str, message: str, user_id: str) -> Dict[str, Any]:
        """Fallback response when DharmaLLM is unavailable"""
        
        rishi_config = self.rishi_personas.get(rishi.lower(), {})
        rishi_name = rishi_config.get("name", f"Sage {rishi.title()}")
        
        wisdom_responses = {
            "valmiki": "🏹 As Sage Valmiki, I remind you that dharma is the foundation of all righteous action. Like in the Ramayana, face challenges with courage and devotion.",
            "vyasa": "📚 Sage Vyasa speaks: The Vedas teach us that knowledge without practice is incomplete. Seek wisdom through both study and righteous living.",
            "narada": "🎵 Narada's wisdom: Divine love is the highest path. Surrender your ego and sing the names of the Divine with pure devotion.",
            "vasishta": "🧘 From Yoga Vasishta: Reality is consciousness itself. What you seek is already within you - realize your true Self.",
            "patanjali": "🕉️ Patanjali teaches: Yoga is the cessation of fluctuations of the mind. Practice with dedication and detachment."
        }
        
        return {
            "response": wisdom_responses.get(rishi.lower(), f"🕉️ The wisdom of {rishi_name} guides you: Seek truth through righteous action and spiritual practice."),
            "rishi_name": rishi_name,
            "session_id": f"{user_id}_{rishi}_fallback_{int(datetime.now().timestamp())}",
            "message_id": f"fallback_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "confidence_score": 0.75,
            "dharmic_alignment": 0.9,
            "service_source": "backend_fallback",
            "backend_processed": True,
            "fallback_reason": "dharmallm_unavailable"
        }

    async def _get_fallback_general_response(self, message: str, user_id: str) -> Dict[str, Any]:
        """General fallback spiritual response"""
        
        return {
            "response": f"🕉️ The path of dharma guides us through all challenges. Your question '{message}' reflects a sincere seeking. May you find peace and wisdom on your spiritual journey.",
            "session_id": f"{user_id}_general_fallback_{int(datetime.now().timestamp())}",
            "message_id": f"fallback_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "confidence_score": 0.7,
            "dharmic_alignment": 0.85,
            "service_source": "backend_fallback",
            "backend_processed": True,
            "guidance_type": "general_spiritual",
            "fallback_reason": "dharmallm_unavailable"
        }

    async def get_available_rishis(self) -> List[Dict[str, Any]]:
        """Get list of available Rishi personas"""
        return [
            {
                "id": key,
                "name": config["name"], 
                "specialty": config["specialty"],
                "style": config["style"],
                "context": config["context"]
            }
            for key, config in self.rishi_personas.items()
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Check DharmaLLM service health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "dharmallm_service": "available",
                        "url": self.base_url
                    }
                else:
                    return {
                        "status": "degraded", 
                        "dharmallm_service": "responding_with_errors",
                        "url": self.base_url
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "dharmallm_service": "unavailable", 
                "error": str(e),
                "url": self.base_url,
                "fallback_available": True
            }

# Singleton instance
_dharmallm_client = None

def get_dharmallm_client() -> DharmaLLMClient:
    """Get singleton DharmaLLM client instance"""
    global _dharmallm_client
    if _dharmallm_client is None:
        _dharmallm_client = DharmaLLMClient()
    return _dharmallm_client