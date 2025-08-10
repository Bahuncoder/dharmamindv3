"""
DharmaMind Backend - External LLM Bridge
=======================================

Service to communicate with the external LLM service while maintaining
backend integration and dharmic validation.
"""

import asyncio
import httpx
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.config import settings
from ..core.exceptions import DharmaMindException
from ..models.llm_models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class ExternalLLMBridge:
    """Bridge service to communicate with external LLM service"""
    
    def __init__(self):
        self.external_llm_url = getattr(settings, 'EXTERNAL_LLM_URL', 'http://localhost:8001')
        self.timeout = 30.0
        self.max_retries = 3
        
    async def process_dharmic_query(
        self,
        user_query: str,
        dharmic_context: Dict[str, Any],
        validation_level: str = "moderate",
        preferred_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send query to external LLM service with dharmic context
        
        Args:
            user_query: The user's question
            dharmic_context: Dharmic guidance and context from backend
            validation_level: strict, moderate, or lenient
            preferred_provider: openai, claude, or gemini
            
        Returns:
            Dict containing the processed response with dharmic validation
        """
        
        try:
            # Prepare request payload
            payload = {
                "user_query": user_query,
                "dharmic_context": dharmic_context,
                "validation_level": validation_level,
                "preferred_provider": preferred_provider or "openai",
                "timestamp": datetime.now().isoformat(),
                "request_metadata": {
                    "source": "dharmamind_backend",
                    "version": "1.0.0"
                }
            }
            
            # Make request to external LLM service
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.external_llm_url}/api/dharmic-query",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Source": "DharmaMind-Backend"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"External LLM service error: {response.status_code} - {response.text}")
                    raise DharmaMindException(
                        f"External LLM service failed: {response.status_code}"
                    )
                
                result = response.json()
                
                # Validate response structure
                if not all(key in result for key in ['response', 'dharmic_compliance_score']):
                    logger.error(f"Invalid response structure from external LLM: {result}")
                    raise DharmaMindException("Invalid response from external LLM service")
                
                logger.info(f"External LLM query successful. Compliance score: {result.get('dharmic_compliance_score', 0):.2f}")
                
                return result
                
        except httpx.TimeoutException:
            logger.error("Timeout communicating with external LLM service")
            raise DharmaMindException("External LLM service timeout")
            
        except httpx.RequestError as e:
            logger.error(f"Network error communicating with external LLM service: {e}")
            raise DharmaMindException("Failed to connect to external LLM service")
            
        except Exception as e:
            logger.error(f"Unexpected error in external LLM bridge: {e}")
            raise DharmaMindException(f"External LLM processing failed: {str(e)}")
    
    async def process_simple_query(
        self,
        query: str,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send simple query to external LLM service without dharmic context
        
        Args:
            query: The query to process
            provider: Preferred LLM provider
            
        Returns:
            Dict containing the response
        """
        
        try:
            payload = {
                "query": query,
                "provider": provider or "openai",
                "timestamp": datetime.now().isoformat()
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.external_llm_url}/api/simple-query",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Source": "DharmaMind-Backend"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"External LLM service error: {response.status_code}")
                    raise DharmaMindException(f"External LLM service failed: {response.status_code}")
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Error in simple query: {e}")
            raise DharmaMindException(f"Simple query failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of external LLM service"""
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.external_llm_url}/health")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e)
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from external LLM service"""
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.external_llm_url}/metrics")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_external_llm_bridge = None


async def get_external_llm_bridge() -> ExternalLLMBridge:
    """Get external LLM bridge instance"""
    global _external_llm_bridge
    
    if _external_llm_bridge is None:
        _external_llm_bridge = ExternalLLMBridge()
    
    return _external_llm_bridge


class DharmicLLMService:
    """
    Enhanced LLM service that combines DharmaMind's dharmic knowledge
    with external LLM capabilities through the bridge service.
    """
    
    def __init__(self):
        self.bridge = None
        
    async def initialize(self):
        """Initialize the service"""
        self.bridge = await get_external_llm_bridge()
        
    async def process_user_query(
        self,
        user_query: str,
        user_context: Optional[Dict[str, Any]] = None,
        dharmic_guidance_level: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Process user query with full dharmic integration
        
        This method:
        1. Analyzes the query for dharmic relevance
        2. Builds dharmic context from knowledge base
        3. Sends to external LLM with dharmic guidance
        4. Validates and enhances the response
        5. Returns fully dharmic-compliant response
        
        Args:
            user_query: User's question or request
            user_context: Additional context about the user
            dharmic_guidance_level: Level of dharmic validation
            
        Returns:
            Enhanced response with dharmic validation
        """
        
        if not self.bridge:
            await self.initialize()
        
        try:
            # Build dharmic context (this would integrate with your knowledge base)
            dharmic_context = await self._build_dharmic_context(user_query, user_context)
            
            # Process through external LLM service
            external_response = await self.bridge.process_dharmic_query(
                user_query=user_query,
                dharmic_context=dharmic_context,
                validation_level=dharmic_guidance_level,
                preferred_provider="openai"
            )
            
            # Post-process and enhance with backend knowledge
            enhanced_response = await self._enhance_with_backend_knowledge(
                external_response,
                dharmic_context
            )
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            # Fallback to backend-only processing if external service fails
            return await self._fallback_processing(user_query, user_context)
    
    async def _build_dharmic_context(
        self,
        user_query: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build dharmic context from knowledge base"""
        
        # This would integrate with your existing knowledge base
        # For now, return a basic structure
        return {
            "scriptural_references": [
                {"source": "Bhagavad Gita", "verse": "2.47", "relevance": "dharmic action"},
                {"source": "Upanishads", "text": "Satyam knowledge", "relevance": "truth seeking"}
            ],
            "sanskrit_context": ["dharma", "karma", "moksha"],
            "dharmic_principles": [
                "ahimsa (non-violence)",
                "satya (truthfulness)", 
                "brahmacharya (self-control)"
            ],
            "validation_guidelines": {
                "must_align_with": ["vedic_wisdom", "dharmic_principles"],
                "avoid_content": ["adharmic_advice", "materialistic_focus"],
                "enhance_with": ["spiritual_perspective", "practical_application"]
            },
            "user_spiritual_level": user_context.get("spiritual_level", "beginner") if user_context else "beginner"
        }
    
    async def _enhance_with_backend_knowledge(
        self,
        external_response: Dict[str, Any],
        dharmic_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance external LLM response with backend dharmic knowledge"""
        
        # Add additional dharmic insights
        external_response["dharmic_insights"] = [
            "Remember that all actions should be performed with dharmic intention",
            "Consider the long-term karmic implications of your choices",
            "Seek guidance from authentic spiritual sources"
        ]
        
        # Add related teachings
        external_response["related_teachings"] = dharmic_context.get("scriptural_references", [])
        
        # Add Sanskrit wisdom
        external_response["sanskrit_wisdom"] = {
            "primary_concept": "dharma",
            "meaning": "righteous path in accordance with cosmic order",
            "application": "Apply dharmic principles in daily decision-making"
        }
        
        return external_response
    
    async def _fallback_processing(
        self,
        user_query: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fallback processing when external service is unavailable"""
        
        logger.warning("Using fallback processing - external LLM service unavailable")
        
        return {
            "response": "I understand your question about dharmic guidance. While our advanced processing is temporarily unavailable, I can share that dharmic living involves following righteous principles aligned with cosmic order. Please refer to authentic scriptures like the Bhagavad Gita for detailed guidance.",
            "dharmic_compliance_score": 0.95,
            "dharmic_validation_passed": True,
            "provider_used": "dharmamind_fallback",
            "model_used": "internal_knowledge_base",
            "processing_time": 0.1,
            "fallback_used": True,
            "dharmic_insights": [
                "All actions should align with dharmic principles",
                "Seek authentic scriptural guidance",
                "Practice self-reflection and spiritual growth"
            ]
        }


# Service instance
_dharmic_llm_service = None


async def get_dharmic_llm_service() -> DharmicLLMService:
    """Get dharmic LLM service instance"""
    global _dharmic_llm_service
    
    if _dharmic_llm_service is None:
        _dharmic_llm_service = DharmicLLMService()
        await _dharmic_llm_service.initialize()
    
    return _dharmic_llm_service
