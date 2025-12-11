"""
🕉️ External LLM Integration Routes

API endpoints for processing external LLM responses (ChatGPT, Claude, etc.)
through the complete DharmaMind dharmic backend system.

Features:
- ChatGPT API integration with dharmic processing
- Claude API integration with spiritual enhancement
- Other LLM provider support
- Subscription-aware processing
- Dharmic alignment and validation
- Spiritual module integration

All external LLM responses are processed through:
1. Spiritual content analysis
2. Dharmic module routing
3. Philosophical framework application
4. Scriptural reference addition
5. Final dharmic response generation

May all AI wisdom be transformed into dharmic guidance 🕉️
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import asyncio
from datetime import datetime

from ..models.subscription import SubscriptionTier
from ..services.dharmic_llm_processor import get_dharmic_llm_processor, DharmicProcessingMode, DharmicResponse
from ..services.subscription_service import SubscriptionService
from ..services.llm_gateway_client import get_llm_gateway_client
from ..routes.auth import get_current_user
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dharmic-llm", tags=["Dharmic LLM Integration"])

# ===============================
# REQUEST/RESPONSE MODELS
# ===============================

class ExternalLLMRequest(BaseModel):
    """Request for external LLM with dharmic processing"""
    query: str = Field(..., description="User query")
    provider: str = Field(..., description="LLM provider (openai, anthropic, huggingface)")
    model: Optional[str] = Field(default=None, description="Specific model (gpt-4, claude-3, etc.)")
    processing_mode: Optional[DharmicProcessingMode] = Field(default=None, description="Dharmic processing level")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="Additional user context")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens for response")
    temperature: Optional[float] = Field(default=0.7, description="Response creativity (0.0-1.0)")

class ChatGPTRequest(BaseModel):
    """Specific ChatGPT request"""
    query: str = Field(..., description="User query")
    model: str = Field(default="gpt-4", description="GPT model to use")
    processing_mode: Optional[DharmicProcessingMode] = Field(default=None, description="Dharmic processing level")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="User context")
    max_tokens: Optional[int] = Field(default=1000, description="Max tokens")
    temperature: Optional[float] = Field(default=0.7, description="Temperature")

class ClaudeRequest(BaseModel):
    """Specific Claude request"""
    query: str = Field(..., description="User query")
    model: str = Field(default="claude-3-opus-20240229", description="Claude model")
    processing_mode: Optional[DharmicProcessingMode] = Field(default=None, description="Dharmic processing level")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="User context")
    max_tokens: Optional[int] = Field(default=1000, description="Max tokens")

class DharmicLLMResponse(BaseModel):
    """Dharmic LLM response"""
    success: bool
    original_response: str
    dharmic_response: str
    spiritual_insights: List[str]
    scriptural_references: List[Dict[str, str]]
    dharmic_alignment_score: float
    processing_mode: DharmicProcessingMode
    modules_used: List[str]
    subscription_tier: SubscriptionTier
    provider: str
    model: str
    metadata: Dict[str, Any]

# ===============================
# EXTERNAL LLM GATEWAY CLIENT
# ===============================

# Use the separate LLM Gateway service instead of direct API calls
llm_gateway_client = get_llm_gateway_client()

# ===============================
# UTILITY FUNCTIONS  
# ===============================

async def get_user_subscription_tier(current_user: Dict[str, Any]) -> SubscriptionTier:
    """Get user's subscription tier"""
    subscription_service = SubscriptionService()
    user_subscription = await subscription_service.get_active_subscription(current_user["id"])
    
    if user_subscription:
        return user_subscription.tier
    return SubscriptionTier.FREE

# ===============================
# API ROUTES
# ===============================

@router.post("/chatgpt", response_model=DharmicLLMResponse)
async def process_chatgpt_with_dharmic_enhancement(
    request: ChatGPTRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    🤖➡️🕉️ Get ChatGPT response via gateway and process through dharmic system
    
    Flow:
    1. Send request to LLM Gateway service
    2. Gateway calls ChatGPT API
    3. Get raw AI response from gateway
    4. Process through complete dharmic backend
    5. Return enhanced dharmic wisdom
    """
    try:
        # Get user subscription tier
        subscription_tier = await get_user_subscription_tier(current_user)
        
        # Get dharmic processor
        dharmic_processor = await get_dharmic_llm_processor()
        
        # Step 1: Get ChatGPT response via gateway
        logger.info(f"🤖 Getting ChatGPT response via gateway for user {current_user['id']}")
        
        system_prompt = request.system_prompt or (
            "You are a helpful AI assistant. Provide clear, accurate, and helpful responses. "
            "Focus on being informative while being respectful of all perspectives."
        )
        
        # Use LLM Gateway client instead of direct API call
        gateway_response = await llm_gateway_client.get_chatgpt_response(
            query=request.query,
            model=request.model,
            system_prompt=system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            user_id=current_user["id"]
        )
        
        if not gateway_response.success:
            raise HTTPException(status_code=502, detail="Failed to get response from LLM Gateway")
        
        logger.info(f"✅ ChatGPT response received via gateway: {len(gateway_response.content)} characters")
        
        # Step 2: Process through dharmic system
        logger.info(f"🕉️ Processing through dharmic system...")
        
        dharmic_response = await dharmic_processor.process_external_llm_response(
            external_response=gateway_response.content,
            original_query=request.query,
            user_id=current_user["id"],
            subscription_tier=subscription_tier,
            processing_mode=request.processing_mode,
            user_context=request.user_context
        )
        
        logger.info(f"✅ Dharmic processing complete. Alignment score: {dharmic_response.dharmic_alignment_score}")
        
        return DharmicLLMResponse(
            success=True,
            original_response=dharmic_response.original_response,
            dharmic_response=dharmic_response.dharmic_response,
            spiritual_insights=dharmic_response.spiritual_insights,
            scriptural_references=dharmic_response.scriptural_references,
            dharmic_alignment_score=dharmic_response.dharmic_alignment_score,
            processing_mode=dharmic_response.processing_mode,
            modules_used=dharmic_response.modules_used,
            subscription_tier=dharmic_response.subscription_tier,
            provider="openai",
            model=request.model,
            metadata={
                **dharmic_response.metadata,
                "gateway_response_time": gateway_response.response_time,
                "gateway_cached": gateway_response.cached,
                "gateway_request_id": gateway_response.request_id
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error in ChatGPT dharmic processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/claude", response_model=DharmicLLMResponse)
async def process_claude_with_dharmic_enhancement(
    request: ClaudeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    🤖➡️🕉️ Get Claude response via gateway and process through dharmic system
    """
    try:
        # Get user subscription tier
        subscription_tier = await get_user_subscription_tier(current_user)
        
        # Get dharmic processor
        dharmic_processor = await get_dharmic_llm_processor()
        
        # Step 1: Get Claude response via gateway
        logger.info(f"🤖 Getting Claude response via gateway for user {current_user['id']}")
        
        system_prompt = request.system_prompt or (
            "You are Claude, a helpful AI assistant created by Anthropic. "
            "Provide thoughtful, nuanced responses that consider multiple perspectives."
        )
        
        # Use LLM Gateway client
        gateway_response = await llm_gateway_client.get_claude_response(
            query=request.query,
            model=request.model,
            system_prompt=system_prompt,
            max_tokens=request.max_tokens,
            user_id=current_user["id"]
        )
        
        if not gateway_response.success:
            raise HTTPException(status_code=502, detail="Failed to get response from LLM Gateway")
        
        logger.info(f"✅ Claude response received via gateway: {len(gateway_response.content)} characters")
        
        # Step 2: Process through dharmic system
        logger.info(f"🕉️ Processing through dharmic system...")
        
        dharmic_response = await dharmic_processor.process_external_llm_response(
            external_response=gateway_response.content,
            original_query=request.query,
            user_id=current_user["id"],
            subscription_tier=subscription_tier,
            processing_mode=request.processing_mode,
            user_context=request.user_context
        )
        
        logger.info(f"✅ Dharmic processing complete. Alignment score: {dharmic_response.dharmic_alignment_score}")
        
        return DharmicLLMResponse(
            success=True,
            original_response=dharmic_response.original_response,
            dharmic_response=dharmic_response.dharmic_response,
            spiritual_insights=dharmic_response.spiritual_insights,
            scriptural_references=dharmic_response.scriptural_references,
            dharmic_alignment_score=dharmic_response.dharmic_alignment_score,
            processing_mode=dharmic_response.processing_mode,
            modules_used=dharmic_response.modules_used,
            subscription_tier=dharmic_response.subscription_tier,
            provider="anthropic",
            model=request.model,
            metadata={
                **dharmic_response.metadata,
                "gateway_response_time": gateway_response.response_time,
                "gateway_cached": gateway_response.cached,
                "gateway_request_id": gateway_response.request_id
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error in Claude dharmic processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-responses")
async def compare_multiple_llm_responses(
    query: str,
    providers: List[str] = ["openai", "anthropic"],
    processing_mode: Optional[DharmicProcessingMode] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    🔄 Compare responses from multiple LLMs processed through dharmic system via gateway
    """
    try:
        # Get user subscription tier
        subscription_tier = await get_user_subscription_tier(current_user)
        
        # Get dharmic processor
        dharmic_processor = await get_dharmic_llm_processor()
        
        # Check available providers from gateway
        available_providers = await llm_gateway_client.get_available_providers()
        
        results = {}
        
        # Process each provider
        for provider in providers:
            try:
                if provider == "openai" and available_providers.get("providers", {}).get("openai", {}).get("available"):
                    gateway_response = await llm_gateway_client.get_chatgpt_response(
                        query=query, user_id=current_user["id"]
                    )
                elif provider == "anthropic" and available_providers.get("providers", {}).get("anthropic", {}).get("available"):
                    gateway_response = await llm_gateway_client.get_claude_response(
                        query=query, user_id=current_user["id"]
                    )
                else:
                    results[provider] = {"error": f"Provider {provider} not available"}
                    continue
                
                if not gateway_response.success:
                    results[provider] = {"error": f"Gateway error for {provider}"}
                    continue
                
                # Process through dharmic system
                dharmic_response = await dharmic_processor.process_external_llm_response(
                    external_response=gateway_response.content,
                    original_query=query,
                    user_id=current_user["id"],
                    subscription_tier=subscription_tier,
                    processing_mode=processing_mode
                )
                
                results[provider] = {
                    "success": True,
                    "original_response": dharmic_response.original_response,
                    "dharmic_response": dharmic_response.dharmic_response,
                    "alignment_score": dharmic_response.dharmic_alignment_score,
                    "modules_used": dharmic_response.modules_used,
                    "processing_mode": dharmic_response.processing_mode,
                    "gateway_response_time": gateway_response.response_time,
                    "gateway_cached": gateway_response.cached
                }
                
            except Exception as e:
                results[provider] = {"error": str(e)}
        
        return {
            "query": query,
            "subscription_tier": subscription_tier,
            "results": results,
            "comparison": {
                "best_alignment": max(
                    (k for k, v in results.items() if v.get("success")),
                    key=lambda k: results[k].get("alignment_score", 0),
                    default=None
                )
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error in LLM comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/subscription-features")
async def get_dharmic_processing_features(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    📋 Get available dharmic processing features for user's subscription
    """
    try:
        subscription_tier = await get_user_subscription_tier(current_user)
        
        features = {
            SubscriptionTier.FREE: {
                "processing_mode": "light",
                "monthly_limit": 50,
                "features": [
                    "Basic dharmic alignment",
                    "Single spiritual insight",
                    "Basic guidance"
                ]
            },
            SubscriptionTier.PRO: {
                "processing_mode": "standard",
                "monthly_limit": -1,  # Unlimited
                "features": [
                    "Full spiritual module processing",
                    "Multiple spiritual insights",
                    "Practical dharmic guidance",
                    "Module coordination"
                ]
            },
            SubscriptionTier.MAX: {
                "processing_mode": "deep",
                "monthly_limit": -1,
                "features": [
                    "Deep philosophical framework",
                    "Darshana engine integration",
                    "Comprehensive spiritual analysis",
                    "Life stage guidance",
                    "Advanced module coordination"
                ]
            },
            SubscriptionTier.ENTERPRISE: {
                "processing_mode": "premium", 
                "monthly_limit": -1,
                "features": [
                    "Premium spiritual processing",
                    "Scriptural references",
                    "Advanced philosophical analysis",
                    "Complete system integration",
                    "Priority processing",
                    "Custom spiritual insights"
                ]
            }
        }
        
        return {
            "current_tier": subscription_tier,
            "available_features": features[subscription_tier],
            "all_tiers": features
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting subscription features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def dharmic_llm_health_check():
    """
    🔍 Health check for dharmic LLM processing system
    """
    try:
        dharmic_processor = await get_dharmic_llm_processor()
        
        # Check LLM Gateway availability
        gateway_health = await llm_gateway_client.health_check()
        gateway_providers = await llm_gateway_client.get_available_providers()
        
        health_status = {
            "dharmic_processor": await dharmic_processor.health_check(),
            "llm_gateway": gateway_health,
            "gateway_providers": gateway_providers.get("providers", {}),
            "spiritual_modules": True,  # Always available
            "darshana_engine": True,    # Always available
            "subscription_service": True # Always available
        }
        
        overall_health = all([
            health_status["dharmic_processor"],
            health_status["llm_gateway"],
            health_status["spiritual_modules"],
            health_status["darshana_engine"]
        ])
        
        return {
            "status": "healthy" if overall_health else "degraded",
            "services": health_status,
            "gateway_info": {
                "connected": gateway_health,
                "available_providers": list(gateway_providers.get("providers", {}).keys())
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
