"""
🕉️ DharmaMind Backend LLM Router
==============================

SIMPLE router that:
1. Authenticates user
2. Checks subscription 
3. Routes to DharmaLLM
4. Tracks usage
5. Returns response

This is NOT a chat interface - just a secure gateway to DharmaLLM.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import models
from ..models.chat import ChatRequest, ChatResponse
from ..models.auth import UserProfile

# Import services  
from ..services.auth_service import get_current_user
from ..services.subscription_service import get_subscription_service
import httpx

logger = logging.getLogger(__name__)
security = HTTPBearer()

# Simple router - just route to DharmaLLM with auth
router = APIRouter(prefix="/api/v1", tags=["llm-routing"])

@router.post("/chat", response_model=ChatResponse)
async def route_to_dharmallm(
    request: ChatRequest,
    current_user: UserProfile = Depends(get_current_user)
):
    """
    Route authenticated chat to DharmaLLM service
    This is just a secure gateway - UI stays in frontend
    """
    try:
        # 1. Check user subscription 
        subscription_service = get_subscription_service()
        usage_result = await subscription_service.check_and_increment_usage(
            user_id=current_user.user_id,
            feature="chat_messages",
            count=1
        )
        
        if not usage_result.allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Usage limit reached: {usage_result.message}"
            )
        
        # 2. Route to DharmaLLM with user context
        dharmallm_url = "http://localhost:8001"
        
        user_context = {
            "user_id": current_user.user_id,
            "subscription": usage_result.subscription.plan_type,
            "authenticated": True
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            dharmallm_response = await client.post(
                f"{dharmallm_url}/api/v1/chat",
                json={
                    "message": request.message,
                    "session_id": request.conversation_id,
                    "user_id": current_user.user_id,
                    "user_context": user_context,
                    "temperature": 0.8,
                    "max_tokens": 1024
                },
                headers={"Content-Type": "application/json"}
            )
            
            if dharmallm_response.status_code == 200:
                result = dharmallm_response.json()
                
                # 3. Add backend metadata and return to frontend
                return ChatResponse(
                    message=result.get("response", ""),
                    conversation_id=result.get("session_id", request.conversation_id or f"chat_{int(datetime.now().timestamp())}"),
                    message_id=f"msg_{int(datetime.now().timestamp())}",
                    timestamp=datetime.now(),
                    confidence=result.get("confidence", 0.8),
                    dharmic_alignment=result.get("dharmic_alignment", 0.85),
                    processing_time=result.get("processing_time", 0.1),
                    model_used="DharmaLLM",
                    modules_used=["DharmaLLM-AI"],
                    evaluation_details={
                        "authenticated": True,
                        "subscription": usage_result.subscription.plan_type,
                        "usage_remaining": usage_result.remaining,
                        "service_source": "dharmallm_via_backend"
                    }
                )
            else:
                raise HTTPException(
                    status_code=502,
                    detail="DharmaLLM service unavailable"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error routing to DharmaLLM: {e}")
        
        # Fallback response
        return ChatResponse(
            message=f"🕉️ I understand your message: '{request.message}'. Due to a temporary service issue, I'm providing this basic response. Please try again shortly for full DharmaLLM wisdom.",
            conversation_id=request.conversation_id or f"fallback_{int(datetime.now().timestamp())}",
            message_id=f"fallback_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            confidence=0.7,
            dharmic_alignment=0.8,
            processing_time=0.1,
            model_used="Backend-Fallback",
            modules_used=["Fallback"],
            evaluation_details={
                "authenticated": True,
                "subscription": current_user.subscription_type,
                "fallback_reason": str(e),
                "service_source": "backend_fallback"
            }
        )

@router.get("/chat/health")
async def chat_routing_health():
    """Check if chat routing is working"""
    try:
        # Test DharmaLLM connectivity
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8001/health")
            dharmallm_status = response.status_code == 200
    except:
        dharmallm_status = False
    
    return {
        "status": "healthy",
        "backend_router": "operational", 
        "dharmallm_connectivity": dharmallm_status,
        "timestamp": datetime.now().isoformat()
    }