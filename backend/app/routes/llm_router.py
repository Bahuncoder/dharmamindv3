"""
DharmaMind Backend LLM Router - Enterprise Gateway
===================================================

Central API gateway that:
1. Authenticates users (JWT or API Key)
2. Validates subscriptions & quotas
3. Routes to DharmaLLM service
4. Tracks usage for billing
5. Applies rate limiting
6. Returns enriched responses

All frontends (Chat, Community, Brand) call this API.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx

from ..config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Router with versioned prefix
router = APIRouter(prefix="/api/v1", tags=["llm-gateway"])

# DharmaLLM service URL
DHARMALLM_URL = settings.DHARMALLM_SERVICE_URL if hasattr(settings, 'DHARMALLM_SERVICE_URL') else "http://localhost:8001"


# =====================
# Request/Response Models  
# =====================

class ChatRequest(BaseModel):
    """Chat request from frontend"""
    message: str
    conversation_id: Optional[str] = None
    rishi_id: Optional[str] = None
    temperature: Optional[float] = 0.8
    max_tokens: Optional[int] = 1024


class ChatResponse(BaseModel):
    """Chat response to frontend"""
    message: str
    conversation_id: str
    message_id: str
    timestamp: datetime
    rishi_id: Optional[str] = None
    rishi_name: Optional[str] = None
    confidence: float = 0.8
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class RishiInfo(BaseModel):
    """Rishi information"""
    id: str
    name: str
    domain: str
    description: str
    available: bool = True


# =====================
# Simple Auth (for dev)
# =====================

async def get_current_user(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Dict[str, Any]:
    """
    Simple auth for development.
    In production, validate JWT token properly.
    """
    # For development, allow anonymous access with limits
    if not authorization and not x_api_key:
        return {
            "user_id": "anonymous",
            "plan": "free",
            "authenticated": False,
            "usage_limit": 10  # Free tier limit
        }
    
    # If token provided, validate it (simplified for dev)
    if authorization:
        token = authorization.credentials
        # TODO: Proper JWT validation
        return {
            "user_id": f"user_{hash(token) % 10000}",
            "plan": "pro",
            "authenticated": True,
            "usage_limit": -1  # Unlimited for authenticated
        }
    
    # API key auth
    if x_api_key:
        return {
            "user_id": f"api_{hash(x_api_key) % 10000}",
            "plan": "enterprise",
            "authenticated": True,
            "usage_limit": -1
        }
    
    return {"user_id": "anonymous", "plan": "free", "authenticated": False, "usage_limit": 10}


# Simple in-memory usage tracking (use Redis in production)
usage_tracker: Dict[str, int] = {}


def check_usage_limit(user_id: str, limit: int) -> bool:
    """Check if user is within usage limits"""
    if limit == -1:
        return True  # Unlimited
    
    current = usage_tracker.get(user_id, 0)
    return current < limit


def increment_usage(user_id: str):
    """Increment usage counter"""
    usage_tracker[user_id] = usage_tracker.get(user_id, 0) + 1


# =====================
# Chat Endpoints
# =====================

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Main chat endpoint - routes to DharmaLLM
    
    Flow:
    1. Validate user authentication
    2. Check usage limits
    3. Forward to DharmaLLM
    4. Track usage
    5. Return enriched response
    """
    user_id = current_user["user_id"]
    usage_limit = current_user["usage_limit"]
    
    # Check usage limits
    if not check_usage_limit(user_id, usage_limit):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "usage_limit_exceeded",
                "message": "You have reached your message limit. Please upgrade your plan.",
                "plan": current_user["plan"],
                "upgrade_url": "/pricing"
            }
        )
    
    try:
        # Build DharmaLLM request
        dharmallm_payload = {
            "message": request.message,
            "session_id": request.conversation_id or f"session_{user_id}_{int(datetime.now().timestamp())}",
            "user_id": user_id,
            "rishi_id": request.rishi_id,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        # Call DharmaLLM service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{DHARMALLM_URL}/api/v1/chat",
                json=dharmallm_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Track usage
                increment_usage(user_id)
                
                # Return enriched response
                return ChatResponse(
                    message=result.get("response", result.get("message", "")),
                    conversation_id=result.get("session_id", dharmallm_payload["session_id"]),
                    message_id=f"msg_{int(datetime.now().timestamp() * 1000)}",
                    timestamp=datetime.now(),
                    rishi_id=result.get("rishi_id"),
                    rishi_name=result.get("rishi_name"),
                    confidence=result.get("confidence", 0.85),
                    usage={
                        "messages_used": usage_tracker.get(user_id, 1),
                        "limit": usage_limit,
                        "plan": current_user["plan"]
                    },
                    metadata={
                        "model": "DharmaLLM",
                        "authenticated": current_user["authenticated"],
                        "processing_time": result.get("processing_time", 0.1)
                    }
                )
            else:
                logger.error(f"DharmaLLM error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=502, detail="AI service temporarily unavailable")
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI service timeout")
    except httpx.ConnectError:
        # Fallback response when DharmaLLM is down
        increment_usage(user_id)
        return ChatResponse(
            message=f"Thank you for your message. Our AI service is currently being updated. Your question about '{request.message[:50]}...' has been noted. Please try again shortly.",
            conversation_id=request.conversation_id or f"fallback_{int(datetime.now().timestamp())}",
            message_id=f"fallback_{int(datetime.now().timestamp() * 1000)}",
            timestamp=datetime.now(),
            confidence=0.5,
            usage={
                "messages_used": usage_tracker.get(user_id, 1),
                "limit": usage_limit,
                "plan": current_user["plan"]
            },
            metadata={
                "model": "Fallback",
                "authenticated": current_user["authenticated"],
                "fallback_reason": "dharmallm_unavailable"
            }
        )


@router.get("/rishis")
async def get_rishis(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get available Rishis/Guides"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{DHARMALLM_URL}/rishis/available")
            
            if response.status_code == 200:
                return response.json()
    except:
        pass
    
    # Fallback Rishi list
    return {
        "rishis": [
            {"id": "vasishtha", "name": "Vasishtha", "domain": "Dharma & Ethics", "available": True},
            {"id": "vishwamitra", "name": "Vishwamitra", "domain": "Willpower & Achievement", "available": True},
            {"id": "bharadvaja", "name": "Bharadvaja", "domain": "Knowledge & Learning", "available": True},
            {"id": "gautama", "name": "Gautama", "domain": "Justice & Logic", "available": True},
            {"id": "jamadagni", "name": "Jamadagni", "domain": "Discipline & Focus", "available": True},
            {"id": "kashyapa", "name": "Kashyapa", "domain": "Creation & Nurturing", "available": True},
            {"id": "atri", "name": "Atri", "domain": "Balance & Harmony", "available": True},
            {"id": "agastya", "name": "Agastya", "domain": "Medicine & Healing", "available": True},
            {"id": "narada", "name": "Narada", "domain": "Devotion & Communication", "available": True},
        ],
        "total": 9
    }


@router.get("/rishi/{rishi_id}")
async def get_rishi_details(
    rishi_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detailed information about a specific Rishi"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{DHARMALLM_URL}/rishi/{rishi_id}")
            if response.status_code == 200:
                return response.json()
    except:
        pass
    
    raise HTTPException(status_code=404, detail=f"Rishi '{rishi_id}' not found")


# =====================
# Subscription Endpoints
# =====================

@router.get("/subscription/status")
async def get_subscription_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current user's subscription status"""
    user_id = current_user["user_id"]
    
    return {
        "user_id": user_id,
        "plan": current_user["plan"],
        "authenticated": current_user["authenticated"],
        "usage": {
            "messages_used": usage_tracker.get(user_id, 0),
            "limit": current_user["usage_limit"],
            "remaining": current_user["usage_limit"] - usage_tracker.get(user_id, 0) if current_user["usage_limit"] > 0 else -1
        },
        "features": {
            "rishis": current_user["plan"] != "free",
            "history": current_user["plan"] != "free",
            "api_access": current_user["plan"] == "enterprise",
            "priority_support": current_user["plan"] in ["pro", "enterprise"]
        }
    }


@router.get("/subscription/plans")
async def get_subscription_plans():
    """Get available subscription plans"""
    return {
        "plans": [
            {
                "id": "free",
                "name": "Free",
                "price": 0,
                "period": "forever",
                "features": ["10 messages/day", "Basic guidance", "Community access"],
                "limits": {"messages": 10, "rishis": 1}
            },
            {
                "id": "pro",
                "name": "Pro",
                "price": 19,
                "period": "month",
                "features": ["Unlimited messages", "All 9 Rishis", "Conversation history", "Priority support"],
                "limits": {"messages": -1, "rishis": 9}
            },
            {
                "id": "enterprise",
                "name": "Enterprise",
                "price": 99,
                "period": "month",
                "features": ["Everything in Pro", "API access", "Custom training", "Dedicated support", "SLA guarantee"],
                "limits": {"messages": -1, "rishis": 9, "api_calls": 10000}
            }
        ]
    }


# =====================
# Health Check
# =====================

@router.get("/health")
async def health_check():
    """Check gateway and DharmaLLM health"""
    dharmallm_status = "unknown"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{DHARMALLM_URL}/health")
            dharmallm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        dharmallm_status = "unavailable"
    
    return {
        "status": "healthy",
        "gateway": "operational",
        "dharmallm": dharmallm_status,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }
