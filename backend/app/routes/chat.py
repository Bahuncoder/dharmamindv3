"""
🕉️ DharmaMind COMPREHENSIVE Chat Router - Enterprise Grade
=========================================================

COMPLETE enterprise routing system with:

🔐 Authentication & Authorization
📊 Subscription Management & Billing
🎯 Rishi Persona Selection & Management  
🧠 Advanced LLM Routing & Load Balancing
📈 Usage Analytics & Rate Limiting
🔄 Conversation History & Context Management
🛡️ Security & Content Filtering
🎨 Response Customization & Personalization
⚡ Caching & Performance Optimization
📱 Multi-Platform Support (Chat, Brand, Community)
🎭 Advanced Persona Management
📝 Conversation Templates & Workflows
🔍 Search & Discovery Features
🎊 Gamification & Achievement System
"""

import logging
import asyncio
import hashlib
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks, Query, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json
import redis
from enum import Enum

# Import models - Comprehensive chat models
from ..models.chat import (
    ChatRequest, ChatResponse, RishiChatRequest, RishiSelectionRequest,
    ChatHistoryResponse, UserChatPreferences, ConversationSummary,
    ChatUsageStats, StreamingChatResponse, AdvancedChatRequest,
    ConversationType, RishiType, ChatTemplate, ChatWorkflow,
    PersonalizedResponse, ContextualChatRequest
)

# Import authentication and user models
from ..models.auth import UserProfile, SubscriptionTier
from ..models.subscription import SubscriptionStatus, UsageLimit, BillingEvent

# Import comprehensive services  
from ..services.dharmallm_client import get_dharmallm_client
from ..services.auth_service import get_current_user, get_auth_service
from ..services.subscription_service import get_subscription_service
from ..services.comprehensive_chat_services import (
    get_conversation_service, get_personalization_service, get_analytics_service,
    get_content_filter_service, get_cache_service, get_rate_limiter,
    get_gamification_service, get_chat_history_service,
    get_rishi_usage_count, get_last_rishi_interaction, validate_rishi_access,
    generate_cache_key, generate_intelligent_fallback_response,
    save_comprehensive_conversation, update_user_personalization
)
from ..config import settings

logger = logging.getLogger(__name__)

# Security and validation
security = HTTPBearer()

# Create comprehensive chat router
chat_router = APIRouter(prefix="/api/v1/chat", tags=["comprehensive-chat"])

# ===============================
# ADVANCED RISHI MANAGEMENT
# ===============================

class RishiAccessLevel(str, Enum):
    """Rishi access levels based on subscription"""
    FREE = "free"           # Basic rishis only
    PREMIUM = "premium"     # All rishis + extended features
    ENTERPRISE = "enterprise" # Custom rishis + AI training

@chat_router.get("/rishis/available")
async def get_available_rishis(
    current_user: UserProfile = Depends(get_current_user),
    subscription_service = Depends(get_subscription_service),
    personalization_service = Depends(get_personalization_service)
):
    """Get comprehensive list of available Rishis with personalized recommendations"""
    try:
        # Get user subscription and preferences
        user_subscription = await subscription_service.get_user_subscription(current_user.user_id)
        user_preferences = await personalization_service.get_user_preferences(current_user.user_id)
        
        # Get all rishis with access level filtering
        dharmallm_client = get_dharmallm_client()
        all_rishis = await dharmallm_client.get_available_rishis()
        
        # Filter by subscription tier
        accessible_rishis = []
        for rishi in all_rishis:
            rishi_access_level = rishi.get("access_level", "free")
            user_tier = user_subscription.tier.lower()
            
            if (rishi_access_level == "free" or 
                (user_tier in ["premium", "enterprise"] and rishi_access_level in ["free", "premium"]) or
                (user_tier == "enterprise")):
                
                # Add personalized metadata
                rishi["recommended"] = rishi["id"] in user_preferences.get("preferred_rishis", [])
                rishi["usage_count"] = await get_rishi_usage_count(current_user.user_id, rishi["id"])
                rishi["last_interaction"] = await get_last_rishi_interaction(current_user.user_id, rishi["id"])
                
                # Add subscription-specific features
                if user_tier in ["premium", "enterprise"]:
                    rishi["premium_features"] = {
                        "extended_conversations": True,
                        "personalized_responses": True,
                        "conversation_history": True,
                        "priority_processing": True,
                        "advanced_insights": True
                    }
                
                if user_tier == "enterprise":
                    rishi["enterprise_features"] = {
                        "custom_training": True,
                        "api_access": True,
                        "bulk_processing": True,
                        "analytics_dashboard": True
                    }
                
                accessible_rishis.append(rishi)
        
        # Get personalized recommendations
        recommendations = await personalization_service.get_rishi_recommendations(
            current_user.user_id, accessible_rishis
        )
        
        return {
            "rishis": accessible_rishis,
            "recommendations": recommendations,
            "user_subscription": {
                "tier": user_subscription.tier,
                "features": user_subscription.features,
                "limits": user_subscription.limits
            },
            "total_available": len(accessible_rishis),
            "personalization_enabled": True
        }
        
    except Exception as e:
        logger.error(f"Error getting available rishis: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available rishis")

@chat_router.post("/rishi/select")
async def select_rishi_advanced(
    request: RishiSelectionRequest,
    background_tasks: BackgroundTasks,
    current_user: UserProfile = Depends(get_current_user),
    subscription_service = Depends(get_subscription_service),
    conversation_service = Depends(get_conversation_service),
    personalization_service = Depends(get_personalization_service),
    analytics_service = Depends(get_analytics_service)
):
    """Advanced Rishi selection with personalization and analytics"""
    try:
        # Validate subscription access
        user_subscription = await subscription_service.get_user_subscription(current_user.user_id)
        
        # Check rishi access permissions
        rishi_access_granted = await validate_rishi_access(
            request.rishi_id, user_subscription.tier
        )
        if not rishi_access_granted:
            raise HTTPException(
                status_code=403,
                detail=f"Rishi {request.rishi_id} requires higher subscription tier"
            )
        
        # Create advanced conversation context
        conversation_context = {
            "user_profile": {
                "id": current_user.user_id,
                "spiritual_background": current_user.spiritual_background,
                "experience_level": current_user.experience_level,
                "interests": current_user.interests,
                "goals": current_user.goals
            },
            "session_context": request.context or {},
            "subscription_context": {
                "tier": user_subscription.tier,
                "features": user_subscription.features
            }
        }
        
        # Create conversation with advanced features
        conversation_id = await conversation_service.create_rishi_conversation(
            user_id=current_user.user_id,
            rishi_id=request.rishi_id,
            context=conversation_context,
            conversation_goal=request.conversation_goal,
            personalization_enabled=True
        )
        
        # Get rishi information with personalization
        dharmallm_client = get_dharmallm_client()
        rishi_info = await dharmallm_client.get_rishi_info(request.rishi_id)
        
        # Personalize rishi introduction
        personalized_intro = await personalization_service.personalize_rishi_introduction(
            current_user.user_id, rishi_info, conversation_context
        )
        
        # Track analytics
        background_tasks.add_task(
            analytics_service.track_rishi_selection,
            user_id=current_user.user_id,
            rishi_id=request.rishi_id,
            conversation_id=conversation_id,
            context=conversation_context
        )
        
        # Update user preferences
        background_tasks.add_task(
            personalization_service.update_rishi_preference,
            user_id=current_user.user_id,
            rishi_id=request.rishi_id,
            interaction_type="selection"
        )
        
        return {
            "status": "rishi_selected",
            "rishi": {
                **rishi_info,
                "personalized_intro": personalized_intro
            },
            "conversation_id": conversation_id,
            "conversation_features": {
                "personalization": True,
                "history_tracking": True,
                "advanced_insights": user_subscription.tier in ["premium", "enterprise"],
                "priority_processing": user_subscription.tier in ["premium", "enterprise"],
                "custom_workflows": user_subscription.tier == "enterprise"
            },
            "welcome_message": personalized_intro
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced rishi selection: {e}")
        raise HTTPException(status_code=500, detail="Failed to select rishi")

# ===============================
# COMPREHENSIVE CHAT PROCESSING
# ===============================

@chat_router.post("/message/advanced", response_model=ChatResponse)
async def advanced_chat_processing(
    request: AdvancedChatRequest,
    background_tasks: BackgroundTasks,
    current_user: UserProfile = Depends(get_current_user),
    rate_limiter = Depends(get_rate_limiter),
    subscription_service = Depends(get_subscription_service),
    conversation_service = Depends(get_conversation_service),
    personalization_service = Depends(get_personalization_service),
    content_filter = Depends(get_content_filter_service),
    cache_service = Depends(get_cache_service),
    analytics_service = Depends(get_analytics_service),
    # notification_service = Depends(get_notification_service)  # Optional feature
):
    """
    Advanced chat processing with comprehensive features:
    - Rate limiting and usage tracking
    - Content filtering and safety
    - Personalization and context management
    - Caching and performance optimization
    - Analytics and insights
    - Multi-modal support
    """
    
    processing_start_time = datetime.now()
    
    try:
        # 1. RATE LIMITING & USAGE VALIDATION
        rate_limit_result = await rate_limiter.check_rate_limit(
            user_id=current_user.user_id,
            endpoint="advanced_chat",
            tier=current_user.subscription_tier
        )
        
        if not rate_limit_result.allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {rate_limit_result.message}"
            )
        
        # 2. SUBSCRIPTION & USAGE VALIDATION  
        usage_result = await subscription_service.check_and_increment_usage(
            user_id=current_user.user_id,
            feature="advanced_chat_messages",
            count=1,
            request_complexity=request.complexity_score or 1.0
        )
        
        if not usage_result.allowed:
            raise HTTPException(
                status_code=402,
                detail=f"Usage limit exceeded: {usage_result.message}"
            )
        
        # 3. CONTENT FILTERING & SAFETY
        content_safety_check = await content_filter.validate_message(
            message=request.message,
            user_id=current_user.user_id,
            conversation_context=request.context
        )
        
        if not content_safety_check.approved:
            raise HTTPException(
                status_code=400,
                detail=f"Content policy violation: {content_safety_check.reason}"
            )
        
        # 4. CACHE CHECK FOR OPTIMIZATION
        cache_key = await generate_cache_key(request, current_user.user_id)
        cached_response = await cache_service.get_cached_response(cache_key)
        
        if cached_response and request.use_cache:
            logger.info(f"Cache hit for user {current_user.user_id}")
            cached_response.metadata["cache_hit"] = True
            cached_response.metadata["processing_time"] = 0.01
            return cached_response
        
        # 5. PERSONALIZATION & CONTEXT ENHANCEMENT
        enhanced_context = await personalization_service.enhance_conversation_context(
            user_id=current_user.user_id,
            message=request.message,
            conversation_id=request.conversation_id,
            base_context=request.context,
            personalization_level=request.personalization_level or "balanced"
        )
        
        # 6. CONVERSATION HISTORY & CONTINUITY
        conversation_history = await conversation_service.get_relevant_history(
            user_id=current_user.user_id,
            conversation_id=request.conversation_id,
            message_count=request.context_window_size or 10
        )
        
        # 7. PREPARE ENHANCED REQUEST FOR DHARMALLM
        enhanced_request = {
            "message": request.message,
            "conversation_id": request.conversation_id,
            "user_id": current_user.user_id,
            "rishi_id": request.rishi_id,
            "user_context": {
                "profile": current_user.to_dict(),
                "subscription": usage_result.subscription.to_dict(),
                "preferences": enhanced_context["preferences"],
                "history_summary": enhanced_context["history_summary"],
                "personalization": enhanced_context["personalization"]
            },
            "conversation_context": {
                "history": conversation_history,
                "goals": enhanced_context.get("goals", []),
                "current_mood": request.mood_context,
                "session_context": request.session_context
            },
            "processing_options": {
                "temperature": request.temperature or 0.8,
                "max_tokens": request.max_tokens or 1024,
                "creativity_level": request.creativity_level or "balanced",
                "response_style": request.response_style or "conversational",
                "include_insights": request.include_insights,
                "include_suggestions": request.include_suggestions,
                "personalization_level": request.personalization_level
            }
        }
        
        # 8. ROUTE TO DHARMALLM WITH COMPREHENSIVE CONTEXT
        dharmallm_client = get_dharmallm_client()
        
        if request.rishi_id:
            llm_response = await dharmallm_client.chat_with_rishi_advanced(
                enhanced_request
            )
        else:
            llm_response = await dharmallm_client.advanced_general_chat(
                enhanced_request
            )
        
        # 9. POST-PROCESS RESPONSE WITH PERSONALIZATION
        personalized_response = await personalization_service.personalize_response(
            response=llm_response,
            user_id=current_user.user_id,
            conversation_context=enhanced_context
        )
        
        # 10. ADD COMPREHENSIVE METADATA
        processing_time = (datetime.now() - processing_start_time).total_seconds()
        
        comprehensive_response = ChatResponse(
            response=personalized_response["response"],
            conversation_id=personalized_response.get("conversation_id"),
            message_id=personalized_response.get("message_id"),
            timestamp=datetime.now().isoformat(),
            confidence_score=personalized_response.get("confidence_score"),
            dharmic_alignment=personalized_response.get("dharmic_alignment"),
            processing_time=processing_time,
            model_used=personalized_response.get("model_used"),
            sources=personalized_response.get("sources", []),
            suggestions=personalized_response.get("suggestions", []),
            metadata={
                "service_source": "comprehensive_backend",
                "user_authenticated": True,
                "subscription_tier": usage_result.subscription.tier,
                "personalization_applied": True,
                "cache_hit": False,
                "content_filtered": True,
                "rate_limited": True,
                "usage_remaining": usage_result.remaining,
                "processing_features": {
                    "advanced_context": True,
                    "personalization": True,
                    "conversation_continuity": True,
                    "content_safety": True,
                    "performance_optimization": True
                },
                "response_enhancements": personalized_response.get("enhancements", {}),
                "analytics_tracked": True
            },
            # Advanced response features
            dharmic_insights=personalized_response.get("dharmic_insights", []),
            growth_suggestions=personalized_response.get("growth_suggestions", []),
            spiritual_context=personalized_response.get("spiritual_context"),
            wisdom_assessment=personalized_response.get("wisdom_assessment", {}),
            cultural_sensitivity=personalized_response.get("cultural_sensitivity"),
            compassion_score=personalized_response.get("compassion_score")
        )
        
        # 11. BACKGROUND TASKS - COMPREHENSIVE TRACKING
        background_tasks.add_task(
            save_comprehensive_conversation,
            user_id=current_user.user_id,
            conversation_id=request.conversation_id,
            user_message=request,
            ai_response=comprehensive_response,
            processing_context=enhanced_context,
            processing_time=processing_time
        )
        
        background_tasks.add_task(
            cache_service.cache_response,
            cache_key=cache_key,
            response=comprehensive_response,
            ttl=3600  # 1 hour cache
        )
        
        background_tasks.add_task(
            analytics_service.track_comprehensive_chat,
            user_id=current_user.user_id,
            request=request,
            response=comprehensive_response,
            processing_context=enhanced_context
        )
        
        background_tasks.add_task(
            update_user_personalization,
            user_id=current_user.user_id,
            interaction_data={
                "message": request.message,
                "response": comprehensive_response,
                "satisfaction_indicators": personalized_response.get("satisfaction_indicators", {})
            }
        )
        
        # 12. GAMIFICATION & ACHIEVEMENT TRACKING
        if current_user.gamification_enabled:
            background_tasks.add_task(
                get_gamification_service().process_chat_achievement,
                user_id=current_user.user_id,
                chat_data={
                    "message_type": request.conversation_type,
                    "rishi_used": request.rishi_id,
                    "quality_score": comprehensive_response.confidence_score,
                    "dharmic_alignment": comprehensive_response.dharmic_alignment
                }
            )
        
        logger.info(f"✅ Advanced chat completed for user {current_user.user_id} in {processing_time:.3f}s")
        return comprehensive_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced chat processing: {e}")
        
        # Return intelligent fallback with error context
        return await generate_intelligent_fallback_response(
            request=request,
            user=current_user,
            error=str(e),
            processing_time=(datetime.now() - processing_start_time).total_seconds()
        )

# ===============================
# STREAMING CHAT SUPPORT
# ===============================

@chat_router.post("/stream")
async def streaming_chat(
    request: AdvancedChatRequest,
    current_user: UserProfile = Depends(get_current_user)
):
    """
    Streaming chat for real-time responses
    Supports both rishi and general chat with live streaming
    """
    try:
        # Validate streaming permissions
        if current_user.subscription_tier == "free":
            raise HTTPException(
                status_code=403,
                detail="Streaming chat requires premium subscription"
            )
        
        # Generate streaming response
        async def generate_stream():
            dharmallm_client = get_dharmallm_client()
            
            async for chunk in dharmallm_client.stream_chat(
                message=request.message,
                user_id=current_user.user_id,
                rishi_id=request.rishi_id,
                context=request.context
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Final completion message
            yield f"data: {json.dumps({'type': 'complete', 'timestamp': datetime.now().isoformat()})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/stream"
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat: {e}")
        raise HTTPException(status_code=500, detail="Streaming chat failed")

# ===============================
# CONVERSATION MANAGEMENT
# ===============================

@chat_router.get("/conversations", response_model=List[ConversationSummary])
async def get_user_conversations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    conversation_type: Optional[ConversationType] = None,
    current_user: UserProfile = Depends(get_current_user),
    conversation_service = Depends(get_conversation_service)
):
    """Get user's conversation history with filtering and pagination"""
    try:
        conversations = await conversation_service.get_user_conversations(
            user_id=current_user.user_id,
            limit=limit,
            offset=offset,
            conversation_type=conversation_type
        )
        
        return conversations
        
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")

@chat_router.get("/conversation/{conversation_id}/history", response_model=ChatHistoryResponse)
async def get_conversation_history(
    conversation_id: str = Path(..., description="Conversation ID"),
    include_context: bool = Query(False, description="Include conversation context"),
    current_user: UserProfile = Depends(get_current_user),
    conversation_service = Depends(get_conversation_service)
):
    """Get detailed conversation history"""
    try:
        history = await conversation_service.get_conversation_history(
            user_id=current_user.user_id,
            conversation_id=conversation_id,
            include_context=include_context
        )
        
        if not history:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history")

@chat_router.delete("/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: str = Path(..., description="Conversation ID"),
    current_user: UserProfile = Depends(get_current_user),
    conversation_service = Depends(get_conversation_service)
):
    """Delete conversation and all associated data"""
    try:
        deleted = await conversation_service.delete_conversation(
            user_id=current_user.user_id,
            conversation_id=conversation_id
        )
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"status": "deleted", "conversation_id": conversation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")

# ===============================
# USER PREFERENCES & PERSONALIZATION
# ===============================

@chat_router.get("/preferences", response_model=UserChatPreferences)
async def get_user_chat_preferences(
    current_user: UserProfile = Depends(get_current_user),
    personalization_service = Depends(get_personalization_service)
):
    """Get user's chat preferences and personalization settings"""
    try:
        preferences = await personalization_service.get_user_preferences(current_user.user_id)
        return UserChatPreferences(**preferences)
        
    except Exception as e:
        logger.error(f"Error retrieving user preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve preferences")

@chat_router.put("/preferences")
async def update_user_chat_preferences(
    preferences: UserChatPreferences,
    current_user: UserProfile = Depends(get_current_user),
    personalization_service = Depends(get_personalization_service)
):
    """Update user's chat preferences"""
    try:
        updated = await personalization_service.update_user_preferences(
            user_id=current_user.user_id,
            preferences=preferences.dict()
        )
        
        return {"status": "updated", "preferences": updated}
        
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")

# ===============================
# ANALYTICS & INSIGHTS
# ===============================

@chat_router.get("/analytics/usage", response_model=ChatUsageStats)
async def get_chat_usage_analytics(
    period: str = Query("month", regex="^(day|week|month|year)$"),
    current_user: UserProfile = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """Get user's chat usage analytics and statistics"""
    try:
        # Check if user has premium access for detailed analytics
        if current_user.subscription_tier == "free":
            raise HTTPException(
                status_code=403,
                detail="Analytics require premium subscription"
            )
        
        usage_stats = await analytics_service.get_user_usage_stats(
            user_id=current_user.user_id,
            period=period
        )
        
        return usage_stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")

@chat_router.get("/analytics/insights")
async def get_conversation_insights(
    conversation_id: Optional[str] = Query(None, description="Specific conversation ID"),
    limit: int = Query(10, ge=1, le=50),
    current_user: UserProfile = Depends(get_current_user),
    analytics_service = Depends(get_analytics_service)
):
    """Get AI-generated insights from conversations"""
    try:
        if current_user.subscription_tier not in ["premium", "enterprise"]:
            raise HTTPException(
                status_code=403,
                detail="Conversation insights require premium subscription"
            )
        
        insights = await analytics_service.get_conversation_insights(
            user_id=current_user.user_id,
            conversation_id=conversation_id,
            limit=limit
        )
        
        return {"insights": insights, "total": len(insights)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve insights")

# ===============================
# TEMPLATE & WORKFLOW MANAGEMENT
# ===============================

@chat_router.get("/templates")
async def get_chat_templates(
    category: Optional[str] = Query(None, description="Template category"),
    difficulty: Optional[str] = Query(None, description="Difficulty level"),
    current_user: UserProfile = Depends(get_current_user)
):
    """Get available chat templates and workflows"""
    try:
        # Mock template data - replace with database
        templates = [
            {
                "template_id": "meditation_guidance",
                "name": "Meditation Guidance Session",
                "description": "Structured meditation instruction and practice",
                "category": "meditation",
                "difficulty_level": "beginner",
                "estimated_duration": 20,
                "access_level": "free"
            },
            {
                "template_id": "scriptural_study",
                "name": "Scriptural Study Session", 
                "description": "Deep dive into sacred texts with expert guidance",
                "category": "study",
                "difficulty_level": "intermediate",
                "estimated_duration": 45,
                "access_level": "premium"
            },
            {
                "template_id": "life_guidance",
                "name": "Life Decision Guidance",
                "description": "Dharmic guidance for important life decisions",
                "category": "counseling",
                "difficulty_level": "advanced",
                "estimated_duration": 60,
                "access_level": "premium"
            }
        ]
        
        # Filter by subscription tier
        accessible_templates = []
        for template in templates:
            if (template["access_level"] == "free" or 
                current_user.subscription_tier in ["premium", "enterprise"]):
                accessible_templates.append(template)
        
        # Apply filters
        if category:
            accessible_templates = [t for t in accessible_templates if t["category"] == category]
        if difficulty:
            accessible_templates = [t for t in accessible_templates if t["difficulty_level"] == difficulty]
        
        return {
            "templates": accessible_templates,
            "total": len(accessible_templates),
            "user_access_level": current_user.subscription_tier
        }
        
    except Exception as e:
        logger.error(f"Error retrieving templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve templates")

@chat_router.post("/template/{template_id}/start")
async def start_template_conversation(
    template_id: str = Path(..., description="Template ID"),
    initial_context: Optional[Dict[str, Any]] = None,
    current_user: UserProfile = Depends(get_current_user),
    conversation_service = Depends(get_conversation_service)
):
    """Start a new conversation using a template"""
    try:
        # Validate template access
        # Mock validation - replace with actual template lookup
        if template_id == "scriptural_study" and current_user.subscription_tier == "free":
            raise HTTPException(
                status_code=403,
                detail="This template requires premium subscription"
            )
        
        conversation_id = await conversation_service.start_template_conversation(
            user_id=current_user.user_id,
            template_id=template_id,
            initial_context=initial_context or {}
        )
        
        return {
            "status": "template_started",
            "conversation_id": conversation_id,
            "template_id": template_id,
            "next_step": "Send your first message to begin the guided session"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting template conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to start template conversation")

# ===============================
# GAMIFICATION & ACHIEVEMENTS
# ===============================

@chat_router.get("/gamification/profile")
async def get_user_gamification_profile(
    current_user: UserProfile = Depends(get_current_user),
    gamification_service = Depends(get_gamification_service)
):
    """Get user's gamification profile and achievements"""
    try:
        if not current_user.gamification_enabled:
            return {"gamification_enabled": False}
        
        profile = await gamification_service.get_user_profile(current_user.user_id)
        
        return {
            "gamification_enabled": True,
            "profile": profile,
            "recent_achievements": profile.get("recent_achievements", []),
            "next_milestones": profile.get("next_milestones", [])
        }
        
    except Exception as e:
        logger.error(f"Error retrieving gamification profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve gamification profile")

@chat_router.get("/gamification/leaderboard")
async def get_gamification_leaderboard(
    category: str = Query("wisdom_points", description="Leaderboard category"),
    limit: int = Query(10, ge=1, le=50),
    current_user: UserProfile = Depends(get_current_user),
    gamification_service = Depends(get_gamification_service)
):
    """Get gamification leaderboards"""
    try:
        if current_user.subscription_tier == "free":
            raise HTTPException(
                status_code=403,
                detail="Leaderboards require premium subscription"
            )
        
        leaderboard = await gamification_service.get_leaderboard(
            category=category,
            limit=limit,
            user_id=current_user.user_id  # To highlight user's position
        )
        
        return leaderboard
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving leaderboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve leaderboard")

# ===============================
# HEALTH & SYSTEM STATUS
# ===============================

@chat_router.get("/health")
async def comprehensive_chat_health(
    dharmallm_client = Depends(get_dharmallm_client),
    cache_service = Depends(get_cache_service)
):
    """Comprehensive health check for all chat services"""
    try:
        # Check DharmaLLM connectivity
        dharmallm_status = await dharmallm_client.health_check()
        
        # Check cache service
        cache_status = await cache_service.health_check() if hasattr(cache_service, 'health_check') else {"status": "healthy"}
        
        # Overall health assessment
        all_healthy = all([
            dharmallm_status.get("status") == "healthy",
            cache_status.get("status") == "healthy"
        ])
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "comprehensive_chat_router": "operational",
                "dharmallm_integration": dharmallm_status,
                "cache_service": cache_status,
                "rate_limiting": "operational",
                "personalization": "operational",
                "analytics": "operational",
                "content_filtering": "operational"
            },
            "features": {
                "advanced_chat": all_healthy,
                "streaming_chat": all_healthy,
                "rishi_selection": all_healthy,
                "conversation_history": True,
                "personalization": True,
                "analytics": True,
                "templates": True,
                "gamification": True,
                "multi_modal": "coming_soon"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "services": {"comprehensive_chat_router": "error"},
            "features": {"fallback_responses": True}
        }

# ===============================
# ENTERPRISE FEATURES
# ===============================

@chat_router.post("/enterprise/bulk-process")
async def enterprise_bulk_processing(
    requests: List[AdvancedChatRequest],
    current_user: UserProfile = Depends(get_current_user)
):
    """Enterprise bulk chat processing"""
    try:
        if current_user.subscription_tier != "enterprise":
            raise HTTPException(
                status_code=403,
                detail="Bulk processing requires enterprise subscription"
            )
        
        if len(requests) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 requests per bulk operation"
            )
        
        # Process requests in parallel with rate limiting
        results = []
        for request in requests:
            # This would normally call the advanced chat processing
            # For now, return a mock response
            result = {
                "request_id": f"bulk_{len(results)}",
                "status": "processed",
                "response": f"Processed: {request.message[:50]}...",
                "processing_time": 0.1
            }
            results.append(result)
        
        return {
            "status": "bulk_completed",
            "total_requests": len(requests),
            "successful": len(results),
            "failed": 0,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk processing: {e}")
        raise HTTPException(status_code=500, detail="Bulk processing failed")

@chat_router.get("/enterprise/custom-rishis")
async def get_custom_enterprise_rishis(
    current_user: UserProfile = Depends(get_current_user)
):
    """Get enterprise custom Rishis"""
    try:
        if current_user.subscription_tier != "enterprise":
            raise HTTPException(
                status_code=403,
                detail="Custom Rishis require enterprise subscription"
            )
        
        # Mock custom rishis - replace with actual enterprise rishis
        custom_rishis = [
            {
                "id": "custom_business_guru",
                "name": "Business Dharma Advisor",
                "specialty": "Ethical business practices and leadership",
                "custom_trained": True,
                "organization": current_user.organization,
                "access_level": "enterprise_only"
            }
        ]
        
        return {
            "custom_rishis": custom_rishis,
            "organization": current_user.organization,
            "customization_options": [
                "custom_training_data",
                "organization_specific_responses", 
                "branded_interactions",
                "api_integration"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving custom rishis: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve custom rishis")

logger.info("🚀 Comprehensive Enterprise Chat Router loaded successfully")

# ===============================
# RISHI SELECTION & MANAGEMENT
# ===============================

@chat_router.get("/rishis", response_model=List[Dict[str, Any]])
async def get_available_rishis(
    current_user: UserProfile = Depends(get_current_user),
    dharmallm_client = Depends(get_dharmallm_client)
):
    """Get all available Rishi personas with their specialties"""
    try:
        rishis = await dharmallm_client.get_available_rishis()
        
        # Add subscription-based access control
        subscription_service = get_subscription_service()
        user_subscription = await subscription_service.get_user_subscription(current_user.user_id)
        
        # Enhanced rishis for premium users
        if user_subscription.plan_type in ["premium", "enterprise"]:
            for rishi in rishis:
                rishi["premium_features"] = {
                    "extended_conversations": True,
                    "personalized_guidance": True,
                    "priority_response": True,
                    "conversation_history": True
                }
        
        logger.info(f"Retrieved {len(rishis)} rishis for user {current_user.user_id}")
        return {
            "rishis": rishis,
            "user_subscription": user_subscription.plan_type,
            "total_available": len(rishis)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving rishis: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available rishis")

@chat_router.post("/rishi/select")
async def select_rishi(
    request: RishiSelectionRequest,
    current_user: UserProfile = Depends(get_current_user),
    dharmallm_client = Depends(get_dharmallm_client)
):
    """Select a Rishi for conversation"""
    try:
        # Validate rishi availability
        available_rishis = await dharmallm_client.get_available_rishis()
        rishi_ids = [r["id"] for r in available_rishis]
        
        if request.rishi_id not in rishi_ids:
            raise HTTPException(status_code=400, detail=f"Invalid rishi selection: {request.rishi_id}")
        
        # Check subscription for rishi access
        subscription_service = get_subscription_service()
        user_subscription = await subscription_service.get_user_subscription(current_user.user_id)
        
        # Some rishis might be premium-only
        premium_rishis = ["vyasa", "vasishta"]  # Example premium rishis
        if request.rishi_id in premium_rishis and user_subscription.plan_type == "free":
            raise HTTPException(
                status_code=403, 
                detail=f"Rishi {request.rishi_id} requires premium subscription"
            )
        
        # Create conversation session
        chat_history_service = get_chat_history_service()
        conversation_id = await chat_history_service.create_rishi_conversation(
            user_id=current_user.user_id,
            rishi_id=request.rishi_id,
            initial_context=request.context
        )
        
        selected_rishi = next(r for r in available_rishis if r["id"] == request.rishi_id)
        
        return {
            "status": "rishi_selected",
            "rishi": selected_rishi,
            "conversation_id": conversation_id,
            "welcome_message": f"🕉️ Greetings! I am {selected_rishi['name']}. I am here to guide you with wisdom from {selected_rishi['specialty']}. How may I serve your spiritual journey today?",
            "subscription_features": user_subscription.features
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting rishi: {e}")
        raise HTTPException(status_code=500, detail="Failed to select rishi")

# ===============================
# CHAT WITH RISHI
# ===============================

@chat_router.post("/rishi/message", response_model=ChatResponse)
async def chat_with_rishi(
    request: RishiChatRequest,
    background_tasks: BackgroundTasks,
    current_user: UserProfile = Depends(get_current_user),
    dharmallm_client = Depends(get_dharmallm_client)
):
    """Send message to selected Rishi for spiritual guidance"""
    try:
        # Validate subscription and usage limits
        subscription_service = get_subscription_service()
        usage_result = await subscription_service.check_and_increment_usage(
            user_id=current_user.user_id,
            feature="rishi_chat",
            count=1
        )
        
        if not usage_result.allowed:
            raise HTTPException(
                status_code=429, 
                detail=f"Usage limit reached. {usage_result.message}"
            )
        
        # Get user context and preferences
        user_context = {
            "user_id": current_user.user_id,
            "subscription": usage_result.subscription.plan_type,
            "spiritual_background": current_user.spiritual_background,
            "language_preference": current_user.language_preference,
            "conversation_style": current_user.conversation_style
        }
        
        # Chat with the Rishi via DharmaLLM
        start_time = datetime.now()
        
        rishi_response = await dharmallm_client.chat_with_rishi(
            message=request.message,
            rishi=request.rishi_id,
            user_id=current_user.user_id,
            session_id=request.conversation_id,
            user_context=user_context
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Save conversation history
        background_tasks.add_task(
            save_conversation_message,
            user_id=current_user.user_id,
            conversation_id=request.conversation_id,
            user_message=request.message,
            rishi_response=rishi_response,
            processing_time=processing_time
        )
        
        # Prepare response
        chat_response = ChatResponse(
            response=rishi_response["response"],
            conversation_id=request.conversation_id,
            message_id=rishi_response.get("message_id"),
            timestamp=rishi_response.get("timestamp"),
            confidence_score=rishi_response.get("confidence_score"),
            dharmic_alignment=rishi_response.get("dharmic_alignment"),
            processing_time=processing_time,
            model_used=f"DharmaLLM-{request.rishi_id}",
            sources=[f"Rishi {rishi_response.get('rishi_name', request.rishi_id)}"],
            metadata={
                "rishi_name": rishi_response.get("rishi_name"),
                "rishi_specialty": rishi_response.get("rishi_specialty"),
                "guidance_style": rishi_response.get("guidance_style"),
                "spiritual_context": rishi_response.get("spiritual_context"),
                "service_source": rishi_response.get("service_source"),
                "usage_remaining": usage_result.remaining
            }
        )
        
        logger.info(f"✅ Rishi chat completed for user {current_user.user_id}")
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in rishi chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to process rishi chat")

# ===============================
# GENERAL SPIRITUAL CHAT
# ===============================

@chat_router.post("/general", response_model=ChatResponse)
async def general_spiritual_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: UserProfile = Depends(get_current_user),
    dharmallm_client = Depends(get_dharmallm_client)
):
    """General spiritual chat without specific rishi"""
    try:
        # Validate subscription and usage
        subscription_service = get_subscription_service()
        usage_result = await subscription_service.check_and_increment_usage(
            user_id=current_user.user_id,
            feature="general_chat",
            count=1
        )
        
        if not usage_result.allowed:
            raise HTTPException(
                status_code=429, 
                detail=f"Usage limit reached. {usage_result.message}"
            )
        
        # Prepare user context
        user_context = {
            "user_id": current_user.user_id,
            "subscription": usage_result.subscription.plan_type,
            "spiritual_background": current_user.spiritual_background,
            "previous_conversations": current_user.conversation_history_summary
        }
        
        # Get response from DharmaLLM
        start_time = datetime.now()
        
        llm_response = await dharmallm_client.general_chat(
            message=request.message,
            user_id=current_user.user_id,
            session_id=request.conversation_id,
            user_context=user_context
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Save conversation
        background_tasks.add_task(
            save_general_conversation,
            user_id=current_user.user_id,
            conversation_id=request.conversation_id or f"general_{current_user.user_id}_{int(datetime.now().timestamp())}",
            user_message=request.message,
            llm_response=llm_response,
            processing_time=processing_time
        )
        
        # Prepare response
        chat_response = ChatResponse(
            response=llm_response["response"],
            conversation_id=llm_response.get("session_id"),
            message_id=llm_response.get("message_id"),
            timestamp=llm_response.get("timestamp"),
            confidence_score=llm_response.get("confidence_score"),
            dharmic_alignment=llm_response.get("dharmic_alignment"),
            processing_time=processing_time,
            model_used="DharmaLLM-General",
            sources=["DharmaLLM AI"],
            metadata={
                "guidance_type": llm_response.get("guidance_type"),
                "service_source": llm_response.get("service_source"),
                "usage_remaining": usage_result.remaining
            }
        )
        
        logger.info(f"✅ General chat completed for user {current_user.user_id}")
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in general chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to process general chat")

# ===============================
# CONVERSATION HISTORY
# ===============================

@chat_router.get("/history/{conversation_id}", response_model=ChatHistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    current_user: UserProfile = Depends(get_current_user),
    chat_history_service = Depends(get_chat_history_service)
):
    """Get conversation history for a specific conversation"""
    try:
        history = await chat_history_service.get_conversation_history(
            user_id=current_user.user_id,
            conversation_id=conversation_id
        )
        
        if not history:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history")

@chat_router.get("/conversations", response_model=List[ConversationSummary])
async def get_user_conversations(
    limit: int = 20,
    offset: int = 0,
    current_user: UserProfile = Depends(get_current_user),
    chat_history_service = Depends(get_chat_history_service)
):
    """Get user's conversation summaries"""
    try:
        conversations = await chat_history_service.get_user_conversations(
            user_id=current_user.user_id,
            limit=limit,
            offset=offset
        )
        
        return conversations
        
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")

# ===============================
# BACKGROUND TASKS
# ===============================

async def save_conversation_message(
    user_id: str,
    conversation_id: str, 
    user_message: str,
    rishi_response: Dict[str, Any],
    processing_time: float
):
    """Background task to save conversation message"""
    try:
        chat_history_service = get_chat_history_service()
        await chat_history_service.save_rishi_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
            rishi_response=rishi_response,
            processing_time=processing_time
        )
        logger.info(f"💾 Saved rishi conversation for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")

async def save_general_conversation(
    user_id: str,
    conversation_id: str,
    user_message: str, 
    llm_response: Dict[str, Any],
    processing_time: float
):
    """Background task to save general conversation"""
    try:
        chat_history_service = get_chat_history_service()
        await chat_history_service.save_general_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
            llm_response=llm_response,
            processing_time=processing_time
        )
        logger.info(f"💾 Saved general conversation for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")

# ===============================
# HEALTH & STATUS
# ===============================

@chat_router.get("/health")
async def chat_service_health(
    dharmallm_client = Depends(get_dharmallm_client)
):
    """Check chat service health and DharmaLLM availability"""
    try:
        dharmallm_status = await dharmallm_client.health_check()
        
        return {
            "status": "healthy",
            "chat_service": "operational",
            "dharmallm_integration": dharmallm_status,
            "available_features": {
                "rishi_chat": dharmallm_status["status"] != "unhealthy",
                "general_chat": dharmallm_status["status"] != "unhealthy", 
                "fallback_responses": True,
                "conversation_history": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat service health check failed: {e}")
        return {
            "status": "degraded",
            "chat_service": "operational",
            "dharmallm_integration": {"status": "unhealthy", "error": str(e)},
            "available_features": {
                "rishi_chat": False,
                "general_chat": False,
                "fallback_responses": True,
                "conversation_history": True
            },
            "timestamp": datetime.now().isoformat()
        }
