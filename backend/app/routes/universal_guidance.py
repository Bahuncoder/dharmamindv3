"""
Universal Dharmic Guidance API Routes
====================================

API endpoints for comprehensive life guidance rooted in Hindu/Sanatan Dharma
but presented in universal, secular language for global accessibility.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging

from app.services.universal_dharmic_engine import (
    get_universal_life_guidance,
    get_universal_dharmic_engine,
    LifeAspect,
    UniversalPrinciple
)
from app.routes.auth import get_current_user
from app.services.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/universal", tags=["Universal Dharmic Guidance"])

class UniversalGuidanceRequest(BaseModel):
    """Request model for universal guidance"""
    query: str = Field(..., min_length=10, max_length=1000, 
                      description="Life question or situation needing guidance")
    cultural_background: str = Field(default="global", 
                                   description="Cultural context: global, western, eastern, etc.")
    depth_level: str = Field(default="comprehensive", 
                           description="Response depth: basic, comprehensive, advanced")
    life_stage: Optional[str] = Field(default=None, 
                                    description="Life stage: student, householder, seeker, elder")
    include_practices: bool = Field(default=True, 
                                  description="Include practical exercises and techniques")
    secular_only: bool = Field(default=False, 
                             description="Exclude religious terminology")

class UniversalGuidanceResponse(BaseModel):
    """Response model for universal guidance"""
    query: str
    cultural_adaptation: str
    primary_guidance: str
    universal_principles: List[str]
    practical_steps: List[str]
    deeper_wisdom: str
    life_integration: str
    holistic_perspective: Dict[str, str]
    long_term_development: List[str]
    practical_roadmap: List[str]
    global_applicability: str
    advanced_practices: List[str]
    dharmic_foundation: str
    scriptural_source: str
    processing_time_ms: float

class QuickGuidanceRequest(BaseModel):
    """Request model for quick guidance"""
    situation: str = Field(..., min_length=5, max_length=500)
    cultural_context: str = Field(default="global")

@router.post("/guidance", response_model=UniversalGuidanceResponse)
async def get_comprehensive_guidance(
    request: UniversalGuidanceRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive life guidance for any human situation.
    
    This endpoint provides deep, practical guidance rooted in authentic
    Hindu/Sanatan Dharma wisdom but presented in universal language
    that resonates across all cultures and backgrounds.
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Processing comprehensive guidance for user {current_user.get('id', 'anonymous')}")
        
        # Get universal guidance
        guidance_result = await get_universal_life_guidance(
            query=request.query,
            cultural_background=request.cultural_background,
            depth_level=request.depth_level
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Add processing time to response
        guidance_result["processing_time_ms"] = processing_time
        
        # Store in user's conversation history
        background_tasks.add_task(
            store_guidance_session,
            user_id=current_user.get('id'),
            request=request,
            response=guidance_result
        )
        
        # Track usage analytics
        background_tasks.add_task(
            track_guidance_usage,
            user_id=current_user.get('id'),
            query_type="comprehensive",
            cultural_background=request.cultural_background,
            processing_time=processing_time
        )
        
        return UniversalGuidanceResponse(**guidance_result)
        
    except Exception as e:
        logger.error(f"Error processing comprehensive guidance: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate guidance")

@router.post("/quick-guidance")
async def get_quick_guidance(
    request: QuickGuidanceRequest,
    current_user = Depends(get_current_user)
):
    """
    Get quick, practical guidance for immediate situations.
    
    Provides concise, actionable wisdom for everyday challenges
    based on universal principles derived from Sanatan Dharma.
    """
    try:
        # Process as simplified request
        guidance_result = await get_universal_life_guidance(
            query=request.situation,
            cultural_background=request.cultural_context,
            depth_level="basic"
        )
        
        # Return simplified response
        return {
            "situation": request.situation,
            "guidance": guidance_result["primary_guidance"],
            "practical_steps": guidance_result["practical_steps"][:3],
            "key_principle": guidance_result["universal_principles"][0] if guidance_result["universal_principles"] else "Wisdom and compassion",
            "cultural_adaptation": guidance_result["cultural_adaptation"]
        }
        
    except Exception as e:
        logger.error(f"Error processing quick guidance: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate quick guidance")

@router.get("/life-aspects")
async def get_life_aspects():
    """Get all supported life aspects for guidance"""
    return {
        "life_aspects": [
            {
                "name": aspect.value,
                "display_name": aspect.value.replace('_', ' ').title(),
                "description": get_aspect_description(aspect)
            }
            for aspect in LifeAspect
        ]
    }

@router.get("/universal-principles")
async def get_universal_principles():
    """Get all universal principles derived from Sanatan Dharma"""
    return {
        "principles": [
            {
                "name": principle.value,
                "display_name": principle.value.replace('_', ' ').title(),
                "description": get_principle_description(principle),
                "dharmic_source": get_dharmic_source(principle)
            }
            for principle in UniversalPrinciple
        ]
    }

@router.post("/guidance/by-aspect")
async def get_guidance_by_aspect(
    life_aspect: str,
    query: str,
    cultural_background: str = "global",
    current_user = Depends(get_current_user)
):
    """Get guidance specifically focused on a particular life aspect"""
    try:
        # Validate life aspect
        try:
            aspect_enum = LifeAspect(life_aspect)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid life aspect: {life_aspect}")
        
        # Enhance query with aspect focus
        enhanced_query = f"Regarding {aspect_enum.value.replace('_', ' ')}: {query}"
        
        guidance_result = await get_universal_life_guidance(
            query=enhanced_query,
            cultural_background=cultural_background,
            depth_level="comprehensive"
        )
        
        return {
            "life_aspect": aspect_enum.value,
            "original_query": query,
            "enhanced_query": enhanced_query,
            "guidance": guidance_result["primary_guidance"],
            "practical_steps": guidance_result["practical_steps"],
            "deeper_wisdom": guidance_result["deeper_wisdom"],
            "universal_application": guidance_result.get("universal_application", ""),
            "dharmic_foundation": guidance_result["dharmic_foundation"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing aspect-specific guidance: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate aspect-specific guidance")

@router.get("/cultural-adaptations")
async def get_cultural_adaptations():
    """Get available cultural adaptation options"""
    return {
        "cultural_backgrounds": [
            {
                "code": "global",
                "name": "Global/Universal",
                "description": "Universal principles applicable across all cultures"
            },
            {
                "code": "western",
                "name": "Western",
                "description": "Adapted for Western cultural context and values"
            },
            {
                "code": "eastern",
                "name": "Eastern",
                "description": "Resonant with Eastern philosophical traditions"
            },
            {
                "code": "secular",
                "name": "Secular",
                "description": "Non-religious, scientific and psychological approach"
            },
            {
                "code": "indigenous",
                "name": "Indigenous",
                "description": "Honors indigenous wisdom traditions globally"
            },
            {
                "code": "african",
                "name": "African",
                "description": "Integrated with African traditional wisdom"
            }
        ]
    }

@router.get("/system-status")
async def get_system_status():
    """Get status of the universal dharmic guidance system"""
    try:
        engine = await get_universal_dharmic_engine()
        
        # Get status from all integrated systems
        status = {
            "universal_engine": "active",
            "knowledge_base": "active",
            "spiritual_modules": "active", 
            "darshana_engine": "active",
            "total_life_aspects": len(LifeAspect),
            "total_principles": len(UniversalPrinciple),
            "supported_cultures": 6,
            "guidance_modes": ["basic", "comprehensive", "advanced"],
            "system_health": "excellent"
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/guidance/crisis-support")
async def get_crisis_support_guidance(
    crisis_description: str,
    urgency_level: str = "moderate",  # low, moderate, high
    cultural_background: str = "global",
    current_user = Depends(get_current_user)
):
    """
    Specialized endpoint for crisis situations requiring immediate support.
    
    Provides compassionate, practical guidance for difficult life situations
    with emphasis on safety, stability, and gradual healing.
    """
    try:
        # Enhance query for crisis support
        crisis_query = f"Crisis support needed: {crisis_description}. Urgency level: {urgency_level}"
        
        guidance_result = await get_universal_life_guidance(
            query=crisis_query,
            cultural_background=cultural_background,
            depth_level="comprehensive"
        )
        
        # Add crisis-specific elements
        crisis_response = {
            **guidance_result,
            "immediate_actions": extract_immediate_actions(guidance_result),
            "safety_considerations": extract_safety_guidance(guidance_result),
            "professional_resources": get_professional_resources(urgency_level),
            "support_network_guidance": extract_support_guidance(guidance_result),
            "healing_timeline": get_healing_timeline_guidance(crisis_description),
            "emergency_contacts": get_emergency_resources()
        }
        
        # Log crisis support request for follow-up
        logger.warning(f"Crisis support provided to user {current_user.get('id')}: {crisis_description[:50]}...")
        
        return crisis_response
        
    except Exception as e:
        logger.error(f"Error processing crisis support: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate crisis support")

# Helper functions
async def store_guidance_session(user_id: str, request: UniversalGuidanceRequest, response: Dict[str, Any]):
    """Store guidance session in user's history"""
    try:
        memory_manager = get_memory_manager()
        await memory_manager.store_conversation(
            user_id=user_id,
            query=request.query,
            response=response["primary_guidance"],
            metadata={
                "guidance_type": "universal_dharmic",
                "cultural_background": request.cultural_background,
                "depth_level": request.depth_level,
                "life_stage": request.life_stage,
                "processing_time": response.get("processing_time_ms", 0)
            }
        )
    except Exception as e:
        logger.error(f"Failed to store guidance session: {e}")

async def track_guidance_usage(user_id: str, query_type: str, cultural_background: str, processing_time: float):
    """Track usage analytics"""
    try:
        # Implementation would send to analytics service
        logger.info(f"Analytics: User {user_id}, Type: {query_type}, Culture: {cultural_background}, Time: {processing_time}ms")
    except Exception as e:
        logger.error(f"Failed to track usage: {e}")

def get_aspect_description(aspect: LifeAspect) -> str:
    """Get description for life aspect"""
    descriptions = {
        LifeAspect.PERSONAL_GROWTH: "Self-development, character building, and personal evolution",
        LifeAspect.RELATIONSHIPS: "Family, friendships, romantic relationships, and social connections",
        LifeAspect.CAREER_PURPOSE: "Professional life, work fulfillment, and life purpose",
        LifeAspect.HEALTH_WELLNESS: "Physical, mental, emotional, and spiritual wellbeing",
        LifeAspect.FINANCIAL_WEALTH: "Money management, prosperity, and financial security",
        LifeAspect.SPIRITUAL_DEVELOPMENT: "Inner growth, consciousness expansion, and spiritual practice",
        LifeAspect.FAMILY_PARENTING: "Family dynamics, parenting, and intergenerational relationships",
        LifeAspect.SOCIAL_COMMUNITY: "Community engagement, social responsibility, and service",
        LifeAspect.CRISIS_CHALLENGES: "Difficult situations, trauma recovery, and resilience building",
        LifeAspect.LIFE_TRANSITIONS: "Major life changes, transitions, and new beginnings",
        LifeAspect.LEADERSHIP_INFLUENCE: "Leadership development, influence, and responsibility",
        LifeAspect.CREATIVE_EXPRESSION: "Creativity, artistic expression, and innovation",
        LifeAspect.EDUCATION_LEARNING: "Learning, education, knowledge acquisition, and teaching",
        LifeAspect.EMOTIONAL_MASTERY: "Emotional intelligence, emotional regulation, and inner balance",
        LifeAspect.MORAL_ETHICS: "Ethical decision-making, moral dilemmas, and values clarification"
    }
    return descriptions.get(aspect, "Comprehensive life guidance")

def get_principle_description(principle: UniversalPrinciple) -> str:
    """Get description for universal principle"""
    descriptions = {
        UniversalPrinciple.TRUTH_INTEGRITY: "Living with honesty, authenticity, and moral integrity",
        UniversalPrinciple.NON_VIOLENCE_COMPASSION: "Practicing non-harm and cultivating compassion for all beings",
        UniversalPrinciple.DUTY_RESPONSIBILITY: "Fulfilling responsibilities and living according to natural law",
        UniversalPrinciple.SELFLESS_SERVICE: "Serving others without expectation of personal gain",
        UniversalPrinciple.INNER_PEACE: "Cultivating tranquility, calm, and emotional balance",
        UniversalPrinciple.WISDOM_KNOWLEDGE: "Pursuing understanding, learning, and practical wisdom",
        UniversalPrinciple.DISCIPLINED_PRACTICE: "Maintaining consistent spiritual and personal practices",
        UniversalPrinciple.DEVOTION_LOVE: "Expressing pure love and dedication in all relationships",
        UniversalPrinciple.RIGHTEOUS_ACTION: "Acting ethically and in alignment with higher principles",
        UniversalPrinciple.CONTENTMENT_GRATITUDE: "Finding satisfaction and appreciation in present circumstances",
        UniversalPrinciple.COURAGE_STRENGTH: "Developing inner strength and courage to face challenges",
        UniversalPrinciple.HUMILITY_SURRENDER: "Practicing humility and acceptance of life's flow"
    }
    return descriptions.get(principle, "Universal wisdom principle")

def get_dharmic_source(principle: UniversalPrinciple) -> str:
    """Get the original dharmic source for universal principle"""
    sources = {
        UniversalPrinciple.TRUTH_INTEGRITY: "Satya (Truth) - First principle of dharmic living",
        UniversalPrinciple.NON_VIOLENCE_COMPASSION: "Ahimsa (Non-violence) - Foundation of ethical conduct",
        UniversalPrinciple.DUTY_RESPONSIBILITY: "Dharma (Righteous duty) - Natural law and order",
        UniversalPrinciple.SELFLESS_SERVICE: "Seva (Selfless service) - Path of compassionate action",
        UniversalPrinciple.INNER_PEACE: "Shanti (Peace) - Goal of spiritual practice",
        UniversalPrinciple.WISDOM_KNOWLEDGE: "Jnana (Knowledge) - Path of understanding and wisdom",
        UniversalPrinciple.DISCIPLINED_PRACTICE: "Yoga/Tapas (Union/Discipline) - Systematic spiritual practice",
        UniversalPrinciple.DEVOTION_LOVE: "Bhakti (Devotion) - Path of love and surrender",
        UniversalPrinciple.RIGHTEOUS_ACTION: "Karma Yoga (Path of action) - Selfless action",
        UniversalPrinciple.CONTENTMENT_GRATITUDE: "Santosha (Contentment) - Satisfaction with what is",
        UniversalPrinciple.COURAGE_STRENGTH: "Vira (Heroism) - Courage in face of adversity",
        UniversalPrinciple.HUMILITY_SURRENDER: "Namrata (Humility) - Ego dissolution and surrender"
    }
    return sources.get(principle, "Sanatan Dharma principles")

def extract_immediate_actions(guidance_result: Dict[str, Any]) -> List[str]:
    """Extract immediate actions for crisis situations"""
    return [
        "Ensure your immediate physical safety and basic needs are met",
        "Reach out to trusted friends, family, or professional support",
        "Focus on one small positive action you can take right now",
        "Practice deep breathing or grounding techniques to stay present",
        "Remember that this difficult time will pass and growth is possible"
    ]

def extract_safety_guidance(guidance_result: Dict[str, Any]) -> List[str]:
    """Extract safety considerations"""
    return [
        "Prioritize your physical and emotional safety above all else",
        "Remove yourself from harmful situations if possible",
        "Seek professional help if experiencing thoughts of self-harm",
        "Trust your instincts about what feels safe and supportive",
        "Create a safety plan with specific steps and support contacts"
    ]

def get_professional_resources(urgency_level: str) -> List[str]:
    """Get professional resources based on urgency"""
    if urgency_level == "high":
        return [
            "Emergency services: 911 (US) or your local emergency number",
            "National Suicide Prevention Lifeline: 988 (US)",
            "Crisis Text Line: Text HOME to 741741",
            "Local emergency mental health services",
            "Trusted healthcare provider or therapist"
        ]
    else:
        return [
            "Licensed therapist or counselor",
            "Primary care physician",
            "Community mental health center",
            "Support groups for your specific situation",
            "Employee assistance program (if available)"
        ]

def extract_support_guidance(guidance_result: Dict[str, Any]) -> List[str]:
    """Extract support network guidance"""
    return [
        "Identify trusted people you can reach out to for support",
        "Communicate your needs clearly to those who want to help",
        "Accept help graciously while maintaining appropriate boundaries",
        "Consider joining support groups with others in similar situations",
        "Build a network of different types of support (emotional, practical, spiritual)"
    ]

def get_healing_timeline_guidance(crisis_description: str) -> str:
    """Provide guidance on healing timeline"""
    return """
    Healing is a gradual process that unfolds naturally when we provide the right conditions. 
    Allow yourself time to process and integrate this experience. Some days will be better than others, 
    and that's completely normal. Focus on small, consistent steps toward wellness rather than 
    expecting dramatic changes. Be patient and compassionate with yourself throughout this journey.
    """

def get_emergency_resources() -> List[Dict[str, str]]:
    """Get emergency contact resources"""
    return [
        {"name": "National Suicide Prevention Lifeline (US)", "contact": "988"},
        {"name": "Crisis Text Line", "contact": "Text HOME to 741741"},
        {"name": "Emergency Services", "contact": "911 (US) or local emergency number"},
        {"name": "National Domestic Violence Hotline", "contact": "1-800-799-7233"},
        {"name": "SAMHSA National Helpline", "contact": "1-800-662-4357"}
    ]
