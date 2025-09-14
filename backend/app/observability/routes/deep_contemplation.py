"""
Deep Contemplation API Routes
============================

REST API endpoints for the Deep Contemplation System,
providing spiritual guidance for authentic contemplative practices.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ...services.deep_contemplation_system import (
    get_contemplation_system as get_deep_contemplation_system,
    begin_contemplation_session,
    guide_contemplation_deepening,
    complete_contemplation_session,
    ContemplationType,
    ContemplationTradition,
    ContemplationDepth
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/contemplation", tags=["Deep Contemplation"])

# Request/Response Models
class ContemplationRequest(BaseModel):
    """Request to start a contemplation session"""
    practice_type: str = Field(..., description="Type of contemplative practice")
    duration_minutes: int = Field(20, ge=5, le=120, description="Duration in minutes")
    tradition: str = Field("universal", description="Spiritual tradition")
    depth_level: str = Field("focused", description="Desired depth level")

class ContemplationGuidanceRequest(BaseModel):
    """Request for contemplation guidance"""
    session_id: str = Field(..., description="Active session ID")
    current_state: str = Field(..., description="Current state (peaceful, distracted, etc.)")

class InsightCaptureRequest(BaseModel):
    """Request to capture an insight"""
    session_id: str = Field(..., description="Active session ID")
    insight: str = Field(..., description="The insight that arose")
    integration_intention: str = Field("", description="How to integrate this insight")

class CompletionRequest(BaseModel):
    """Request to complete a session"""
    session_id: str = Field(..., description="Session to complete")
    completion_reflection: str = Field("", description="Final reflection")

class ContemplationResponse(BaseModel):
    """Response with contemplation session details"""
    session_id: str
    practice_type: str
    tradition: str
    depth_level: str
    duration_minutes: int
    guidance_text: str
    sanskrit_wisdom: Optional[str]
    reflection_prompts: List[str]
    mantras: List[str]
    created_at: datetime

@router.post("/begin", response_model=ContemplationResponse)
async def begin_contemplation(request: ContemplationRequest):
    """
    Begin a new deep contemplation session
    
    This endpoint starts a guided contemplation session based on authentic
    spiritual practices from various traditions.
    """
    try:
        # Validate inputs
        try:
            practice_type = ContemplationType(request.practice_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid practice type. Valid options: {[t.value for t in ContemplationType]}"
            )
        
        try:
            tradition = ContemplationTradition(request.tradition)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tradition. Valid options: {[t.value for t in ContemplationTradition]}"
            )
        
        try:
            depth_level = ContemplationDepth(request.depth_level)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid depth level. Valid options: {[d.value for d in ContemplationDepth]}"
            )
        
        # Generate user ID (in production, this would come from authentication)
        user_id = "demo_user"  # Replace with actual user authentication
        
        system = await get_deep_contemplation_system()
        session = await system.begin_contemplation(
            user_id=user_id,
            practice_type=practice_type,
            duration_minutes=request.duration_minutes,
            tradition=tradition,
            depth_level=depth_level
        )
        
        return ContemplationResponse(
            session_id=session.id,
            practice_type=session.practice_type.value,
            tradition=session.tradition.value,
            depth_level=session.depth_level.value,
            duration_minutes=session.duration_minutes,
            guidance_text=session.guidance_text,
            sanskrit_wisdom=session.sanskrit_wisdom,
            reflection_prompts=session.reflection_prompts,
            mantras=session.mantras,
            created_at=session.created_at
        )
        
    except Exception as e:
        logger.error(f"Error beginning contemplation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to begin contemplation session")

@router.post("/guide")
async def guide_contemplation(request: ContemplationGuidanceRequest):
    """
    Get adaptive guidance to deepen contemplation
    
    This endpoint provides personalized guidance based on the current
    state of the practitioner during their contemplation session.
    """
    try:
        guidance = await guide_contemplation_deepening(
            session_id=request.session_id,
            current_state=request.current_state
        )
        
        return JSONResponse(content={
            "status": "success",
            "guidance": guidance,
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error providing contemplation guidance: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to provide guidance")

@router.post("/insight")
async def capture_insight(request: InsightCaptureRequest):
    """
    Capture insights that arise during contemplation
    
    This endpoint allows practitioners to record insights, realizations,
    or significant experiences during their contemplation practice.
    """
    try:
        system = await get_deep_contemplation_system()
        success = await system.capture_insight(
            session_id=request.session_id,
            insight=request.insight,
            integration_intention=request.integration_intention
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return JSONResponse(content={
            "status": "success",
            "message": "Insight captured successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error capturing insight: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to capture insight")

@router.post("/complete")
async def complete_contemplation(request: CompletionRequest):
    """
    Complete a contemplation session and receive integration guidance
    
    This endpoint concludes the contemplation session and provides
    personalized guidance for integrating insights into daily life.
    """
    try:
        completion_summary = await complete_contemplation_session(
            session_id=request.session_id,
            completion_reflection=request.completion_reflection
        )
        
        return JSONResponse(content={
            "status": "success",
            "completion_summary": completion_summary,
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error completing contemplation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to complete session")

@router.get("/practices")
async def get_available_practices():
    """
    Get available contemplation practices and traditions
    
    This endpoint returns all available practice types, traditions,
    and depth levels for contemplation sessions.
    """
    try:
        return JSONResponse(content={
            "practice_types": {
                practice.value: {
                    "name": practice.value.replace("_", " ").title(),
                    "description": _get_practice_description(practice)
                }
                for practice in ContemplationType
            },
            "traditions": {
                tradition.value: {
                    "name": tradition.value.replace("_", " ").title(),
                    "description": _get_tradition_description(tradition)
                }
                for tradition in ContemplationTradition
            },
            "depth_levels": {
                depth.value: {
                    "name": depth.value.title(),
                    "description": _get_depth_description(depth)
                }
                for depth in ContemplationDepth
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting available practices: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve practices")

@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get the current status of an active contemplation session
    """
    try:
        system = await get_deep_contemplation_system()
        
        if session_id not in system.active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = system.active_sessions[session_id]
        
        return JSONResponse(content={
            "session_id": session.id,
            "status": session.completion_status,
            "practice_type": session.practice_type.value,
            "duration_minutes": session.duration_minutes,
            "insights_captured": len(session.insights_captured),
            "created_at": session.created_at.isoformat(),
            "time_elapsed": (datetime.now() - session.created_at).total_seconds() / 60
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get session status")

def _get_practice_description(practice: ContemplationType) -> str:
    """Get description for a practice type"""
    descriptions = {
        ContemplationType.BREATH_AWARENESS: "Mindful awareness of the natural breath rhythm",
        ContemplationType.LOVING_KINDNESS: "Cultivation of unconditional love and compassion",
        ContemplationType.WISDOM_REFLECTION: "Deep contemplation of spiritual teachings",
        ContemplationType.DEATH_CONTEMPLATION: "Profound reflection on mortality and impermanence",
        ContemplationType.IMPERMANENCE: "Understanding the transient nature of all phenomena",
        ContemplationType.INTERCONNECTEDNESS: "Recognizing the unity underlying all existence",
        ContemplationType.GRATITUDE_PRACTICE: "Cultivating appreciation and thankfulness",
        ContemplationType.SCRIPTURE_STUDY: "Contemplative engagement with sacred texts",
        ContemplationType.SELF_INQUIRY: "Investigation into the nature of the self",
        ContemplationType.COMPASSION_PRACTICE: "Development of empathy and care for suffering",
        ContemplationType.EQUANIMITY: "Cultivation of balanced, peaceful awareness",
        ContemplationType.DHARMIC_REFLECTION: "Contemplation of righteous living and duty"
    }
    return descriptions.get(practice, "Authentic spiritual contemplation practice")

def _get_tradition_description(tradition: ContemplationTradition) -> str:
    """Get description for a spiritual tradition"""
    descriptions = {
        ContemplationTradition.VEDANTA: "Non-dual wisdom tradition from ancient India",
        ContemplationTradition.BUDDHIST: "Mindfulness and wisdom practices from Buddhism",
        ContemplationTradition.YOGA: "Union practices integrating body, mind, and spirit",
        ContemplationTradition.SUFI: "Islamic mystical tradition of divine love",
        ContemplationTradition.CHRISTIAN_MYSTIC: "Contemplative practices from Christian mysticism",
        ContemplationTradition.ZEN: "Direct pointing to awakened awareness",
        ContemplationTradition.UNIVERSAL: "Universal spiritual principles accessible to all"
    }
    return descriptions.get(tradition, "Authentic spiritual tradition")

def _get_depth_description(depth: ContemplationDepth) -> str:
    """Get description for a contemplation depth level"""
    descriptions = {
        ContemplationDepth.SURFACE: "Basic mindfulness and awareness practices",
        ContemplationDepth.FOCUSED: "Single-pointed concentration and presence",
        ContemplationDepth.ABSORBED: "Deep meditative states of absorption",
        ContemplationDepth.INSIGHTFUL: "Wisdom-generating contemplation practices",
        ContemplationDepth.TRANSFORMATIVE: "Profound practices for spiritual transformation"
    }
    return descriptions.get(depth, "Contemplative awareness practice")
