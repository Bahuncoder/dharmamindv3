"""
🧘 Rishi Mode Routes - Direct Guidance from Enlightened Sages
==========================================================

Simple extension of existing system to provide Rishi personality-based guidance.
Leverages all existing dharmic systems (darshana, spiritual modules, knowledge base).
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi import Request
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ...services.universal_dharmic_engine import get_universal_dharmic_engine, DharmicEngine
from ...services.auth_service import get_current_user
from ...models.user_profile import UserProfile

logger = logging.getLogger(__name__)

router = APIRouter()

class RishiQuery(BaseModel):
    """Request for Rishi guidance"""
    query: str = Field(..., description="Question for the Rishi", min_length=1, max_length=2000)
    rishi_name: str = Field(..., description="Name of the Saptarishi (atri, bhrigu, vashishta, vishwamitra, gautama, jamadagni, kashyapa)")
    context: Optional[str] = Field(default=None, description="Additional context")
    spiritual_level: Optional[str] = Field(default="beginner", description="Spiritual experience level")
    preferred_style: Optional[str] = Field(default="practical", description="Preferred guidance style")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Previous conversation messages")

class RishiResponse(BaseModel):
    """Response from Rishi guidance"""
    mode: str
    rishi_info: Dict[str, Any]
    greeting: str
    guidance: Dict[str, Any]
    dharmic_foundation: Any  # Can be string or list of scriptural references
    practical_steps: List[str]
    wisdom_synthesis: str
    growth_opportunities: List[str]
    processing_time: float
    enhanced: Optional[bool] = False
    session_continuity: Optional[Dict[str, Any]] = None

@router.get("/rishi/available")
async def get_available_rishis(
    current_user: UserProfile = Depends(get_current_user),
    dharmic_engine: DharmicEngine = Depends(get_universal_dharmic_engine)
):
    """Get list of available Rishis based on subscription"""
    try:
        all_rishis = dharmic_engine.rishi_personalities
        
        # Filter based on user subscription
        available_rishis = []
        for rishi_id, rishi_info in all_rishis.items():
            # Check if user has access
            has_access = (
                rishi_info.get('available_free', False) or 
                current_user.subscription_tier in ['premium', 'master']
            )
            
            available_rishis.append({
                'id': rishi_id,
                'name': rishi_info['name'],
                'sanskrit': rishi_info['sanskrit'],
                'specialization': rishi_info['specialization'],
                'greeting': rishi_info['greeting'],
                'available': has_access,
                'requires_upgrade': not has_access
            })
        
        return {
            'available_rishis': available_rishis,
            'user_subscription': current_user.subscription_tier
        }
        
    except Exception as e:
        logger.error(f"Error retrieving Rishis: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving available Rishis")

@router.post("/rishi/guidance", response_model=RishiResponse)
async def get_rishi_guidance(
    request: RishiQuery,
    background_tasks: BackgroundTasks,
    current_user: UserProfile = Depends(get_current_user),
    dharmic_engine: DharmicEngine = Depends(get_universal_dharmic_engine)
):
    """Get guidance from specific Rishi"""
    try:
        start_time = datetime.now()
        
        # Check if user has access to this Rishi
        rishi_info = dharmic_engine.rishi_personalities.get(request.rishi_name)
        if not rishi_info:
            raise HTTPException(status_code=404, detail="Rishi not found")
        
        # Check subscription access
        if not rishi_info.get('available_free', False):
            if current_user.subscription_tier not in ['premium', 'master']:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Access to {rishi_info['name']} requires premium subscription"
                )
        
        # Get Rishi guidance (uses enhanced system if available)
        user_context = {
            'user_id': current_user.id,
            'subscription': current_user.subscription_tier,
            'context': request.context,
            'spiritual_level': request.spiritual_level,
            'preferred_style': request.preferred_style,
            'conversation_history': request.conversation_history or []
        }
        
        rishi_response = await dharmic_engine.get_rishi_guidance(
            query=request.query,
            rishi_name=request.rishi_name,
            user_context=user_context
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        rishi_response['processing_time'] = processing_time
        
        # Log successful interaction
        logger.info(f"Rishi guidance provided: {request.rishi_name} for user {current_user.id}")
        
        return RishiResponse(**rishi_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Rishi guidance: {e}")
        raise HTTPException(status_code=500, detail="Error processing Rishi guidance")

@router.get("/rishi/specializations")
async def get_rishi_specializations():
    """Get information about each Saptarishi's specializations"""
    return {
        'atri': {
            'best_for': ['meditation practice', 'spiritual discipline', 'austerity', 'cosmic consciousness', 'deep contemplation'],
            'example_questions': [
                'How can I deepen my meditation practice?',
                'What is the path of tapasya?',
                'How do I develop spiritual discipline?',
                'How to connect with cosmic consciousness?'
            ]
        },
        'bhrigu': {
            'best_for': ['astrology guidance', 'karma understanding', 'life purpose', 'cosmic order', 'divine knowledge'],
            'example_questions': [
                'What does my karma indicate?',
                'How to understand astrological influences?',
                'What is my life purpose according to cosmic design?',
                'How does karma shape destiny?'
            ]
        },
        'vashishta': {
            'best_for': ['royal guidance', 'leadership wisdom', 'dharmic decision-making', 'spiritual mastery', 'divine wisdom'],
            'example_questions': [
                'How should a leader act dharmaically?',
                'What wisdom guided Lord Rama?',
                'How to make decisions aligned with dharma?',
                'How to achieve spiritual mastery?'
            ]
        },
        'vishwamitra': {
            'best_for': ['spiritual transformation', 'Gayatri mantra', 'divine power', 'overcoming obstacles', 'spiritual achievement'],
            'example_questions': [
                'How can I transform spiritually?',
                'What is the power of Gayatri mantra?',
                'How to overcome spiritual obstacles?',
                'How did you become a Brahmarishi?'
            ]
        },
        'gautama': {
            'best_for': ['meditation guidance', 'dharmic living', 'righteousness', 'spiritual discipline', 'ethical conduct'],
            'example_questions': [
                'How to live righteously?',
                'What is the essence of dharma?',
                'How to develop ethical conduct?',
                'How to deepen meditation practice?'
            ]
        },
        'jamadagni': {
            'best_for': ['spiritual discipline', 'tapas practice', 'divine power', 'righteous action', 'spiritual strength'],
            'example_questions': [
                'How to develop spiritual discipline?',
                'What is the power of tapas?',
                'How to act righteously with strength?',
                'How to channel divine power?'
            ]
        },
        'kashyapa': {
            'best_for': ['universal consciousness', 'cosmic creation', 'pranic wisdom', 'life force', 'spiritual fatherhood'],
            'example_questions': [
                'How to connect with universal consciousness?',
                'What is the nature of cosmic creation?',
                'How to work with life force energy?',
                'How to understand my cosmic heritage?'
            ]
        }
    }
