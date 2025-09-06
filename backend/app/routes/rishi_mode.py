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

from ..services.universal_dharmic_engine import get_universal_dharmic_engine, DharmicEngine
from ..services.auth_service import get_current_user
from ..models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()

class RishiQuery(BaseModel):
    """Request for Rishi guidance"""
    query: str = Field(..., description="Question for the Rishi", min_length=1, max_length=2000)
    rishi_name: str = Field(..., description="Name of the Rishi (patanjali, vyasa, valmiki)")
    context: Optional[str] = Field(default=None, description="Additional context")

class RishiResponse(BaseModel):
    """Response from Rishi guidance"""
    mode: str
    rishi_info: Dict[str, Any]
    greeting: str
    guidance: Dict[str, Any]
    dharmic_foundation: str
    practical_steps: List[str]
    wisdom_synthesis: str
    growth_opportunities: List[str]
    processing_time: float

@router.get("/rishi/available")
async def get_available_rishis(
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
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
        
        # Get Rishi guidance (uses all existing systems)
        user_context = {
            'user_id': current_user.id,
            'subscription': current_user.subscription_tier,
            'context': request.context
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
    """Get information about each Rishi's specializations"""
    return {
        'patanjali': {
            'best_for': ['meditation questions', 'mind control', 'yoga practice', 'concentration issues'],
            'example_questions': [
                'How can I control my wandering mind?',
                'What is the best meditation technique for beginners?',
                'How to develop concentration?'
            ]
        },
        'vyasa': {
            'best_for': ['life decisions', 'dharmic dilemmas', 'family issues', 'purpose questions'],
            'example_questions': [
                'What is my dharma in this situation?',
                'How should I handle this life challenge?',
                'What does the Gita say about duty?'
            ]
        },
        'valmiki': {
            'best_for': ['devotion practices', 'transformation', 'overcoming past', 'spiritual growth'],
            'example_questions': [
                'How can I develop devotion?',
                'Is transformation possible despite my past?',
                'How to surrender to the divine?'
            ]
        }
    }
