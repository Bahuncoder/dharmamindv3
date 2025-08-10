"""
Darshana Routes - Classical Hindu Philosophy Integration
======================================================

Handles philosophical query processing through the Six Classical Darshanas:
- Query classification and routing
- Philosophical response generation
- Sanskrit term integration
- Scriptural reference provision
- Multi-darshana perspective integration

🕉️ Serving wisdom through systematic philosophical inquiry
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import uuid

from ..chakra_modules.darshana_engine import (
    DarshanaEngine, get_darshana_engine, DarshanaType, PhilosophicalResponse
)
from ..services.memory_manager import MemoryManager, get_memory_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class PhilosophicalQuery(BaseModel):
    """Request for philosophical wisdom through darshana analysis"""
    query: str = Field(..., description="Philosophical question", min_length=1, max_length=2000)
    context: Optional[str] = Field(default=None, description="Additional context")
    preferred_darshana: Optional[str] = Field(default=None, description="Preferred philosophical school")
    user_background: str = Field(default="beginner", description="User philosophical background")
    sanskrit_preference: bool = Field(default=False, description="Include Sanskrit terms")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")


class DarshanaResponse(BaseModel):
    """Response from darshana philosophical analysis"""
    response: str
    primary_darshana: str
    secondary_perspectives: Dict[str, str]
    sanskrit_terms: List[Dict[str, str]]
    scriptural_references: List[str]
    practical_guidance: Optional[str]
    philosophical_depth_score: float
    integration_notes: Optional[str]
    conversation_id: str
    timestamp: datetime
    processing_time: float


class DarshanaInfo(BaseModel):
    """Information about available darshanas"""
    name: str
    sanskrit: str
    focus: str
    available: bool


class ComparativeQuery(BaseModel):
    """Request for comparative darshana analysis"""
    query: str = Field(..., description="Philosophical question", min_length=1, max_length=2000)
    darshanas: List[str] = Field(..., description="List of darshana names to compare")
    context: Optional[str] = Field(default=None, description="Additional context")
    user_background: str = Field(default="beginner", description="User philosophical background")
    sanskrit_preference: bool = Field(default=False, description="Include Sanskrit terms")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")


@router.post("/darshana/query", response_model=DarshanaResponse)
async def process_philosophical_query(
    query: PhilosophicalQuery,
    background_tasks: BackgroundTasks,
    darshana_engine: DarshanaEngine = Depends(get_darshana_engine),
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Process a philosophical query through appropriate darshana
    
    This endpoint:
    1. Classifies the query into appropriate darshana(s)
    2. Processes through philosophical frameworks
    3. Provides comprehensive response with Sanskrit terms
    4. Includes scriptural references and practical guidance
    5. Stores interaction for learning and context
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = query.conversation_id or str(uuid.uuid4())
        
        logger.info(f"Processing philosophical query: {query.query[:50]}...")
        
        # Prepare user preferences
        user_preferences = {
            "background": query.user_background,
            "sanskrit_preference": query.sanskrit_preference
        }
        
        if query.preferred_darshana:
            user_preferences["preferred_darshana"] = query.preferred_darshana
        
        # Process through darshana engine
        start_time = datetime.now()
        
        philosophical_response = await darshana_engine.process_philosophical_query(
            query=query.query,
            context=query.context,
            user_preferences=user_preferences
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store philosophical interaction
        background_tasks.add_task(
            memory_manager.store_philosophical_interaction,
            conversation_id,
            query.query,
            philosophical_response,
            query.user_id
        )
        
        # Prepare response
        response = DarshanaResponse(
            response=philosophical_response.primary_perspective,
            primary_darshana=philosophical_response.darshana_used.value,
            secondary_perspectives={
                darshana.value: perspective 
                for darshana, perspective in philosophical_response.secondary_perspectives.items()
            },
            sanskrit_terms=philosophical_response.sanskrit_terms,
            scriptural_references=philosophical_response.scriptural_references,
            practical_guidance=philosophical_response.practical_guidance,
            philosophical_depth_score=philosophical_response.philosophical_depth_score,
            integration_notes=philosophical_response.integration_notes,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            processing_time=processing_time
        )
        
        logger.info(f"Philosophical query processed successfully via {philosophical_response.darshana_used.value}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing philosophical query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing philosophical query: {str(e)}")


@router.get("/darshana/schools", response_model=List[DarshanaInfo])
async def get_available_darshanas(
    darshana_engine: DarshanaEngine = Depends(get_darshana_engine)
):
    """Get information about all available darshana schools"""
    try:
        darshana_info = darshana_engine.get_darshana_info()
        
        return [
            DarshanaInfo(
                name=darshana["name"],
                sanskrit=darshana["sanskrit"],
                focus=darshana["focus"],
                available=darshana["available"]
            )
            for darshana in darshana_info["available_darshanas"]
        ]
        
    except Exception as e:
        logger.error(f"Error retrieving darshana information: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving darshana information")


@router.get("/darshana/metrics")
async def get_darshana_metrics(
    darshana_engine: DarshanaEngine = Depends(get_darshana_engine)
):
    """Get darshana processing metrics"""
    try:
        darshana_info = darshana_engine.get_darshana_info()
        return {
            "available_schools": len([d for d in darshana_info["available_darshanas"] if d["available"]]),
            "total_schools": len(darshana_info["available_darshanas"]),
            "processing_metrics": darshana_info["processing_metrics"],
            "most_used_darshana": max(
                darshana_info["processing_metrics"]["darshana_usage"],
                key=darshana_info["processing_metrics"]["darshana_usage"].get
            ) if darshana_info["processing_metrics"]["darshana_usage"] else None
        }
        
    except Exception as e:
        logger.error(f"Error retrieving darshana metrics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving darshana metrics")


@router.post("/darshana/comparative")
async def comparative_analysis(
    query: ComparativeQuery,
    darshana_engine: DarshanaEngine = Depends(get_darshana_engine)
):
    """
    Get comparative analysis across multiple darshanas
    
    This endpoint processes the same query through multiple philosophical
    schools to show different perspectives on the same question.
    """
    try:
        if len(query.darshanas) > 4:
            raise HTTPException(status_code=400, detail="Maximum 4 darshanas allowed for comparison")
        
        logger.info(f"Comparative analysis across: {', '.join(query.darshanas)}")
        
        comparative_responses = {}
        
        for darshana_name in query.darshanas:
            try:
                # Validate darshana name
                darshana_type = DarshanaType(darshana_name)
                
                # Process with forced darshana preference
                user_preferences = {
                    "preferred_darshana": darshana_name,
                    "background": query.user_background,
                    "sanskrit_preference": query.sanskrit_preference
                }
                
                response = await darshana_engine.process_philosophical_query(
                    query=query.query,
                    context=query.context,
                    user_preferences=user_preferences
                )
                
                comparative_responses[darshana_name] = {
                    "perspective": response.primary_perspective,
                    "sanskrit_terms": response.sanskrit_terms,
                    "practical_guidance": response.practical_guidance,
                    "depth_score": response.philosophical_depth_score
                }
                
            except ValueError:
                logger.warning(f"Invalid darshana name: {darshana_name}")
                comparative_responses[darshana_name] = {
                    "error": f"Invalid darshana: {darshana_name}"
                }
        
        return {
            "query": query.query,
            "comparative_responses": comparative_responses,
            "synthesis": _generate_synthesis(comparative_responses),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error in comparative analysis: {str(e)}")


def _generate_synthesis(responses: Dict[str, Any]) -> str:
    """Generate a synthesis of multiple darshana perspectives"""
    valid_responses = {k: v for k, v in responses.items() if "error" not in v}
    
    if len(valid_responses) < 2:
        return "Insufficient valid responses for synthesis."
    
    synthesis = "Synthesis of Philosophical Perspectives:\n\n"
    
    # Identify common themes
    synthesis += "Each darshana offers a unique lens for understanding this question. "
    
    if "vedanta" in valid_responses:
        synthesis += "Vedanta provides the metaphysical foundation, "
    if "yoga" in valid_responses:
        synthesis += "Yoga offers practical methods, "
    if "nyaya" in valid_responses:
        synthesis += "Nyaya ensures logical rigor, "
    if "samkhya" in valid_responses:
        synthesis += "Samkhya explains the cosmic process, "
    
    synthesis += "together creating a comprehensive understanding that honors the systematic approach of Hindu philosophy."
    
    return synthesis


@router.get("/darshana/health")
async def darshana_health_check(
    darshana_engine: DarshanaEngine = Depends(get_darshana_engine)
):
    """Health check for darshana engine"""
    try:
        status = await darshana_engine.get_status()
        return {
            "status": "healthy",
            "service": "darshana_engine",
            "engine_status": status,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Darshana health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Darshana service health check failed: {str(e)}")
