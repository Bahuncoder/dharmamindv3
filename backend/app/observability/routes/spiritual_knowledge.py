"""
Spiritual Knowledge API Routes
=============================

API endpoints for accessing the spiritual knowledge base,
including semantic search, wisdom retrieval, and practice recommendations.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/knowledge", tags=["spiritual-knowledge"])

# Pydantic models for request/response
class WisdomSearchRequest(BaseModel):
    """Request model for wisdom search"""
    query: str = Field(..., description="Search query for spiritual wisdom")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    category: Optional[str] = Field(None, description="Filter by category")
    tradition: Optional[str] = Field(None, description="Filter by spiritual tradition")
    wisdom_level: Optional[str] = Field(None, description="Filter by wisdom level (practical, deep, fundamental)")

class WisdomItem(BaseModel):
    """Response model for a wisdom item"""
    title: str
    text: str
    source: str
    category: str
    tradition: str
    wisdom_level: str
    tags: List[str]
    relevance_score: float
    match_reasons: List[str]

class WisdomSearchResponse(BaseModel):
    """Response model for wisdom search"""
    query: str
    results: List[WisdomItem]
    total_found: int
    search_metadata: Dict[str, Any]

class GuidanceRequest(BaseModel):
    """Request model for situational guidance"""
    situation: str = Field(..., description="Life situation or challenge")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    limit: int = Field(default=3, ge=1, le=10, description="Maximum number of guidance items")

class PracticeRequest(BaseModel):
    """Request model for practice recommendations"""
    area_of_focus: str = Field(..., description="Area of spiritual focus")
    difficulty_level: Optional[str] = Field(None, description="Difficulty level (beginner, intermediate, advanced)")
    duration_preference: Optional[str] = Field(None, description="Time preference (short, medium, long)")
    limit: int = Field(default=3, ge=1, le=10, description="Maximum number of practices")

class EmotionRequest(BaseModel):
    """Request model for emotional guidance"""
    emotion: str = Field(..., description="Emotion to address")
    intensity: Optional[str] = Field(None, description="Intensity level (mild, moderate, intense)")
    context: Optional[str] = Field(None, description="Context around the emotion")
    limit: int = Field(default=3, ge=1, le=10, description="Maximum number of wisdom items")

# Dependency to get knowledge base
async def get_knowledge_base():
    """Get the spiritual knowledge base instance"""
    try:
        from ...knowledge_base.spiritual_knowledge_retrieval import get_knowledge_base
        return await get_knowledge_base()
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Knowledge base unavailable")

# API Endpoints

@router.post("/search", response_model=WisdomSearchResponse)
async def search_wisdom(
    request: WisdomSearchRequest,
    kb = Depends(get_knowledge_base)
):
    """
    Search for spiritual wisdom using semantic similarity.
    
    This endpoint provides intelligent search across the spiritual knowledge base,
    returning relevant wisdom, teachings, and guidance based on semantic similarity.
    """
    try:
        logger.info(f"Wisdom search request: '{request.query}'")
        
        # Import here to avoid circular imports
        from ...knowledge_base.spiritual_knowledge_retrieval import search_spiritual_wisdom
        
        # Perform the search
        results = await search_spiritual_wisdom(
            query=request.query,
            limit=request.limit,
            category=request.category,
            tradition=request.tradition
        )
        
        # Convert to response format
        wisdom_items = [
            WisdomItem(
                title=item['title'],
                text=item['text'],
                source=item['source'],
                category=item['category'],
                tradition=item['tradition'],
                wisdom_level=item['wisdom_level'],
                tags=item['tags'],
                relevance_score=item['relevance_score'],
                match_reasons=item['match_reasons']
            )
            for item in results
        ]
        
        return WisdomSearchResponse(
            query=request.query,
            results=wisdom_items,
            total_found=len(wisdom_items),
            search_metadata={
                "category_filter": request.category,
                "tradition_filter": request.tradition,
                "wisdom_level_filter": request.wisdom_level
            }
        )
        
    except Exception as e:
        logger.error(f"Wisdom search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/guidance", response_model=WisdomSearchResponse)
async def get_situational_guidance(
    request: GuidanceRequest,
    kb = Depends(get_knowledge_base)
):
    """
    Get spiritual guidance for specific life situations.
    
    This endpoint provides targeted wisdom and guidance for life challenges,
    decisions, and situations based on contextual understanding.
    """
    try:
        logger.info(f"Guidance request for situation: '{request.situation}'")
        
        # Get guidance from knowledge base
        results = await kb.get_guidance_for_situation(
            situation=request.situation,
            context=request.context
        )
        
        # Limit results
        results = results[:request.limit]
        
        # Convert to response format
        wisdom_items = [
            WisdomItem(
                title=item['title'],
                text=item['text'],
                source=item['source'],
                category=item['category'],
                tradition=item['tradition'],
                wisdom_level=item['wisdom_level'],
                tags=item['tags'],
                relevance_score=item['relevance_score'],
                match_reasons=item['match_reasons']
            )
            for item in results
        ]
        
        return WisdomSearchResponse(
            query=f"Guidance for: {request.situation}",
            results=wisdom_items,
            total_found=len(wisdom_items),
            search_metadata={
                "situation": request.situation,
                "context": request.context,
                "guidance_type": "situational"
            }
        )
        
    except Exception as e:
        logger.error(f"Guidance request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Guidance request failed: {str(e)}")

@router.post("/practices", response_model=WisdomSearchResponse)
async def get_practice_recommendations(
    request: PracticeRequest,
    kb = Depends(get_knowledge_base)
):
    """
    Get spiritual practice recommendations.
    
    This endpoint provides personalized spiritual practice recommendations
    based on areas of focus, difficulty level, and time preferences.
    """
    try:
        logger.info(f"Practice request for: '{request.area_of_focus}'")
        
        # Get practice recommendations
        results = await kb.get_practice_recommendations(
            area_of_focus=request.area_of_focus,
            difficulty_level=request.difficulty_level
        )
        
        # Limit results
        results = results[:request.limit]
        
        # Convert to response format
        wisdom_items = [
            WisdomItem(
                title=item['title'],
                text=item['text'],
                source=item['source'],
                category=item['category'],
                tradition=item['tradition'],
                wisdom_level=item['wisdom_level'],
                tags=item['tags'],
                relevance_score=item['relevance_score'],
                match_reasons=item['match_reasons']
            )
            for item in results
        ]
        
        return WisdomSearchResponse(
            query=f"Practices for: {request.area_of_focus}",
            results=wisdom_items,
            total_found=len(wisdom_items),
            search_metadata={
                "area_of_focus": request.area_of_focus,
                "difficulty_level": request.difficulty_level,
                "duration_preference": request.duration_preference,
                "recommendation_type": "practice"
            }
        )
        
    except Exception as e:
        logger.error(f"Practice recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Practice recommendation failed: {str(e)}")

@router.post("/emotions", response_model=WisdomSearchResponse)
async def get_emotional_guidance(
    request: EmotionRequest,
    kb = Depends(get_knowledge_base)
):
    """
    Get wisdom for dealing with specific emotions.
    
    This endpoint provides targeted wisdom and guidance for processing
    and transforming difficult emotions into wisdom and growth.
    """
    try:
        logger.info(f"Emotional guidance request for: '{request.emotion}'")
        
        # Get emotional wisdom
        results = await kb.get_wisdom_for_emotion(request.emotion)
        
        # Limit results
        results = results[:request.limit]
        
        # Convert to response format
        wisdom_items = [
            WisdomItem(
                title=item['title'],
                text=item['text'],
                source=item['source'],
                category=item['category'],
                tradition=item['tradition'],
                wisdom_level=item['wisdom_level'],
                tags=item['tags'],
                relevance_score=item['relevance_score'],
                match_reasons=item['match_reasons']
            )
            for item in results
        ]
        
        return WisdomSearchResponse(
            query=f"Wisdom for emotion: {request.emotion}",
            results=wisdom_items,
            total_found=len(wisdom_items),
            search_metadata={
                "emotion": request.emotion,
                "intensity": request.intensity,
                "context": request.context,
                "guidance_type": "emotional"
            }
        )
        
    except Exception as e:
        logger.error(f"Emotional guidance failed: {e}")
        raise HTTPException(status_code=500, detail=f"Emotional guidance failed: {str(e)}")

@router.get("/categories")
async def get_available_categories(kb = Depends(get_knowledge_base)):
    """Get all available knowledge categories"""
    try:
        stats = kb.get_statistics()
        return {
            "categories": sorted(stats['categories']),
            "total_categories": len(stats['categories'])
        }
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve categories")

@router.get("/traditions")
async def get_available_traditions(kb = Depends(get_knowledge_base)):
    """Get all available spiritual traditions"""
    try:
        stats = kb.get_statistics()
        return {
            "traditions": sorted(stats['traditions']),
            "total_traditions": len(stats['traditions'])
        }
    except Exception as e:
        logger.error(f"Failed to get traditions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve traditions")

@router.get("/stats")
async def get_knowledge_base_statistics(kb = Depends(get_knowledge_base)):
    """Get comprehensive knowledge base statistics"""
    try:
        stats = kb.get_statistics()
        return {
            "total_items": stats['total_items'],
            "categories": {
                "list": sorted(stats['categories']),
                "count": len(stats['categories'])
            },
            "traditions": {
                "list": sorted(stats['traditions']),
                "count": len(stats['traditions'])
            },
            "wisdom_levels": sorted(stats['wisdom_levels']),
            "total_tags": stats['total_tags'],
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@router.get("/random")
async def get_random_wisdom(
    limit: int = Query(default=1, ge=1, le=5, description="Number of random wisdom items"),
    kb = Depends(get_knowledge_base)
):
    """Get random wisdom items for inspiration"""
    try:
        random_items = await kb.get_random_wisdom(limit=limit)
        
        wisdom_items = [
            WisdomItem(
                title=item.title,
                text=item.text,
                source=item.source,
                category=item.category,
                tradition=item.tradition,
                wisdom_level=item.wisdom_level,
                tags=item.tags,
                relevance_score=1.0,  # Random items have perfect "relevance" to randomness
                match_reasons=["Random selection"]
            )
            for item in random_items
        ]
        
        return {
            "results": wisdom_items,
            "total_found": len(wisdom_items),
            "selection_type": "random"
        }
        
    except Exception as e:
        logger.error(f"Random wisdom failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get random wisdom")

# Health check endpoint
@router.get("/health")
async def knowledge_health_check():
    """Health check for the spiritual knowledge system"""
    try:
        # Try to initialize knowledge base
        kb = await get_knowledge_base()
        stats = kb.get_statistics()
        
        return {
            "status": "healthy",
            "knowledge_items": stats['total_items'],
            "categories": len(stats['categories']),
            "traditions": len(stats['traditions']),
            "system": "operational"
        }
    except Exception as e:
        logger.error(f"Knowledge system health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "system": "degraded"
        }
