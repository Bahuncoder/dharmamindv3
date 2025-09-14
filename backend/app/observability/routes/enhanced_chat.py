"""
Enhanced DharmaLLM Chat Route - Demonstrating Enterprise Integration

This new endpoint showcases how the frontend will use DharmaLLM enterprise features:
- Advanced evaluation metrics
- Cultural sensitivity analysis
- Model selection and quality scoring
- Real-time wisdom assessment
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import uuid
import sys
import os

# Add DharmaLLM to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import DharmaLLM enterprise components
try:
    from dharmallm import (
        create_config, 
        create_evaluator, 
        create_data_processor,
        DharmaLLMAdvancedEvaluator,
        WisdomTradition,
        DharmicPrinciple
    )
    DHARMALLM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"DharmaLLM not available: {e}")
    DHARMALLM_AVAILABLE = False

from ..models import ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()

class EnhancedChatRequest(BaseModel):
    """Enhanced chat request with DharmaLLM enterprise features"""
    message: str = Field(..., description="User message", min_length=1, max_length=2000)
    user_id: Optional[str] = Field(default=None, description="User identifier")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")
    
    # DharmaLLM Enterprise Features
    model_preference: str = Field(default="dharmallm-7b", description="Preferred AI model")
    tradition: str = Field(default="universal", description="Wisdom tradition preference")
    cultural_context: Optional[Dict[str, Any]] = Field(default=None, description="Cultural context")
    evaluation_level: str = Field(default="standard", description="Evaluation depth: basic, standard, comprehensive")
    quality_threshold: float = Field(default=0.7, description="Minimum quality threshold")

class EnhancedChatResponse(BaseModel):
    """Enhanced response with enterprise metrics"""
    response: str
    conversation_id: str
    
    # Standard metrics
    confidence_score: float
    dharmic_alignment: float
    modules_used: List[str]
    timestamp: str
    model_used: str
    processing_time: float
    
    # Enterprise metrics
    cultural_sensitivity: Optional[float] = None
    compassion_score: Optional[float] = None
    wisdom_assessment: Optional[Dict[str, float]] = None
    safety_score: Optional[float] = None
    tradition_alignment: Optional[float] = None
    
    # Quality indicators
    quality_gates_passed: bool = True
    evaluation_details: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    """Available model information"""
    id: str
    name: str
    tradition: str
    size: str
    description: str
    available: bool

# Initialize DharmaLLM components if available
dharmallm_config = None
dharmallm_evaluator = None

if DHARMALLM_AVAILABLE:
    try:
        dharmallm_config = create_config('development')
        dharmallm_evaluator = create_evaluator(dharmallm_config)
        logger.info("DharmaLLM enterprise components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DharmaLLM components: {e}")
        DHARMALLM_AVAILABLE = False

@router.post("/v1/chat/enhanced", response_model=EnhancedChatResponse)
async def enhanced_chat_message(
    request: EnhancedChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Enhanced chat endpoint with DharmaLLM enterprise features
    
    Demonstrates integration of:
    - Advanced evaluation metrics
    - Cultural sensitivity analysis
    - Quality scoring and validation
    - Multiple wisdom traditions
    """
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Processing enhanced chat for conversation {conversation_id}")
        
        # Generate base response (using mock for now, would integrate with actual LLM)
        base_response = generate_dharmic_response(request.message, request.tradition)
        
        # Initialize response object
        response_data = {
            "response": base_response,
            "conversation_id": conversation_id,
            "confidence_score": 0.85,
            "dharmic_alignment": 0.92,
            "modules_used": ["consciousness_core", "dharma_engine", "wisdom_validator"],
            "timestamp": datetime.now().isoformat(),
            "model_used": request.model_preference,
            "processing_time": 0.0,
            "quality_gates_passed": True
        }
        
        # Apply DharmaLLM enterprise evaluation if available
        if DHARMALLM_AVAILABLE and dharmallm_evaluator:
            try:
                # Perform comprehensive evaluation
                evaluation_result = await perform_dharma_evaluation(
                    base_response, 
                    request.message,
                    request.tradition,
                    request.evaluation_level
                )
                
                # Add enterprise metrics to response
                response_data.update({
                    "cultural_sensitivity": evaluation_result.get("cultural_sensitivity", 0.85),
                    "compassion_score": evaluation_result.get("compassion_score", 0.88),
                    "wisdom_assessment": evaluation_result.get("wisdom_assessment", {
                        "tradition_alignment": 0.90,
                        "practical_applicability": 0.85,
                        "spiritual_depth": 0.87
                    }),
                    "safety_score": evaluation_result.get("safety_score", 0.95),
                    "tradition_alignment": evaluation_result.get("tradition_alignment", 0.90),
                    "evaluation_details": evaluation_result.get("details", {})
                })
                
                # Check quality gates
                if response_data["cultural_sensitivity"] < request.quality_threshold:
                    response_data["quality_gates_passed"] = False
                    
            except Exception as e:
                logger.error(f"DharmaLLM evaluation failed: {e}")
                # Continue with basic response if evaluation fails
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        response_data["processing_time"] = processing_time
        
        logger.info(f"Enhanced chat completed in {processing_time:.3f}s")
        
        return EnhancedChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process enhanced chat: {str(e)}")

@router.get("/v1/models/available", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of available DharmaLLM models"""
    models = [
        ModelInfo(
            id="dharmallm-7b",
            name="DharmaLLM 7B Enhanced",
            tradition="universal",
            size="7B parameters",
            description="Universal dharmic wisdom model with multi-tradition support",
            available=DHARMALLM_AVAILABLE
        ),
        ModelInfo(
            id="dharmallm-hindu-specialized",
            name="DharmaLLM Hindu Specialized",
            tradition="hindu",
            size="7B parameters",
            description="Specialized model for Hindu dharmic wisdom and scriptures",
            available=DHARMALLM_AVAILABLE
        ),
        ModelInfo(
            id="dharmallm-buddhist-specialized", 
            name="DharmaLLM Buddhist Specialized",
            tradition="buddhist",
            size="7B parameters",
            description="Specialized model for Buddhist teachings and mindfulness",
            available=DHARMALLM_AVAILABLE
        ),
        ModelInfo(
            id="dharmallm-compassion-focused",
            name="DharmaLLM Compassion Focus",
            tradition="universal",
            size="7B parameters", 
            description="Model optimized for compassionate and empathetic responses",
            available=DHARMALLM_AVAILABLE
        )
    ]
    
    return models

@router.post("/v1/evaluate/response")
async def evaluate_response_quality(request: Dict[str, Any]):
    """Evaluate response quality using DharmaLLM enterprise evaluator"""
    if not DHARMALLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="DharmaLLM evaluation not available")
    
    try:
        text = request.get("text", "")
        context = request.get("context", "")
        tradition = request.get("tradition", "universal")
        
        evaluation = await perform_dharma_evaluation(text, context, tradition, "comprehensive")
        
        return {
            "status": "success",
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# Helper functions

def generate_dharmic_response(message: str, tradition: str) -> str:
    """Generate a dharmic response based on message and tradition"""
    # This would integrate with actual LLM, using mock response for demo
    tradition_context = {
        "hindu": "Drawing from the eternal wisdom of the Vedas and Upanishads",
        "buddhist": "Following the Noble Eightfold Path and Buddha's teachings",
        "universal": "Embracing wisdom from all traditions for universal truth",
        "jain": "Following the path of ahimsa and spiritual liberation",
        "sikh": "Guided by the teachings of the Gurus and service to humanity"
    }
    
    context = tradition_context.get(tradition, tradition_context["universal"])
    
    return f"🕉️ {context}, I offer this guidance: Every challenge you face is an opportunity for spiritual growth. Your question shows deep contemplation, and the very act of seeking wisdom demonstrates your commitment to dharmic living. May you find clarity and peace on your journey."

async def perform_dharma_evaluation(
    response: str, 
    original_message: str, 
    tradition: str,
    level: str
) -> Dict[str, Any]:
    """Perform DharmaLLM enterprise evaluation"""
    # Mock evaluation results - would use actual DharmaLLM evaluator
    base_scores = {
        "cultural_sensitivity": 0.85 + (len(response) / 1000) * 0.1,
        "compassion_score": 0.88 + (response.count("compassion") + response.count("love")) * 0.02,
        "safety_score": 0.95,
        "tradition_alignment": 0.90 if tradition in response.lower() else 0.85,
        "wisdom_assessment": {
            "tradition_alignment": 0.90,
            "practical_applicability": 0.85,
            "spiritual_depth": 0.87
        },
        "details": {
            "evaluation_level": level,
            "tradition_context": tradition,
            "response_length": len(response),
            "spiritual_keywords_found": response.count("dharma") + response.count("wisdom") + response.count("spiritual")
        }
    }
    
    # Adjust scores based on evaluation level
    if level == "comprehensive":
        # More detailed analysis
        base_scores["details"]["comprehensive_analysis"] = {
            "emotional_intelligence": 0.87,
            "practical_guidance": 0.84,
            "scriptural_alignment": 0.91,
            "universal_applicability": 0.88
        }
    
    return base_scores
