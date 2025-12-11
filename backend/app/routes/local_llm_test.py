"""
Local LLM Test Endpoint

This endpoint demonstrates how to use local LLMs without API keys.
Shows real text generation using Hugging Face models running locally.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

from ..services.local_llm import get_local_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/local", tags=["Local LLM"])

class LocalLLMRequest(BaseModel):
    message: str
    model_name: Optional[str] = "distilgpt2"
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    context: Optional[str] = None

class LocalLLMResponse(BaseModel):
    response: str
    model_name: str
    processing_time: float
    tokens_used: int
    device: str
    metadata: dict

@router.post("/generate", response_model=LocalLLMResponse)
async def generate_local_response(request: LocalLLMRequest):
    """
    Generate response using local LLM models (no API key required)
    
    Available models:
    - distilgpt2 (lightweight, fast)
    - gpt2 (medium size)
    - microsoft/DialoGPT-small (conversational)
    - microsoft/DialoGPT-medium (better quality)
    - facebook/blenderbot-400M-distill (chatbot optimized)
    """
    try:
        logger.info(f"Generating local response with model: {request.model_name}")
        
        # Get local LLM service
        local_service = await get_local_llm_service()
        
        # Generate response
        result = await local_service.generate_response(
            message=request.message,
            model_name=request.model_name,
            max_length=request.max_length,
            temperature=request.temperature,
            context=request.context
        )
        
        return LocalLLMResponse(
            response=result["content"],
            model_name=result["model_name"],
            processing_time=result["processing_time"],
            tokens_used=result["tokens_used"],
            device=result["metadata"]["device"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Local LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Local LLM error: {str(e)}")

@router.get("/models")
async def get_available_models():
    """Get list of available local models"""
    try:
        local_service = await get_local_llm_service()
        models = await local_service.get_available_models()
        return {"available_models": models}
        
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")

@router.get("/memory")
async def get_memory_usage():
    """Get current memory usage"""
    try:
        local_service = await get_local_llm_service()
        memory_info = await local_service.get_memory_usage()
        return memory_info
        
    except Exception as e:
        logger.error(f"Failed to get memory info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting memory info: {str(e)}")

@router.post("/load-model")
async def load_model(model_name: str):
    """Load a specific model"""
    try:
        local_service = await get_local_llm_service()
        success = await local_service.load_model(model_name)
        
        if success:
            return {"message": f"Model {model_name} loaded successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@router.post("/unload-model")
async def unload_model(model_name: str):
    """Unload a specific model to free memory"""
    try:
        local_service = await get_local_llm_service()
        await local_service.unload_model(model_name)
        return {"message": f"Model {model_name} unloaded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(status_code=500, detail=f"Error unloading model: {str(e)}")

# Example usage with different models
@router.post("/test-models")
async def test_multiple_models(message: str = "Hello, how are you?"):
    """Test the same message with different local models"""
    try:
        local_service = await get_local_llm_service()
        
        # Test with lightweight models
        test_models = ["distilgpt2", "microsoft/DialoGPT-small"]
        results = []
        
        for model_name in test_models:
            try:
                result = await local_service.generate_response(
                    message=message,
                    model_name=model_name,
                    max_length=256,
                    temperature=0.8
                )
                results.append({
                    "model": model_name,
                    "response": result["content"],
                    "processing_time": result["processing_time"],
                    "device": result["metadata"]["device"]
                })
            except Exception as model_error:
                results.append({
                    "model": model_name,
                    "error": str(model_error)
                })
        
        return {"message": message, "results": results}
        
    except Exception as e:
        logger.error(f"Failed to test models: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing models: {str(e)}")
