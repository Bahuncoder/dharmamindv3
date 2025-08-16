"""
ChatGPT-Style Response Configuration

This configuration allows DharmaMind to respond more like ChatGPT
while optionally maintaining spiritual enhancements.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

from ..services.llm_router import get_llm_router
from ..services.local_llm import get_local_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/simple", tags=["Simple Chat"])

class SimpleChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"
    use_spiritual_enhancement: Optional[bool] = False
    model_preference: Optional[str] = "external"  # "external", "local", "auto"

class SimpleChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    model_used: str
    processing_time: float
    spiritual_enhancement: bool
    timestamp: str

@router.post("/chat", response_model=SimpleChatResponse)
async def simple_chat(request: SimpleChatRequest):
    """
    Simple ChatGPT-style chat endpoint (minimal spiritual processing)
    
    This endpoint provides more direct, ChatGPT-like responses by:
    1. Bypassing heavy Chakra module processing
    2. Using direct LLM responses 
    3. Optional spiritual enhancement only
    """
    import time
    from datetime import datetime
    
    start_time = time.time()
    
    try:
        logger.info(f"Simple chat request: {request.message[:50]}...")
        
        # Choose response method based on preference
        if request.model_preference == "local":
            # Use local LLM for direct response
            local_service = await get_local_llm_service()
            result = await local_service.generate_response(
                message=request.message,
                model_name="distilgpt2",
                max_length=512,
                temperature=0.7
            )
            response_text = result["content"]
            model_used = f"local-{result['model_name']}"
            
        else:
            # Use external LLM router for ChatGPT-style response
            llm_router = get_llm_router()
            
            # Direct LLM call without heavy Chakra processing
            response_data = await llm_router.generate_response(
                message=request.message,
                context="You are a helpful AI assistant. Respond naturally and conversationally.",
                max_tokens=512,
                temperature=0.7
            )
            
            response_text = response_data.get("content", "I apologize, but I couldn't generate a response.")
            model_used = response_data.get("model", "external-llm")
        
        # Optional light spiritual enhancement
        if request.use_spiritual_enhancement:
            # Add minimal dharmic wisdom without heavy processing
            if any(word in request.message.lower() for word in ["meaning", "purpose", "life", "wisdom", "spiritual"]):
                response_text += "\n\n🕉️ *May this response serve your highest good.*"
        
        processing_time = time.time() - start_time
        
        return SimpleChatResponse(
            response=response_text,
            conversation_id=request.conversation_id or f"simple-{int(time.time())}",
            message_id=f"msg-{int(time.time() * 1000)}",
            model_used=model_used,
            processing_time=processing_time,
            spiritual_enhancement=request.use_spiritual_enhancement,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Simple chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.post("/direct-llm")
async def direct_llm_chat(message: str, model: str = "auto"):
    """
    Direct LLM access - closest to ChatGPT behavior
    No spiritual processing, no Chakra modules, just raw LLM
    """
    try:
        if model == "local":
            local_service = await get_local_llm_service()
            result = await local_service.generate_response(
                message=message,
                model_name="distilgpt2",
                max_length=512,
                temperature=0.8
            )
            return {
                "response": result["content"],
                "model": result["model_name"],
                "processing_time": result["processing_time"],
                "type": "direct_local"
            }
        else:
            llm_router = get_llm_router()
            response = await llm_router.generate_response(
                message=message,
                context="Be helpful and conversational.",
                max_tokens=512
            )
            return {
                "response": response.get("content", "No response generated"),
                "model": response.get("model", "external"),
                "type": "direct_external"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_available_chat_models():
    """Get available models for simple chat"""
    return {
        "external_models": ["gpt-3.5-turbo", "gpt-4", "claude-3"],
        "local_models": ["distilgpt2", "microsoft/DialoGPT-small"],
        "recommendation": "Use 'external' for ChatGPT-like responses, 'local' for privacy"
    }
