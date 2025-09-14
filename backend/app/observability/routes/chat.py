"""
Chat Routes - Main interaction endpoints for DharmaMind

Handles all chat-related API endpoints including:
- Message processing and response generation
- Conversation history management
- Module routing and selection
- Response evaluation and scoring

🕉️ Serving wisdom through technology
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import uuid

# Import services with fallbacks
try:
    from ...services.llm_router import get_llm_router, LLMRouter
    from ...services.module_selector import get_module_selector, ModuleSelector
    from ...services.evaluator import get_response_evaluator, ResponseEvaluator
    from ...services.memory_manager import get_memory_manager, MemoryManager
    from ...engines.rishi import create_authentic_rishi_engine
    from ...engines.emotional import create_emotional_engine
except ImportError as e:
    logging.warning(f"Service imports failed: {e}")
    
    # Fallback classes
    class LLMRouter:
        pass
    class ModuleSelector:
        pass  
    class ResponseEvaluator:
        pass
    class MemoryManager:
        pass
    
    def get_llm_router():
        return LLMRouter()
    def get_module_selector():
        return ModuleSelector()
    def get_response_evaluator():
        return ResponseEvaluator()
    def get_memory_manager():
        return MemoryManager()
    def create_authentic_rishi_engine():
        return None
    def create_emotional_engine():
        return None

from ..models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatMessage(BaseModel):
    """Individual chat message"""
    message: str = Field(..., description="User message", min_length=1, max_length=2000)
    context: Optional[str] = Field(default=None, description="Additional context")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")
    language: str = Field(default="en", description="Response language preference")
    category: Optional[str] = Field(default=None, description="Wisdom category (dharma, karma, etc.)")
    urgency: str = Field(default="normal", description="Request urgency: low, normal, high")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="User context")

class ConversationHistory(BaseModel):
    """Conversation history model"""
    conversation_id: str
    messages: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    total_messages: int
    
class WisdomRequest(BaseModel):
    """Request for dharmic wisdom"""
    question: str = Field(..., description="Wisdom question", min_length=1)
@router.post("/chat", response_model=ChatResponse)
async def chat_message(
    chat_request: ChatMessage,
    background_tasks: BackgroundTasks,
    llm_router: LLMRouter = Depends(get_llm_router),
    module_selector: ModuleSelector = Depends(get_module_selector),
    evaluator: ResponseEvaluator = Depends(get_response_evaluator),
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Process a chat message and return AI response
    
    This endpoint orchestrates the complete DharmaMind response flow:
    1. Select best-fit Dharma modules
    2. Route to appropriate LLM (GPT-4, DharmaLLM, etc.)
    3. Generate and evaluate response
    4. Store in memory/vector DB
    5. Return response with metadata
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = chat_request.conversation_id or str(uuid.uuid4())
        
        logger.info(f"Processing chat message for conversation {conversation_id}")
        
        # 1. Retrieve conversation history
        conversation_history = await memory_manager.get_conversation_history(
            conversation_id, limit=10
        )
        
        # 2. Select best-fit modules based on message content
        selected_modules = await module_selector.select_modules(
            message=chat_request.message,
            context=chat_request.context,
            history=conversation_history
        )
        
        logger.info(f"Selected modules: {[m.name for m in selected_modules]}")
        
        # 3. Route to appropriate LLM and generate response
        llm_response = await llm_router.generate_response(
            message=chat_request.message,
            context=chat_request.context,
            modules=selected_modules,
            language=chat_request.language,
            history=conversation_history
        )
        
        # 4. Evaluate response quality and dharmic alignment
        evaluation = await evaluator.evaluate_response(
            question=chat_request.message,
            response=llm_response.content,
            modules=selected_modules,
            context=chat_request.context
        )
        
        # 5. Store conversation in memory
        background_tasks.add_task(
            memory_manager.store_conversation,
            conversation_id,
            chat_request.message,
            llm_response.content,
            selected_modules,
            evaluation
        )
        
        # 6. Prepare response
        response = ChatResponse(
            response=llm_response.content,
            conversation_id=conversation_id,
            modules_used=[m.name for m in selected_modules],
            confidence_score=evaluation.confidence_score,
            dharmic_alignment=evaluation.dharmic_alignment,
            sources=evaluation.sources,
            suggestions=evaluation.suggestions,
            timestamp=datetime.now(),
            model_used=llm_response.model_name,
            processing_time=llm_response.processing_time
        )
        
        logger.info(f"Chat response generated successfully for conversation {conversation_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.post("/wisdom", response_model=ChatResponse)
async def ask_wisdom(
    wisdom_request: WisdomRequest,
    background_tasks: BackgroundTasks,
    llm_router: LLMRouter = Depends(get_llm_router),
    module_selector: ModuleSelector = Depends(get_module_selector),
    evaluator: ResponseEvaluator = Depends(get_response_evaluator),
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Ask for specific dharmic wisdom
    
    Specialized endpoint for seeking spiritual guidance and wisdom.
    Optimized for dharmic content and scriptural references.
    """
    try:
        logger.info(f"Wisdom request: {wisdom_request.question[:50]}...")
        
        # Select wisdom-focused modules
        selected_modules = await module_selector.select_wisdom_modules(
            question=wisdom_request.question,
            category=wisdom_request.category,
            urgency=wisdom_request.urgency
        )
        
        # Generate wisdom response
        llm_response = await llm_router.generate_wisdom_response(
            question=wisdom_request.question,
            modules=selected_modules,
            category=wisdom_request.category,
            user_context=wisdom_request.user_context
        )
        
        # Evaluate dharmic alignment
        evaluation = await evaluator.evaluate_wisdom_response(
            question=wisdom_request.question,
            response=llm_response.content,
            modules=selected_modules
        )
        
        # Store wisdom interaction
        conversation_id = str(uuid.uuid4())
        background_tasks.add_task(
            memory_manager.store_wisdom_interaction,
            conversation_id,
            wisdom_request.question,
            llm_response.content,
            selected_modules,
            evaluation
        )
        
        response = ChatResponse(
            response=llm_response.content,
            conversation_id=conversation_id,
            modules_used=[m.name for m in selected_modules],
            confidence_score=evaluation.confidence_score,
            dharmic_alignment=evaluation.dharmic_alignment,
            sources=evaluation.sources,
            suggestions=evaluation.suggestions,
            timestamp=datetime.now(),
            model_used=llm_response.model_name,
            processing_time=llm_response.processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing wisdom request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing wisdom request: {str(e)}")

@router.get("/conversation/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(
    conversation_id: str,
    limit: int = 20,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Get conversation history"""
    try:
        history = await memory_manager.get_conversation_history(conversation_id, limit)
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=history,
            total_messages=len(history)
        )
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving conversation history")

@router.delete("/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Delete conversation history"""
    try:
        await memory_manager.delete_conversation(conversation_id)
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail="Error deleting conversation")

@router.get("/modules")
async def get_available_modules(
    module_selector: ModuleSelector = Depends(get_module_selector)
):
    """Get list of available Dharma modules"""
    try:
        modules = await module_selector.get_available_modules()
        return {
            "modules": [
                {
                    "name": module.name,
                    "description": module.description,
                    "category": module.category,
                    "expertise": module.expertise_areas
                }
                for module in modules
            ]
        }
    except Exception as e:
        logger.error(f"Error retrieving modules: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving modules")

@router.get("/health")
async def chat_health_check():
    """Health check for chat service"""
    return {
        "status": "healthy",
        "service": "chat",
        "timestamp": datetime.now()
    }
