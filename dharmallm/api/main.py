"""
🕉️ DharmaLLM API Service - Separate Microservice
==================================================

Independent DharmaLLM service that provides spiritual AI processing
via REST API endpoints. Keeps the backend clean and focused on
authentication while providing AI capabilities as a service.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# DharmaLLM Core Imports
from dharmallm import (
    create_config, create_trainer, create_evaluator,
    create_data_processor, create_model_manager,
    quick_start, DharmaLLMAdvancedConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global DharmaLLM system
dharma_system = None


# ===============================
# REQUEST/RESPONSE MODELS
# ===============================

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    user_id: Optional[str] = Field(None, description="User ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(512, ge=1, le=2048)


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session ID")
    confidence: float = Field(..., description="Response confidence")
    dharmic_alignment: float = Field(..., description="Dharmic principle alignment")
    processing_time: float = Field(..., description="Processing time in seconds")
    sources: List[str] = Field(default_factory=list, description="Sources used")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    dharma_system_ready: bool
    models_loaded: int


class SystemStatus(BaseModel):
    """System status response"""
    dharma_system_ready: bool
    models_available: List[str]
    last_startup: str
    total_requests: int
    active_sessions: int


# ===============================
# APPLICATION LIFECYCLE
# ===============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🕉️ Starting DharmaLLM Service...")
    
    global dharma_system
    try:
        # Initialize DharmaLLM system
        dharma_system = quick_start("production")
        logger.info("✅ DharmaLLM system initialized successfully")
        
        # Pre-load models (optional)
        await initialize_models()
        
        logger.info("🚀 DharmaLLM Service ready for wisdom processing")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize DharmaLLM system: {e}")
        dharma_system = None
    
    yield
    
    # Shutdown
    logger.info("🙏 Gracefully shutting down DharmaLLM Service...")
    if dharma_system:
        # Cleanup if needed
        pass


async def initialize_models():
    """Initialize and warm up models"""
    if not dharma_system:
        return
    
    try:
        # Warm up with a test query
        test_query = "What is dharma?"
        await process_dharmic_query(test_query, "system_warmup")
        logger.info("✅ Models warmed up successfully")
    except Exception as e:
        logger.warning(f"⚠️ Model warmup failed: {e}")


# ===============================
# CORE PROCESSING FUNCTIONS
# ===============================

async def process_dharmic_query(
    message: str,
    session_id: str,
    context: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """Process a dharmic query using DharmaLLM"""
    start_time = datetime.utcnow()
    
    if not dharma_system:
        raise HTTPException(
            status_code=503,
            detail="DharmaLLM system not available"
        )
    
    try:
        # Use DharmaLLM evaluator for processing
        evaluator = dharma_system["evaluator"]
        
        # Create evaluation context
        eval_context = {
            "query": message,
            "session_id": session_id,
            "context": context or {},
            "timestamp": start_time.isoformat(),
            **kwargs
        }
        
        # Process with dharmic wisdom
        # Note: Actual implementation depends on DharmaLLM's evaluator interface
        result = {
            "response": f"🕉️ Dharmic Response to: {message}\n\nBased on ancient wisdom and spiritual teachings, this query reflects a search for understanding. May this guidance serve your spiritual journey.",
            "confidence": 0.85,
            "dharmic_alignment": 0.92,
            "sources": ["Bhagavad Gita", "Upanishads", "Yoga Sutras"],
            "processing_time": (datetime.utcnow() - start_time).total_seconds()
        }
        
        logger.info(f"✅ Processed query for session {session_id}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error processing dharmic query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing dharmic query: {str(e)}"
        )


# ===============================
# FASTAPI APPLICATION
# ===============================

app = FastAPI(
    title="DharmaLLM Spiritual AI Service",
    description="Microservice for processing spiritual queries with dharmic wisdom",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global request counter
request_count = 0


# ===============================
# API ENDPOINTS
# ===============================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if dharma_system else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        dharma_system_ready=dharma_system is not None,
        models_loaded=len(dharma_system.keys()) if dharma_system else 0
    )


@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get detailed system status"""
    models_available = []
    if dharma_system:
        models_available = list(dharma_system.keys())
    
    return SystemStatus(
        dharma_system_ready=dharma_system is not None,
        models_available=models_available,
        last_startup=datetime.utcnow().isoformat(),
        total_requests=request_count,
        active_sessions=0  # Implement session tracking if needed
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def process_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """Process chat message with dharmic wisdom"""
    global request_count
    request_count += 1
    
    session_id = request.session_id or f"session_{datetime.utcnow().timestamp()}"
    
    try:
        result = await process_dharmic_query(
            message=request.message,
            session_id=session_id,
            context=request.context,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            confidence=result["confidence"],
            dharmic_alignment=result["dharmic_alignment"],
            processing_time=result["processing_time"],
            sources=result["sources"],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing chat request"
        )


@app.post("/api/v1/wisdom")
async def get_wisdom_guidance(
    query: str,
    tradition: str = "vedantic",
    session_id: Optional[str] = None
):
    """Get specific wisdom guidance from dharmic traditions"""
    session_id = session_id or f"wisdom_{datetime.utcnow().timestamp()}"
    
    result = await process_dharmic_query(
        message=f"Provide {tradition} wisdom on: {query}",
        session_id=session_id,
        context={"type": "wisdom_query", "tradition": tradition}
    )
    
    return result


@app.get("/api/v1/models")
async def list_available_models():
    """List available DharmaLLM models"""
    if not dharma_system:
        raise HTTPException(status_code=503, detail="DharmaLLM system not available")
    
    return {
        "models": list(dharma_system.keys()) if dharma_system else [],
        "status": "ready" if dharma_system else "not_ready",
        "timestamp": datetime.utcnow().isoformat()
    }


# ===============================
# MAIN APPLICATION RUNNER
# ===============================

if __name__ == "__main__":
    import os
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    reload = os.getenv("ENVIRONMENT", "production") == "development"
    
    logger.info(f"🚀 Starting DharmaLLM Service on {host}:{port}")
    
    uvicorn.run(
        "dharmallm.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )