"""
🕉️ DharmaLLM Production API Service
====================================

Full-featured DharmaLLM microservice with complete spiritual intelligence:
- Nava Manas Putra (9 Rishis) personalities with wisdom
- Spiritual Intelligence Engine
- Consciousness Core
- Knowledge Base with Vedic scriptures
- Emotional Intelligence
- Dharma Engine for ethical guidance
- Custom DharmaLLM (Pure PyTorch - NO GPT-2!)

This is the COMPLETE system with custom AI.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global system components - will be initialized on startup
rishi_engine = None
spiritual_intelligence = None
consciousness_core = None
knowledge_base = None
emotional_intelligence = None
dharma_engine = None
llm_service = None  # Custom DharmaLLM inference service

# Configuration
USE_LLM_GENERATION = True  # Set to False to disable LLM and use templates only

# Track system stats
system_stats = {
    "total_requests": 0,
    "active_sessions": 0,
    "startup_time": None,
    "models_loaded": []
}

# ===============================
# REQUEST/RESPONSE MODELS
# ===============================

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    user_id: Optional[str] = Field(None, description="User ID")
    rishi_id: Optional[str] = Field(None, description="Specific Rishi to consult")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(512, ge=1, le=2048)
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context")

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session ID")
    confidence: float = Field(..., description="Response confidence")
    dharmic_alignment: float = Field(..., description="Dharmic principle alignment")
    processing_time: float = Field(..., description="Processing time in seconds")
    sources: List[str] = Field(default_factory=list, description="Sources used")
    timestamp: str = Field(..., description="Response timestamp")
    rishi_name: Optional[str] = Field(None, description="Name of Rishi who responded")
    wisdom_metrics: Optional[Dict[str, Any]] = Field(None, description="Wisdom metrics")

class WisdomRequest(BaseModel):
    """Spiritual wisdom request"""
    query: str = Field(..., description="Wisdom query")
    tradition: Optional[str] = Field("vedantic", description="Spiritual tradition")
    depth: Optional[str] = Field("intermediate", description="Depth level")
    include_sanskrit: bool = Field(True, description="Include Sanskrit texts")

class WisdomResponse(BaseModel):
    """Wisdom response"""
    wisdom: str
    sanskrit_text: Optional[str] = None
    translation: Optional[str] = None
    sources: List[str]
    tradition: str
    depth: str

class RishiInfo(BaseModel):
    """Rishi information"""
    id: str
    name: str
    sanskrit_name: str
    specialization: List[str]
    greeting: str
    archetype: str
    available: bool

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    components_loaded: Dict[str, bool]
    models_loaded: int

class SystemStatus(BaseModel):
    """System status response"""
    components_ready: Dict[str, bool]
    models_available: List[str]
    last_startup: str
    total_requests: int
    active_sessions: int

# ===============================
# SYSTEM INITIALIZATION
# ===============================

async def initialize_spiritual_systems():
    """Initialize all spiritual intelligence systems"""
    global rishi_engine, spiritual_intelligence, consciousness_core
    global knowledge_base, emotional_intelligence, dharma_engine, llm_service, system_stats
    
    logger.info("🕉️ Initializing DharmaLLM Spiritual Systems...")
    
    components_status = {
        "rishi_engine": False,
        "spiritual_intelligence": False,
        "consciousness_core": False,
        "knowledge_base": False,
        "emotional_intelligence": False,
        "dharma_engine": False,
        "llm_service": False
    }
    
    # 1. Initialize Custom DharmaLLM Service FIRST
    if USE_LLM_GENERATION:
        try:
            from inference.llm_service import DharmaLLMService
            llm_service = DharmaLLMService()
            await llm_service.load_model()
            
            if llm_service.is_loaded():
                components_status["llm_service"] = True
                system_stats["models_loaded"].append("Custom DharmaLLM (Pure PyTorch)")
                logger.info("✅ Custom DharmaLLM Service initialized - AI GENERATION ENABLED 🤖")
            else:
                logger.warning("⚠️ LLM Service loaded but model failed to load")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize LLM Service: {e}")
    else:
        logger.info("ℹ️ LLM Generation disabled - using templates only")
    
    # 2. Initialize Nava Manas Putra Engine (9 Rishis)
    try:
        from engines.rishi.enhanced_saptarishi_engine import EnhancedSaptarishiEngine
        rishi_engine = EnhancedSaptarishiEngine(llm_service=llm_service)
        components_status["rishi_engine"] = True
        system_stats["models_loaded"].append("Nava Manas Putra Engine (9 Rishis)")
        logger.info("✅ Nava Manas Putra Engine initialized (9 Rishis ready)")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize Rishi Engine: {e}")
    
    # 3. Initialize Spiritual Intelligence
    try:
        from engines.spiritual_intelligence import SpiritualIntelligenceEngine
        spiritual_intelligence = SpiritualIntelligenceEngine()
        components_status["spiritual_intelligence"] = True
        system_stats["models_loaded"].append("Spiritual Intelligence")
        logger.info("✅ Spiritual Intelligence initialized")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize Spiritual Intelligence: {e}")
    
    # 4. Initialize Consciousness Core
    try:
        from engines.consciousness_core import ConsciousnessCore
        consciousness_core = ConsciousnessCore()
        components_status["consciousness_core"] = True
        system_stats["models_loaded"].append("Consciousness Core")
        logger.info("✅ Consciousness Core initialized")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize Consciousness Core: {e}")
    
    # 5. Initialize Knowledge Base
    try:
        from engines.knowledge_base import AdvancedKnowledgeBase
        knowledge_base = AdvancedKnowledgeBase()
        components_status["knowledge_base"] = True
        system_stats["models_loaded"].append("Knowledge Base")
        logger.info("✅ Knowledge Base initialized")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize Knowledge Base: {e}")
    
    # 6. Initialize Emotional Intelligence
    try:
        from engines.emotional_intelligence import EmotionalIntelligence
        emotional_intelligence = EmotionalIntelligence()
        components_status["emotional_intelligence"] = True
        system_stats["models_loaded"].append("Emotional Intelligence")
        logger.info("✅ Emotional Intelligence initialized")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize Emotional Intelligence: {e}")
    
    # 7. Initialize Dharma Engine
    try:
        from engines.dharma_engine import DharmaEngine
        dharma_engine = DharmaEngine()
        components_status["dharma_engine"] = True
        system_stats["models_loaded"].append("Dharma Engine")
        logger.info("✅ Dharma Engine initialized")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize Dharma Engine: {e}")
    
    system_stats["startup_time"] = datetime.now().isoformat()
    
    loaded_count = sum(1 for v in components_status.values() if v)
    total_components = len(components_status)
    logger.info(f"🎉 DharmaLLM initialized with {loaded_count}/{total_components} components ready")
    
    return components_status

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting DharmaLLM API Service...")
    components = await initialize_spiritual_systems()
    app.state.components = components
    logger.info("✅ DharmaLLM API Service ready!")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down DharmaLLM API Service...")
    logger.info("✅ Shutdown complete")

# ===============================
# FASTAPI APPLICATION
# ===============================

app = FastAPI(
    title="DharmaLLM Spiritual Intelligence API",
    description="Custom AI system with Nava Manas Putra wisdom - Pure PyTorch, NO GPT-2",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# API ENDPOINTS
# ===============================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    llm_loaded = llm_service.is_loaded() if llm_service else False
    
    components_loaded = {
        "rishi_engine": rishi_engine is not None,
        "spiritual_intelligence": spiritual_intelligence is not None,
        "consciousness_core": consciousness_core is not None,
        "knowledge_base": knowledge_base is not None,
        "emotional_intelligence": emotional_intelligence is not None,
        "dharma_engine": dharma_engine is not None,
        "llm_service": llm_loaded,
    }
    
    return HealthResponse(
        status="healthy" if llm_loaded or any(components_loaded.values()) else "degraded",
        timestamp=datetime.now().isoformat(),
        version="2.0.0-CustomLLM",
        components_loaded=components_loaded,
        models_loaded=len(system_stats["models_loaded"])
    )


@app.get("/api/v1/llm/status")
async def llm_status():
    """Get LLM service status and statistics"""
    if not llm_service:
        return {
            "status": "not_initialized",
            "llm_enabled": USE_LLM_GENERATION,
            "message": "LLM service not initialized",
            "model_type": "Custom DharmaLLM (Pure PyTorch)"
        }
    
    status = llm_service.get_status()
    return {
        "status": "active" if status["is_loaded"] else "inactive",
        "llm_enabled": USE_LLM_GENERATION,
        "is_loaded": status["is_loaded"],
        "model_path": status["model_path"],
        "device": status["device"],
        "total_generations": status["generation_count"],
        "model_parameters": f"{status['model_params']/1e6:.1f}M" if status["model_params"] else None,
        "tokenizer_vocab_size": status.get("tokenizer_vocab_size", 0),
        "model_type": status.get("model_type", "Custom DharmaLLM"),
        "message": "Custom DharmaLLM ready for spiritual guidance" if status["is_loaded"] else "Model not loaded"
    }


@app.get("/status", response_model=SystemStatus)
async def system_status():
    """Detailed system status"""
    llm_loaded = llm_service.is_loaded() if llm_service else False
    
    components_ready = {
        "rishi_engine": rishi_engine is not None,
        "spiritual_intelligence": spiritual_intelligence is not None,
        "consciousness_core": consciousness_core is not None,
        "knowledge_base": knowledge_base is not None,
        "emotional_intelligence": emotional_intelligence is not None,
        "dharma_engine": dharma_engine is not None,
        "llm_service": llm_loaded,
    }
    
    return SystemStatus(
        components_ready=components_ready,
        models_available=system_stats["models_loaded"],
        last_startup=system_stats["startup_time"] or datetime.now().isoformat(),
        total_requests=system_stats["total_requests"],
        active_sessions=system_stats["active_sessions"]
    )


@app.get("/api/v1/rishis", response_model=List[RishiInfo])
async def get_available_rishis():
    """Get list of all 9 Nava Manas Putra Rishis"""
    rishis = [
        RishiInfo(
            id="marichi",
            name="Sage Marichi",
            sanskrit_name="मरीचि",
            specialization=["Cosmic Light", "Divine Radiance", "Creation"],
            greeting="I am Marichi, the ray of cosmic light. Let my radiance illuminate your path.",
            archetype="light_bearer",
            available=True
        ),
        RishiInfo(
            id="atri",
            name="Sage Atri",
            sanskrit_name="अत्रि",
            specialization=["Meditation", "Tapasya", "Inner Peace"],
            greeting="Om Shanti. I am Atri, master of meditation. Let us journey inward together.",
            archetype="ascetic",
            available=True
        ),
        RishiInfo(
            id="angiras",
            name="Sage Angiras",
            sanskrit_name="अंगिरस",
            specialization=["Sacred Mantras", "Rituals", "Divine Knowledge"],
            greeting="I am Angiras, keeper of sacred mantras. The Vedic fire illuminates all wisdom.",
            archetype="ritual_master",
            available=True
        ),
        RishiInfo(
            id="pulastya",
            name="Sage Pulastya",
            sanskrit_name="पुलस्त्य",
            specialization=["Creation", "Cosmic Stories", "Universal Balance"],
            greeting="I am Pulastya, seer of cosmic order. All creation follows divine law.",
            archetype="cosmic_seer",
            available=True
        ),
        RishiInfo(
            id="pulaha",
            name="Sage Pulaha",
            sanskrit_name="पुलह",
            specialization=["Dharmic Living", "Righteousness", "Ethical Guidance"],
            greeting="I am Pulaha, guardian of dharma. Let righteousness guide your actions.",
            archetype="dharma_guardian",
            available=True
        ),
        RishiInfo(
            id="kratu",
            name="Sage Kratu",
            sanskrit_name="क्रतु",
            specialization=["Yogic Wisdom", "Discipline", "Sacrifice"],
            greeting="I am Kratu, master of yoga and sacrifice. Through discipline, liberation is attained.",
            archetype="yoga_master",
            available=True
        ),
        RishiInfo(
            id="bhrigu",
            name="Sage Bhrigu",
            sanskrit_name="भृगु",
            specialization=["Vedic Astrology", "Karma", "Destiny"],
            greeting="I am Bhrigu, seer of destinies. The stars speak of your karmic journey.",
            archetype="astrologer",
            available=True
        ),
        RishiInfo(
            id="vasishtha",
            name="Sage Vasishtha",
            sanskrit_name="वशिष्ठ",
            specialization=["Supreme Knowledge", "Royal Wisdom", "Consciousness"],
            greeting="I am Vasishtha, holder of supreme wisdom. As guru to Lord Rama, I offer divine guidance.",
            archetype="royal_guru",
            available=True
        ),
        RishiInfo(
            id="daksha",
            name="Sage Daksha",
            sanskrit_name="दक्ष",
            specialization=["Creation", "Manifestation", "Skillful Action"],
            greeting="I am Daksha, lord of skillful creation. Through right action, we manifest our destiny.",
            archetype="creator",
            available=True
        )
    ]
    
    return rishis


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint - routes to appropriate Rishi or general wisdom"""
    start_time = datetime.now()
    system_stats["total_requests"] += 1
    
    try:
        session_id = request.session_id or f"session_{int(datetime.now().timestamp())}"
        
        # Route to specific Rishi if requested
        if request.rishi_id and rishi_engine:
            logger.info(f"Routing to Rishi: {request.rishi_id}")
            response_text, metrics = await chat_with_rishi(
                rishi_id=request.rishi_id,
                message=request.message,
                session_id=session_id,
                user_id=request.user_id or "anonymous",
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            rishi_name = f"Sage {request.rishi_id.title()}"
        else:
            logger.info("Providing general spiritual guidance")
            response_text, metrics = await general_spiritual_guidance(
                message=request.message,
                session_id=session_id,
                user_id=request.user_id or "anonymous"
            )
            rishi_name = "DharmaMind Collective Wisdom"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            confidence=metrics.get("confidence", 0.85),
            dharmic_alignment=metrics.get("dharmic_alignment", 0.90),
            processing_time=processing_time,
            sources=metrics.get("sources", ["Custom DharmaLLM", "Vedic Wisdom"]),
            timestamp=datetime.now().isoformat(),
            rishi_name=rishi_name,
            wisdom_metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/wisdom", response_model=WisdomResponse)
async def wisdom_endpoint(request: WisdomRequest):
    """Dedicated wisdom query endpoint for deep spiritual questions"""
    try:
        if spiritual_intelligence:
            result = await query_spiritual_intelligence(
                query=request.query,
                tradition=request.tradition,
                depth=request.depth,
                include_sanskrit=request.include_sanskrit
            )
            return WisdomResponse(**result)
        else:
            return WisdomResponse(
                wisdom="The spiritual intelligence system is initializing. Please try again shortly.",
                sources=["System Status"],
                tradition=request.tradition,
                depth=request.depth
            )
    
    except Exception as e:
        logger.error(f"Error in wisdom endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models")
async def list_models():
    """List available AI models and components"""
    llm_loaded = llm_service.is_loaded() if llm_service else False
    llm_info = llm_service.get_status() if llm_service else {}
    
    return {
        "models": system_stats["models_loaded"],
        "total_count": len(system_stats["models_loaded"]),
        "llm": {
            "active": llm_loaded,
            "enabled": USE_LLM_GENERATION,
            "model_type": "Custom DharmaLLM (Pure PyTorch)",
            "model_path": llm_info.get("model_path"),
            "device": llm_info.get("device"),
            "parameters": f"{llm_info.get('model_params', 0)/1e6:.1f}M" if llm_info.get("model_params") else None,
            "vocab_size": llm_info.get("tokenizer_vocab_size", 0),
            "generations": llm_info.get("generation_count", 0)
        },
        "components": {
            "nava_manas_putra": rishi_engine is not None,
            "spiritual_intelligence": spiritual_intelligence is not None,
            "consciousness_core": consciousness_core is not None,
            "knowledge_base": knowledge_base is not None,
            "emotional_intelligence": emotional_intelligence is not None,
            "dharma_engine": dharma_engine is not None,
            "llm_service": llm_loaded,
        }
    }

# ===============================
# HELPER FUNCTIONS
# ===============================

async def chat_with_rishi(
    rishi_id: str, 
    message: str, 
    session_id: str, 
    user_id: str,
    temperature: float = 0.7,
    max_tokens: int = 256
) -> tuple[str, Dict]:
    """Chat with specific Rishi using Custom DharmaLLM."""
    try:
        if not rishi_engine:
            return "The Rishi system is currently initializing.", {}
        
        # Try LLM-powered generation
        if USE_LLM_GENERATION and llm_service and llm_service.is_loaded():
            try:
                response = await llm_service.generate_response(
                    prompt=message,
                    rishi_name=rishi_id,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                
                metrics = {
                    "confidence": 0.88,
                    "dharmic_alignment": 0.90,
                    "sources": ["Custom DharmaLLM", f"Sage {rishi_id.title()}"],
                    "llm_generated": True,
                    "model_type": "Custom DharmaLLM (Pure PyTorch)"
                }
                
                logger.info(f"🤖 Custom LLM response generated for {rishi_id}")
                return response, metrics
                
            except Exception as llm_error:
                logger.warning(f"LLM generation failed: {llm_error}")
        
        # Fallback to Rishi engine templates
        if hasattr(rishi_engine, 'get_rishi_guidance'):
            response = rishi_engine.get_rishi_guidance(
                rishi_name=rishi_id,
                user_question=message,
                user_id=user_id,
                session_id=session_id
            )
            
            response_text = response.get("response", response.get("message", "Namaste, seeker."))
            
            metrics = {
                "confidence": 0.85,
                "dharmic_alignment": 0.90,
                "sources": [f"Sage {rishi_id.title()} Templates"],
                "llm_generated": False
            }
            
            return response_text, metrics
        
        # Emergency fallback
        return get_rishi_fallback(rishi_id), {
            "confidence": 0.70, 
            "dharmic_alignment": 0.85, 
            "sources": ["Emergency Fallback"],
            "llm_generated": False
        }
        
    except Exception as e:
        logger.error(f"Error in Rishi chat: {e}")
        return get_rishi_fallback(rishi_id), {
            "confidence": 0.70, 
            "dharmic_alignment": 0.85, 
            "sources": ["Emergency Fallback"],
            "llm_generated": False
        }


def get_rishi_fallback(rishi_id: str) -> str:
    """Get fallback response for a Rishi."""
    fallback_messages = {
        "marichi": "🌟 I am Marichi, the ray of cosmic light. May divine radiance illuminate your path.",
        "atri": "🧘 Om Shanti. Through meditation, all answers arise from within.",
        "angiras": "🔱 The sacred mantras contain all wisdom. May the Vedic fire guide you.",
        "pulastya": "📖 The cosmic stories reveal all truths. Trust in the divine narrative.",
        "pulaha": "⚖️ Dharma is the foundation of righteous living. Act with integrity.",
        "kratu": "🎯 Through discipline and yoga, liberation is attained.",
        "bhrigu": "✨ The stars speak of your journey. Your karma is unfolding as it should.",
        "vasishtha": "🏛️ As guru to Lord Rama, I remind you: dharma guides all righteous action.",
        "daksha": "🌺 Through skillful action, we manifest our destiny. Create with wisdom."
    }
    
    return fallback_messages.get(
        rishi_id.lower(), 
        "🕉️ Namaste, seeker. How may I guide you on your spiritual path?"
    )


async def general_spiritual_guidance(
    message: str, 
    session_id: str, 
    user_id: str
) -> tuple[str, Dict]:
    """Provide general spiritual guidance using Custom DharmaLLM."""
    try:
        # Try LLM-powered generation
        if USE_LLM_GENERATION and llm_service and llm_service.is_loaded():
            try:
                response = await llm_service.generate_response(
                    prompt=message,
                    max_new_tokens=256,
                    temperature=0.75
                )
                
                if response and len(response) > 30:
                    return response, {
                        "confidence": 0.88,
                        "dharmic_alignment": 0.88,
                        "sources": ["Custom DharmaLLM", "Vedic Wisdom"],
                        "llm_generated": True,
                        "model_type": "Custom DharmaLLM (Pure PyTorch)"
                    }
                    
            except Exception as llm_error:
                logger.warning(f"LLM general guidance failed: {llm_error}")
        
        # Fallback to template
        response_text = (
            "🕉️ Namaste, dear seeker.\n\n"
            "Your question touches upon the eternal truths of dharma. "
            "The path to wisdom lies in self-inquiry, compassionate action, and devoted practice. "
            "Remember the teaching from the Bhagavad Gita: 'Yoga is the journey of the self, "
            "through the self, to the self.'\n\n"
            "How may I assist you further on your spiritual journey?"
        )
        
        return response_text, {
            "confidence": 0.82,
            "dharmic_alignment": 0.88,
            "sources": ["Universal Wisdom", "Bhagavad Gita"],
            "llm_generated": False
        }
        
    except Exception as e:
        logger.error(f"Error in general guidance: {e}")
        return (
            "🕉️ Namaste. The answers you seek lie within. Practice meditation and self-inquiry.",
            {"confidence": 0.70, "dharmic_alignment": 0.85, "sources": ["Fallback Wisdom"], "llm_generated": False}
        )


async def query_spiritual_intelligence(
    query: str, 
    tradition: str, 
    depth: str, 
    include_sanskrit: bool
) -> Dict:
    """Query the spiritual intelligence engine"""
    try:
        if not spiritual_intelligence:
            return {
                "wisdom": "The spiritual intelligence system is initializing...",
                "sources": ["System Status"],
                "tradition": tradition,
                "depth": depth
            }
        
        result = spiritual_intelligence.process_spiritual_query(
            query=query,
            tradition=tradition,
            wisdom_level=depth
        )
        
        return {
            "wisdom": result.get("guidance", "Spiritual wisdom is emerging..."),
            "sanskrit_text": result.get("sanskrit_verse") if include_sanskrit else None,
            "translation": result.get("translation") if include_sanskrit else None,
            "sources": result.get("sources", ["Spiritual Intelligence Engine"]),
            "tradition": tradition,
            "depth": depth
        }
    
    except Exception as e:
        logger.error(f"Error querying spiritual intelligence: {e}")
        return {
            "wisdom": "🕉️ Through contemplation and practice, wisdom unfolds naturally.",
            "sources": ["Universal Wisdom"],
            "tradition": tradition,
            "depth": depth
        }

# ===============================
# MAIN ENTRY POINT
# ===============================

if __name__ == "__main__":
    logger.info("🕉️ Starting DharmaLLM Production API Service...")
    logger.info("   Using Custom DharmaLLM (Pure PyTorch - NO GPT-2!)")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
