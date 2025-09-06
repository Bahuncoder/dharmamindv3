#!/usr/bin/env python3
"""
DharmaMind Backend - Complete System Integration

This is the main FastAPI application that orchestrates all DharmaMind services:
- Complete Chakra module integration (consciousness, knowledge, AI, etc.)
- Chat endpoints for user interactions
- LLM routing and advanced language processing
- Dharmic validation and compliance
- System analysis and monitoring
- Memory management and persistence
- Enterprise security framework
- Session management and authentication

🕉️ May this serve all beings with wisdom and compassion
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import asyncio
import uvicorn
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import redis.asyncio as redis

# Import our application modules
from .routes.chat import router as chat_router
from .routes.auth import router as auth_router
from .routes.admin_auth import router as admin_auth_router
from .routes.feedback import router as feedback_router
from .routes.spiritual_knowledge import router as knowledge_router
from .routes.enhanced_chat import router as enhanced_chat_router
from .routes.darshana import router as darshana_router
from .routes.universal_guidance import router as universal_router
from .routes.local_llm_test import router as local_llm_router
from .routes.dharmic_chat import router as dharmic_chat_router
from .routes.external_llm import router as external_llm_router
from .routes.deep_contemplation import router as deep_contemplation_router
from .routes.mfa_auth import router as mfa_router
from .services.llm_router import LLMRouter
from .services.module_selector import ModuleSelector
from .services.evaluator import ResponseEvaluator
from .services.memory_manager import MemoryManager
from .services.llm_gateway_client import get_llm_gateway_client
from .db.database import DatabaseManager
from .config import settings

# Import observability components
from .observability.distributed_tracing import initialize_tracing, TracingConfig, TracingMiddleware
from .observability.analytics_engine import AdvancedAnalyticsEngine
from .observability.realtime_dashboard import initialize_dashboard, dashboard_router

# Import all Chakra modules
from .chakra_modules import (
    get_consciousness_core, get_knowledge_base, get_emotional_intelligence,
    get_dharma_engine, get_ai_core, get_protection_layer, 
    get_system_orchestrator, get_llm_engine, get_module_info, initialize_all_modules
)

# Import analysis engine
from .chakra_modules.analysis_engine import (
    get_analysis_engine, AnalysisRequest, AnalysisType, AnalysisLevel
)

# Import security components
from .security.jwt_manager import init_jwt_manager
from .security.monitoring import init_security_monitor
from .security.session_middleware import init_session_security_middleware
from .security.advanced_middleware import AdvancedSecurityMiddleware
from .middleware.rate_limiting import RateLimitMiddleware
from .middleware.security import SecurityHeadersMiddleware, BruteForceProtectionMiddleware, RequestValidationMiddleware

# Import enhanced performance components
from .services.advanced_llm_router import init_advanced_llm_router
from .services.intelligent_cache import init_intelligent_cache, CacheMiddleware
from .db.advanced_pool import init_db_pool_manager
from .monitoring.performance_monitor import init_performance_monitoring, PerformanceMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances
llm_router: Optional[LLMRouter] = None
module_selector: Optional[ModuleSelector] = None
response_evaluator: Optional[ResponseEvaluator] = None
memory_manager: Optional[MemoryManager] = None
db_manager: Optional[DatabaseManager] = None
redis_client: Optional[redis.Redis] = None

# Global Chakra module instances
chakra_modules: Dict[str, Any] = {}
system_status: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup and shutdown logic"""
    global llm_router, module_selector, response_evaluator, memory_manager
    global db_manager, redis_client, chakra_modules, system_status
    
    logger.info("🔱 Starting DharmaMind Complete System...")
    logger.info("=" * 60)
    
    try:
        # Phase 1: Initialize Core Infrastructure
        logger.info("📦 Phase 1: Initializing Core Infrastructure...")
        
        # Initialize Redis client with fallback to fakeredis
        try:
            redis_client = redis.from_url(settings.REDIS_URL)
            await redis_client.ping()  # Test connection
            logger.info("✅ Real Redis client initialized")
        except Exception as e:
            logger.warning(f"⚠️ Real Redis not available: {e}")
            logger.info("� Using FakeRedis for development...")
            try:
                import fakeredis.aioredis
                redis_client = fakeredis.aioredis.FakeRedis()
                await redis_client.ping()  # Test fake connection
                logger.info("✅ FakeRedis client initialized")
            except Exception as fake_e:
                logger.error(f"❌ Failed to initialize FakeRedis: {fake_e}")
                redis_client = None
        
        # Initialize database
        db_manager = DatabaseManager()
        await db_manager.initialize()
        logger.info("✅ Database initialized")
        
        # Initialize security components
        if redis_client:
            logger.info("🔐 Initializing enterprise security framework...")
            
            # Initialize JWT manager
            jwt_manager = init_jwt_manager(redis_client, settings.JWT_SECRET_KEY or "dev-secret-key-for-development-only")
            logger.info("✅ JWT manager initialized")
            
            # Initialize MFA manager
            from .security.mfa_manager import MFAManager
            mfa_manager = MFAManager()
            await mfa_manager.initialize()
            logger.info("✅ MFA manager initialized")
            
            # Initialize security monitoring
            security_monitor = init_security_monitor(redis_client)
            await security_monitor.start_monitoring()
            logger.info("✅ Security monitoring started")
            
            # Initialize session security middleware
            session_middleware = init_session_security_middleware(redis_client)
            logger.info("✅ Session security middleware initialized")
            
            logger.info("🛡️ Enterprise security framework ready")
        else:
            logger.warning("⚠️ Security features disabled - Redis unavailable")
        
        # Initialize enhanced performance components
        if redis_client:
            logger.info("⚡ Initializing performance enhancement framework...")
            
            # Initialize distributed tracing
            env_value = settings.ENVIRONMENT.value if hasattr(settings.ENVIRONMENT, 'value') else str(settings.ENVIRONMENT)
            tracing_config = TracingConfig(
                service_name="dharmamind-backend",
                service_version="2.0.0",
                environment=env_value,
                jaeger_endpoint="http://localhost:14268/api/traces"
            )
            tracer = initialize_tracing(tracing_config)
            logger.info("✅ Distributed tracing system initialized")
            
            # Initialize analytics engine
            analytics_engine = AdvancedAnalyticsEngine(redis_client)
            logger.info("✅ Advanced analytics engine initialized")
            
            # Initialize real-time dashboard
            dashboard = initialize_dashboard(redis_client, analytics_engine)
            logger.info("✅ Real-time dashboard system initialized")
            
            # Initialize advanced LLM router
            advanced_router = init_advanced_llm_router(redis_client)
            logger.info("✅ Advanced LLM router initialized")
            
            # Initialize intelligent caching
            intelligent_cache = init_intelligent_cache(redis_client)
            logger.info("✅ Intelligent caching system initialized")
            
            # Initialize database pool manager
            db_pool_mgr = init_db_pool_manager(redis_client)
            await db_pool_mgr.add_pool(
                name="default",
                database_url="sqlite+aiosqlite:///./data/dharma_knowledge.db",
                min_size=5,
                max_size=20
            )
            await db_pool_mgr.start_monitoring()
            logger.info("✅ Advanced database pool manager initialized")
            
            # Initialize performance monitoring
            metrics_collector, profiler = init_performance_monitoring(redis_client)
            await metrics_collector.start_collection()
            logger.info("✅ Performance monitoring system initialized")
            
            logger.info("🚀 Enhanced observability & performance framework ready")
        else:
            logger.warning("⚠️ Performance features limited - Redis unavailable")
        
        # Initialize LLM Gateway client
        # LLM Gateway client will be initialized when first used
        logger.info("✅ LLM Gateway client ready")
        
        # Initialize memory manager (vector DB, embeddings)
        memory_manager = MemoryManager()
        await memory_manager.initialize()
        logger.info("✅ Memory manager initialized")
        
        # Phase 2: Initialize Chakra Modules
        logger.info("🕉️ Phase 2: Initializing Chakra Modules...")
        
        # Initialize all Chakra modules
        chakra_init_results = await initialize_all_modules()
        logger.info(f"🔮 Chakra modules initialization: {chakra_init_results}")
        
        # Get all Chakra module instances (update global variable)
        global chakra_modules
        chakra_modules = {
            "consciousness_core": get_consciousness_core(),
            "knowledge_base": get_knowledge_base(),
            "emotional_intelligence": get_emotional_intelligence(),
            "dharma_engine": get_dharma_engine(),
            "ai_core": get_ai_core(),
            "security_protection": get_protection_layer(),
            "system_orchestrator": get_system_orchestrator(),
            "llm_engine": get_llm_engine(),
            "analysis_engine": get_analysis_engine()
        }
        
        logger.info(f"✅ {len(chakra_modules)} Chakra modules loaded")
        
        # Phase 3: Initialize Service Layer
        logger.info("⚙️ Phase 3: Initializing Service Layer...")
        
        # Initialize LLM router with Chakra integration
        llm_router = LLMRouter()
        await llm_router.initialize()
        logger.info("✅ LLM router initialized")
        
        # Initialize module selector with Chakra awareness
        module_selector = ModuleSelector()
        await module_selector.initialize()
        logger.info("✅ Module selector initialized")
        
        # Initialize response evaluator with dharmic validation
        response_evaluator = ResponseEvaluator()
        await response_evaluator.initialize()
        logger.info("✅ Response evaluator initialized")
        
        # Phase 4: System Integration and Validation
        logger.info("� Phase 4: System Integration and Validation...")
        
        # Perform initial system analysis
        analysis_engine = get_analysis_engine()
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.SYSTEM_HEALTH,
            level=AnalysisLevel.STANDARD,
            include_metrics=True,
            include_recommendations=True
        )
        
        initial_analysis = await analysis_engine.analyze_system(analysis_request)
        system_status = {
            "overall_health": initial_analysis.overall_health_score,
            "system_status": initial_analysis.system_status.value,
            "active_modules": len([a for a in initial_analysis.component_analyses if a.health_score > 0.5]),
            "startup_time": datetime.now().isoformat()
        }
        
        logger.info(f"📊 System health: {initial_analysis.overall_health_score:.2f}")
        logger.info(f"🔋 System status: {initial_analysis.system_status.value}")
        
        # Final startup message
        logger.info("=" * 60)
        logger.info("🕉️ DharmaMind Complete System Ready!")
        logger.info("💫 All Chakra modules aligned and operational")
        logger.info("🌟 Serving universal wisdom with consciousness and compassion")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize complete system: {e}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("🔱 Shutting down DharmaMind Complete System...")
        
        if redis_client:
            await redis_client.close()
        if memory_manager:
            await memory_manager.cleanup()
        if db_manager:
            await db_manager.cleanup()
            
        logger.info("🕉️ Graceful shutdown complete - May all beings be well")

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="DharmaMind Complete API",
    description="""
    🕉️ Universal Wisdom AI Platform - Complete System Integration
    
    This API provides access to the complete DharmaMind consciousness system:
    
    **Core Capabilities:**
    - 🧠 Advanced consciousness processing and awareness
    - 📚 Universal spiritual knowledge and wisdom
    - ❤️ Emotional intelligence and empathetic responses  
    - ⚖️ Dharmic compliance and righteousness validation
    - 🤖 AI-powered intelligent processing and guidance
    - 🔒 Comprehensive security and protection
    - 🎯 System orchestration and harmony
    - 🗣️ Advanced language model processing
    - 📊 Real-time system analysis and monitoring
    
    **Spiritual Principles:**
    - Non-violence (Ahimsa) in all interactions
    - Truthfulness (Satya) in knowledge sharing
    - Service (Seva) to all beings
    - Wisdom (Prajna) in guidance
    - Compassion (Karuna) in responses
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Add enhanced middleware stack
app.add_middleware(AdvancedSecurityMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BruteForceProtectionMiddleware)
app.add_middleware(RequestValidationMiddleware)

# Distributed tracing middleware
if redis_client:
    try:
        from .observability.distributed_tracing import get_tracer
        tracer = get_tracer()
        if tracer:
            app.add_middleware(TracingMiddleware, tracer=tracer)
            logger.info("✅ Distributed tracing middleware added")
    except Exception as e:
        logger.warning(f"⚠️ Tracing middleware not available: {e}")

# Performance monitoring middleware
if redis_client:
    try:
        from .monitoring.performance_monitor import get_metrics_collector
        metrics = get_metrics_collector()
        app.add_middleware(PerformanceMiddleware, metrics_collector=metrics)
        logger.info("✅ Performance monitoring middleware added")
    except RuntimeError:
        logger.warning("⚠️ Performance monitoring middleware not available")

# Intelligent caching middleware
if redis_client:
    try:
        from .services.intelligent_cache import get_intelligent_cache
        cache = get_intelligent_cache()
        app.add_middleware(CacheMiddleware, cache=cache)
        logger.info("✅ Intelligent caching middleware added")
    except RuntimeError:
        logger.warning("⚠️ Caching middleware not available")

# Session security middleware (if Redis is available)
if redis_client:
    from .security.session_middleware import get_session_security_middleware
    try:
        session_middleware = get_session_security_middleware()
        app.middleware("http")(session_middleware)
        logger.info("✅ Session security middleware added")
    except RuntimeError:
        logger.warning("⚠️ Session security middleware not initialized")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Host Middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Rate Limiting Middleware
if redis_client:
    app.add_middleware(RateLimitMiddleware, redis_client=redis_client)

# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(enhanced_chat_router, prefix="/api", tags=["enhanced-chat"])
app.include_router(darshana_router, prefix="/api/v1", tags=["philosophy"])
app.include_router(universal_router, prefix="/api/v1", tags=["universal-guidance"])
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(mfa_router, prefix="/api/v1/mfa", tags=["multi-factor-auth"])
app.include_router(admin_auth_router, prefix="/api/admin", tags=["admin-authentication"])
app.include_router(feedback_router, prefix="/api/v1", tags=["feedback"])
app.include_router(knowledge_router, tags=["spiritual-knowledge"])
app.include_router(local_llm_router, tags=["local-llm"])
app.include_router(dharmic_chat_router, tags=["dharmic-chat"])
app.include_router(external_llm_router, prefix="/api/v1", tags=["external-llm"])
app.include_router(deep_contemplation_router, prefix="/api/v1", tags=["deep-contemplation"])

# Import and include internal spiritual processing router
from .routes.internal_spiritual import router as internal_spiritual_router
app.include_router(internal_spiritual_router, tags=["internal-spiritual"])

# Import and include security dashboard router
from .routes.security_dashboard import router as security_dashboard_router
app.include_router(security_dashboard_router, prefix="/api/v1", tags=["security-dashboard"])

# Import and include performance dashboard router
from .routes.performance_dashboard import router as performance_dashboard_router
app.include_router(performance_dashboard_router, prefix="/api/v1", tags=["performance-dashboard"])

# Import and include Rishi Mode router
from .routes.rishi_mode import router as rishi_mode_router
app.include_router(rishi_mode_router, prefix="/api/v1", tags=["rishi-mode"])

# Include observability dashboard router
app.include_router(dashboard_router, tags=["observability-dashboard"])

@app.get("/", tags=["system"])
async def root():
    """Root endpoint - API health check with Chakra module status"""
    module_info = get_module_info()
    
    return {
        "message": "🕉️ DharmaMind Complete API - Universal Wisdom Platform",
        "status": "active",
        "version": "2.0.0",
        "system": "DharmaMind Chakra Integration",
        "modules": {
            "total": module_info["total_modules"],
            "active": module_info["active_modules"],
            "capabilities": list(module_info.get("capabilities", {}).keys())
        },
        "consciousness": "awakened",
        "dharma": "aligned",
        "wisdom": "accessible",
        "docs": "/docs",
        "health": "/health",
        "chakra_status": "/chakra/status"
    }

@app.get("/health", tags=["monitoring"])
async def health_check():
    """Comprehensive health check including all Chakra modules"""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": system_status,
            "services": {},
            "chakra_modules": {},
            "overall_score": 0.0
        }
        
        # Check core services
        if db_manager:
            health_data["services"]["database"] = "healthy" if await db_manager.health_check() else "unhealthy"
        
        if memory_manager:
            health_data["services"]["memory_manager"] = "healthy" if await memory_manager.health_check() else "unhealthy"
        
        if llm_router:
            health_data["services"]["llm_router"] = "healthy" if await llm_router.health_check() else "unhealthy"
        
        # Check Chakra modules
        module_scores = []
        for module_name, module_instance in chakra_modules.items():
            try:
                if hasattr(module_instance, 'get_status'):
                    status_result = module_instance.get_status()
                    # Check if it's a coroutine and await it
                    if hasattr(status_result, '__await__'):
                        status = await status_result
                    else:
                        status = status_result
                    health_score = status.get('health_score', 0.8) if isinstance(status, dict) else 0.8
                    health_data["chakra_modules"][module_name] = {
                        "status": "healthy" if health_score > 0.7 else "warning" if health_score > 0.5 else "unhealthy",
                        "score": health_score
                    }
                    module_scores.append(health_score)
                else:
                    health_data["chakra_modules"][module_name] = {
                        "status": "healthy",
                        "score": 0.8
                    }
                    module_scores.append(0.8)
                    
            except Exception as e:
                health_data["chakra_modules"][module_name] = {
                    "status": "error",
                    "error": str(e),
                    "score": 0.0
                }
                module_scores.append(0.0)
        
        # Calculate overall health score
        if module_scores:
            health_data["overall_score"] = sum(module_scores) / len(module_scores)
        
        # Determine overall status
        if health_data["overall_score"] >= 0.8:
            health_data["status"] = "excellent"
        elif health_data["overall_score"] >= 0.7:
            health_data["status"] = "healthy"
        elif health_data["overall_score"] >= 0.5:
            health_data["status"] = "warning"
        else:
            health_data["status"] = "critical"
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service health check failed: {str(e)}")

@app.get("/chakra/status", tags=["monitoring"])
async def chakra_status():
    """Get detailed status of all Chakra modules"""
    try:
        module_info = get_module_info()
        
        detailed_status = {
            "package_info": module_info,
            "modules": {},
            "system_harmony": "unknown",
            "consciousness_level": "unknown",
            "dharma_alignment": "unknown"
        }
        
        # Get detailed status from each module
        for module_name, module_instance in chakra_modules.items():
            try:
                if hasattr(module_instance, 'get_status'):
                    status_result = module_instance.get_status()
                    # Handle both async and sync get_status methods
                    if asyncio.iscoroutine(status_result):
                        detailed_status["modules"][module_name] = await status_result
                    else:
                        detailed_status["modules"][module_name] = status_result
                else:
                    detailed_status["modules"][module_name] = {
                        "status": "active",
                        "type": type(module_instance).__name__
                    }
            except Exception as e:
                detailed_status["modules"][module_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Calculate system harmony
        active_modules = len([m for m in detailed_status["modules"].values() 
                            if m.get("status") not in ["error", "inactive"]])
        total_modules = len(detailed_status["modules"])
        
        if active_modules == total_modules:
            detailed_status["system_harmony"] = "perfect"
        elif active_modules >= total_modules * 0.8:
            detailed_status["system_harmony"] = "excellent"
        elif active_modules >= total_modules * 0.6:
            detailed_status["system_harmony"] = "good"
        else:
            detailed_status["system_harmony"] = "needs_attention"
        
        # Check consciousness level
        if "consciousness_core" in detailed_status["modules"]:
            consciousness_status = detailed_status["modules"]["consciousness_core"]
            if consciousness_status.get("status") == "active":
                detailed_status["consciousness_level"] = "awakened"
            else:
                detailed_status["consciousness_level"] = "sleeping"
        
        # Check dharma alignment
        if "dharma_engine" in detailed_status["modules"]:
            dharma_status = detailed_status["modules"]["dharma_engine"]
            if dharma_status.get("status") == "active":
                detailed_status["dharma_alignment"] = "aligned"
            else:
                detailed_status["dharma_alignment"] = "misaligned"
        
        return detailed_status
        
    except Exception as e:
        logger.error(f"Chakra status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chakra status check failed: {str(e)}")

@app.get("/system/analysis", tags=["monitoring"])
async def system_analysis(analysis_type: str = "system_health", level: str = "standard"):
    """Perform comprehensive system analysis"""
    try:
        analysis_engine = get_analysis_engine()
        
        # Parse analysis parameters
        try:
            analysis_type_enum = AnalysisType(analysis_type)
        except ValueError:
            analysis_type_enum = AnalysisType.SYSTEM_HEALTH
        
        try:
            level_enum = AnalysisLevel(level)
        except ValueError:
            level_enum = AnalysisLevel.STANDARD
        
        # Create analysis request
        request = AnalysisRequest(
            analysis_type=analysis_type_enum,
            level=level_enum,
            include_metrics=True,
            include_recommendations=True
        )
        
        # Perform analysis
        analysis_report = await analysis_engine.analyze_system(request)
        
        # Convert to JSON-serializable format
        return {
            "analysis_id": analysis_report.analysis_id,
            "timestamp": analysis_report.timestamp.isoformat(),
            "analysis_type": analysis_report.analysis_type.value,
            "level": analysis_report.level.value,
            "overall_health_score": analysis_report.overall_health_score,
            "system_status": analysis_report.system_status.value,
            "component_count": len(analysis_report.component_analyses),
            "issues_summary": analysis_report.issues_summary,
            "recommendations": analysis_report.recommendations,
            "system_metrics": analysis_report.system_metrics,
            "performance_summary": analysis_report.performance_summary,
            "dharmic_compliance": analysis_report.dharmic_compliance
        }
        
    except Exception as e:
        logger.error(f"System analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"System analysis failed: {str(e)}")

# Dependency injection for services
async def get_llm_router() -> LLMRouter:
    """Get LLM router instance"""
    if not llm_router:
        raise HTTPException(status_code=503, detail="LLM router not initialized")
    return llm_router

async def get_module_selector() -> ModuleSelector:
    """Get module selector instance"""
    if not module_selector:
        raise HTTPException(status_code=503, detail="Module selector not initialized")
    return module_selector

async def get_response_evaluator() -> ResponseEvaluator:
    """Get response evaluator instance"""
    if not response_evaluator:
        raise HTTPException(status_code=503, detail="Response evaluator not initialized")
    return response_evaluator

async def get_memory_manager() -> MemoryManager:
    """Get memory manager instance"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    return memory_manager

async def get_db_manager() -> DatabaseManager:
    """Get database manager instance"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database manager not initialized")
    return db_manager

# Chakra module dependency injection
async def get_chakra_modules() -> Dict[str, Any]:
    """Get all Chakra module instances"""
    if not chakra_modules:
        raise HTTPException(status_code=503, detail="Chakra modules not initialized")
    return chakra_modules

if __name__ == "__main__":
    # Run the application
    logger.info("🚀 Starting DharmaMind Complete System...")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
