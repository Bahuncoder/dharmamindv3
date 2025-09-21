#!/usr/bin/env python3
"""
🕉️ DharmaMind Backend Integration Test Suite
==============================================

Comprehensive testing of all backend components, integrations, and dependencies.
This will verify:

1. Core engine imports and functionality
2. FastAPI app initialization 
3. Database connections
4. Redis connectivity
5. Security middleware
6. API endpoints
7. Chakra module integrations
8. Service layer integrations
"""

import sys
import os
import asyncio
import importlib
import traceback
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

class BackendIntegrationTester:
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = []
        
    def log_result(self, test_name: str, success: bool, details: str = "", warning: bool = False):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        if warning:
            status = "⚠️ WARNING"
            self.warnings.append(f"{test_name}: {details}")
        
        self.test_results[test_name] = {
            "success": success,
            "details": details,
            "warning": warning
        }
        
        if success and not warning:
            self.passed_tests += 1
        elif not warning:
            self.failed_tests += 1
            
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
    
    async def test_core_imports(self):
        """Test core module imports"""
        print("\n🔍 Testing Core Module Imports...")
        
        # Test engine imports
        try:
            from app.engines.emotional import AdvancedEmotionalEngine, create_emotional_engine
            self.log_result("Emotional Engine Import", True, "Advanced emotional engine imported successfully")
        except Exception as e:
            self.log_result("Emotional Engine Import", False, f"Failed: {e}")
        
        try:
            from app.engines.rishi import AuthenticRishiEngine, create_authentic_rishi_engine
            self.log_result("Rishi Engine Import", True, "Authentic Rishi engine imported successfully")
        except Exception as e:
            self.log_result("Rishi Engine Import", False, f"Failed: {e}")
        
        try:
            from app.engines.personalization_engine import PersonalizationIntegration
            self.log_result("Personalization Engine Import", True, "Personalization engine imported successfully")
        except Exception as e:
            self.log_result("Personalization Engine Import", False, f"Failed: {e}")
        
        # Test core app imports
        try:
            from app.db.database import DatabaseManager
            self.log_result("Database Manager Import", True, "Database manager imported successfully")
        except Exception as e:
            self.log_result("Database Manager Import", False, f"Failed: {e}")
        
        try:
            from app.services.memory_manager import MemoryManager
            self.log_result("Memory Manager Import", True, "Memory manager imported successfully")
        except Exception as e:
            self.log_result("Memory Manager Import", False, f"Failed: {e}")
        
        try:
            from app.services.llm_router import LLMRouter
            self.log_result("LLM Router Import", True, "LLM router imported successfully")
        except Exception as e:
            self.log_result("LLM Router Import", False, f"Failed: {e}")
    
    async def test_engine_functionality(self):
        """Test core engine functionality"""
        print("\n🧠 Testing Engine Functionality...")
        
        # Test Emotional Engine
        try:
            from app.engines.emotional import create_emotional_engine
            engine = create_emotional_engine()
            
            # Test emotion analysis
            result = await engine.analyze_emotional_state("I feel happy and peaceful today")
            if result and hasattr(result, 'primary_emotion'):
                self.log_result("Emotional Analysis", True, f"Detected emotion: {result.primary_emotion.value}")
            else:
                self.log_result("Emotional Analysis", False, "Invalid analysis result")
                
        except Exception as e:
            self.log_result("Emotional Analysis", False, f"Failed: {e}")
        
        # Test Rishi Engine
        try:
            from app.engines.rishi import create_authentic_rishi_engine
            engine = create_authentic_rishi_engine()
            
            # Test Rishi response
            result = await engine.get_authentic_response("atri", "Namaste, guide me on meditation")
            if result and isinstance(result, dict):
                self.log_result("Rishi Response", True, f"Generated response from Atri")
            else:
                self.log_result("Rishi Response", False, "Invalid Rishi response")
                
        except Exception as e:
            self.log_result("Rishi Response", False, f"Failed: {e}")
    
    async def test_fastapi_app(self):
        """Test FastAPI application"""
        print("\n🚀 Testing FastAPI Application...")
        
        try:
            from app.main import app
            self.log_result("FastAPI App Import", True, "Main FastAPI app imported successfully")
            
            # Check if app has required attributes
            if hasattr(app, 'routes') and len(app.routes) > 0:
                route_count = len(app.routes)
                self.log_result("FastAPI Routes", True, f"Found {route_count} routes")
            else:
                self.log_result("FastAPI Routes", False, "No routes found")
                
        except Exception as e:
            self.log_result("FastAPI App Import", False, f"Failed: {e}")
    
    async def test_database_integration(self):
        """Test database connectivity"""
        print("\n💾 Testing Database Integration...")
        
        try:
            from app.db.database import DatabaseManager
            db_manager = DatabaseManager()
            
            # Test initialization
            await db_manager.initialize()
            self.log_result("Database Initialization", True, "Database initialized successfully")
            
            # Test health check
            is_healthy = await db_manager.health_check()
            if is_healthy:
                self.log_result("Database Health Check", True, "Database is healthy")
            else:
                self.log_result("Database Health Check", False, "Database health check failed")
                
            await db_manager.cleanup()
            
        except Exception as e:
            self.log_result("Database Integration", False, f"Failed: {e}")
    
    async def test_redis_fallback(self):
        """Test Redis connectivity and fallback"""
        print("\n🔴 Testing Redis Connectivity...")
        
        try:
            import redis.asyncio as redis
            
            # Try real Redis first
            try:
                redis_client = redis.from_url("redis://localhost:6379")
                await redis_client.ping()
                self.log_result("Real Redis Connection", True, "Connected to real Redis")
                await redis_client.close()
            except Exception as e:
                self.log_result("Real Redis Connection", False, f"Real Redis unavailable: {e}", warning=True)
                
                # Test fakeredis fallback
                try:
                    import fakeredis
                    fake_redis = fakeredis.FakeRedis()
                    await fake_redis.ping()
                    self.log_result("FakeRedis Fallback", True, "FakeRedis fallback working")
                except Exception as fake_e:
                    self.log_result("FakeRedis Fallback", False, f"FakeRedis failed: {fake_e}")
                    
        except Exception as e:
            self.log_result("Redis Integration", False, f"Failed: {e}")
    
    async def test_chakra_modules(self):
        """Test Chakra module integration"""
        print("\n🕉️ Testing Chakra Module Integration...")
        
        try:
            from app.chakra_modules import get_module_info, initialize_all_modules
            
            # Test module info
            module_info = get_module_info()
            if module_info and isinstance(module_info, dict):
                total_modules = module_info.get('total_modules', 0)
                self.log_result("Chakra Module Info", True, f"Found {total_modules} modules")
            else:
                self.log_result("Chakra Module Info", False, "Invalid module info")
            
            # Test module initialization
            init_results = await initialize_all_modules()
            if init_results:
                self.log_result("Chakra Module Initialization", True, "Modules initialized successfully")
            else:
                self.log_result("Chakra Module Initialization", False, "Module initialization failed")
                
        except Exception as e:
            self.log_result("Chakra Module Integration", False, f"Failed: {e}")
    
    async def test_api_routes(self):
        """Test API route imports"""
        print("\n🛣️ Testing API Route Imports...")
        
        routes_to_test = [
            ("app.routes.chat", "Chat Routes"),
            ("app.routes.auth", "Auth Routes"),
            ("app.routes.enhanced_chat", "Enhanced Chat Routes"),
            ("app.routes.spiritual_knowledge", "Spiritual Knowledge Routes"),
            ("app.routes.rishi_mode", "Rishi Mode Routes")
        ]
        
        for route_module, route_name in routes_to_test:
            try:
                importlib.import_module(route_module)
                self.log_result(f"{route_name} Import", True, f"{route_module} imported successfully")
            except Exception as e:
                self.log_result(f"{route_name} Import", False, f"Failed to import {route_module}: {e}")
    
    async def test_security_middleware(self):
        """Test security middleware imports"""
        print("\n🔒 Testing Security Middleware...")
        
        security_components = [
            ("app.security.jwt_manager", "JWT Manager"),
            ("app.security.monitoring", "Security Monitoring"), 
            ("app.middleware.security", "Security Middleware"),
            ("app.middleware.rate_limiting", "Rate Limiting")
        ]
        
        for component_module, component_name in security_components:
            try:
                importlib.import_module(component_module)
                self.log_result(f"{component_name} Import", True, f"{component_module} imported successfully")
            except Exception as e:
                self.log_result(f"{component_name} Import", False, f"Failed to import {component_module}: {e}")
    
    async def test_service_integration(self):
        """Test service layer integration"""
        print("\n⚙️ Testing Service Layer Integration...")
        
        try:
            from app.services.llm_router import LLMRouter
            router = LLMRouter()
            await router.initialize()
            self.log_result("LLM Router Service", True, "LLM router initialized successfully")
        except Exception as e:
            self.log_result("LLM Router Service", False, f"Failed: {e}")
        
        try:
            from app.services.module_selector import ModuleSelector
            selector = ModuleSelector()
            await selector.initialize()
            self.log_result("Module Selector Service", True, "Module selector initialized successfully")
        except Exception as e:
            self.log_result("Module Selector Service", False, f"Failed: {e}")
        
        try:
            from app.services.evaluator import ResponseEvaluator
            evaluator = ResponseEvaluator()
            await evaluator.initialize()
            self.log_result("Response Evaluator Service", True, "Response evaluator initialized successfully")
        except Exception as e:
            self.log_result("Response Evaluator Service", False, f"Failed: {e}")
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🕉️ DharmaMind Backend Integration Test Suite")
        print("=" * 60)
        
        # Run all test suites
        await self.test_core_imports()
        await self.test_engine_functionality()
        await self.test_fastapi_app()
        await self.test_database_integration()
        await self.test_redis_fallback()
        await self.test_chakra_modules()
        await self.test_api_routes()
        await self.test_security_middleware()
        await self.test_service_integration()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏁 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        
        if self.warnings:
            print(f"\nWarnings:")
            for warning in self.warnings:
                print(f"  ⚠️ {warning}")
        
        if self.failed_tests == 0:
            print(f"\n🌟 ALL TESTS PASSED! Backend integration is healthy.")
            return True
        else:
            print(f"\n🚨 {self.failed_tests} TESTS FAILED! Backend needs attention.")
            return False

if __name__ == "__main__":
    tester = BackendIntegrationTester()
    success = asyncio.run(tester.run_all_tests())
    exit(0 if success else 1)
