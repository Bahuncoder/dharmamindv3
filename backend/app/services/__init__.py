"""
🕉️ Services Module
==================

Core services for DharmaMind backend:

- llm_router.py        - Intelligent LLM routing and provider management
- module_selector.py   - Smart module selection based on context
- evaluator.py         - Response quality evaluation and scoring
- memory_manager.py    - Conversation and user memory management
- auth_service.py      - User authentication and session management

These services provide the core business logic and orchestration for the DharmaMind platform.

May these services bring wisdom and compassion to all beings 🙏
"""

# Import all services with graceful fallback
try:
    from .llm_router import LLMRouter, get_llm_router
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import LLMRouter: {e}")
    
    class LLMRouter:
        async def initialize(self): pass
        async def health_check(self): return False
    
    async def get_llm_router():
        return LLMRouter()

try:
    from .module_selector import ModuleSelector, get_module_selector
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import ModuleSelector: {e}")
    
    class ModuleSelector:
        async def initialize(self): pass
        async def health_check(self): return False
    
    async def get_module_selector():
        return ModuleSelector()

try:
    from .evaluator import ResponseEvaluator, get_response_evaluator
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import ResponseEvaluator: {e}")
    
    class ResponseEvaluator:
        async def initialize(self): pass
        async def health_check(self): return False
    
    async def get_response_evaluator():
        return ResponseEvaluator()

try:
    from .memory_manager import MemoryManager, get_memory_manager
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import MemoryManager: {e}")
    
    class MemoryManager:
        async def initialize(self): pass
        async def health_check(self): return False
        async def cleanup(self): pass
    
    async def get_memory_manager():
        return MemoryManager()

try:
    from .auth_service import AuthService, get_auth_service
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import AuthService: {e}")
    
    class AuthService:
        async def initialize(self): pass
        async def health_check(self): return False
    
    async def get_auth_service():
        return AuthService()

__all__ = [
    'LLMRouter', 'get_llm_router',
    'ModuleSelector', 'get_module_selector', 
    'ResponseEvaluator', 'get_response_evaluator',
    'MemoryManager', 'get_memory_manager',
    'AuthService', 'get_auth_service'
]
