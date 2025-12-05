"""
🛠️ Utilities Module
===================

Contains utility functions and helper services:

- logging_service.py - Logging and monitoring utilities
- data_manager.py    - Data processing utilities
- evaluator.py       - System evaluation utilities
- module_selector.py - Dynamic module selection utilities
"""

# Import main classes with fallback handling
try:
    from .logging_service import DharmicLogger
    # Create alias for backward compatibility
    LoggingService = DharmicLogger
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import DharmicLogger: {e}")
    class DharmicLogger:
        pass
    class LoggingService:
        pass

try:
    from .data_manager import DataManager
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import DataManager: {e}")
    class DataManager:
        pass

try:
    from .evaluator import Evaluator
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import Evaluator: {e}")
    class Evaluator:
        pass

try:
    from .module_selector import ModuleSelector
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import ModuleSelector: {e}")
    class ModuleSelector:
        pass

try:
    from .monitoring import MonitoringService
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import MonitoringService: {e}")
    class MonitoringService:
        pass

__all__ = [
    'LoggingService',
    'DataManager', 
    'Evaluator',
    'ModuleSelector',
    'MonitoringService'
]
