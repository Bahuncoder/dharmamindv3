"""
🌟 DharmaMind Vision - Revolutionary AI Yoga & Meditation System

A groundbreaking computer vision system that combines cutting-edge AI technology
with ancient yogic wisdom to create the most sophisticated yoga and meditation
guidance platform ever developed.

Core Capabilities:
- Real-time pose estimation with traditional alignment principles
- Advanced breath pattern analysis and Pranayama guidance
- Meditation state detection through micro-movement analysis
- Progressive learning with personalized AI coaching
- Life integration with continuous mindfulness tracking
- Multi-dimensional feedback with traditional wisdom

The system integrates:
- MediaPipe Holistic for advanced body tracking
- Traditional Sanskrit terminology and concepts
- Chakra-based analysis and energy flow mapping
- Classical yoga posture assessment
- Pranayama breathing pattern recognition
- Dhyana meditation state classification

Author: DharmaMind Development Team
License: MIT
Version: 1.0.0 Revolutionary Release
"""

import logging
import sys
from pathlib import Path

# Add the package root to Python path for imports
package_root = Path(__file__).parent
sys.path.insert(0, str(package_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "DharmaMind Development Team"
__license__ = "MIT"
__status__ = "Revolutionary Release"

# Import core components with error handling
try:
    # Core components - use lazy imports to avoid dependency conflicts
    POSE_ESTIMATION_AVAILABLE = True
    BREATH_DETECTION_AVAILABLE = True 
    MEDITATION_DETECTION_AVAILABLE = True
    
    # Only import if specifically requested
    def _lazy_import_pose():
        from .core.pose_estimation import PoseEstimator, PoseQualityAnalyzer
        return PoseEstimator, PoseQualityAnalyzer
    
    def _lazy_import_breath():
        from .core.breath_detection import BreathDetector, PranayamaClassifier
        return BreathDetector, PranayamaClassifier
    
    def _lazy_import_meditation():
        from .core.meditation_detection import MeditationDetector, StillnessAnalyzer
        return MeditationDetector, StillnessAnalyzer
    
    COMPONENTS_LOADED = True
    
except ImportError as e:
    logger.warning(f"Some components not available: {e}")
    COMPONENTS_LOADED = False
    POSE_ESTIMATION_AVAILABLE = False
    BREATH_DETECTION_AVAILABLE = False
    MEDITATION_DETECTION_AVAILABLE = False

# Revolutionary subsystems - lazy loading
try:
    def _lazy_import_revolutionary():
        from .core.realtime_posture_corrector import RealTimePostureCorrector as AdvancedPostureCorrection
        from .core.meditation_detection import MeditationDetector as MeditationAnalysis  
        from .core.progressive_learning_system import ProgressiveLearningSystem as ProgressiveLearning
        from .core.dharmamind_map_integration import DharmaMindMapIntegration as LifeIntegration
        from .core.session_manager import SessionManager as SessionManagement
        from .core.intelligent_feedback_engine import IntelligentFeedbackEngine as IntelligentFeedback
        from .dharma_mind_vision_master import DharmaMindVisionMaster
        return (AdvancedPostureCorrection, MeditationAnalysis, ProgressiveLearning,
                LifeIntegration, SessionManagement, IntelligentFeedback, DharmaMindVisionMaster)
    
    REVOLUTIONARY_AVAILABLE = True
    
except ImportError:
    REVOLUTIONARY_AVAILABLE = False

# Utilities and models - lazy loading
try:
    def _lazy_import_utils():
        from .utils import VisionUtils, YogaGeometry, PerformanceOptimizer, TraditionalWisdom
        return VisionUtils, YogaGeometry, PerformanceOptimizer, TraditionalWisdom
    
    def _lazy_import_models():
        from .models import (
            PoseFrame, SessionMetrics, MeditationAnalysis as MeditationData,
            AsanaInstruction, PostureCorrection, BreathingPattern,
            TraditionalConcept, ChakraAnalysis, EnergyFlow
        )
        return (PoseFrame, SessionMetrics, MeditationData, AsanaInstruction, 
                PostureCorrection, BreathingPattern, TraditionalConcept, 
                ChakraAnalysis, EnergyFlow)
    
    UTILS_AVAILABLE = True
    MODELS_AVAILABLE = True
    
except ImportError:
    UTILS_AVAILABLE = False
    MODELS_AVAILABLE = False

# Legacy component fallback
try:
    def _lazy_import_legacy():
        from .core.vision_engine import DharmaMindVisionEngine
        from .core.pose_detector import HathaYogaPoseDetector
        from .core.asana_classifier import AsanaClassifier
        from .core.alignment_checker import AlignmentChecker
        from .api.vision_api import VisionAPI
        return (DharmaMindVisionEngine, HathaYogaPoseDetector, AsanaClassifier, 
                AlignmentChecker, VisionAPI)
    
    LEGACY_COMPONENTS = True
except ImportError:
    LEGACY_COMPONENTS = False

# Package-level constants
DEFAULT_CONFIG = {
    'pose_confidence_threshold': 0.5,
    'breath_detection_sensitivity': 0.8,
    'meditation_stillness_threshold': 0.1,
    'traditional_wisdom_integration': True,
    'sanskrit_terminology': True,
    'chakra_analysis': True,
    'real_time_processing': True,
    'progressive_learning': True,
    'life_integration': True,
    'multi_dimensional_feedback': True
}

# System capabilities
SYSTEM_CAPABILITIES = [
    "Real-time Pose Estimation",
    "Advanced Breath Analysis", 
    "Meditation State Detection",
    "Progressive AI Learning",
    "Life Integration Tracking",
    "Traditional Wisdom Integration",
    "Multi-dimensional Feedback",
    "Sanskrit Terminology Support",
    "Chakra-based Analysis",
    "Energy Flow Mapping",
    "Classical Posture Assessment",
    "Pranayama Pattern Recognition",
    "Dhyana State Classification"
]

# Component availability status
COMPONENT_STATUS = {
    'pose_estimation': POSE_ESTIMATION_AVAILABLE,
    'breath_detection': BREATH_DETECTION_AVAILABLE,
    'meditation_detection': MEDITATION_DETECTION_AVAILABLE,
    'revolutionary_subsystems': REVOLUTIONARY_AVAILABLE,
    'utils': UTILS_AVAILABLE,
    'models': MODELS_AVAILABLE,
    'legacy_components': LEGACY_COMPONENTS
}

# Traditional 15 Asanas from Hatha Yoga Pradipika Chapter 2
TRADITIONAL_ASANAS = [
    'Swastikasana',      # Auspicious Pose
    'Gomukhasana',       # Cow's Face
    'Virasana',          # Hero's Pose
    'Kurmasana',         # Tortoise
    'Kukkutasana',       # Cockerel (Advanced)
    'Uttana Kurmasana',  # Stretched Tortoise (Advanced)
    'Dhanurasana',       # Bow (Advanced)
    'Matsyendrasana',    # Lord of Fishes Twist (Advanced)
    'Paschimottanasana', # Seated Forward Bend (Advanced)
    'Mayurasana',        # Peacock (Advanced)
    'Shavasana',         # Corpse
    'Siddhasana',        # Accomplished
    'Padmasana',         # Lotus
    'Simhasana',         # Lion
    'Bhadrasana'         # Gracious
]

def get_system_info():
    """Get comprehensive system information."""
    return {
        'name': 'DharmaMind Vision',
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'status': __status__,
        'capabilities': SYSTEM_CAPABILITIES,
        'config': DEFAULT_CONFIG,
        'component_status': COMPONENT_STATUS,
        'supported_asanas': TRADITIONAL_ASANAS
    }

def initialize_system(config=None):
    """Initialize the complete DharmaMind Vision system."""
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    logger.info("🌟 Initializing DharmaMind Vision - Revolutionary AI System")
    logger.info(f"Version: {__version__}")
    logger.info(f"Capabilities: {len(SYSTEM_CAPABILITIES)} advanced features")
    
    try:
        if REVOLUTIONARY_AVAILABLE:
            # Initialize master system with lazy import
            revolutionary_classes = _lazy_import_revolutionary()
            DharmaMindVisionMaster = revolutionary_classes[-1]  # Last item is master
            vision_master = DharmaMindVisionMaster(config)
            logger.info("✅ DharmaMind Vision Master System initialized successfully")
            return vision_master
        else:
            logger.warning("⚠️ Revolutionary components not available")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize DharmaMind Vision: {str(e)}")
        return None

def create_vision_engine(model_path=None):
    """Create a vision engine instance based on available components."""
    if REVOLUTIONARY_AVAILABLE:
        return initialize_system()
    elif LEGACY_COMPONENTS:
        legacy_classes = _lazy_import_legacy()
        DharmaMindVisionEngine = legacy_classes[0]  # First item is vision engine
        return DharmaMindVisionEngine(model_path=model_path)
    else:
        raise ImportError("No vision engine components available")

def create_vision_api():
    """Create a Vision API instance for web integration."""
    if LEGACY_COMPONENTS:
        legacy_classes = _lazy_import_legacy()
        VisionAPI = legacy_classes[-1]  # Last item is API
        return VisionAPI()
    else:
        raise ImportError("Vision API not available")

def get_supported_asanas():
    """Get list of supported traditional asanas."""
    return TRADITIONAL_ASANAS.copy()

def get_documentation():
    """Get comprehensive documentation about the vision system."""
    return {
        'version': __version__,
        'description': 'Revolutionary AI Yoga & Meditation System',
        'supported_asanas': len(TRADITIONAL_ASANAS),
        'asana_list': TRADITIONAL_ASANAS,
        'capabilities': SYSTEM_CAPABILITIES,
        'features': [
            'Real-time pose estimation with MediaPipe Holistic',
            'Advanced breath pattern analysis',
            'Meditation state detection',
            'Progressive AI learning system',
            'Life integration tracking',
            'Traditional wisdom integration',
            'Sanskrit terminology support',
            'Chakra-based energy analysis',
            'Multi-dimensional feedback'
        ],
        'source_texts': [
            'Hatha Yoga Pradipika',
            'Gheranda Samhita',
            'Shiva Samhita',
            'Patanjali Yoga Sutras'
        ],
        'usage': {
            'basic': 'from dharmamind_vision import initialize_system',
            'api': 'from dharmamind_vision import create_vision_api',
            'advanced': 'from dharmamind_vision import DharmaMindVisionMaster'
        }
    }

# Export main classes for easy import based on availability
__all__ = [
    # Package functions (always available)
    'get_system_info', 'initialize_system', 'create_vision_engine',
    'create_vision_api', 'get_supported_asanas', 'get_documentation',
    
    # Constants (always available)
    'DEFAULT_CONFIG', 'SYSTEM_CAPABILITIES', 'TRADITIONAL_ASANAS'
]

# Add component-specific exports based on availability
if POSE_ESTIMATION_AVAILABLE:
    __all__.extend(['_lazy_import_pose'])

if BREATH_DETECTION_AVAILABLE:
    __all__.extend(['_lazy_import_breath'])

if MEDITATION_DETECTION_AVAILABLE:
    __all__.extend(['_lazy_import_meditation'])

if REVOLUTIONARY_AVAILABLE:
    __all__.extend(['_lazy_import_revolutionary'])

if UTILS_AVAILABLE:
    __all__.extend(['_lazy_import_utils'])

if MODELS_AVAILABLE:
    __all__.extend(['_lazy_import_models'])

if LEGACY_COMPONENTS:
    __all__.extend(['_lazy_import_legacy'])

# Package banner
BANNER = """
🌟 ══════════════════════════════════════════════════════════════════════════════════════════ 🌟
                           DharmaMind Vision - Revolutionary AI System
                        Advanced Yoga & Meditation Guidance Through Computer Vision
                                    Version 1.0.0 - Revolutionary Release
🌟 ══════════════════════════════════════════════════════════════════════════════════════════ 🌟
"""

if __name__ == "__main__":
    print(BANNER)
    doc = get_documentation()
    print("🕉️ DharmaMind Vision System")
    print("=" * 50)
    print(f"Version: {doc['version']}")
    print(f"Description: {doc['description']}")
    print(f"Components Loaded: {COMPONENTS_LOADED}")
    print(f"Supported Asanas: {doc['supported_asanas']}")
    print("\nTraditional Asanas:")
    for i, asana in enumerate(doc['asana_list'], 1):
        print(f"  {i:2}. {asana}")
    print("\nRevolutionary Capabilities:")
    for capability in SYSTEM_CAPABILITIES:
        print(f"  ✅ {capability}")
    print(f"\nBased on classical texts: {', '.join(doc['source_texts'])}")
    print("\n🙏 May this serve all beings on the path to liberation")