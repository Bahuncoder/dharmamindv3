"""
🌟 DharmaMind Vision Core Components

Essential computer vision components for the revolutionary AI yoga and meditation system.
This module contains the fundamental building blocks for pose estimation, breath detection,
and meditation analysis using advanced computer vision techniques.

Components:
- PoseEstimator: Advanced pose detection using MediaPipe Holistic
- BreathDetector: Breathing pattern analysis with Pranayama classification
- MeditationDetector: Meditation state detection through micro-movement analysis
- Quality analyzers and traditional wisdom integration

Author: DharmaMind Development Team
License: MIT
Version: 1.0.0
"""

import logging

logger = logging.getLogger(__name__)

# Core component imports with error handling
try:
    from .pose_estimation import PoseEstimator, PoseQualityAnalyzer
    logger.info("✅ Pose estimation components loaded")
    POSE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Pose estimation not available: {e}")
    POSE_AVAILABLE = False

try:
    from .breath_detection import BreathDetector, PranayamaClassifier
    logger.info("✅ Breath detection components loaded")
    BREATH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Breath detection not available: {e}")
    BREATH_AVAILABLE = False

try:
    from .meditation_detection import MeditationDetector, StillnessAnalyzer
    logger.info("✅ Meditation detection components loaded")
    MEDITATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Meditation detection not available: {e}")
    MEDITATION_AVAILABLE = False

# Legacy component imports
try:
    from .pose_detector import HathaYogaPoseDetector, PoseKeypoints, ChakraAlignment
    from .asana_classifier import AsanaClassifier, ClassificationResult, AsanaInfo, AsanaDifficulty, AsanaCategory
    from .alignment_checker import AlignmentChecker, AlignmentFeedback, AlignmentLevel, ChakraState
    from .vision_engine import DharmaMindVisionEngine, VisionSession, RealTimeAnalysis, VisionConfig
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

# Component availability status
CORE_STATUS = {
    'pose_estimation': POSE_AVAILABLE,
    'breath_detection': BREATH_AVAILABLE,
    'meditation_detection': MEDITATION_AVAILABLE,
    'legacy_components': LEGACY_AVAILABLE
}

def get_core_status():
    """Get the availability status of core components."""
    return CORE_STATUS.copy()

def get_available_components():
    """Get list of available core components."""
    available = []
    
    if POSE_AVAILABLE:
        available.extend(['PoseEstimator', 'PoseQualityAnalyzer'])
    if BREATH_AVAILABLE:
        available.extend(['BreathDetector', 'PranayamaClassifier'])
    if MEDITATION_AVAILABLE:
        available.extend(['MeditationDetector', 'StillnessAnalyzer'])
    if LEGACY_AVAILABLE:
        available.extend(['HathaYogaPoseDetector', 'PoseKeypoints', 'ChakraAlignment',
                         'AsanaClassifier', 'ClassificationResult', 'AsanaInfo', 
                         'AsanaDifficulty', 'AsanaCategory', 'AlignmentChecker',
                         'AlignmentFeedback', 'AlignmentLevel', 'ChakraState',
                         'DharmaMindVisionEngine', 'VisionSession', 'RealTimeAnalysis', 'VisionConfig'])
    
    return available

# Legacy exports (maintain compatibility)
if LEGACY_AVAILABLE:
    __all__ = [
        "HathaYogaPoseDetector",
        "PoseKeypoints", 
        "ChakraAlignment",
        "AsanaClassifier",
        "ClassificationResult",
        "AsanaInfo",
        "AsanaDifficulty",
        "AsanaCategory",
        "AlignmentChecker",
        "AlignmentFeedback",
        "AlignmentLevel",
        "ChakraState",
        "DharmaMindVisionEngine",
        "VisionSession",
        "RealTimeAnalysis",
        "VisionConfig"
    ]
else:
    __all__ = []

# Add new components if available
if POSE_AVAILABLE:
    __all__.extend(['PoseEstimator', 'PoseQualityAnalyzer'])
if BREATH_AVAILABLE:
    __all__.extend(['BreathDetector', 'PranayamaClassifier'])
if MEDITATION_AVAILABLE:
    __all__.extend(['MeditationDetector', 'StillnessAnalyzer'])

# Add utility functions
__all__.extend(['get_core_status', 'get_available_components'])

# Version info
__version__ = "1.0.0"
__author__ = "DharmaMind Development Team"

if __name__ == "__main__":
    print("🌟 DharmaMind Vision Core Components")
    print("=" * 40)
    print(f"Version: {__version__}")
    status = get_core_status()
    print("\nComponent Status:")
    for component, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component}")
    
    available_comps = get_available_components()
    print(f"\nAvailable Components ({len(available_comps)}):")
    for comp in available_comps:
        print(f"  📦 {comp}")