"""
🕉️ DharmaMind Vision - Traditional Hindu Yoga Pose Detection System

A standalone computer vision system for detecting and analyzing traditional yoga asanas
based on the Hatha Yoga Pradipika and other classical Hindu texts.

This module provides:
- Real-time pose estimation using MediaPipe
- Classification of 15+ traditional asanas  
- Scriptural alignment feedback
- Chakra-based energy flow analysis
- Integration with spiritual guidance systems

Author: DharmaMind AI Team
Based on: Hatha Yoga Pradipika by Yogi Svatmarama (15th century)
Version: 1.0.0
"""

# Import main classes when available
try:
    from .core.vision_engine import DharmaMindVisionEngine
    from .core.pose_detector import HathaYogaPoseDetector
    from .core.asana_classifier import AsanaClassifier
    from .core.alignment_checker import AlignmentChecker
    from .api.vision_api import VisionAPI
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Some dependencies not available: {e}")
    DharmaMindVisionEngine = None
    HathaYogaPoseDetector = None
    AsanaClassifier = None
    AlignmentChecker = None
    VisionAPI = None

__version__ = "1.0.0"
__author__ = "DharmaMind AI Team"
__license__ = "MIT"

__all__ = [
    "DharmaMindVisionEngine",
    "HathaYogaPoseDetector", 
    "AsanaClassifier",
    "AlignmentChecker",
    "VisionAPI"
]

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

def get_version():
    """Get the version of DharmaMind Vision."""
    return __version__

def get_supported_asanas():
    """Get list of supported traditional asanas."""
    return TRADITIONAL_ASANAS.copy()

def create_vision_engine(model_path=None):
    """
    Create a new DharmaMind Vision Engine instance.
    
    Args:
        model_path: Optional path to pre-trained model
        
    Returns:
        DharmaMindVisionEngine instance
    """
    return DharmaMindVisionEngine(model_path=model_path)

def create_vision_api():
    """
    Create a new Vision API instance for web integration.
    
    Returns:
        VisionAPI instance
    """
    return VisionAPI()

# Module-level documentation
def get_documentation():
    """Get comprehensive documentation about the vision system."""
    return {
        'version': __version__,
        'description': 'Traditional Hindu Yoga Pose Detection System',
        'supported_asanas': len(TRADITIONAL_ASANAS),
        'asana_list': TRADITIONAL_ASANAS,
        'features': [
            'Real-time pose estimation',
            'Traditional asana classification',
            'Scriptural alignment feedback', 
            'Chakra energy analysis',
            'Spiritual guidance integration'
        ],
        'source_texts': [
            'Hatha Yoga Pradipika',
            'Gheranda Samhita',
            'Shiva Samhita'
        ],
        'usage': {
            'basic': 'from dharmamind_vision import create_vision_engine',
            'api': 'from dharmamind_vision import create_vision_api'
        }
    }

if __name__ == "__main__":
    # Print module information when run directly
    doc = get_documentation()
    print("🕉️ DharmaMind Vision System")
    print("=" * 50)
    print(f"Version: {doc['version']}")
    print(f"Description: {doc['description']}")
    print(f"Supported Asanas: {doc['supported_asanas']}")
    print("\nTraditional Asanas:")
    for i, asana in enumerate(doc['asana_list'], 1):
        print(f"  {i:2}. {asana}")
    print("\nFeatures:")
    for feature in doc['features']:
        print(f"  ✅ {feature}")
    print(f"\nBased on classical texts: {', '.join(doc['source_texts'])}")
    print("\n🙏 May this serve all beings on the path to liberation")