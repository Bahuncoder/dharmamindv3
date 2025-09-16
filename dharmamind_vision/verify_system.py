#!/usr/bin/env python3
"""
🌟 DharmaMind Vision System Verification Script

This script verifies that the DharmaMind Vision system is properly structured
and all components are accessible. It provides a comprehensive health check
of the revolutionary AI yoga and meditation system.

Author: DharmaMind Development Team
Version: 1.0.0 Revolutionary Release
"""

import sys
import logging
from pathlib import Path
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print the system verification banner."""
    banner = """
🌟 ══════════════════════════════════════════════════════════════════════════════════════════ 🌟
                           DharmaMind Vision - System Verification
                        Revolutionary AI Yoga & Meditation Platform
                                    Checking System Health...
🌟 ══════════════════════════════════════════════════════════════════════════════════════════ 🌟
"""
    print(banner)

def check_python_version():
    """Check if Python version meets requirements."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= min_version:
        print(f"✅ Python Version: {sys.version.split()[0]} (Supported)")
        return True
    else:
        print(f"❌ Python Version: {sys.version.split()[0]} (Requires 3.8+)")
        return False

def check_package_imports():
    """Check if the main package and components can be imported."""
    print("\n🔍 Checking Package Imports...")
    
    # Main package
    try:
        import dharmamind_vision
        print("✅ Main package: dharmamind_vision")
        
        # Check version info
        if hasattr(dharmamind_vision, '__version__'):
            print(f"   Version: {dharmamind_vision.__version__}")
        
        # Check system info
        if hasattr(dharmamind_vision, 'get_system_info'):
            info = dharmamind_vision.get_system_info()
            print(f"   System: {info.get('name', 'Unknown')}")
            print(f"   Status: {info.get('status', 'Unknown')}")
        
    except ImportError as e:
        print(f"❌ Main package import failed: {e}")
        return False
    
    # Core components
    core_components = [
        'dharmamind_vision.core',
        'dharmamind_vision.utils', 
        'dharmamind_vision.models'
    ]
    
    core_success = True
    for component in core_components:
        try:
            importlib.import_module(component)
            print(f"✅ Core component: {component}")
        except ImportError as e:
            print(f"❌ Core component failed: {component} - {e}")
            core_success = False
    
    # Check specific core classes
    core_classes = [
        ('dharmamind_vision.core.pose_estimation', 'PoseEstimator'),
        ('dharmamind_vision.core.breath_detection', 'BreathDetector'),
        ('dharmamind_vision.core.meditation_detection', 'MeditationDetector')
    ]
    
    for module_name, class_name in core_classes:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"✅ Essential class: {class_name}")
        except (ImportError, AttributeError) as e:
            print(f"⚠️  Essential class not available: {class_name} - {e}")
    
    return core_success

def check_dependencies():
    """Check if required dependencies are available."""
    print("\n📦 Checking Dependencies...")
    
    critical_deps = [
        'cv2',          # opencv-python
        'mediapipe',    # mediapipe
        'numpy',        # numpy
        'scipy',        # scipy
        'sklearn',      # scikit-learn
    ]
    
    optional_deps = [
        'torch',        # PyTorch
        'transformers', # Transformers
        'fastapi',      # FastAPI
        'uvicorn',      # Uvicorn
        'pandas',       # Pandas
        'matplotlib',   # Matplotlib
    ]
    
    critical_success = True
    for dep in critical_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ Critical dependency: {dep}")
        except ImportError:
            print(f"❌ Critical dependency missing: {dep}")
            critical_success = False
    
    optional_count = 0
    for dep in optional_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ Optional dependency: {dep}")
            optional_count += 1
        except ImportError:
            print(f"⚠️  Optional dependency missing: {dep}")
    
    print(f"\n📊 Optional Dependencies: {optional_count}/{len(optional_deps)} available")
    return critical_success

def check_file_structure():
    """Check if the file structure is complete."""
    print("\n📁 Checking File Structure...")
    
    base_path = Path(__file__).parent
    
    critical_files = [
        'dharma_mind_vision_master.py',
        'requirements.txt',
        '__init__.py',
        'core/__init__.py',
        'core/pose_estimation.py',
        'core/breath_detection.py', 
        'core/meditation_detection.py',
        'utils/__init__.py',
        'models/__init__.py'
    ]
    
    optional_files = [
        'advanced_posture_correction.py',
        'meditation_analysis.py',
        'progressive_learning.py',
        'life_integration.py',
        'session_management.py',
        'intelligent_feedback.py',
        'README.md',
        'api/vision_api.py'
    ]
    
    critical_success = True
    for file_path in critical_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ Critical file: {file_path}")
        else:
            print(f"❌ Critical file missing: {file_path}")
            critical_success = False
    
    optional_count = 0
    for file_path in optional_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ Optional file: {file_path}")
            optional_count += 1
        else:
            print(f"⚠️  Optional file missing: {file_path}")
    
    print(f"\n📊 Optional Files: {optional_count}/{len(optional_files)} present")
    return critical_success

def check_system_functionality():
    """Test basic system functionality."""
    print("\n🧪 Testing System Functionality...")
    
    try:
        # Test system initialization
        import dharmamind_vision
        if hasattr(dharmamind_vision, 'initialize_system'):
            print("✅ System initialization function available")
        
        # Test core status
        from dharmamind_vision.core import get_core_status
        status = get_core_status()
        print("✅ Core status check successful")
        
        for component, available in status.items():
            status_icon = "✅" if available else "⚠️ "
            print(f"   {status_icon} {component}: {available}")
        
        return True
        
    except Exception as e:
        print(f"❌ System functionality test failed: {e}")
        return False

def generate_system_report():
    """Generate a comprehensive system report."""
    print("\n📋 System Health Report")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Package Imports", check_package_imports()),
        ("Dependencies", check_dependencies()),
        ("File Structure", check_file_structure()),
        ("System Functionality", check_system_functionality())
    ]
    
    passed = sum(1 for _, success in checks if success)
    total = len(checks)
    
    print(f"\n📊 Overall Health: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ✅ EXCELLENT: System is fully functional and ready for revolutionary AI yoga guidance!")
    elif passed >= total * 0.8:
        print("👍 ✅ GOOD: System is mostly functional with minor issues")
    elif passed >= total * 0.6:
        print("⚠️  WARNING: System has significant issues that should be addressed")
    else:
        print("❌ CRITICAL: System requires major fixes before use")
    
    return passed / total

def main():
    """Main verification function."""
    print_banner()
    
    try:
        health_score = generate_system_report()
        
        print("\n🌟 Next Steps:")
        if health_score == 1.0:
            print("   🚀 Ready to revolutionize yoga and meditation with AI!")
            print("   🧘 Try: python dharma_mind_vision_master.py")
        elif health_score >= 0.8:
            print("   📦 Install missing optional dependencies for full functionality")
            print("   🧘 Basic system should work for testing")
        else:
            print("   🔧 Fix critical issues before using the system")
            print("   📋 Check logs above for specific problems")
        
        print(f"\n🕉️  May this technology serve your practice with wisdom and compassion")
        
    except Exception as e:
        logger.error(f"Verification failed with unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())