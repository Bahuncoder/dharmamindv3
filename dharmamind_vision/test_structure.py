#!/usr/bin/env python3
"""
🌟 DharmaMind Vision - Simple Demo and Structure Test

This demo tests the package structure without requiring heavy dependencies
like MediaPipe, TensorFlow, etc. It verifies that our revolutionary system
architecture is properly organized and accessible.

Author: DharmaMind Development Team
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to import dharmamind_vision
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def print_banner():
    """Print the demo banner."""
    banner = """
🌟 ══════════════════════════════════════════════════════════════════════════════════════════ 🌟
                        DharmaMind Vision - Structure & Demo Test
                        Revolutionary AI Yoga & Meditation System
                                Version 1.0.0 - Testing Mode
🌟 ══════════════════════════════════════════════════════════════════════════════════════════ 🌟
"""
    print(banner)

def test_package_structure():
    """Test the basic package structure without heavy imports."""
    print("🔍 Testing Package Structure...")
    
    # Test basic package import
    try:
        import dharmamind_vision
        print("✅ Main package imported successfully")
        
        # Test version info
        if hasattr(dharmamind_vision, '__version__'):
            print(f"   📦 Version: {dharmamind_vision.__version__}")
        
        if hasattr(dharmamind_vision, '__author__'):
            print(f"   👨‍💻 Author: {dharmamind_vision.__author__}")
        
        # Test constants
        if hasattr(dharmamind_vision, 'TRADITIONAL_ASANAS'):
            asanas = dharmamind_vision.TRADITIONAL_ASANAS
            print(f"   🧘 Traditional Asanas: {len(asanas)} poses available")
            print(f"      First 3: {', '.join(asanas[:3])}")
        
        if hasattr(dharmamind_vision, 'SYSTEM_CAPABILITIES'):
            capabilities = dharmamind_vision.SYSTEM_CAPABILITIES
            print(f"   🚀 System Capabilities: {len(capabilities)} features")
            print(f"      Examples: {', '.join(capabilities[:2])}")
        
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False
    
    return True

def test_core_structure():
    """Test core module structure."""
    print("\n🧠 Testing Core Structure...")
    
    try:
        from dharmamind_vision import core
        print("✅ Core module accessible")
        
        # Test core status function
        if hasattr(core, 'get_core_status'):
            status = core.get_core_status()
            print("   📊 Core Component Status:")
            for component, available in status.items():
                status_icon = "✅" if available else "⚠️ "
                print(f"      {status_icon} {component}")
        
        # Test available components
        if hasattr(core, 'get_available_components'):
            components = core.get_available_components()
            print(f"   📦 Available Components: {len(components)}")
            if components:
                print(f"      Examples: {', '.join(components[:3])}")
        
    except Exception as e:
        print(f"⚠️  Core structure test: {e}")
        return False
    
    return True

def test_utilities():
    """Test utilities structure."""
    print("\n🛠️  Testing Utilities...")
    
    try:
        from dharmamind_vision import utils
        print("✅ Utils module accessible")
        
        # Test if utility classes are available
        if hasattr(utils, 'VisionUtils'):
            print("   🎯 VisionUtils class available")
        
        if hasattr(utils, 'YogaGeometry'):
            print("   📐 YogaGeometry class available")
            
        if hasattr(utils, 'TraditionalWisdom'):
            print("   🕉️  TraditionalWisdom class available")
            
    except Exception as e:
        print(f"⚠️  Utils test: {e}")
        return False
    
    return True

def test_models():
    """Test models structure."""
    print("\n📊 Testing Models...")
    
    try:
        from dharmamind_vision import models
        print("✅ Models module accessible")
        
        # Test data model availability
        model_classes = [
            'PoseFrame', 'SessionMetrics', 'AsanaInstruction',
            'PostureCorrection', 'BreathingPattern', 'TraditionalConcept'
        ]
        
        available_models = []
        for model_class in model_classes:
            if hasattr(models, model_class):
                available_models.append(model_class)
                print(f"   📋 {model_class} available")
        
        print(f"   📊 Total Models: {len(available_models)}/{len(model_classes)}")
        
    except Exception as e:
        print(f"⚠️  Models test: {e}")
        return False
    
    return True

def test_documentation_functions():
    """Test documentation and info functions."""
    print("\n📚 Testing Documentation Functions...")
    
    try:
        import dharmamind_vision
        
        # Test system info
        if hasattr(dharmamind_vision, 'get_system_info'):
            info = dharmamind_vision.get_system_info()
            print("✅ System info function works")
            print(f"   📋 System: {info.get('name', 'Unknown')}")
            print(f"   🔢 Capabilities: {len(info.get('capabilities', []))}")
        
        # Test documentation
        if hasattr(dharmamind_vision, 'get_documentation'):
            docs = dharmamind_vision.get_documentation()
            print("✅ Documentation function works")
            print(f"   📖 Description: {docs.get('description', 'Unknown')}")
            print(f"   🧘 Supported Asanas: {docs.get('supported_asanas', 0)}")
        
        # Test asanas list
        if hasattr(dharmamind_vision, 'get_supported_asanas'):
            asanas = dharmamind_vision.get_supported_asanas()
            print("✅ Supported asanas function works")
            print(f"   🕉️  Traditional Poses: {len(asanas)} available")
        
    except Exception as e:
        print(f"⚠️  Documentation test: {e}")
        return False
    
    return True

def display_system_summary():
    """Display a comprehensive system summary."""
    print("\n📋 System Summary")
    print("=" * 50)
    
    try:
        import dharmamind_vision
        
        # Get documentation
        if hasattr(dharmamind_vision, 'get_documentation'):
            docs = dharmamind_vision.get_documentation()
            
            print(f"🌟 System: {docs.get('description', 'DharmaMind Vision')}")
            print(f"📦 Version: {getattr(dharmamind_vision, '__version__', '1.0.0')}")
            
            # Show capabilities
            capabilities = docs.get('capabilities', [])
            if capabilities:
                print(f"\n🚀 Revolutionary Capabilities ({len(capabilities)}):")
                for i, capability in enumerate(capabilities[:8], 1):  # Show first 8
                    print(f"   {i}. {capability}")
                if len(capabilities) > 8:
                    print(f"   ... and {len(capabilities) - 8} more!")
            
            # Show traditional asanas
            asanas = docs.get('asana_list', [])
            if asanas:
                print(f"\n🧘 Traditional Asanas ({len(asanas)}):")
                for i, asana in enumerate(asanas[:5], 1):  # Show first 5
                    print(f"   {i}. {asana}")
                if len(asanas) > 5:
                    print(f"   ... and {len(asanas) - 5} more classical poses!")
            
            # Show source texts
            sources = docs.get('source_texts', [])
            if sources:
                print(f"\n📚 Based on Classical Texts:")
                for source in sources:
                    print(f"   📖 {source}")
        
    except Exception as e:
        print(f"Could not generate full summary: {e}")

def main():
    """Main demo function."""
    print_banner()
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Core Structure", test_core_structure),
        ("Utilities", test_utilities),
        ("Models", test_models),
        ("Documentation Functions", test_documentation_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Results
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ✅ EXCELLENT: All structure tests passed!")
        success_rate = "Perfect"
    elif passed >= total * 0.8:
        print("👍 ✅ GOOD: Most tests passed with minor issues")
        success_rate = "Good" 
    elif passed >= total * 0.6:
        print("⚠️  WARNING: Some tests failed")
        success_rate = "Fair"
    else:
        print("❌ CRITICAL: Major structural issues")
        success_rate = "Poor"
    
    # Display summary
    display_system_summary()
    
    print(f"\n🌟 Structure Status: {success_rate}")
    print("🕉️  May this technology serve your practice with wisdom and compassion")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())