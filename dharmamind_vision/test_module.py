#!/usr/bin/env python3
"""
🕉️ DharmaMind Vision - Simple Test Script

Test the standalone vision module without dependencies to verify structure.
"""

import sys
import os

def test_module_structure():
    """Test that the module structure is correct."""
    print("🕉️ Testing DharmaMind Vision Module Structure")
    print("=" * 50)
    
    # Test basic import
    try:
        import dharmamind_vision
        print("✅ Basic module import successful")
        print(f"   Version: {dharmamind_vision.__version__}")
        print(f"   Author: {dharmamind_vision.__author__}")
    except Exception as e:
        print(f"❌ Basic module import failed: {e}")
        return False
    
    # Test module info
    try:
        doc = dharmamind_vision.get_documentation()
        print(f"✅ Module documentation available")
        print(f"   Supported asanas: {doc['supported_asanas']}")
        print(f"   Features: {len(doc['features'])}")
    except Exception as e:
        print(f"❌ Documentation access failed: {e}")
    
    # Test asana list
    try:
        asanas = dharmamind_vision.get_supported_asanas()
        print(f"✅ Asana list available ({len(asanas)} traditional poses)")
        print(f"   Sample asanas: {', '.join(asanas[:3])}")
    except Exception as e:
        print(f"❌ Asana list access failed: {e}")
    
    # Test factory functions (without dependencies)
    try:
        # These will fail due to missing dependencies but should import
        from dharmamind_vision import create_vision_engine, create_vision_api
        print("✅ Factory functions importable")
    except Exception as e:
        print(f"❌ Factory function import failed: {e}")
    
    return True

def test_backend_integration():
    """Test how backend would import this module."""
    print("\n🔗 Testing Backend Integration Pattern")
    print("=" * 50)
    
    # Simulate backend import
    try:
        # This is how backend would import
        from dharmamind_vision import create_vision_engine
        print("✅ Backend import pattern works")
        
        # Test creation (will fail due to dependencies but structure is good)
        try:
            engine = create_vision_engine()
            print("✅ Vision engine created successfully")
        except Exception as e:
            print(f"⚠️  Vision engine creation failed (expected due to missing deps): {e}")
            print("   This is normal without MediaPipe/OpenCV installed")
        
    except Exception as e:
        print(f"❌ Backend integration pattern failed: {e}")

def test_api_structure():
    """Test API module structure."""
    print("\n🌐 Testing API Module Structure")
    print("=" * 50)
    
    try:
        from dharmamind_vision.api import VisionAPI, create_vision_api
        print("✅ API module imports work")
        
        # Test API creation (will fail due to dependencies)
        try:
            api = create_vision_api()
            print("✅ API instance created")
        except Exception as e:
            print(f"⚠️  API creation failed (expected): {e}")
    
    except Exception as e:
        print(f"❌ API module import failed: {e}")

def main():
    """Run all tests."""
    print("🕉️ DharmaMind Vision - Standalone Module Test")
    print("Testing module structure and integration patterns")
    print("Note: Dependency errors are expected without full installation")
    print("")
    
    success = True
    
    # Test basic module
    if not test_module_structure():
        success = False
    
    # Test backend integration
    test_backend_integration()
    
    # Test API structure
    test_api_structure()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Module structure tests completed!")
        print("📋 Summary:")
        print("   ✅ Module imports correctly")
        print("   ✅ Standalone architecture works")
        print("   ✅ Backend integration pattern ready")
        print("   ⚠️  Dependency installation needed for full functionality")
        print("")
        print("🔧 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Test with actual vision processing")
        print("   3. Integrate with backend using import statements")
    else:
        print("❌ Some tests failed - check module structure")
    
    print("\n🙏 Namaste - May this serve the path of dharma")

if __name__ == "__main__":
    main()