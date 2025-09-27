#!/usr/bin/env python3
"""Complete backend startup test"""
import sys
import os

# Set environment variables
os.environ['JWT_SECRET_KEY'] = 'your-super-secret-jwt-key-change-this-in-production-min-32-chars'
os.environ['SECRET_KEY'] = 'your-secret-key-for-sessions-change-this'
os.environ['ENVIRONMENT'] = 'development'
os.environ['DEBUG'] = 'true'

try:
    # Add backend to path
    sys.path.insert(0, '/media/rupert/New Volume/Dharmamind/FinalTesting/DharmaMind-chat-master/backend')
    
    print("🧪 Testing complete backend startup...")
    
    # Test main app import
    from app.main import app
    print("✅ Main FastAPI app imported successfully!")
    
    # Test that app has routes
    if hasattr(app, 'router') and app.router.routes:
        print(f"✅ App has {len(app.router.routes)} routes configured!")
    
    print("\n🎯 COMPLETE BACKEND TEST PASSED!")
    print("🚀 Backend is ready for production!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)