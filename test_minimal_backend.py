#!/usr/bin/env python3
"""Minimal backend test"""
import sys
import os

# Set minimal environment variables
os.environ['JWT_SECRET_KEY'] = 'test-jwt-secret'
os.environ['SECRET_KEY'] = 'test-secret'
os.environ['ENVIRONMENT'] = 'development'
os.environ['DEBUG'] = 'true'

try:
    # Add backend to path
    sys.path.insert(0, '/media/rupert/New Volume/Dharmamind/FinalTesting/DharmaMind-chat-master/backend')
    
    print("🧪 Testing minimal backend imports...")
    
    # Test basic imports only
    from app.config import settings
    print("✅ Config loaded!")
    
    from app.routes.health import router as health_router
    print("✅ Health routes imported!")
    
    print("\n🎯 MINIMAL BACKEND TEST PASSED!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)