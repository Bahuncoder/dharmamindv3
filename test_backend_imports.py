#!/usr/bin/env python3
"""Simple test script to verify backend imports"""
import sys
import os

try:
    # Add backend to path
    sys.path.insert(0, '/media/rupert/New Volume/Dharmamind/FinalTesting/DharmaMind-chat-master/backend')
    
    # Test imports
    from app.main import app
    print("✅ Backend FastAPI app imported successfully!")
    
    # Test config
    from app.config import settings
    print("✅ Backend configuration loaded successfully!")
    
    # Test auth routes
    from app.routes.auth import router as auth_router
    print("✅ Authentication routes imported successfully!")
    
    # Test health routes
    from app.routes.health import router as health_router
    print("✅ Health check routes imported successfully!")
    
    print("\n🎯 ALL BACKEND IMPORTS SUCCESSFUL!")
    print("✅ Backend is ready to run!")
    
except Exception as e:
    print(f"❌ Backend import error: {e}")
    sys.exit(1)