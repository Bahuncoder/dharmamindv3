#!/usr/bin/env python3
"""
Startup script for DharmaMind Enhanced Enterprise Authentication
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now try to import and start the app
try:
    from enhanced_enterprise_auth import app
    import uvicorn
    
    print("🚀 Starting DharmaMind Enhanced Enterprise Authentication...")
    print("📍 Server will be available at: http://localhost:8081")
    print("📖 API Documentation: http://localhost:8081/docs")
    print("🔐 Enterprise Authentication Features:")
    print("   ✅ User Registration & Login")
    print("   ✅ Password Security Validation")
    print("   ✅ Profile Management")
    print("   ✅ Session Management")
    print("   ✅ Security Logging")
    print("")
    
    uvicorn.run(app, host="127.0.0.1", port=8081, reload=False)
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Attempting to fix import issues...")
    
    # Try direct execution
    import enhanced_enterprise_auth
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)
