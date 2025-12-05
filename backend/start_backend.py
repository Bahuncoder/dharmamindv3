#!/usr/bin/env python3
"""
DharmaMind Backend Startup Script
Handles environment loading and server startup
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def load_environment():
    """Load environment variables from .env file"""
    env_file = current_dir / ".env"
    
    if env_file.exists():
        print(f"📄 Loading environment from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
                    print(f"✅ Set {key.strip()}")
    else:
        print("⚠️  No .env file found, using system environment variables")

def main():
    """Main startup function"""
    print("🚀 DharmaMind Backend Starting...")
    print("=" * 40)
    
    # Load environment variables first
    load_environment()
    
    # Verify critical environment variables
    required_vars = ["JWT_SECRET_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            print(f"❌ Missing required environment variable: {var}")
            sys.exit(1)
        else:
            print(f"✅ Found {var}")
    
    print("🎯 Environment loaded successfully!")
    print("📡 Starting FastAPI server...")
    
    # Now import and start the app
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[str(current_dir)],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
