#!/usr/bin/env python3
"""
DharmaMind Simple Backend Startup
Simplified version for basic chat functionality
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

def create_simple_main():
    """Create a simplified main.py for basic functionality"""
    simple_main = '''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Dict, Any

# Load environment
load_environment()

app = FastAPI(
    title="DharmaMind Chat API",
    description="Spiritual AI Chat Backend",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    context: str = ""

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

def load_environment():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

@app.get("/")
async def root():
    return {
        "message": "🕉️ DharmaMind Chat API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Backend is running"
    }

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage) -> ChatResponse:
    """Simple chat endpoint"""
    try:
        # Simple spiritual response logic for testing
        spiritual_response = generate_simple_response(message.message)
        
        return ChatResponse(
            response=spiritual_response,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_simple_response(message: str) -> str:
    """Generate a simple spiritual response for testing"""
    
    # Simple keyword-based spiritual responses
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['peace', 'calm', 'stress']):
        return "🕉️ Peace comes from within. Take a deep breath and remember that this moment is temporary. Find your center through mindful breathing and let tranquility flow through you."
    
    elif any(word in message_lower for word in ['love', 'heart', 'compassion']):
        return "💝 Love is the essence of our being. Cultivate compassion for yourself and others. Remember that love multiplies when shared and heals when received."
    
    elif any(word in message_lower for word in ['wisdom', 'knowledge', 'learn']):
        return "📚 True wisdom comes from experience and reflection. The greatest teachers are often our challenges. Embrace learning as a lifelong journey toward understanding."
    
    elif any(word in message_lower for word in ['purpose', 'meaning', 'life']):
        return "🌟 Your purpose unfolds naturally when you align with your authentic self. Listen to your inner voice and trust the journey. Every experience has meaning in the larger tapestry of life."
    
    elif any(word in message_lower for word in ['meditation', 'mindfulness']):
        return "🧘‍♂️ Meditation is the practice of returning home to yourself. Even a few minutes of mindful breathing can transform your day. Start where you are, use what you have, do what you can."
    
    else:
        return "🙏 Thank you for sharing with me. Remember that every moment is an opportunity for growth and every challenge is a chance to develop strength. How can I help guide you further on your spiritual journey?"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open(current_dir / "simple_main.py", "w") as f:
        f.write(simple_main)

def main():
    """Main startup function"""
    print("🚀 DharmaMind Simple Backend Starting...")
    print("=" * 45)
    
    # Load environment variables first
    load_environment()
    
    # Create simple main if needed
    create_simple_main()
    
    print("🎯 Environment loaded successfully!")
    print("📡 Starting Simple FastAPI server...")
    
    # Now import and start the simple app
    try:
        import uvicorn
        uvicorn.run(
            "simple_main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[str(current_dir)],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
