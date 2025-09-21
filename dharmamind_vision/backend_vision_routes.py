"""
🕉️ DharmaMind Backend Vision Routes

Example of how to integrate the standalone dharmamind_vision module 
into the existing DharmaMind backend.

Add these routes to your main FastAPI backend application.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64
import io
from PIL import Image
import json

# Import the standalone vision module
try:
    # In production, this would be: from dharmamind_vision import create_vision_engine
    import sys
    from pathlib import Path
    vision_path = Path(__file__).parent / "dharmamind_vision"
    sys.path.insert(0, str(vision_path))
    
    from dharmamind_vision import create_vision_engine
    VISION_AVAILABLE = True
    
    # Initialize global vision engine
    vision_engine = create_vision_engine({
        'detection_confidence': 0.7,
        'classification_confidence': 0.6,
        'enable_spiritual_guidance': True
    })
    
except Exception as e:
    print(f"⚠️  Vision system not available: {e}")
    print("Install dependencies: pip install -r dharmamind_vision/requirements.txt")
    VISION_AVAILABLE = False
    vision_engine = None

# Create router for vision endpoints
vision_router = APIRouter(prefix="/api/vision", tags=["Vision"])

class VisionAnalysisRequest(BaseModel):
    """Request model for vision analysis."""
    image_base64: str
    target_asana: Optional[str] = None
    include_spiritual_guidance: bool = True

class VisionSessionRequest(BaseModel):
    """Request model for starting vision session."""
    session_name: Optional[str] = None
    user_id: Optional[str] = None

@vision_router.get("/status")
async def get_vision_status():
    """Get vision system status."""
    return {
        "vision_available": VISION_AVAILABLE,
        "supported_asanas": 15 if VISION_AVAILABLE else 0,
        "traditional_reference": "Hatha Yoga Pradipika",
        "message": "🕉️ Traditional yoga vision system ready" if VISION_AVAILABLE else "⚠️ Install vision dependencies"
    }

@vision_router.get("/asanas")
async def get_supported_asanas():
    """Get list of supported traditional asanas."""
    if not VISION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Vision system not available")
    
    try:
        asanas = vision_engine.asana_classifier.get_supported_asanas()
        asana_details = []
        
        for asana_name in asanas:
            info = vision_engine.asana_classifier.get_asana_info(asana_name)
            if info:
                asana_details.append({
                    "name": info.name,
                    "sanskrit_name": info.sanskrit_name,
                    "english_name": info.english_name,
                    "difficulty": info.difficulty.value,
                    "category": info.category.value,
                    "benefits": info.benefits[:3],  # Limit for API response
                    "description": info.description
                })
        
        return {
            "total_asanas": len(asanas),
            "traditional_source": "Hatha Yoga Pradipika Chapter 2",
            "asanas": asana_details
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get asanas: {str(e)}")

@vision_router.post("/analyze")
async def analyze_yoga_pose(request: VisionAnalysisRequest):
    """
    Analyze yoga pose from base64 image.
    
    Returns pose detection, asana classification, and alignment feedback.
    """
    if not VISION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Vision system not available - install dependencies")
    
    try:
        # Process the image through vision engine
        result = {}
        
        # Decode base64 image
        import cv2
        import numpy as np
        
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Analyze with vision engine
        analysis = vision_engine.analyze_frame(frame, request.target_asana)
        
        # Prepare response
        result = {
            "timestamp": analysis.timestamp.isoformat(),
            "pose_detected": analysis.pose_detected,
            "processing_time": analysis.processing_time
        }
        
        if analysis.classification:
            result["classification"] = {
                "predicted_asana": analysis.classification.predicted_asana,
                "confidence": analysis.classification.confidence,
                "top_predictions": analysis.classification.top_predictions[:3]
            }
        
        if analysis.alignment:
            result["alignment"] = {
                "overall_score": analysis.alignment.overall_score,
                "level": analysis.alignment.level.value,
                "corrections": analysis.alignment.corrections[:2],
                "benefits": analysis.alignment.benefits_achieved[:2]
            }
        
        if request.include_spiritual_guidance and analysis.spiritual_guidance:
            result["spiritual_guidance"] = analysis.spiritual_guidance[:2]
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@vision_router.post("/analyze/upload")
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    target_asana: Optional[str] = None,
    include_guidance: bool = True
):
    """Analyze uploaded image file for yoga pose."""
    if not VISION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Vision system not available")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to base64 for processing
        import cv2
        import numpy as np
        
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Analyze
        analysis = vision_engine.analyze_frame(frame, target_asana)
        
        # Prepare response
        result = {
            "filename": file.filename,
            "pose_detected": analysis.pose_detected
        }
        
        if analysis.classification:
            result["asana"] = analysis.classification.predicted_asana
            result["confidence"] = analysis.classification.confidence
        
        if analysis.alignment:
            result["alignment_score"] = analysis.alignment.overall_score
            result["alignment_level"] = analysis.alignment.level.value
        
        if include_guidance and analysis.spiritual_guidance:
            result["guidance"] = analysis.spiritual_guidance[0]
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload analysis failed: {str(e)}")

@vision_router.post("/session/start")
async def start_vision_session(request: VisionSessionRequest):
    """Start a new yoga practice session with vision tracking."""
    if not VISION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Vision system not available")
    
    try:
        session_id = vision_engine.start_session(request.session_name)
        
        return {
            "session_id": session_id,
            "status": "started",
            "message": "🙏 Vision-guided yoga session started",
            "supported_asanas": len(vision_engine.asana_classifier.get_supported_asanas())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@vision_router.post("/session/end")
async def end_vision_session():
    """End current vision session and get summary."""
    if not VISION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Vision system not available")
    
    try:
        session = vision_engine.end_session()
        
        if not session:
            raise HTTPException(status_code=404, detail="No active session")
        
        summary = vision_engine.get_session_summary()
        
        return {
            "session_completed": True,
            "summary": summary,
            "message": "🙏 Practice session completed - review your journey"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")

@vision_router.websocket("/live")
async def live_yoga_analysis(websocket: WebSocket):
    """WebSocket endpoint for live yoga pose analysis."""
    if not VISION_AVAILABLE:
        await websocket.close(code=1003, reason="Vision system not available")
        return
    
    await websocket.accept()
    
    # Send welcome message
    await websocket.send_text(json.dumps({
        "type": "connected",
        "message": "🕉️ Connected to live yoga analysis",
        "supported_asanas": len(vision_engine.asana_classifier.get_supported_asanas())
    }))
    
    session_id = None
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start_session":
                session_id = vision_engine.start_session("live_session")
                await websocket.send_text(json.dumps({
                    "type": "session_started",
                    "session_id": session_id
                }))
            
            elif message["type"] == "analyze_image":
                # Process image data
                try:
                    import cv2
                    import numpy as np
                    
                    image_data = base64.b64decode(message["image_data"])
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        analysis = vision_engine.analyze_frame(frame)
                        
                        response = {
                            "type": "analysis_result",
                            "pose_detected": analysis.pose_detected,
                            "timestamp": analysis.timestamp.isoformat()
                        }
                        
                        if analysis.classification:
                            response["asana"] = analysis.classification.predicted_asana
                            response["confidence"] = analysis.classification.confidence
                        
                        if analysis.alignment:
                            response["alignment_score"] = analysis.alignment.overall_score
                            response["level"] = analysis.alignment.level.value
                        
                        if analysis.spiritual_guidance:
                            response["guidance"] = analysis.spiritual_guidance[0]
                        
                        await websocket.send_text(json.dumps(response))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Analysis failed: {str(e)}"
                    }))
            
            elif message["type"] == "end_session":
                if session_id:
                    session = vision_engine.end_session()
                    summary = vision_engine.get_session_summary()
                    
                    await websocket.send_text(json.dumps({
                        "type": "session_ended",
                        "summary": summary
                    }))
                    session_id = None
    
    except WebSocketDisconnect:
        # Clean up session if active
        if session_id:
            vision_engine.end_session()
    
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": str(e)
        }))

# Integration function for main backend
def add_vision_routes(app):
    """
    Add vision routes to the main FastAPI app.
    
    Usage in main.py:
        from backend.vision_routes import add_vision_routes
        add_vision_routes(app)
    """
    app.include_router(vision_router)
    print(f"🕉️ Vision routes added - Available: {VISION_AVAILABLE}")

# Alternative: export the router for manual inclusion
__all__ = ["vision_router", "add_vision_routes", "VISION_AVAILABLE"]

if __name__ == "__main__":
    # Test the vision routes
    print("🕉️ Testing Vision Routes Integration")
    print("=" * 40)
    print(f"Vision Available: {VISION_AVAILABLE}")
    
    if VISION_AVAILABLE:
        print("✅ Ready to add vision routes to backend")
        print("   Add to main.py: app.include_router(vision_router)")
    else:
        print("⚠️  Install vision dependencies first")
        print("   pip install -r dharmamind_vision/requirements.txt")
    
    print("\n🙏 Namaste - Traditional wisdom meets modern API")