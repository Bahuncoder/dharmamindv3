"""
🕉️ VisionAPI - FastAPI Interface for DharmaMind Vision System

RESTful API and WebSocket interface for traditional yoga pose detection,
classification, and alignment analysis.

Provides endpoints for:
- Real-time pose analysis
- Session management  
- Asana classification
- Alignment feedback
- Spiritual guidance integration
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
import base64
import json
import asyncio
from datetime import datetime
import io
from PIL import Image

from ..core.vision_engine import DharmaMindVisionEngine, VisionConfig, create_vision_engine

# Pydantic models for API
class VisionConfigModel(BaseModel):
    """Vision configuration model."""
    detection_confidence: float = Field(0.7, ge=0.0, le=1.0)
    tracking_confidence: float = Field(0.5, ge=0.0, le=1.0)
    classification_confidence: float = Field(0.6, ge=0.0, le=1.0)
    enable_alignment_feedback: bool = True
    enable_spiritual_guidance: bool = True
    save_session_data: bool = True
    session_timeout: int = Field(300, ge=60, le=3600)

class ImageAnalysisRequest(BaseModel):
    """Request model for image analysis."""
    image_base64: str = Field(..., description="Base64 encoded image")
    target_asana: Optional[str] = Field(None, description="Expected asana for targeted feedback")
    include_alignment: bool = Field(True, description="Include alignment analysis")
    include_spiritual_guidance: bool = Field(True, description="Include spiritual guidance")

class SessionStartRequest(BaseModel):
    """Request model for starting a session."""
    session_name: Optional[str] = Field(None, description="Optional session name")
    config: Optional[VisionConfigModel] = Field(None, description="Session configuration")

class AsanaInfo(BaseModel):
    """Asana information model."""
    name: str
    sanskrit_name: str
    english_name: str
    difficulty: str
    category: str
    benefits: List[str]
    precautions: List[str]
    chakras_activated: List[str]
    description: str
    traditional_text_reference: str

class VisionAPI:
    """
    FastAPI application for DharmaMind Vision System.
    
    Provides REST and WebSocket endpoints for yoga pose analysis.
    """
    
    def __init__(self, enable_rishi_integration: bool = True):
        """
        Initialize the Vision API.
        
        Args:
            enable_rishi_integration: Enable Rishi spiritual guidance integration
        """
        self.app = FastAPI(
            title="DharmaMind Vision API",
            description="Traditional Yoga Pose Detection and Analysis System",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize vision engine
        self.vision_engine = DharmaMindVisionEngine(
            enable_rishi_integration=enable_rishi_integration
        )
        
        # Active WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", tags=["General"])
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "🕉️ DharmaMind Vision API - Traditional Yoga Analysis",
                "version": "1.0.0",
                "description": "AI-powered traditional yoga pose detection and spiritual guidance",
                "endpoints": {
                    "docs": "/docs",
                    "health": "/health",
                    "analyze": "/analyze",
                    "session": "/session",
                    "asanas": "/asanas",
                    "websocket": "/ws"
                },
                "traditional_reference": "Based on Hatha Yoga Pradipika by Yogi Svatmarama"
            }
        
        @self.app.get("/health", tags=["General"])
        async def health_check():
            """Health check endpoint."""
            try:
                stats = self.vision_engine.get_performance_stats()
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "vision_engine": "ready",
                    "classifier_trained": self.vision_engine.asana_classifier.is_trained,
                    "supported_asanas": len(self.vision_engine.asana_classifier.get_supported_asanas()),
                    "performance": stats
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
        
        @self.app.post("/analyze", tags=["Analysis"])
        async def analyze_image(request: ImageAnalysisRequest):
            """
            Analyze a single image for yoga pose detection and alignment.
            
            Returns comprehensive analysis including:
            - Pose detection results
            - Asana classification
            - Alignment feedback
            - Spiritual guidance
            """
            try:
                # Decode base64 image
                image_data = base64.b64decode(request.image_base64)
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    raise HTTPException(status_code=400, detail="Invalid image data")
                
                # Analyze frame
                analysis = self.vision_engine.analyze_frame(frame, request.target_asana)
                
                # Prepare response
                result = {
                    "frame_id": analysis.frame_id,
                    "timestamp": analysis.timestamp.isoformat(),
                    "pose_detected": analysis.pose_detected,
                    "processing_time": analysis.processing_time
                }
                
                if analysis.pose_detected and analysis.keypoints:
                    result["pose_confidence"] = analysis.keypoints.confidence
                    result["chakra_points"] = len(analysis.keypoints.chakra_points)
                
                if analysis.classification:
                    result["classification"] = {
                        "predicted_asana": analysis.classification.predicted_asana,
                        "confidence": analysis.classification.confidence,
                        "top_predictions": analysis.classification.top_predictions[:5],
                        "features_used": analysis.classification.features_used
                    }
                
                if analysis.alignment and request.include_alignment:
                    result["alignment"] = {
                        "overall_score": analysis.alignment.overall_score,
                        "level": analysis.alignment.level.value,
                        "geometric_scores": analysis.alignment.geometric_scores,
                        "chakra_analysis": {k: v.value for k, v in analysis.alignment.chakra_analysis.items()},
                        "corrections": analysis.alignment.corrections,
                        "benefits_achieved": analysis.alignment.benefits_achieved,
                        "areas_for_improvement": analysis.alignment.areas_for_improvement,
                        "traditional_quotes": analysis.alignment.traditional_quotes
                    }
                
                if request.include_spiritual_guidance:
                    result["spiritual_guidance"] = analysis.spiritual_guidance
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.post("/analyze/upload", tags=["Analysis"])
        async def analyze_uploaded_image(
            file: UploadFile = File(...),
            target_asana: Optional[str] = None,
            include_alignment: bool = True,
            include_spiritual_guidance: bool = True
        ):
            """
            Analyze an uploaded image file for yoga pose detection.
            
            Supports common image formats: JPEG, PNG, etc.
            """
            try:
                # Read and validate file
                if not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="File must be an image")
                
                # Read image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                
                # Convert to OpenCV format
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Analyze
                analysis = self.vision_engine.analyze_frame(frame, target_asana)
                
                # Prepare response (similar to analyze_image)
                result = {
                    "filename": file.filename,
                    "frame_id": analysis.frame_id,
                    "timestamp": analysis.timestamp.isoformat(),
                    "pose_detected": analysis.pose_detected,
                    "processing_time": analysis.processing_time
                }
                
                if analysis.classification:
                    result["classification"] = {
                        "predicted_asana": analysis.classification.predicted_asana,
                        "confidence": analysis.classification.confidence,
                        "top_predictions": analysis.classification.top_predictions[:5]
                    }
                
                if analysis.alignment and include_alignment:
                    result["alignment"] = {
                        "overall_score": analysis.alignment.overall_score,
                        "level": analysis.alignment.level.value,
                        "corrections": analysis.alignment.corrections[:3],
                        "benefits_achieved": analysis.alignment.benefits_achieved[:3]
                    }
                
                if include_spiritual_guidance:
                    result["spiritual_guidance"] = analysis.spiritual_guidance
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Upload analysis failed: {str(e)}")
        
        @self.app.post("/session/start", tags=["Session Management"])
        async def start_session(request: SessionStartRequest):
            """Start a new yoga practice session."""
            try:
                # Update configuration if provided
                if request.config:
                    config = VisionConfig(**request.config.dict())
                    # Note: In a full implementation, you'd update the engine config
                
                session_id = self.vision_engine.start_session(request.session_name)
                
                return {
                    "session_id": session_id,
                    "status": "started",
                    "timestamp": datetime.now().isoformat(),
                    "message": "🙏 New yoga session started. Begin your practice!"
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")
        
        @self.app.post("/session/end", tags=["Session Management"])
        async def end_session():
            """End the current yoga session and get summary."""
            try:
                session = self.vision_engine.end_session()
                
                if not session:
                    raise HTTPException(status_code=404, detail="No active session found")
                
                summary = self.vision_engine.get_session_summary()
                
                return {
                    "session_ended": True,
                    "summary": summary,
                    "message": "🙏 Session completed. Review your practice journey!"
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")
        
        @self.app.get("/session/current", tags=["Session Management"])
        async def get_current_session():
            """Get current session information."""
            try:
                summary = self.vision_engine.get_session_summary()
                
                if not summary:
                    return {"active_session": False, "message": "No active session"}
                
                return {
                    "active_session": True,
                    "session_info": summary
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")
        
        @self.app.get("/asanas", tags=["Asana Information"])
        async def get_supported_asanas():
            """Get list of all supported traditional asanas."""
            try:
                asanas = self.vision_engine.asana_classifier.get_supported_asanas()
                asana_details = []
                
                for asana_name in asanas:
                    info = self.vision_engine.asana_classifier.get_asana_info(asana_name)
                    if info:
                        asana_details.append({
                            "name": info.name,
                            "sanskrit_name": info.sanskrit_name,
                            "english_name": info.english_name,
                            "difficulty": info.difficulty.value,
                            "category": info.category.value,
                            "benefits": info.benefits,
                            "precautions": info.precautions,
                            "chakras_activated": info.chakras_activated,
                            "description": info.description,
                            "traditional_text_reference": info.traditional_text_reference
                        })
                
                return {
                    "total_asanas": len(asanas),
                    "traditional_text": "Based on Hatha Yoga Pradipika Chapter 2",
                    "asanas": asana_details
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get asanas: {str(e)}")
        
        @self.app.get("/asanas/{asana_name}", tags=["Asana Information"])
        async def get_asana_details(asana_name: str):
            """Get detailed information about a specific asana."""
            try:
                info = self.vision_engine.asana_classifier.get_asana_info(asana_name)
                
                if not info:
                    raise HTTPException(status_code=404, detail=f"Asana '{asana_name}' not found")
                
                return {
                    "name": info.name,
                    "sanskrit_name": info.sanskrit_name,
                    "english_name": info.english_name,
                    "difficulty": info.difficulty.value,
                    "category": info.category.value,
                    "benefits": info.benefits,
                    "precautions": info.precautions,
                    "chakras_activated": info.chakras_activated,
                    "description": info.description,
                    "traditional_text_reference": info.traditional_text_reference
                }
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get asana details: {str(e)}")
        
        @self.app.get("/stats", tags=["Statistics"])
        async def get_performance_stats():
            """Get vision engine performance statistics."""
            try:
                stats = self.vision_engine.get_performance_stats()
                
                return {
                    "performance_stats": stats,
                    "timestamp": datetime.now().isoformat(),
                    "engine_status": "operational"
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """
            WebSocket endpoint for real-time yoga pose analysis.
            
            Supports live video stream analysis with immediate feedback.
            """
            await self.connect_websocket(websocket)
            
            try:
                while True:
                    # Receive image data
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message["type"] == "image":
                        # Process image
                        result = await self.process_websocket_image(message["data"])
                        await websocket.send_text(json.dumps(result))
                    
                    elif message["type"] == "start_session":
                        session_id = self.vision_engine.start_session(
                            message.get("session_name")
                        )
                        await websocket.send_text(json.dumps({
                            "type": "session_started",
                            "session_id": session_id
                        }))
                    
                    elif message["type"] == "end_session":
                        session = self.vision_engine.end_session()
                        summary = self.vision_engine.get_session_summary()
                        await websocket.send_text(json.dumps({
                            "type": "session_ended",
                            "summary": summary
                        }))
                    
            except WebSocketDisconnect:
                self.disconnect_websocket(websocket)
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
                self.disconnect_websocket(websocket)
    
    async def connect_websocket(self, websocket: WebSocket):
        """Accept and track WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send welcome message
        welcome = {
            "type": "connected",
            "message": "🕉️ Connected to DharmaMind Vision - Begin your practice!",
            "supported_asanas": len(self.vision_engine.asana_classifier.get_supported_asanas())
        }
        await websocket.send_text(json.dumps(welcome))
    
    def disconnect_websocket(self, websocket: WebSocket):
        """Remove WebSocket connection from tracking."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def process_websocket_image(self, image_base64: str) -> Dict[str, Any]:
        """Process image received via WebSocket."""
        try:
            # Decode and analyze image
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"type": "error", "message": "Invalid image data"}
            
            # Analyze frame
            analysis = self.vision_engine.analyze_frame(frame)
            
            # Prepare WebSocket response
            result = {
                "type": "analysis",
                "timestamp": analysis.timestamp.isoformat(),
                "pose_detected": analysis.pose_detected,
                "processing_time": analysis.processing_time
            }
            
            if analysis.classification:
                result["asana"] = analysis.classification.predicted_asana
                result["confidence"] = analysis.classification.confidence
            
            if analysis.alignment:
                result["alignment_score"] = analysis.alignment.overall_score
                result["alignment_level"] = analysis.alignment.level.value
                result["feedback"] = analysis.alignment.corrections[:2]
            
            if analysis.spiritual_guidance:
                result["guidance"] = analysis.spiritual_guidance[0]
            
            return result
            
        except Exception as e:
            return {"type": "error", "message": str(e)}
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app

# Factory function for creating API instance
def create_vision_api(enable_rishi: bool = True) -> VisionAPI:
    """
    Create a VisionAPI instance.
    
    Args:
        enable_rishi: Enable Rishi spiritual guidance integration
        
    Returns:
        VisionAPI instance
    """
    return VisionAPI(enable_rishi_integration=enable_rishi)

# Usage example
if __name__ == "__main__":
    import uvicorn
    
    print("🕉️ Starting DharmaMind Vision API Server...")
    print("Traditional Yoga Pose Detection and Analysis System")
    print("=" * 60)
    
    # Create API instance
    vision_api = create_vision_api()
    app = vision_api.get_app()
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )