"""
🕉️ DharmaMindVisionEngine - Integrated Traditional Yoga Vision System

The main engine that orchestrates pose detection, asana classification, and 
alignment analysis for traditional Hindu yoga practice.

Combines modern computer vision with ancient wisdom from:
- Hatha Yoga Pradipika
- Gheranda Samhita  
- Shiva Samhita
- Yoga Sutras of Patanjali
"""

import cv2
import numpy as np
import asyncio
import json
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import queue
import time
from pathlib import Path

from .pose_detector import HathaYogaPoseDetector, PoseKeypoints
from .asana_classifier import AsanaClassifier, ClassificationResult, AsanaInfo
from .alignment_checker import AlignmentChecker, AlignmentFeedback, AlignmentLevel

@dataclass
class VisionSession:
    """Complete yoga session analysis."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_frames: int
    poses_detected: int
    asanas_identified: List[str]
    average_alignment: float
    peak_alignment: float
    session_feedback: List[str]
    spiritual_insights: List[str]
    recommended_next_poses: List[str]

@dataclass
class RealTimeAnalysis:
    """Real-time pose analysis result."""
    frame_id: int
    timestamp: datetime
    pose_detected: bool
    keypoints: Optional[PoseKeypoints]
    classification: Optional[ClassificationResult]
    alignment: Optional[AlignmentFeedback]
    spiritual_guidance: List[str]
    processing_time: float

@dataclass
class VisionConfig:
    """Configuration for vision engine."""
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.5
    classification_confidence: float = 0.6
    enable_alignment_feedback: bool = True
    enable_spiritual_guidance: bool = True
    save_session_data: bool = True
    session_timeout: int = 300  # seconds
    max_frames_per_second: int = 30

class DharmaMindVisionEngine:
    """
    Integrated traditional yoga vision system.
    
    Features:
    - Real-time pose detection using MediaPipe
    - Traditional asana classification (15 classical poses)
    - Scriptural alignment feedback
    - Chakra energy analysis
    - Spiritual guidance integration
    - Session tracking and progression
    
    Designed for integration with DharmaMind AI systems.
    """
    
    def __init__(self, 
                 config: Optional[VisionConfig] = None,
                 model_path: Optional[str] = None,
                 enable_rishi_integration: bool = True):
        """
        Initialize the vision engine.
        
        Args:
            config: Vision configuration
            model_path: Path to pre-trained classification model
            enable_rishi_integration: Enable integration with Rishi spiritual guidance
        """
        self.config = config or VisionConfig()
        self.enable_rishi = enable_rishi_integration
        
        # Initialize core components
        print("🕉️ Initializing DharmaMind Vision Engine...")
        
        self.pose_detector = HathaYogaPoseDetector(
            min_detection_confidence=self.config.detection_confidence,
            min_tracking_confidence=self.config.tracking_confidence
        )
        
        self.asana_classifier = AsanaClassifier(model_path=model_path)
        self.alignment_checker = AlignmentChecker()
        
        # Session management
        self.current_session: Optional[VisionSession] = None
        self.session_history: List[VisionSession] = []
        self.frame_buffer = queue.Queue(maxsize=100)
        
        # Performance tracking
        self.frame_count = 0
        self.detection_stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'classification_accuracy': [],
            'average_processing_time': []
        }
        
        # Spiritual guidance system
        self.spiritual_context = {
            'current_asana': None,
            'practice_duration': 0,
            'alignment_progress': [],
            'chakra_focus': [],
            'traditional_guidance': []
        }
        
        # Initialize classifier if not pre-trained
        if not self.asana_classifier.is_trained:
            print("🧘‍♀️ Training asana classifier...")
            self.asana_classifier.train()
        
        print("✨ Vision Engine initialized successfully!")
    
    def start_session(self, session_name: Optional[str] = None) -> str:
        """
        Start a new yoga practice session.
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            Session ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if session_name:
            session_id = f"{session_name}_{session_id}"
        
        self.current_session = VisionSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            total_frames=0,
            poses_detected=0,
            asanas_identified=[],
            average_alignment=0.0,
            peak_alignment=0.0,
            session_feedback=[],
            spiritual_insights=[],
            recommended_next_poses=[]
        )
        
        print(f"🙏 Started new session: {session_id}")
        return session_id
    
    def end_session(self) -> Optional[VisionSession]:
        """
        End the current session and return summary.
        
        Returns:
            Completed session data
        """
        if not self.current_session:
            return None
        
        self.current_session.end_time = datetime.now()
        
        # Calculate session statistics
        if self.current_session.poses_detected > 0:
            alignment_scores = [feedback.overall_score for feedback in self.detection_stats.get('alignment_history', [])]
            if alignment_scores:
                self.current_session.average_alignment = np.mean(alignment_scores)
                self.current_session.peak_alignment = max(alignment_scores)
        
        # Generate session feedback
        self._generate_session_feedback()
        
        # Save to history
        self.session_history.append(self.current_session)
        completed_session = self.current_session
        self.current_session = None
        
        print(f"📊 Session completed: {completed_session.session_id}")
        return completed_session
    
    def analyze_frame(self, frame: np.ndarray, 
                     target_asana: Optional[str] = None) -> RealTimeAnalysis:
        """
        Analyze a single frame for pose, classification, and alignment.
        
        Args:
            frame: Input image frame
            target_asana: Expected asana for targeted feedback
            
        Returns:
            Complete real-time analysis
        """
        start_time = time.time()
        frame_id = self.frame_count
        self.frame_count += 1
        
        analysis = RealTimeAnalysis(
            frame_id=frame_id,
            timestamp=datetime.now(),
            pose_detected=False,
            keypoints=None,
            classification=None,
            alignment=None,
            spiritual_guidance=[],
            processing_time=0.0
        )
        
        try:
            # Step 1: Pose Detection
            keypoints = self.pose_detector.detect_pose(frame)
            if keypoints is None:
                analysis.processing_time = time.time() - start_time
                return analysis
            
            analysis.pose_detected = True
            analysis.keypoints = keypoints
            
            # Update session stats
            if self.current_session:
                self.current_session.total_frames += 1
                self.current_session.poses_detected += 1
            
            # Step 2: Asana Classification
            if keypoints.confidence > self.config.classification_confidence:
                classification = self.asana_classifier.predict(keypoints)
                analysis.classification = classification
                
                # Track identified asanas
                if (self.current_session and 
                    classification.predicted_asana not in self.current_session.asanas_identified):
                    self.current_session.asanas_identified.append(classification.predicted_asana)
                
                # Update spiritual context
                self.spiritual_context['current_asana'] = classification.predicted_asana
            
            # Step 3: Alignment Analysis
            if self.config.enable_alignment_feedback and analysis.classification:
                alignment = self.alignment_checker.check_alignment(
                    keypoints, 
                    analysis.classification.predicted_asana
                )
                analysis.alignment = alignment
                
                # Track alignment progress
                if not hasattr(self.detection_stats, 'alignment_history'):
                    self.detection_stats['alignment_history'] = []
                self.detection_stats['alignment_history'].append(alignment)
            
            # Step 4: Spiritual Guidance
            if self.config.enable_spiritual_guidance:
                guidance = self._generate_spiritual_guidance(analysis, target_asana)
                analysis.spiritual_guidance = guidance
            
            # Update performance stats
            self.detection_stats['total_frames'] += 1
            self.detection_stats['successful_detections'] += 1
            
        except Exception as e:
            print(f"Error analyzing frame {frame_id}: {e}")
        
        analysis.processing_time = time.time() - start_time
        self.detection_stats['average_processing_time'].append(analysis.processing_time)
        
        return analysis
    
    def analyze_video_stream(self, video_source: Any, 
                           callback: Optional[callable] = None) -> None:
        """
        Analyze continuous video stream (webcam, video file, etc.).
        
        Args:
            video_source: Video source (int for webcam, str for file path)
            callback: Optional callback function for each frame analysis
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        print(f"🎥 Starting video stream analysis from: {video_source}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze frame
                analysis = self.analyze_frame(frame)
                
                # Call callback if provided
                if callback:
                    callback(analysis, frame)
                
                # Draw annotations
                annotated_frame = self.draw_annotations(frame, analysis)
                
                # Display frame
                cv2.imshow('DharmaMind Vision - Traditional Yoga Analysis', annotated_frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Respect FPS limit
                time.sleep(1.0 / self.config.max_frames_per_second)
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🙏 Video stream analysis completed")
    
    def draw_annotations(self, frame: np.ndarray, 
                        analysis: RealTimeAnalysis) -> np.ndarray:
        """
        Draw pose landmarks, classification, and alignment feedback on frame.
        
        Args:
            frame: Input frame
            analysis: Analysis results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw pose landmarks and chakras
        if analysis.keypoints:
            annotated_frame = self.pose_detector.draw_pose_landmarks(
                annotated_frame, analysis.keypoints
            )
        
        # Draw classification results
        if analysis.classification:
            self._draw_classification_info(annotated_frame, analysis.classification)
        
        # Draw alignment feedback
        if analysis.alignment:
            self._draw_alignment_info(annotated_frame, analysis.alignment)
        
        # Draw spiritual guidance
        if analysis.spiritual_guidance:
            self._draw_spiritual_guidance(annotated_frame, analysis.spiritual_guidance)
        
        # Draw session info
        if self.current_session:
            self._draw_session_info(annotated_frame)
        
        return annotated_frame
    
    def _draw_classification_info(self, frame: np.ndarray, 
                                classification: ClassificationResult) -> None:
        """Draw classification information on frame."""
        height, width = frame.shape[:2]
        
        # Main prediction
        text = f"Asana: {classification.predicted_asana}"
        confidence_text = f"Confidence: {classification.confidence:.2f}"
        
        # Draw background box
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 100), (255, 255, 255), 2)
        
        # Draw text
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Top predictions
        for i, (asana, conf) in enumerate(classification.top_predictions[:3]):
            y_pos = 85 + i * 20
            pred_text = f"{i+1}. {asana}: {conf:.2f}"
            cv2.putText(frame, pred_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _draw_alignment_info(self, frame: np.ndarray, 
                           alignment: AlignmentFeedback) -> None:
        """Draw alignment feedback on frame."""
        height, width = frame.shape[:2]
        
        # Alignment score and level
        score_text = f"Alignment: {alignment.overall_score:.2f}"
        level_text = f"Level: {alignment.level.value}"
        
        # Choose color based on level
        color_map = {
            AlignmentLevel.EXCELLENT: (0, 255, 0),    # Green
            AlignmentLevel.GOOD: (0, 200, 255),       # Orange
            AlignmentLevel.MODERATE: (0, 255, 255),   # Yellow
            AlignmentLevel.NEEDS_WORK: (0, 165, 255), # Orange
            AlignmentLevel.POOR: (0, 0, 255)          # Red
        }
        color = color_map.get(alignment.level, (255, 255, 255))
        
        # Draw alignment info
        cv2.rectangle(frame, (width-300, 10), (width-10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (width-300, 10), (width-10, 120), color, 2)
        
        cv2.putText(frame, score_text, (width-290, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, level_text, (width-290, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show top correction
        if alignment.corrections:
            correction = alignment.corrections[0][:30] + "..." if len(alignment.corrections[0]) > 30 else alignment.corrections[0]
            cv2.putText(frame, correction, (width-290, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_spiritual_guidance(self, frame: np.ndarray, 
                               guidance: List[str]) -> None:
        """Draw spiritual guidance on frame."""
        if not guidance:
            return
        
        height, width = frame.shape[:2]
        
        # Show first guidance message
        message = guidance[0]
        if len(message) > 50:
            message = message[:47] + "..."
        
        # Draw at bottom of frame
        cv2.rectangle(frame, (10, height-60), (width-10, height-10), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, height-60), (width-10, height-10), (255, 215, 0), 2)
        
        cv2.putText(frame, "🕉️ Guidance:", (20, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 2)
        cv2.putText(frame, message, (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_session_info(self, frame: np.ndarray) -> None:
        """Draw current session information."""
        if not self.current_session:
            return
        
        height, width = frame.shape[:2]
        
        # Session duration
        duration = datetime.now() - self.current_session.start_time
        duration_text = f"Session: {str(duration).split('.')[0]}"
        poses_text = f"Poses: {len(self.current_session.asanas_identified)}"
        
        # Draw session info
        cv2.rectangle(frame, (10, height-120), (250, height-70), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, height-120), (250, height-70), (100, 100, 100), 1)
        
        cv2.putText(frame, duration_text, (20, height-100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, poses_text, (20, height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _generate_spiritual_guidance(self, analysis: RealTimeAnalysis, 
                                   target_asana: Optional[str] = None) -> List[str]:
        """Generate spiritual guidance based on current analysis."""
        guidance = []
        
        # Classification-based guidance
        if analysis.classification:
            asana_name = analysis.classification.predicted_asana
            asana_info = self.asana_classifier.get_asana_info(asana_name)
            
            if asana_info:
                if analysis.classification.confidence > 0.8:
                    guidance.append(f"Beautiful {asana_info.english_name}! Focus on the breath.")
                else:
                    guidance.append(f"Approaching {asana_info.english_name}. Refine the form.")
        
        # Alignment-based guidance
        if analysis.alignment:
            if analysis.alignment.level == AlignmentLevel.EXCELLENT:
                guidance.append("Perfect alignment! Feel the divine energy flowing.")
            elif analysis.alignment.level == AlignmentLevel.GOOD:
                guidance.append("Excellent form. Deepen your awareness.")
            else:
                if analysis.alignment.spiritual_guidance:
                    guidance.extend(analysis.alignment.spiritual_guidance[:1])
        
        # General encouragement
        if not guidance:
            guidance.append("Breathe mindfully and honor your practice.")
        
        # Rishi integration (if enabled)
        if self.enable_rishi and analysis.alignment:
            rishi_guidance = self._get_rishi_guidance(analysis)
            if rishi_guidance:
                guidance.extend(rishi_guidance)
        
        return guidance[:3]  # Limit to 3 messages
    
    def _get_rishi_guidance(self, analysis: RealTimeAnalysis) -> List[str]:
        """Get guidance from Rishi spiritual AI system."""
        # This would integrate with the actual Rishi system
        # For now, return placeholder guidance
        
        if not analysis.alignment:
            return []
        
        rishi_guidance = []
        
        # Simulate Rishi responses based on alignment
        if analysis.alignment.overall_score > 0.8:
            rishi_guidance.append("The ancient sages smile upon your practice.")
        elif analysis.alignment.overall_score > 0.6:
            rishi_guidance.append("Your dedication to dharma is evident.")
        else:
            rishi_guidance.append("Every step on the path brings you closer to truth.")
        
        return rishi_guidance
    
    def _generate_session_feedback(self) -> None:
        """Generate comprehensive session feedback."""
        if not self.current_session:
            return
        
        session = self.current_session
        
        # Duration feedback
        duration = (session.end_time - session.start_time).total_seconds() / 60
        session.session_feedback.append(f"🕰️ Practice duration: {duration:.1f} minutes")
        
        # Pose variety feedback
        if len(session.asanas_identified) >= 5:
            session.session_feedback.append("🌟 Excellent variety in your practice!")
        elif len(session.asanas_identified) >= 3:
            session.session_feedback.append("👍 Good diversity in asanas explored")
        else:
            session.session_feedback.append("🧘‍♀️ Focus on exploring more poses next time")
        
        # Alignment feedback
        if session.peak_alignment > 0.9:
            session.session_feedback.append("💫 You achieved excellent alignment!")
        elif session.average_alignment > 0.7:
            session.session_feedback.append("✨ Consistent good form throughout")
        else:
            session.session_feedback.append("🌱 Great foundation - keep practicing!")
        
        # Spiritual insights
        session.spiritual_insights = [
            "Your practice contributes to your spiritual evolution",
            "Each asana is a step toward self-realization",
            "Consistency in practice leads to inner transformation"
        ]
        
        # Recommended next poses
        if len(session.asanas_identified) > 0:
            current_poses = set(session.asanas_identified)
            all_poses = set(self.asana_classifier.get_supported_asanas())
            not_practiced = list(all_poses - current_poses)
            
            if not_practiced:
                session.recommended_next_poses = not_practiced[:3]
    
    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current or last session."""
        session = self.current_session or (self.session_history[-1] if self.session_history else None)
        
        if not session:
            return None
        
        return {
            'session_id': session.session_id,
            'duration_minutes': ((session.end_time or datetime.now()) - session.start_time).total_seconds() / 60,
            'poses_detected': session.poses_detected,
            'unique_asanas': len(session.asanas_identified),
            'asanas_practiced': session.asanas_identified,
            'average_alignment': session.average_alignment,
            'peak_alignment': session.peak_alignment,
            'feedback': session.session_feedback,
            'spiritual_insights': session.spiritual_insights,
            'recommendations': session.recommended_next_poses
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        stats = self.detection_stats.copy()
        
        if stats['total_frames'] > 0:
            stats['detection_rate'] = stats['successful_detections'] / stats['total_frames']
        else:
            stats['detection_rate'] = 0.0
        
        if stats['average_processing_time']:
            stats['avg_processing_time'] = np.mean(stats['average_processing_time'])
            stats['fps_estimate'] = 1.0 / stats['avg_processing_time']
        else:
            stats['avg_processing_time'] = 0.0
            stats['fps_estimate'] = 0.0
        
        return stats
    
    def export_session_data(self, session_id: str, output_path: str) -> bool:
        """
        Export session data to JSON file.
        
        Args:
            session_id: Session to export
            output_path: Output file path
            
        Returns:
            Success status
        """
        session = None
        
        # Find session
        if self.current_session and self.current_session.session_id == session_id:
            session = self.current_session
        else:
            for hist_session in self.session_history:
                if hist_session.session_id == session_id:
                    session = hist_session
                    break
        
        if not session:
            return False
        
        try:
            # Convert to dictionary
            session_data = asdict(session)
            
            # Convert datetime objects to strings
            session_data['start_time'] = session.start_time.isoformat()
            if session.end_time:
                session_data['end_time'] = session.end_time.isoformat()
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            print(f"📄 Session data exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting session data: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.pose_detector:
            self.pose_detector.release()
        
        print("🙏 DharmaMind Vision Engine cleaned up. Namaste!")

# Utility functions for integration
def create_vision_engine(config_dict: Optional[Dict] = None) -> DharmaMindVisionEngine:
    """
    Factory function to create a vision engine with configuration.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configured vision engine
    """
    config = VisionConfig()
    
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return DharmaMindVisionEngine(config=config)

def process_image_base64(engine: DharmaMindVisionEngine, 
                        image_base64: str) -> Dict[str, Any]:
    """
    Process base64 encoded image and return analysis.
    
    Args:
        engine: Vision engine instance
        image_base64: Base64 encoded image
        
    Returns:
        Analysis results as dictionary
    """
    try:
        # Decode image
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze
        analysis = engine.analyze_frame(frame)
        
        # Convert to dictionary
        result = {
            'frame_id': analysis.frame_id,
            'timestamp': analysis.timestamp.isoformat(),
            'pose_detected': analysis.pose_detected,
            'processing_time': analysis.processing_time
        }
        
        if analysis.classification:
            result['classification'] = {
                'asana': analysis.classification.predicted_asana,
                'confidence': analysis.classification.confidence,
                'top_predictions': analysis.classification.top_predictions
            }
        
        if analysis.alignment:
            result['alignment'] = {
                'score': analysis.alignment.overall_score,
                'level': analysis.alignment.level.value,
                'feedback': analysis.alignment.corrections[:3]
            }
        
        result['spiritual_guidance'] = analysis.spiritual_guidance
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

# Usage example
if __name__ == "__main__":
    print("🕉️ DharmaMind Vision Engine - Traditional Yoga AI")
    print("=" * 60)
    
    # Create engine
    engine = DharmaMindVisionEngine()
    
    # Start a session
    session_id = engine.start_session("test_practice")
    
    # Get performance stats
    stats = engine.get_performance_stats()
    print(f"Engine ready - Detection rate: {stats['detection_rate']:.2f}")
    
    # Example: analyze webcam (uncomment to test)
    # try:
    #     engine.analyze_video_stream(0)  # Use webcam
    # except KeyboardInterrupt:
    #     print("\n🙏 Practice session interrupted")
    # finally:
    #     # End session
    #     session = engine.end_session()
    #     if session:
    #         summary = engine.get_session_summary()
    #         print(f"Session Summary: {summary}")
    #     
    #     engine.cleanup()
    
    print("\n🙏 May this technology serve the path of dharma and liberation")
    print("\"Yoga is the journey of the self, through the self, to the Self\" - Bhagavad Gita")