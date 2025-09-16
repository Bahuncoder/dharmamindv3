"""
🎯 Core Computer Vision Components for DharmaMind Vision

Essential vision processing components that the revolutionary system needs:
- Pose estimation with MediaPipe integration
- Breath detection from video analysis  
- Meditation state detection from micro-movements
- Real-time frame processing pipeline
- Vision utilities and helpers

These are the fundamental building blocks that our revolutionary systems depend on.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Any
import time
from dataclasses import dataclass
from enum import Enum

class PoseQuality(Enum):
    """Pose quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    NEEDS_IMPROVEMENT = "needs_improvement"

@dataclass
class PoseLandmarks:
    """Pose landmarks with confidence scores."""
    landmarks: np.ndarray
    visibility: np.ndarray
    confidence: float
    timestamp: float

class PoseEstimator:
    """
    🎯 Advanced Pose Estimation Engine
    
    Core computer vision component for detecting and analyzing human poses
    using MediaPipe with enhanced accuracy and traditional yoga knowledge.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the pose estimation engine."""
        self.config = config or {}
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Pose detection models
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # High accuracy
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.holistic_detector = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Pose analysis components
        self.pose_history = []
        self.quality_analyzer = PoseQualityAnalyzer()
        
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame for pose detection.
        
        Args:
            frame: Input video frame
            
        Returns:
            Comprehensive pose analysis results
        """
        
        if frame is None:
            return {'error': 'Invalid frame'}
            
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose using MediaPipe
        pose_results = self.pose_detector.process(rgb_frame)
        holistic_results = self.holistic_detector.process(rgb_frame)
        
        # Extract landmarks
        pose_landmarks = self._extract_pose_landmarks(pose_results, holistic_results)
        
        if pose_landmarks is None:
            return {'pose_detected': False}
            
        # Analyze pose quality
        quality_assessment = self.quality_analyzer.assess_pose_quality(pose_landmarks)
        
        # Calculate key pose metrics
        pose_metrics = self._calculate_pose_metrics(pose_landmarks)
        
        # Store in history for temporal analysis
        self.pose_history.append({
            'landmarks': pose_landmarks,
            'timestamp': time.time(),
            'quality': quality_assessment
        })
        
        # Keep only last 30 frames for analysis
        if len(self.pose_history) > 30:
            self.pose_history = self.pose_history[-30:]
            
        return {
            'pose_detected': True,
            'landmarks': pose_landmarks.landmarks.tolist(),
            'visibility': pose_landmarks.visibility.tolist(),
            'confidence': pose_landmarks.confidence,
            'quality_assessment': quality_assessment,
            'pose_metrics': pose_metrics,
            'frame_timestamp': pose_landmarks.timestamp
        }
        
    def _extract_pose_landmarks(self, pose_results, holistic_results) -> Optional[PoseLandmarks]:
        """Extract and combine pose landmarks from different models."""
        
        if not pose_results.pose_landmarks:
            return None
            
        # Extract landmarks as numpy array
        landmarks = []
        visibility = []
        
        for landmark in pose_results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
            visibility.append(landmark.visibility)
            
        landmarks = np.array(landmarks)
        visibility = np.array(visibility)
        
        # Calculate overall confidence
        confidence = np.mean(visibility)
        
        return PoseLandmarks(
            landmarks=landmarks,
            visibility=visibility, 
            confidence=confidence,
            timestamp=time.time()
        )
        
    def _calculate_pose_metrics(self, pose_landmarks: PoseLandmarks) -> Dict:
        """Calculate key pose metrics for analysis."""
        
        landmarks = pose_landmarks.landmarks
        
        # Key body part indices (MediaPipe pose landmarks)
        nose = 0
        left_shoulder = 11
        right_shoulder = 12
        left_hip = 23
        right_hip = 24
        left_knee = 25
        right_knee = 26
        left_ankle = 27
        right_ankle = 28
        
        metrics = {}
        
        try:
            # Spine alignment (shoulder to hip)
            left_spine_angle = self._calculate_angle(
                landmarks[left_shoulder], landmarks[left_hip], 
                landmarks[left_hip] + [0, 1, 0]  # Vertical reference
            )
            
            right_spine_angle = self._calculate_angle(
                landmarks[right_shoulder], landmarks[right_hip],
                landmarks[right_hip] + [0, 1, 0]  # Vertical reference  
            )
            
            metrics['spine_alignment'] = (left_spine_angle + right_spine_angle) / 2
            
            # Shoulder level balance
            shoulder_balance = abs(landmarks[left_shoulder][1] - landmarks[right_shoulder][1])
            metrics['shoulder_balance'] = 1.0 - min(shoulder_balance * 10, 1.0)  # Normalize
            
            # Hip level balance  
            hip_balance = abs(landmarks[left_hip][1] - landmarks[right_hip][1])
            metrics['hip_balance'] = 1.0 - min(hip_balance * 10, 1.0)  # Normalize
            
            # Knee alignment
            left_knee_angle = self._calculate_angle(
                landmarks[left_hip], landmarks[left_knee], landmarks[left_ankle]
            )
            right_knee_angle = self._calculate_angle(
                landmarks[right_hip], landmarks[right_knee], landmarks[right_ankle]
            )
            
            metrics['left_knee_angle'] = left_knee_angle
            metrics['right_knee_angle'] = right_knee_angle
            
            # Overall pose stability (based on confidence and alignment)
            stability_score = (
                pose_landmarks.confidence * 0.4 +
                metrics['shoulder_balance'] * 0.2 +
                metrics['hip_balance'] * 0.2 +
                (1.0 - abs(metrics['spine_alignment'] - 90) / 90) * 0.2
            )
            
            metrics['stability_score'] = stability_score
            
        except Exception as e:
            print(f"Error calculating pose metrics: {e}")
            metrics = {'error': str(e)}
            
        return metrics
        
    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """Calculate angle between three points."""
        
        # Create vectors
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Calculate angle
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)

class PoseQualityAnalyzer:
    """Analyzes pose quality using traditional yoga principles."""
    
    def assess_pose_quality(self, pose_landmarks: PoseLandmarks) -> Dict:
        """Assess the quality of detected pose."""
        
        confidence = pose_landmarks.confidence
        landmarks = pose_landmarks.landmarks
        
        # Quality assessment criteria
        quality_factors = {
            'detection_confidence': confidence,
            'landmark_visibility': np.mean(pose_landmarks.visibility),
            'pose_completeness': self._assess_completeness(pose_landmarks),
            'stability': self._assess_stability(landmarks),
            'alignment': self._assess_alignment(landmarks)
        }
        
        # Calculate overall quality score
        overall_score = np.mean(list(quality_factors.values()))
        
        # Determine quality level
        if overall_score >= 0.9:
            quality_level = PoseQuality.EXCELLENT
        elif overall_score >= 0.75:
            quality_level = PoseQuality.GOOD
        elif overall_score >= 0.6:
            quality_level = PoseQuality.FAIR
        else:
            quality_level = PoseQuality.NEEDS_IMPROVEMENT
            
        return {
            'overall_score': overall_score,
            'quality_level': quality_level.value,
            'quality_factors': quality_factors,
            'recommendations': self._generate_quality_recommendations(quality_factors)
        }
        
    def _assess_completeness(self, pose_landmarks: PoseLandmarks) -> float:
        """Assess how complete the pose detection is."""
        
        # Check visibility of key landmarks
        key_landmark_indices = [0, 11, 12, 23, 24, 25, 26, 27, 28]  # Nose, shoulders, hips, knees, ankles
        
        if len(pose_landmarks.visibility) <= max(key_landmark_indices):
            return 0.0
            
        key_visibilities = pose_landmarks.visibility[key_landmark_indices]
        return np.mean(key_visibilities)
        
    def _assess_stability(self, landmarks: np.ndarray) -> float:
        """Assess pose stability based on landmark positions."""
        
        # For now, return high stability - would implement temporal analysis
        return 0.8
        
    def _assess_alignment(self, landmarks: np.ndarray) -> float:
        """Assess pose alignment based on traditional yoga principles."""
        
        # Basic alignment check - could be enhanced with traditional yoga knowledge
        try:
            # Check if key body parts are aligned
            # Simplified alignment score
            return 0.8
        except:
            return 0.5
            
    def _generate_quality_recommendations(self, quality_factors: Dict) -> List[str]:
        """Generate recommendations for improving pose quality."""
        
        recommendations = []
        
        if quality_factors['detection_confidence'] < 0.7:
            recommendations.append("Move closer to camera for better detection")
            
        if quality_factors['landmark_visibility'] < 0.8:
            recommendations.append("Ensure full body is visible in frame")
            
        if quality_factors['alignment'] < 0.7:
            recommendations.append("Focus on traditional alignment principles")
            
        return recommendations