"""
🧘 Meditation Detection System for DharmaMind Vision

Advanced meditation state detection from micro-movements:
- Stillness quality assessment
- Micro-movement pattern analysis  
- Meditation depth estimation
- Traditional meditation state mapping
- Real-time meditation quality feedback

Essential component for meditation practice analysis and guidance.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from enum import Enum
import statistics

class MeditationState(Enum):
    """Traditional meditation states from yoga philosophy."""
    SETTLING = "settling"           # Initial settling phase
    FOCUSED = "focused"             # Concentrated attention
    ABSORBED = "absorbed"           # Deep absorption
    TRANSCENDENT = "transcendent"   # Beyond ordinary awareness
    UNKNOWN = "unknown"

@dataclass
class MeditationMetrics:
    """Meditation quality metrics."""
    stillness_score: float
    micro_movement_level: float
    meditation_depth: float
    state_duration: float
    traditional_state: MeditationState

class MeditationDetector:
    """
    🧘 Advanced Meditation Detection Engine
    
    Detects meditation states through micro-movement analysis and stillness assessment.
    Maps detected states to traditional meditation classifications.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the meditation detection system."""
        self.config = config or {}
        
        # Movement tracking
        self.position_history = []
        self.movement_threshold = 0.001  # Very sensitive to micro-movements
        self.stillness_window = 30  # Frames to analyze for stillness
        
        # Meditation state tracking
        self.current_state = MeditationState.UNKNOWN
        self.state_start_time = time.time()
        self.meditation_session_start = None
        
        # Analysis components
        self.stillness_analyzer = StillnessAnalyzer()
        self.state_classifier = MeditationStateClassifier()
        
    def analyze_frame(self, frame: np.ndarray, pose_data: Dict) -> Dict:
        """
        Analyze meditation state from current frame and pose data.
        
        Args:
            frame: Current video frame
            pose_data: Pose landmarks from pose estimation
            
        Returns:
            Meditation analysis results
        """
        
        if not pose_data.get('pose_detected', False):
            return {'meditation_detected': False}
            
        # Extract key position data for movement analysis
        position_data = self._extract_position_data(pose_data)
        
        if position_data is None:
            return {'meditation_detected': False}
            
        # Analyze stillness and micro-movements
        stillness_analysis = self.stillness_analyzer.analyze_stillness(self.position_history)
        
        # Classify meditation state
        state_classification = self.state_classifier.classify_state(
            stillness_analysis, self.position_history
        )
        
        # Update meditation state tracking
        self._update_meditation_state(state_classification)
        
        # Calculate meditation metrics
        meditation_metrics = self._calculate_meditation_metrics(stillness_analysis, state_classification)
        
        # Traditional meditation assessment
        traditional_assessment = self._assess_traditional_meditation(meditation_metrics)
        
        return {
            'meditation_detected': True,
            'current_state': self.current_state.value,
            'state_duration': time.time() - self.state_start_time,
            'stillness_score': stillness_analysis['stillness_score'],
            'micro_movement_level': stillness_analysis['micro_movement_level'],
            'meditation_depth': meditation_metrics.meditation_depth,
            'traditional_assessment': traditional_assessment,
            'session_duration': self._get_session_duration(),
            'timestamp': time.time()
        }
        
    def _extract_position_data(self, pose_data: Dict) -> Optional[Dict]:
        """Extract key position data for movement analysis."""
        
        try:
            landmarks = np.array(pose_data['landmarks'])
            
            # Key landmarks for meditation posture analysis
            nose = 0
            left_shoulder = 11
            right_shoulder = 12
            left_hip = 23
            right_hip = 24
            
            # Calculate key reference points
            head_position = landmarks[nose]
            shoulder_center = (landmarks[left_shoulder] + landmarks[right_shoulder]) / 2
            hip_center = (landmarks[left_hip] + landmarks[right_hip]) / 2
            
            position_data = {
                'head_position': head_position,
                'shoulder_center': shoulder_center,
                'hip_center': hip_center,
                'overall_centroid': np.mean([head_position, shoulder_center, hip_center], axis=0),
                'timestamp': time.time()
            }
            
            # Store in position history
            self.position_history.append(position_data)
            
            # Keep only recent positions for analysis
            if len(self.position_history) > 100:
                self.position_history = self.position_history[-100:]
                
            # Start meditation session tracking if not already started
            if self.meditation_session_start is None:
                self.meditation_session_start = time.time()
                
            return position_data
            
        except Exception as e:
            print(f"Error extracting position data: {e}")
            return None
            
    def _update_meditation_state(self, state_classification: Dict):
        """Update current meditation state based on classification."""
        
        new_state = state_classification.get('predicted_state', MeditationState.UNKNOWN)
        
        # Update state if changed
        if new_state != self.current_state:
            self.current_state = new_state
            self.state_start_time = time.time()
            
    def _calculate_meditation_metrics(self, stillness_analysis: Dict, state_classification: Dict) -> MeditationMetrics:
        """Calculate comprehensive meditation metrics."""
        
        # Extract key metrics
        stillness_score = stillness_analysis.get('stillness_score', 0.0)
        micro_movement_level = stillness_analysis.get('micro_movement_level', 1.0)
        
        # Calculate meditation depth based on multiple factors
        state_duration = time.time() - self.state_start_time
        depth_from_stillness = stillness_score
        depth_from_duration = min(state_duration / 300, 1.0)  # Normalize to 5 minutes max
        depth_from_state = self._get_state_depth_factor(self.current_state)
        
        meditation_depth = (depth_from_stillness * 0.5 + 
                          depth_from_duration * 0.3 + 
                          depth_from_state * 0.2)
        
        return MeditationMetrics(
            stillness_score=stillness_score,
            micro_movement_level=micro_movement_level,
            meditation_depth=meditation_depth,
            state_duration=state_duration,
            traditional_state=self.current_state
        )
        
    def _get_state_depth_factor(self, state: MeditationState) -> float:
        """Get depth factor for different meditation states."""
        
        depth_factors = {
            MeditationState.SETTLING: 0.2,
            MeditationState.FOCUSED: 0.5,
            MeditationState.ABSORBED: 0.8,
            MeditationState.TRANSCENDENT: 1.0,
            MeditationState.UNKNOWN: 0.0
        }
        
        return depth_factors.get(state, 0.0)
        
    def _assess_traditional_meditation(self, metrics: MeditationMetrics) -> Dict:
        """Assess meditation quality from traditional perspective."""
        
        assessment = {
            'dharana_quality': 'developing',  # Concentration
            'dhyana_indicators': [],          # Meditation indicators
            'samadhi_proximity': 0.0,         # Closeness to absorption
            'traditional_guidance': [],
            'state_progression': self._assess_state_progression()
        }
        
        # Assess Dharana (concentration) quality
        if metrics.stillness_score > 0.8 and metrics.state_duration > 60:
            assessment['dharana_quality'] = 'excellent'
            assessment['dhyana_indicators'].append("Sustained one-pointed concentration")
        elif metrics.stillness_score > 0.6:
            assessment['dharana_quality'] = 'good'
            assessment['dhyana_indicators'].append("Developing concentration")
        else:
            assessment['traditional_guidance'].append("Focus on breath to develop concentration")
            
        # Assess proximity to Samadhi (absorption)
        if metrics.traditional_state == MeditationState.ABSORBED:
            assessment['samadhi_proximity'] = 0.8
            assessment['dhyana_indicators'].append("Approaching absorbed state")
        elif metrics.traditional_state == MeditationState.TRANSCENDENT:
            assessment['samadhi_proximity'] = 1.0
            assessment['dhyana_indicators'].append("Deep meditative absorption")
            
        # Traditional guidance based on state
        if metrics.traditional_state == MeditationState.SETTLING:
            assessment['traditional_guidance'].append("Allow the mind to naturally settle")
        elif metrics.traditional_state == MeditationState.FOCUSED:
            assessment['traditional_guidance'].append("Maintain gentle awareness on your chosen object")
            
        return assessment
        
    def _assess_state_progression(self) -> Dict:
        """Assess progression through meditation states."""
        
        return {
            'current_stage': self.current_state.value,
            'time_in_stage': time.time() - self.state_start_time,
            'progression_quality': 'natural',
            'traditional_mapping': self._map_to_traditional_stages()
        }
        
    def _map_to_traditional_stages(self) -> str:
        """Map current state to traditional meditation stages."""
        
        state_mapping = {
            MeditationState.SETTLING: "Pratyahara - Withdrawal of senses",
            MeditationState.FOCUSED: "Dharana - Concentration developing",
            MeditationState.ABSORBED: "Dhyana - Meditative flow",
            MeditationState.TRANSCENDENT: "Samadhi - Absorbed awareness",
            MeditationState.UNKNOWN: "Initial stages of practice"
        }
        
        return state_mapping.get(self.current_state, "Unknown stage")
        
    def _get_session_duration(self) -> float:
        """Get total meditation session duration."""
        
        if self.meditation_session_start is None:
            return 0.0
            
        return time.time() - self.meditation_session_start

class StillnessAnalyzer:
    """Analyzes stillness quality and micro-movements."""
    
    def analyze_stillness(self, position_history: List[Dict]) -> Dict:
        """Analyze stillness from position history."""
        
        if len(position_history) < 5:
            return {
                'stillness_score': 0.0,
                'micro_movement_level': 1.0,
                'movement_consistency': 0.0
            }
            
        # Extract recent positions for analysis
        recent_positions = position_history[-30:] if len(position_history) >= 30 else position_history
        
        # Calculate movement metrics
        movement_analysis = self._analyze_micro_movements(recent_positions)
        stillness_score = self._calculate_stillness_score(movement_analysis)
        
        return {
            'stillness_score': stillness_score,
            'micro_movement_level': movement_analysis['average_movement'],
            'movement_consistency': movement_analysis['movement_consistency'],
            'stillness_trend': movement_analysis['stillness_trend']
        }
        
    def _analyze_micro_movements(self, positions: List[Dict]) -> Dict:
        """Analyze micro-movements between consecutive positions."""
        
        movements = []
        centroids = [pos['overall_centroid'] for pos in positions]
        
        # Calculate movements between consecutive frames
        for i in range(1, len(centroids)):
            movement = np.linalg.norm(centroids[i] - centroids[i-1])
            movements.append(movement)
            
        if not movements:
            return {
                'average_movement': 0.0,
                'movement_consistency': 0.0,
                'stillness_trend': 'stable'
            }
            
        # Calculate movement statistics
        average_movement = np.mean(movements)
        movement_std = np.std(movements)
        movement_consistency = 1.0 - (movement_std / (average_movement + 0.0001))  # Avoid division by zero
        
        # Determine stillness trend
        if len(movements) >= 10:
            recent_movements = movements[-10:]
            earlier_movements = movements[-20:-10] if len(movements) >= 20 else movements[:-10]
            
            if recent_movements and earlier_movements:
                recent_avg = np.mean(recent_movements)
                earlier_avg = np.mean(earlier_movements)
                
                if recent_avg < earlier_avg * 0.8:
                    stillness_trend = 'improving'
                elif recent_avg > earlier_avg * 1.2:
                    stillness_trend = 'declining'
                else:
                    stillness_trend = 'stable'
            else:
                stillness_trend = 'stable'
        else:
            stillness_trend = 'developing'
            
        return {
            'average_movement': average_movement,
            'movement_consistency': max(0, min(1, movement_consistency)),
            'stillness_trend': stillness_trend,
            'raw_movements': movements
        }
        
    def _calculate_stillness_score(self, movement_analysis: Dict) -> float:
        """Calculate overall stillness score."""
        
        average_movement = movement_analysis['average_movement']
        movement_consistency = movement_analysis['movement_consistency']
        
        # Stillness score based on low movement and high consistency
        movement_score = 1.0 - min(average_movement / 0.01, 1.0)  # Normalize to 0.01 threshold
        consistency_score = movement_consistency
        
        # Trend bonus
        trend_bonus = 0.0
        if movement_analysis['stillness_trend'] == 'improving':
            trend_bonus = 0.1
        elif movement_analysis['stillness_trend'] == 'declining':
            trend_bonus = -0.1
            
        stillness_score = (movement_score * 0.7 + consistency_score * 0.3) + trend_bonus
        
        return max(0, min(1, stillness_score))

class MeditationStateClassifier:
    """Classifies meditation states based on movement and duration patterns."""
    
    def classify_state(self, stillness_analysis: Dict, position_history: List[Dict]) -> Dict:
        """Classify current meditation state."""
        
        stillness_score = stillness_analysis['stillness_score']
        session_duration = len(position_history) * 0.033  # Assuming 30 FPS
        
        # Rule-based state classification
        predicted_state = self._rule_based_classification(stillness_score, session_duration)
        
        return {
            'predicted_state': predicted_state,
            'confidence': self._calculate_confidence(stillness_score, session_duration),
            'classification_factors': {
                'stillness_score': stillness_score,
                'session_duration': session_duration,
                'primary_indicator': self._get_primary_indicator(stillness_score, session_duration)
            }
        }
        
    def _rule_based_classification(self, stillness_score: float, session_duration: float) -> MeditationState:
        """Rule-based meditation state classification."""
        
        # Classification based on stillness and duration
        if session_duration < 30:  # First 30 seconds
            return MeditationState.SETTLING
        elif stillness_score < 0.4:
            return MeditationState.SETTLING
        elif stillness_score < 0.7:
            return MeditationState.FOCUSED
        elif stillness_score < 0.9 and session_duration > 120:
            return MeditationState.ABSORBED
        elif stillness_score >= 0.9 and session_duration > 300:
            return MeditationState.TRANSCENDENT
        else:
            return MeditationState.FOCUSED
            
    def _calculate_confidence(self, stillness_score: float, session_duration: float) -> float:
        """Calculate confidence in state classification."""
        
        # Higher confidence with longer duration and clearer stillness patterns
        duration_confidence = min(session_duration / 300, 1.0)  # Max confidence at 5 minutes
        stillness_confidence = stillness_score
        
        return (duration_confidence * 0.4 + stillness_confidence * 0.6)
        
    def _get_primary_indicator(self, stillness_score: float, session_duration: float) -> str:
        """Get primary indicator for state classification."""
        
        if session_duration < 60:
            return "session_duration"
        else:
            return "stillness_quality"