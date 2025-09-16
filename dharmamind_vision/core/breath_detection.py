"""
🌬️ Breath Detection System for DharmaMind Vision

Advanced breath detection from video analysis:
- Chest and abdomen movement tracking
- Breathing pattern classification (Pranayama types)
- Breath rate calculation and rhythm analysis
- Traditional breathing technique recognition
- Real-time breathing quality assessment

Essential component for meditation and pranayama practice analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from enum import Enum
from scipy import signal
import statistics

class BreathingPattern(Enum):
    """Traditional breathing patterns in yoga practice."""
    NATURAL = "natural_breathing"
    UJJAYI = "ujjayi_pranayama"
    NADI_SHODHANA = "nadi_shodhana"
    BHASTRIKA = "bhastrika"
    KAPALABHATI = "kapalabhati"
    BHRAMARI = "bhramari"
    UNKNOWN = "unknown_pattern"

@dataclass
class BreathCycle:
    """Single breath cycle data."""
    inhale_duration: float
    exhale_duration: float
    pause_duration: float
    cycle_quality: float
    timestamp: float
    pattern_type: BreathingPattern

class BreathDetector:
    """
    🌬️ Advanced Breath Detection Engine
    
    Detects breathing patterns from video by analyzing chest and abdomen movement.
    Integrates traditional Pranayama knowledge for pattern classification.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the breath detection system."""
        self.config = config or {}
        
        # Breathing analysis components
        self.movement_history = []
        self.breath_cycles = []
        self.current_phase = "unknown"  # inhale, exhale, pause
        self.phase_start_time = time.time()
        
        # Pattern recognition
        self.pattern_classifier = BreathPatternClassifier()
        
        # Smoothing parameters
        self.smoothing_window = 10
        self.movement_threshold = 0.002
        
    def analyze_frame(self, frame: np.ndarray, pose_data: Dict) -> Dict:
        """
        Analyze breathing from current frame and pose data.
        
        Args:
            frame: Current video frame
            pose_data: Pose landmarks from pose estimation
            
        Returns:
            Breathing analysis results
        """
        
        if not pose_data.get('pose_detected', False):
            return {'breathing_detected': False}
            
        # Extract chest and abdomen movement
        movement_data = self._extract_movement_data(pose_data)
        
        if movement_data is None:
            return {'breathing_detected': False}
            
        # Analyze breathing pattern
        breathing_analysis = self._analyze_breathing_pattern(movement_data)
        
        # Classify breathing type
        pattern_classification = self.pattern_classifier.classify_pattern(
            self.movement_history[-20:] if len(self.movement_history) >= 20 else self.movement_history
        )
        
        # Calculate breathing metrics
        breathing_metrics = self._calculate_breathing_metrics()
        
        return {
            'breathing_detected': True,
            'current_phase': self.current_phase,
            'movement_amplitude': movement_data['amplitude'],
            'breathing_rate': breathing_metrics.get('breathing_rate', 0),
            'pattern_classification': pattern_classification,
            'breathing_quality': breathing_analysis.get('quality_score', 0),
            'traditional_assessment': self._assess_traditional_breathing(pattern_classification),
            'timestamp': time.time()
        }
        
    def _extract_movement_data(self, pose_data: Dict) -> Optional[Dict]:
        """Extract chest and abdomen movement from pose landmarks."""
        
        try:
            landmarks = np.array(pose_data['landmarks'])
            
            # Key landmark indices for breathing analysis
            left_shoulder = 11
            right_shoulder = 12
            left_hip = 23  
            right_hip = 24
            
            # Calculate chest center (midpoint between shoulders)
            chest_center = (landmarks[left_shoulder] + landmarks[right_shoulder]) / 2
            
            # Calculate abdomen center (midpoint between hips, slightly above)
            abdomen_center = (landmarks[left_hip] + landmarks[right_hip]) / 2
            abdomen_center[1] -= 0.1  # Move up slightly from hip line
            
            # Calculate movement amplitude (primarily Z-axis for depth)
            chest_z = chest_center[2] if len(chest_center) > 2 else 0
            abdomen_z = abdomen_center[2] if len(abdomen_center) > 2 else 0
            
            movement_data = {
                'chest_position': chest_center,
                'abdomen_position': abdomen_center,
                'chest_depth': chest_z,
                'abdomen_depth': abdomen_z,
                'amplitude': abs(chest_z - abdomen_z),
                'timestamp': time.time()
            }
            
            # Store in movement history
            self.movement_history.append(movement_data)
            
            # Keep only last 100 frames for analysis
            if len(self.movement_history) > 100:
                self.movement_history = self.movement_history[-100:]
                
            return movement_data
            
        except Exception as e:
            print(f"Error extracting movement data: {e}")
            return None
            
    def _analyze_breathing_pattern(self, movement_data: Dict) -> Dict:
        """Analyze breathing pattern from movement data."""
        
        if len(self.movement_history) < 5:
            return {'quality_score': 0.5}
            
        # Extract amplitude over time
        amplitudes = [data['amplitude'] for data in self.movement_history[-20:]]
        timestamps = [data['timestamp'] for data in self.movement_history[-20:]]
        
        # Smooth the signal
        if len(amplitudes) >= self.smoothing_window:
            smoothed_amplitudes = self._smooth_signal(amplitudes)
        else:
            smoothed_amplitudes = amplitudes
            
        # Detect breathing phases
        self._detect_breathing_phases(smoothed_amplitudes, timestamps)
        
        # Calculate quality metrics
        quality_score = self._calculate_breathing_quality(smoothed_amplitudes)
        
        return {
            'quality_score': quality_score,
            'smoothed_signal': smoothed_amplitudes,
            'phase_detection': self.current_phase
        }
        
    def _smooth_signal(self, signal_data: List[float]) -> List[float]:
        """Smooth breathing signal to reduce noise."""
        
        # Apply moving average smoothing
        window = min(self.smoothing_window, len(signal_data))
        smoothed = []
        
        for i in range(len(signal_data)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(signal_data), i + window // 2 + 1)
            smoothed.append(statistics.mean(signal_data[start_idx:end_idx]))
            
        return smoothed
        
    def _detect_breathing_phases(self, amplitudes: List[float], timestamps: List[float]):
        """Detect inhale, exhale, and pause phases."""
        
        if len(amplitudes) < 3:
            return
            
        # Simple phase detection based on amplitude changes
        current_amplitude = amplitudes[-1]
        previous_amplitude = amplitudes[-2] if len(amplitudes) > 1 else current_amplitude
        
        amplitude_change = current_amplitude - previous_amplitude
        
        # Phase detection thresholds
        inhale_threshold = self.movement_threshold
        exhale_threshold = -self.movement_threshold
        
        new_phase = self.current_phase
        
        if amplitude_change > inhale_threshold:
            new_phase = "inhale"
        elif amplitude_change < exhale_threshold:
            new_phase = "exhale"
        elif abs(amplitude_change) < self.movement_threshold / 2:
            new_phase = "pause"
            
        # Update phase if changed
        if new_phase != self.current_phase:
            phase_duration = time.time() - self.phase_start_time
            
            # Record breath cycle if complete
            if self.current_phase in ["inhale", "exhale"] and phase_duration > 0.5:
                self._record_breath_cycle(self.current_phase, phase_duration)
                
            self.current_phase = new_phase
            self.phase_start_time = time.time()
            
    def _record_breath_cycle(self, phase: str, duration: float):
        """Record a completed breath cycle phase."""
        
        # For now, create a simple breath cycle record
        # In full implementation, would track complete inhale-exhale cycles
        breath_cycle = BreathCycle(
            inhale_duration=duration if phase == "inhale" else 0,
            exhale_duration=duration if phase == "exhale" else 0,
            pause_duration=duration if phase == "pause" else 0,
            cycle_quality=0.8,  # Placeholder
            timestamp=time.time(),
            pattern_type=BreathingPattern.NATURAL
        )
        
        self.breath_cycles.append(breath_cycle)
        
        # Keep only last 20 cycles
        if len(self.breath_cycles) > 20:
            self.breath_cycles = self.breath_cycles[-20:]
            
    def _calculate_breathing_quality(self, amplitudes: List[float]) -> float:
        """Calculate breathing quality score."""
        
        if len(amplitudes) < 5:
            return 0.5
            
        # Quality factors
        consistency = 1.0 - (np.std(amplitudes) / (np.mean(amplitudes) + 0.001))
        consistency = max(0, min(1, consistency))
        
        # Amplitude adequacy (not too shallow, not too deep)
        mean_amplitude = np.mean(amplitudes)
        amplitude_score = 1.0 - abs(mean_amplitude - 0.01) / 0.01  # Ideal around 0.01
        amplitude_score = max(0, min(1, amplitude_score))
        
        # Overall quality
        quality_score = (consistency * 0.6 + amplitude_score * 0.4)
        
        return quality_score
        
    def _calculate_breathing_metrics(self) -> Dict:
        """Calculate breathing rate and other metrics."""
        
        metrics = {}
        
        if len(self.breath_cycles) >= 2:
            # Calculate breathing rate (breaths per minute)
            recent_cycles = self.breath_cycles[-10:]  # Last 10 cycles
            if recent_cycles:
                total_duration = recent_cycles[-1].timestamp - recent_cycles[0].timestamp
                if total_duration > 0:
                    breathing_rate = (len(recent_cycles) / total_duration) * 60
                    metrics['breathing_rate'] = breathing_rate
                    
                # Average cycle durations
                inhale_durations = [cycle.inhale_duration for cycle in recent_cycles if cycle.inhale_duration > 0]
                exhale_durations = [cycle.exhale_duration for cycle in recent_cycles if cycle.exhale_duration > 0]
                
                if inhale_durations:
                    metrics['average_inhale_duration'] = statistics.mean(inhale_durations)
                if exhale_durations:
                    metrics['average_exhale_duration'] = statistics.mean(exhale_durations)
                    
        return metrics
        
    def _assess_traditional_breathing(self, pattern_classification: Dict) -> Dict:
        """Assess breathing quality from traditional yoga perspective."""
        
        assessment = {
            'traditional_quality': 'developing',
            'pranayama_recommendations': [],
            'dharmic_insights': []
        }
        
        pattern_type = pattern_classification.get('pattern_type', BreathingPattern.NATURAL)
        
        if pattern_type == BreathingPattern.UJJAYI:
            assessment['traditional_quality'] = 'excellent'
            assessment['dharmic_insights'].append("Ujjayi breathing calms the mind and enhances concentration")
        elif pattern_type == BreathingPattern.NATURAL:
            assessment['traditional_quality'] = 'good'
            assessment['pranayama_recommendations'].append("Consider developing Ujjayi breath for deeper practice")
        else:
            assessment['pranayama_recommendations'].append("Focus on natural, deep breathing first")
            
        return assessment

class BreathPatternClassifier:
    """Classifies breathing patterns according to traditional Pranayama types."""
    
    def classify_pattern(self, movement_history: List[Dict]) -> Dict:
        """Classify the breathing pattern from movement history."""
        
        if len(movement_history) < 10:
            return {
                'pattern_type': BreathingPattern.UNKNOWN,
                'confidence': 0.0,
                'characteristics': []
            }
            
        # Extract features for classification
        features = self._extract_pattern_features(movement_history)
        
        # Simple rule-based classification (would use ML in full implementation)
        pattern_type = self._rule_based_classification(features)
        
        return {
            'pattern_type': pattern_type,
            'confidence': 0.7,  # Placeholder confidence
            'characteristics': features,
            'traditional_name': self._get_traditional_name(pattern_type)
        }
        
    def _extract_pattern_features(self, movement_history: List[Dict]) -> Dict:
        """Extract features that characterize breathing patterns."""
        
        amplitudes = [data['amplitude'] for data in movement_history]
        
        features = {
            'rhythm_regularity': self._calculate_rhythm_regularity(amplitudes),
            'depth_consistency': 1.0 - (np.std(amplitudes) / (np.mean(amplitudes) + 0.001)),
            'average_amplitude': np.mean(amplitudes),
            'breathing_smoothness': self._calculate_smoothness(amplitudes)
        }
        
        return features
        
    def _calculate_rhythm_regularity(self, amplitudes: List[float]) -> float:
        """Calculate how regular the breathing rhythm is."""
        
        if len(amplitudes) < 5:
            return 0.5
            
        # Simple regularity measure based on amplitude variation
        variation = np.std(amplitudes) / (np.mean(amplitudes) + 0.001)
        regularity = 1.0 - min(variation, 1.0)
        
        return regularity
        
    def _calculate_smoothness(self, amplitudes: List[float]) -> float:
        """Calculate breathing smoothness (lack of abrupt changes)."""
        
        if len(amplitudes) < 3:
            return 0.5
            
        # Calculate differences between consecutive amplitudes
        differences = [abs(amplitudes[i] - amplitudes[i-1]) for i in range(1, len(amplitudes))]
        smoothness = 1.0 - (np.mean(differences) / (np.mean(amplitudes) + 0.001))
        
        return max(0, min(1, smoothness))
        
    def _rule_based_classification(self, features: Dict) -> BreathingPattern:
        """Simple rule-based pattern classification."""
        
        # Ujjayi characteristics: deep, regular, smooth
        if (features['depth_consistency'] > 0.7 and 
            features['rhythm_regularity'] > 0.7 and
            features['breathing_smoothness'] > 0.8):
            return BreathingPattern.UJJAYI
            
        # Natural breathing: moderate regularity
        elif features['rhythm_regularity'] > 0.5:
            return BreathingPattern.NATURAL
            
        else:
            return BreathingPattern.UNKNOWN
            
    def _get_traditional_name(self, pattern_type: BreathingPattern) -> str:
        """Get traditional Sanskrit name for breathing pattern."""
        
        traditional_names = {
            BreathingPattern.UJJAYI: "उज्जायी (Ujjayi) - Victorious Breath",
            BreathingPattern.NADI_SHODHANA: "नाडी शोधन (Nadi Shodhana) - Alternate Nostril",
            BreathingPattern.BHASTRIKA: "भस्त्रिका (Bhastrika) - Bellows Breath",
            BreathingPattern.KAPALABHATI: "कपालभाति (Kapalabhati) - Skull Shining",
            BreathingPattern.BHRAMARI: "भ्रामरी (Bhramari) - Bee Breath",
            BreathingPattern.NATURAL: "Natural Breathing",
            BreathingPattern.UNKNOWN: "Pattern Recognition in Progress"
        }
        
        return traditional_names.get(pattern_type, "Unknown Pattern")