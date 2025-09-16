"""
🎯 DharmaMind Vision - Essential Utilities

Core utility functions for the revolutionary vision system:
- Image processing utilities
- Coordinate transformations
- Mathematical calculations for yoga analysis
- Traditional yoga geometry calculations
- Performance optimization helpers

These utilities support all 6 revolutionary subsystems.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import math

class VisionUtils:
    """Core vision processing utilities."""
    
    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Normalize landmarks to image coordinates.
        
        Args:
            landmarks: Raw landmarks from MediaPipe (0-1 normalized)
            image_shape: (height, width) of the image
            
        Returns:
            Landmarks in pixel coordinates
        """
        
        height, width = image_shape[:2]
        normalized = landmarks.copy()
        
        # Convert normalized coordinates to pixel coordinates
        normalized[:, 0] *= width   # X coordinates
        normalized[:, 1] *= height  # Y coordinates
        # Z coordinates remain as-is (depth)
        
        return normalized
    
    @staticmethod
    def calculate_angle_3d(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        Calculate angle between three 3D points.
        
        Args:
            point1: First point (apex of angle)
            point2: Second point (vertex)
            point3: Third point
            
        Returns:
            Angle in degrees
        """
        
        # Create vectors
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Calculate angle using dot product
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    @staticmethod
    def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(point1 - point2)
    
    @staticmethod
    def get_body_center(landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate the center of the body from pose landmarks.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            3D coordinates of body center
        """
        
        # Use key landmarks to find body center
        left_shoulder = 11
        right_shoulder = 12
        left_hip = 23
        right_hip = 24
        
        if len(landmarks) <= max(left_shoulder, right_shoulder, left_hip, right_hip):
            return np.array([0.5, 0.5, 0.0])  # Default center
            
        # Calculate center as average of shoulder and hip midpoints
        shoulder_center = (landmarks[left_shoulder] + landmarks[right_shoulder]) / 2
        hip_center = (landmarks[left_hip] + landmarks[right_hip]) / 2
        body_center = (shoulder_center + hip_center) / 2
        
        return body_center

class YogaGeometry:
    """Traditional yoga geometry calculations."""
    
    @staticmethod
    def calculate_spine_alignment(landmarks: np.ndarray) -> Dict:
        """
        Calculate spine alignment based on traditional yoga principles.
        
        Args:
            landmarks: Pose landmarks
            
        Returns:
            Spine alignment analysis
        """
        
        # Key spine reference points
        nose = 0
        left_shoulder = 11
        right_shoulder = 12
        left_hip = 23
        right_hip = 24
        
        if len(landmarks) <= max(nose, left_shoulder, right_shoulder, left_hip, right_hip):
            return {'error': 'Insufficient landmarks'}
            
        # Calculate spine reference line
        shoulder_center = (landmarks[left_shoulder] + landmarks[right_shoulder]) / 2
        hip_center = (landmarks[left_hip] + landmarks[right_hip]) / 2
        
        # Spine vector
        spine_vector = shoulder_center - hip_center
        
        # Ideal vertical vector (traditional straight spine)
        vertical_vector = np.array([0, 1, 0])
        
        # Calculate deviation from vertical
        deviation_angle = VisionUtils.calculate_angle_3d(
            shoulder_center, hip_center, hip_center + vertical_vector
        )
        
        # Traditional alignment assessment
        if deviation_angle <= 5:
            alignment_quality = "excellent"
            traditional_note = "Sthira - steady mountain-like posture"
        elif deviation_angle <= 15:
            alignment_quality = "good"
            traditional_note = "Approaching ideal alignment"
        elif deviation_angle <= 30:
            alignment_quality = "needs_attention"
            traditional_note = "Focus on lengthening the spine"
        else:
            alignment_quality = "poor"
            traditional_note = "Return to basic alignment principles"
            
        return {
            'deviation_angle': deviation_angle,
            'alignment_quality': alignment_quality,
            'traditional_assessment': traditional_note,
            'spine_vector': spine_vector.tolist(),
            'shoulder_center': shoulder_center.tolist(),
            'hip_center': hip_center.tolist()
        }
    
    @staticmethod
    def assess_balance_symmetry(landmarks: np.ndarray) -> Dict:
        """
        Assess left-right balance and symmetry.
        
        Args:
            landmarks: Pose landmarks
            
        Returns:
            Balance and symmetry analysis
        """
        
        # Key symmetry points
        left_shoulder = 11
        right_shoulder = 12
        left_elbow = 13
        right_elbow = 14
        left_wrist = 15
        right_wrist = 16
        left_hip = 23
        right_hip = 24
        left_knee = 25
        right_knee = 26
        left_ankle = 27
        right_ankle = 28
        
        symmetry_analysis = {}
        
        # Shoulder level balance
        if len(landmarks) > max(left_shoulder, right_shoulder):
            shoulder_diff = abs(landmarks[left_shoulder][1] - landmarks[right_shoulder][1])
            symmetry_analysis['shoulder_balance'] = 1.0 - min(shoulder_diff * 10, 1.0)
            
        # Hip level balance
        if len(landmarks) > max(left_hip, right_hip):
            hip_diff = abs(landmarks[left_hip][1] - landmarks[right_hip][1])
            symmetry_analysis['hip_balance'] = 1.0 - min(hip_diff * 10, 1.0)
            
        # Overall symmetry score
        balance_scores = [score for score in symmetry_analysis.values() if isinstance(score, float)]
        overall_symmetry = np.mean(balance_scores) if balance_scores else 0.5
        
        # Traditional assessment
        if overall_symmetry >= 0.9:
            balance_quality = "excellent"
            traditional_note = "Sama - perfect balance achieved"
        elif overall_symmetry >= 0.75:
            balance_quality = "good"
            traditional_note = "Approaching balanced state"
        else:
            balance_quality = "needs_improvement"
            traditional_note = "Focus on equal distribution of weight and attention"
            
        symmetry_analysis.update({
            'overall_symmetry': overall_symmetry,
            'balance_quality': balance_quality,
            'traditional_assessment': traditional_note
        })
        
        return symmetry_analysis

class PerformanceOptimizer:
    """Performance optimization utilities for real-time processing."""
    
    @staticmethod
    def resize_for_processing(frame: np.ndarray, max_dimension: int = 640) -> Tuple[np.ndarray, float]:
        """
        Resize frame for optimal processing while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            max_dimension: Maximum dimension (width or height)
            
        Returns:
            Resized frame and scale factor
        """
        
        height, width = frame.shape[:2]
        
        # Calculate scale factor
        scale = min(max_dimension / width, max_dimension / height)
        
        if scale >= 1.0:
            return frame, 1.0  # No resizing needed
            
        # Resize frame
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        return resized_frame, scale
    
    @staticmethod
    def optimize_landmarks_for_analysis(landmarks: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Optimize landmarks for analysis by filtering low-confidence points.
        
        Args:
            landmarks: Input landmarks
            confidence_threshold: Minimum confidence to keep landmarks
            
        Returns:
            Filtered landmarks
        """
        
        # For now, return all landmarks (would implement confidence filtering)
        return landmarks
    
    @staticmethod
    def batch_process_frames(frames: List[np.ndarray], processor_func, batch_size: int = 4) -> List:
        """
        Process multiple frames in batches for efficiency.
        
        Args:
            frames: List of frames to process
            processor_func: Function to process each frame
            batch_size: Number of frames to process together
            
        Returns:
            List of processing results
        """
        
        results = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = [processor_func(frame) for frame in batch]
            results.extend(batch_results)
            
        return results

class TraditionalWisdom:
    """Traditional yoga wisdom integration utilities."""
    
    @staticmethod
    def get_pose_philosophy(pose_name: str) -> Dict:
        """
        Get traditional philosophy associated with a pose.
        
        Args:
            pose_name: Name of the yoga pose
            
        Returns:
            Traditional wisdom and philosophy
        """
        
        pose_wisdom = {
            'mountain_pose': {
                'sanskrit': 'Tadasana (ताडासन)',
                'meaning': 'Palm tree pose - standing tall like a mountain',
                'philosophy': 'Represents stability, grounding, and reaching toward the divine',
                'traditional_benefits': ['Improves posture', 'Develops awareness', 'Cultivates steadiness'],
                'energetic_qualities': 'Grounding and centering'
            },
            'warrior_ii': {
                'sanskrit': 'Virabhadrasana II (वीरभद्रासन)',
                'meaning': 'Fierce warrior pose',
                'philosophy': 'Embodies the spiritual warrior fighting against ego and ignorance',
                'traditional_benefits': ['Builds strength', 'Develops focus', 'Cultivates courage'],
                'energetic_qualities': 'Strength and determination'
            },
            'tree_pose': {
                'sanskrit': 'Vrikshasana (वृक्षासन)',
                'meaning': 'Tree pose',
                'philosophy': 'Like a tree, finding balance between earth and sky',
                'traditional_benefits': ['Improves balance', 'Develops concentration', 'Connects with nature'],
                'energetic_qualities': 'Balance and growth'
            }
        }
        
        return pose_wisdom.get(pose_name, {
            'sanskrit': 'Traditional Asana',
            'meaning': 'Sacred yoga posture',
            'philosophy': 'Each pose is a meditation in movement',
            'traditional_benefits': ['Physical health', 'Mental clarity', 'Spiritual growth'],
            'energetic_qualities': 'Harmonizing body, mind, and spirit'
        })
    
    @staticmethod
    def get_chakra_association(body_region: str) -> Dict:
        """
        Get chakra associations for different body regions.
        
        Args:
            body_region: Body region (e.g., 'spine', 'heart', 'throat')
            
        Returns:
            Chakra information and traditional wisdom
        """
        
        chakra_associations = {
            'base_spine': {
                'chakra': 'Muladhara (मूलाधार)',
                'element': 'Earth',
                'qualities': 'Grounding, stability, survival',
                'traditional_practices': ['Standing poses', 'Grounding meditation']
            },
            'sacrum': {
                'chakra': 'Svadhisthana (स्वाधिष्ठान)',
                'element': 'Water',
                'qualities': 'Creativity, sexuality, emotion',
                'traditional_practices': ['Hip openers', 'Flow sequences']
            },
            'solar_plexus': {
                'chakra': 'Manipura (मणिपुर)',
                'element': 'Fire',
                'qualities': 'Personal power, confidence, transformation',
                'traditional_practices': ['Core strengthening', 'Breathing exercises']
            },
            'heart': {
                'chakra': 'Anahata (अनाहत)',
                'element': 'Air',
                'qualities': 'Love, compassion, connection',
                'traditional_practices': ['Heart opening poses', 'Loving-kindness meditation']
            },
            'throat': {
                'chakra': 'Vishuddha (विशुद्ध)',
                'element': 'Space',
                'qualities': 'Communication, truth, expression',
                'traditional_practices': ['Neck stretches', 'Chanting']
            },
            'third_eye': {
                'chakra': 'Ajna (आज्ञा)',
                'element': 'Light',
                'qualities': 'Intuition, wisdom, insight',
                'traditional_practices': ['Forward folds', 'Meditation']
            },
            'crown': {
                'chakra': 'Sahasrara (सहस्रार)',
                'element': 'Thought',
                'qualities': 'Spiritual connection, enlightenment',
                'traditional_practices': ['Inversions', 'Silent meditation']
            }
        }
        
        return chakra_associations.get(body_region, {
            'chakra': 'Energy center',
            'element': 'Universal energy',
            'qualities': 'Balance and harmony',
            'traditional_practices': ['Mindful movement', 'Breath awareness']
        })