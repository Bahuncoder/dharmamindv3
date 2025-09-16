"""
🕉️ AsanaClassifier - Traditional Hatha Yoga Pose Classification

Machine Learning classifier for identifying traditional yoga asanas based on 
the 15 classical poses described in the Hatha Yoga Pradipika.

Combines traditional wisdom with modern ML techniques for accurate pose recognition.
"""

import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

class AsanaDifficulty(Enum):
    """Traditional difficulty levels based on Hatha Yoga Pradipika."""
    BEGINNER = "Beginner"         # Easy poses for new practitioners
    INTERMEDIATE = "Intermediate" # Moderate challenge
    ADVANCED = "Advanced"         # Requires significant practice
    MASTER = "Master"             # For advanced yogis

class AsanaCategory(Enum):
    """Traditional categories of asanas."""
    SEATED = "Seated Meditation"     # Seated poses for pranayama
    STANDING = "Standing"            # Standing poses for strength
    TWISTING = "Spinal Twist"        # Twisting poses for flexibility
    BACKBEND = "Backbend"           # Heart opening poses
    FORWARD_FOLD = "Forward Fold"    # Introspective poses
    ARM_BALANCE = "Arm Balance"      # Strength and balance
    PRONE = "Prone"                  # Face-down poses
    SUPINE = "Supine"               # Face-up poses

@dataclass
class AsanaInfo:
    """Information about a traditional asana."""
    name: str
    sanskrit_name: str
    english_name: str
    difficulty: AsanaDifficulty
    category: AsanaCategory
    benefits: List[str]
    precautions: List[str]
    chakras_activated: List[str]
    description: str
    traditional_text_reference: str

@dataclass
class ClassificationResult:
    """Result of asana classification."""
    predicted_asana: str
    confidence: float
    top_predictions: List[Tuple[str, float]]
    features_used: int
    processing_time: float
    timestamp: datetime

class AsanaClassifier:
    """
    Traditional Hatha Yoga asana classifier using Random Forest.
    
    Classifies poses based on:
    - Joint angles and positions
    - Body geometry measurements
    - Chakra alignment patterns
    - Traditional pose characteristics
    
    Supports 15 classical asanas from Hatha Yoga Pradipika Chapter 2.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the asana classifier.
        
        Args:
            model_path: Path to pre-trained model file
        """
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Traditional 15 asanas from Hatha Yoga Pradipika
        self.asana_info = self._initialize_traditional_asanas()
        self.asana_labels = list(self.asana_info.keys())
        
        # Feature names for pose analysis
        self.feature_names = [
            'spine_angle', 'head_position', 'shoulder_width',
            'left_arm_angle', 'right_arm_angle', 'torso_angle',
            'left_hip_angle', 'right_hip_angle', 'left_knee_angle', 'right_knee_angle',
            'left_ankle_angle', 'right_ankle_angle', 'balance_point_x', 'balance_point_y',
            'pose_width', 'pose_height', 'symmetry_score',
            # Chakra alignment features
            'root_chakra_y', 'sacral_chakra_y', 'solar_plexus_y', 'heart_chakra_y',
            'throat_chakra_y', 'third_eye_y', 'crown_chakra_y',
            # Advanced geometric features
            'body_compactness', 'arm_extension', 'leg_extension', 'spinal_curve'
        ]
        
        self.is_trained = False
        self.model_path = model_path
        
        # Load pre-trained model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _initialize_traditional_asanas(self) -> Dict[str, AsanaInfo]:
        """Initialize the 15 traditional asanas with their properties."""
        asanas = {
            'Swastikasana': AsanaInfo(
                name='Swastikasana',
                sanskrit_name='स्वस्तिकासन',
                english_name='Auspicious Pose',
                difficulty=AsanaDifficulty.BEGINNER,
                category=AsanaCategory.SEATED,
                benefits=['Meditation', 'Pranayama', 'Mental calm'],
                precautions=['Knee issues'],
                chakras_activated=['Root', 'Sacral'],
                description='Comfortable cross-legged sitting position for meditation',
                traditional_text_reference='Hatha Yoga Pradipika 1.19'
            ),
            'Gomukhasana': AsanaInfo(
                name='Gomukhasana',
                sanskrit_name='गोमुखासन',
                english_name='Cow Face Pose',
                difficulty=AsanaDifficulty.INTERMEDIATE,
                category=AsanaCategory.SEATED,
                benefits=['Hip flexibility', 'Shoulder mobility', 'Spinal health'],
                precautions=['Hip injuries', 'Shoulder problems'],
                chakras_activated=['Heart', 'Throat'],
                description='Seated pose resembling the face of a cow',
                traditional_text_reference='Hatha Yoga Pradipika 1.20'
            ),
            'Virasana': AsanaInfo(
                name='Virasana',
                sanskrit_name='वीरासन',
                english_name='Hero Pose',
                difficulty=AsanaDifficulty.BEGINNER,
                category=AsanaCategory.SEATED,
                benefits=['Digestion', 'Posture', 'Meditation'],
                precautions=['Knee problems', 'Ankle issues'],
                chakras_activated=['Root', 'Solar Plexus'],
                description='Kneeling pose of the spiritual warrior',
                traditional_text_reference='Hatha Yoga Pradipika 1.21'
            ),
            'Kurmasana': AsanaInfo(
                name='Kurmasana',
                sanskrit_name='कूर्मासन',
                english_name='Tortoise Pose',
                difficulty=AsanaDifficulty.ADVANCED,
                category=AsanaCategory.FORWARD_FOLD,
                benefits=['Spinal flexibility', 'Introspection', 'Calming'],
                precautions=['Back injuries', 'Hip problems'],
                chakras_activated=['Root', 'Sacral', 'Solar Plexus'],
                description='Forward fold resembling a tortoise withdrawing',
                traditional_text_reference='Hatha Yoga Pradipika 1.22'
            ),
            'Kukkutasana': AsanaInfo(
                name='Kukkutasana',
                sanskrit_name='कुक्कुटासन',
                english_name='Cockerel Pose',
                difficulty=AsanaDifficulty.ADVANCED,
                category=AsanaCategory.ARM_BALANCE,
                benefits=['Arm strength', 'Core power', 'Balance'],
                precautions=['Wrist injuries', 'Shoulder problems'],
                chakras_activated=['Solar Plexus', 'Heart'],
                description='Arm balance in lotus position like a rooster',
                traditional_text_reference='Hatha Yoga Pradipika 1.23'
            ),
            'Uttana_Kurmasana': AsanaInfo(
                name='Uttana Kurmasana',
                sanskrit_name='उत्तान कूर्मासन',
                english_name='Stretched Tortoise Pose',
                difficulty=AsanaDifficulty.MASTER,
                category=AsanaCategory.FORWARD_FOLD,
                benefits=['Deep flexibility', 'Spiritual withdrawal', 'Advanced practice'],
                precautions=['Extreme flexibility required', 'Advanced practitioners only'],
                chakras_activated=['All chakras'],
                description='Advanced variation of tortoise pose',
                traditional_text_reference='Hatha Yoga Pradipika 1.24'
            ),
            'Dhanurasana': AsanaInfo(
                name='Dhanurasana',
                sanskrit_name='धनुरासन',
                english_name='Bow Pose',
                difficulty=AsanaDifficulty.ADVANCED,
                category=AsanaCategory.BACKBEND,
                benefits=['Spinal strength', 'Heart opening', 'Digestive fire'],
                precautions=['Back injuries', 'Neck problems'],
                chakras_activated=['Heart', 'Solar Plexus', 'Throat'],
                description='Backbend resembling an archer\'s bow',
                traditional_text_reference='Hatha Yoga Pradipika 1.25'
            ),
            'Matsyendrasana': AsanaInfo(
                name='Matsyendrasana',
                sanskrit_name='मत्स्येन्द्रासन',
                english_name='Lord of the Fishes Pose',
                difficulty=AsanaDifficulty.ADVANCED,
                category=AsanaCategory.TWISTING,
                benefits=['Spinal mobility', 'Organ massage', 'Energy activation'],
                precautions=['Spinal injuries', 'Recent surgery'],
                chakras_activated=['Solar Plexus', 'Heart', 'Throat'],
                description='Seated spinal twist named after sage Matsyendra',
                traditional_text_reference='Hatha Yoga Pradipika 1.26-27'
            ),
            'Paschimottanasana': AsanaInfo(
                name='Paschimottanasana',
                sanskrit_name='पश्चिमोत्तानासन',
                english_name='Seated Forward Bend',
                difficulty=AsanaDifficulty.INTERMEDIATE,
                category=AsanaCategory.FORWARD_FOLD,
                benefits=['Hamstring flexibility', 'Introspection', 'Calming'],
                precautions=['Lower back issues', 'Hamstring injuries'],
                chakras_activated=['Root', 'Sacral', 'Solar Plexus'],
                description='Seated forward fold of the west (back) side',
                traditional_text_reference='Hatha Yoga Pradipika 1.28'
            ),
            'Mayurasana': AsanaInfo(
                name='Mayurasana',
                sanskrit_name='मयूरासन',
                english_name='Peacock Pose',
                difficulty=AsanaDifficulty.MASTER,
                category=AsanaCategory.ARM_BALANCE,
                benefits=['Digestive fire', 'Arm strength', 'Detoxification'],
                precautions=['Wrist problems', 'High blood pressure'],
                chakras_activated=['Solar Plexus', 'Heart'],
                description='Arm balance resembling a peacock',
                traditional_text_reference='Hatha Yoga Pradipika 1.29-30'
            ),
            'Shavasana': AsanaInfo(
                name='Shavasana',
                sanskrit_name='शवासन',
                english_name='Corpse Pose',
                difficulty=AsanaDifficulty.BEGINNER,
                category=AsanaCategory.SUPINE,
                benefits=['Deep relaxation', 'Stress relief', 'Integration'],
                precautions=['None'],
                chakras_activated=['All chakras'],
                description='Complete relaxation pose like a corpse',
                traditional_text_reference='Hatha Yoga Pradipika 1.32'
            ),
            'Siddhasana': AsanaInfo(
                name='Siddhasana',
                sanskrit_name='सिद्धासन',
                english_name='Accomplished Pose',
                difficulty=AsanaDifficulty.INTERMEDIATE,
                category=AsanaCategory.SEATED,
                benefits=['Meditation', 'Energy conservation', 'Spiritual progress'],
                precautions=['Knee problems', 'Hip tightness'],
                chakras_activated=['Root', 'Sacral', 'Solar Plexus'],
                description='The accomplished pose for advanced meditation',
                traditional_text_reference='Hatha Yoga Pradipika 1.35-44'
            ),
            'Padmasana': AsanaInfo(
                name='Padmasana',
                sanskrit_name='पद्मासन',
                english_name='Lotus Pose',
                difficulty=AsanaDifficulty.ADVANCED,
                category=AsanaCategory.SEATED,
                benefits=['Deep meditation', 'Energy conservation', 'Spiritual awakening'],
                precautions=['Knee injuries', 'Hip problems'],
                chakras_activated=['All chakras'],
                description='The sacred lotus pose for highest meditation',
                traditional_text_reference='Hatha Yoga Pradipika 1.44-49'
            ),
            'Simhasana': AsanaInfo(
                name='Simhasana',
                sanskrit_name='सिंहासन',
                english_name='Lion Pose',
                difficulty=AsanaDifficulty.BEGINNER,
                category=AsanaCategory.SEATED,
                benefits=['Throat chakra activation', 'Facial tension release', 'Confidence'],
                precautions=['Jaw problems'],
                chakras_activated=['Throat', 'Heart'],
                description='Fierce pose of the lion with roaring breath',
                traditional_text_reference='Hatha Yoga Pradipika 1.50-52'
            ),
            'Bhadrasana': AsanaInfo(
                name='Bhadrasana',
                sanskrit_name='भद्रासन',
                english_name='Gracious Pose',
                difficulty=AsanaDifficulty.BEGINNER,
                category=AsanaCategory.SEATED,
                benefits=['Hip flexibility', 'Meditation preparation', 'Grounding'],
                precautions=['Hip problems'],
                chakras_activated=['Root', 'Sacral'],
                description='Gracious butterfly-like seated pose',
                traditional_text_reference='Hatha Yoga Pradipika 1.53'
            )
        }
        return asanas
    
    def extract_features(self, pose_keypoints) -> np.ndarray:
        """
        Extract features from pose keypoints for classification.
        
        Args:
            pose_keypoints: PoseKeypoints object from pose detector
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        landmarks = pose_keypoints.landmarks
        
        try:
            # Basic geometric features
            # Spine angle
            spine_angle = self._calculate_spine_angle(landmarks)
            features.append(spine_angle)
            
            # Head position relative to body
            head_pos = self._calculate_head_position(landmarks)
            features.append(head_pos)
            
            # Shoulder width
            shoulder_width = self._calculate_shoulder_width(landmarks)
            features.append(shoulder_width)
            
            # Arm angles
            left_arm_angle = self._calculate_arm_angle(landmarks, 'left')
            right_arm_angle = self._calculate_arm_angle(landmarks, 'right')
            features.extend([left_arm_angle, right_arm_angle])
            
            # Torso angle
            torso_angle = self._calculate_torso_angle(landmarks)
            features.append(torso_angle)
            
            # Leg angles
            left_hip_angle = self._calculate_hip_angle(landmarks, 'left')
            right_hip_angle = self._calculate_hip_angle(landmarks, 'right')
            left_knee_angle = self._calculate_knee_angle(landmarks, 'left')
            right_knee_angle = self._calculate_knee_angle(landmarks, 'right')
            left_ankle_angle = self._calculate_ankle_angle(landmarks, 'left')
            right_ankle_angle = self._calculate_ankle_angle(landmarks, 'right')
            features.extend([left_hip_angle, right_hip_angle, left_knee_angle, 
                           right_knee_angle, left_ankle_angle, right_ankle_angle])
            
            # Balance and symmetry
            balance_x, balance_y = self._calculate_balance_point(landmarks)
            features.extend([balance_x, balance_y])
            
            # Pose dimensions
            pose_width, pose_height = self._calculate_pose_dimensions(landmarks)
            features.extend([pose_width, pose_height])
            
            # Symmetry score
            symmetry = self._calculate_symmetry_score(landmarks)
            features.append(symmetry)
            
            # Chakra positions (normalized y-coordinates)
            chakra_features = self._extract_chakra_features(pose_keypoints.chakra_points)
            features.extend(chakra_features)
            
            # Advanced geometric features
            compactness = self._calculate_body_compactness(landmarks)
            arm_extension = self._calculate_arm_extension(landmarks)
            leg_extension = self._calculate_leg_extension(landmarks)
            spinal_curve = self._calculate_spinal_curve(landmarks)
            features.extend([compactness, arm_extension, leg_extension, spinal_curve])
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default features if extraction fails
            features = [0.0] * len(self.feature_names)
        
        # Ensure we have the right number of features
        while len(features) < len(self.feature_names):
            features.append(0.0)
        
        return np.array(features[:len(self.feature_names)])
    
    def _calculate_spine_angle(self, landmarks) -> float:
        """Calculate the angle of the spine."""
        try:
            head = landmarks['head']
            shoulder = landmarks['left_shoulder']
            hip = landmarks['left_hip']
            
            # Calculate angle from vertical
            spine_vector = np.array([hip[0] - head[0], hip[1] - head[1]])
            vertical_vector = np.array([0, 1])
            
            cosine_angle = np.dot(spine_vector, vertical_vector) / (
                np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector)
            )
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 90.0  # Default upright
    
    def _calculate_head_position(self, landmarks) -> float:
        """Calculate head position relative to shoulders."""
        try:
            head = landmarks['head']
            left_shoulder = landmarks['left_shoulder']
            right_shoulder = landmarks['right_shoulder']
            
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            return head[0] - shoulder_center_x
        except:
            return 0.0
    
    def _calculate_shoulder_width(self, landmarks) -> float:
        """Calculate shoulder width."""
        try:
            left_shoulder = landmarks['left_shoulder']
            right_shoulder = landmarks['right_shoulder']
            return abs(right_shoulder[0] - left_shoulder[0])
        except:
            return 100.0  # Default width
    
    def _calculate_arm_angle(self, landmarks, side) -> float:
        """Calculate arm angle."""
        try:
            shoulder = landmarks[f'{side}_shoulder']
            elbow = landmarks[f'{side}_elbow']
            wrist = landmarks[f'{side}_wrist']
            
            # Calculate angle at elbow
            v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
            v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
            
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 180.0  # Default straight arm
    
    def _calculate_torso_angle(self, landmarks) -> float:
        """Calculate torso angle from vertical."""
        try:
            left_shoulder = landmarks['left_shoulder']
            right_shoulder = landmarks['right_shoulder']
            left_hip = landmarks['left_hip']
            right_hip = landmarks['right_hip']
            
            shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2,
                             (left_shoulder[1] + right_shoulder[1]) / 2]
            hip_center = [(left_hip[0] + right_hip[0]) / 2,
                         (left_hip[1] + right_hip[1]) / 2]
            
            torso_vector = np.array([hip_center[0] - shoulder_center[0],
                                   hip_center[1] - shoulder_center[1]])
            vertical_vector = np.array([0, 1])
            
            cosine_angle = np.dot(torso_vector, vertical_vector) / (
                np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector)
            )
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 0.0
    
    def _calculate_hip_angle(self, landmarks, side) -> float:
        """Calculate hip angle."""
        try:
            shoulder = landmarks[f'{side}_shoulder']
            hip = landmarks[f'{side}_hip']
            knee = landmarks[f'{side}_knee']
            
            v1 = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])
            v2 = np.array([knee[0] - hip[0], knee[1] - hip[1]])
            
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 180.0
    
    def _calculate_knee_angle(self, landmarks, side) -> float:
        """Calculate knee angle."""
        try:
            hip = landmarks[f'{side}_hip']
            knee = landmarks[f'{side}_knee']
            ankle = landmarks[f'{side}_ankle']
            
            v1 = np.array([hip[0] - knee[0], hip[1] - knee[1]])
            v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
            
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 180.0
    
    def _calculate_ankle_angle(self, landmarks, side) -> float:
        """Calculate ankle flexion angle."""
        try:
            knee = landmarks[f'{side}_knee']
            ankle = landmarks[f'{side}_ankle']
            
            # Simplified: angle from vertical
            leg_vector = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
            vertical_vector = np.array([0, 1])
            
            cosine_angle = np.dot(leg_vector, vertical_vector) / (
                np.linalg.norm(leg_vector) * np.linalg.norm(vertical_vector)
            )
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 90.0
    
    def _calculate_balance_point(self, landmarks) -> Tuple[float, float]:
        """Calculate center of balance."""
        try:
            points = []
            for landmark in landmarks.values():
                points.append([landmark[0], landmark[1]])
            
            points = np.array(points)
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            return center_x, center_y
        except:
            return 0.0, 0.0
    
    def _calculate_pose_dimensions(self, landmarks) -> Tuple[float, float]:
        """Calculate bounding box dimensions of pose."""
        try:
            points = []
            for landmark in landmarks.values():
                points.append([landmark[0], landmark[1]])
            
            points = np.array(points)
            width = np.max(points[:, 0]) - np.min(points[:, 0])
            height = np.max(points[:, 1]) - np.min(points[:, 1])
            return width, height
        except:
            return 100.0, 200.0
    
    def _calculate_symmetry_score(self, landmarks) -> float:
        """Calculate left-right symmetry score."""
        try:
            # Compare left and right limb positions
            left_arm_y = landmarks['left_wrist'][1]
            right_arm_y = landmarks['right_wrist'][1]
            arm_symmetry = 1.0 - abs(left_arm_y - right_arm_y) / 100.0
            
            left_leg_y = landmarks['left_ankle'][1]
            right_leg_y = landmarks['right_ankle'][1]
            leg_symmetry = 1.0 - abs(left_leg_y - right_leg_y) / 100.0
            
            return (arm_symmetry + leg_symmetry) / 2.0
        except:
            return 0.5
    
    def _extract_chakra_features(self, chakra_points) -> List[float]:
        """Extract normalized chakra positions."""
        features = []
        
        # Order chakras from root to crown
        chakra_order = ['Root Chakra', 'Sacral Chakra', 'Solar Plexus', 'Heart Chakra',
                       'Throat Chakra', 'Third Eye', 'Crown Chakra']
        
        for chakra_name in chakra_order:
            found = False
            for chakra, (x, y) in chakra_points.items():
                if chakra.value == chakra_name:
                    features.append(y)  # Normalized y-coordinate
                    found = True
                    break
            if not found:
                features.append(0.0)
        
        return features
    
    def _calculate_body_compactness(self, landmarks) -> float:
        """Calculate how compact/spread out the pose is."""
        try:
            points = []
            for landmark in landmarks.values():
                points.append([landmark[0], landmark[1]])
            
            points = np.array(points)
            center = np.mean(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            return np.mean(distances)
        except:
            return 50.0
    
    def _calculate_arm_extension(self, landmarks) -> float:
        """Calculate average arm extension."""
        try:
            left_arm_length = np.linalg.norm(
                np.array(landmarks['left_wrist'][:2]) - np.array(landmarks['left_shoulder'][:2])
            )
            right_arm_length = np.linalg.norm(
                np.array(landmarks['right_wrist'][:2]) - np.array(landmarks['right_shoulder'][:2])
            )
            return (left_arm_length + right_arm_length) / 2.0
        except:
            return 50.0
    
    def _calculate_leg_extension(self, landmarks) -> float:
        """Calculate average leg extension."""
        try:
            left_leg_length = np.linalg.norm(
                np.array(landmarks['left_ankle'][:2]) - np.array(landmarks['left_hip'][:2])
            )
            right_leg_length = np.linalg.norm(
                np.array(landmarks['right_ankle'][:2]) - np.array(landmarks['right_hip'][:2])
            )
            return (left_leg_length + right_leg_length) / 2.0
        except:
            return 100.0
    
    def _calculate_spinal_curve(self, landmarks) -> float:
        """Calculate spinal curvature."""
        try:
            head = landmarks['head']
            shoulder = landmarks['left_shoulder']
            hip = landmarks['left_hip']
            
            # Calculate deviation from straight line
            spine_length = np.linalg.norm(np.array(hip[:2]) - np.array(head[:2]))
            shoulder_deviation = np.linalg.norm(
                np.array(shoulder[:2]) - 
                (np.array(head[:2]) + np.array(hip[:2])) / 2
            )
            
            return shoulder_deviation / spine_length if spine_length > 0 else 0.0
        except:
            return 0.1
    
    def generate_training_data(self, num_samples_per_asana: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for traditional asanas.
        
        Args:
            num_samples_per_asana: Number of training samples per asana
            
        Returns:
            Tuple of (features, labels)
        """
        X = []
        y = []
        
        print("🕉️ Generating traditional asana training data...")
        
        for asana_name in self.asana_labels:
            asana_info = self.asana_info[asana_name]
            
            for _ in range(num_samples_per_asana):
                # Generate synthetic features based on asana characteristics
                features = self._generate_asana_features(asana_name, asana_info)
                X.append(features)
                y.append(asana_name)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Generated {len(X)} training samples for {len(self.asana_labels)} asanas")
        return X, y
    
    def _generate_asana_features(self, asana_name: str, asana_info: AsanaInfo) -> List[float]:
        """Generate synthetic features for specific asana."""
        features = []
        
        # Base features with some randomness
        np.random.seed(hash(asana_name) % 2**32)
        
        # Asana-specific feature generation
        if asana_info.category == AsanaCategory.SEATED:
            # Seated poses have specific characteristics
            spine_angle = np.random.normal(5, 10)  # Mostly upright
            torso_angle = np.random.normal(0, 5)
            knee_angles = [np.random.normal(90, 20), np.random.normal(90, 20)]  # Bent knees
        elif asana_info.category == AsanaCategory.STANDING:
            spine_angle = np.random.normal(0, 5)  # Very upright
            torso_angle = np.random.normal(0, 3)
            knee_angles = [np.random.normal(170, 10), np.random.normal(170, 10)]  # Straight legs
        elif asana_info.category == AsanaCategory.ARM_BALANCE:
            spine_angle = np.random.normal(45, 15)  # Forward lean
            torso_angle = np.random.normal(30, 10)
            knee_angles = [np.random.normal(120, 30), np.random.normal(120, 30)]
        else:
            # Default values with variation
            spine_angle = np.random.normal(15, 20)
            torso_angle = np.random.normal(10, 15)
            knee_angles = [np.random.normal(120, 40), np.random.normal(120, 40)]
        
        # Generate all required features
        features = [
            spine_angle,  # spine_angle
            np.random.normal(0, 10),  # head_position
            np.random.normal(100, 20),  # shoulder_width
            np.random.normal(140, 30),  # left_arm_angle
            np.random.normal(140, 30),  # right_arm_angle
            torso_angle,  # torso_angle
            np.random.normal(120, 20),  # left_hip_angle
            np.random.normal(120, 20),  # right_hip_angle
            knee_angles[0],  # left_knee_angle
            knee_angles[1],  # right_knee_angle
            np.random.normal(90, 15),  # left_ankle_angle
            np.random.normal(90, 15),  # right_ankle_angle
            np.random.normal(0, 20),  # balance_point_x
            np.random.normal(0, 20),  # balance_point_y
            np.random.normal(100, 30),  # pose_width
            np.random.normal(200, 50),  # pose_height
            np.random.normal(0.7, 0.2),  # symmetry_score
            # Chakra positions (normalized y-coordinates)
            np.random.normal(0.8, 0.1),  # root_chakra_y
            np.random.normal(0.7, 0.1),  # sacral_chakra_y
            np.random.normal(0.6, 0.1),  # solar_plexus_y
            np.random.normal(0.4, 0.1),  # heart_chakra_y
            np.random.normal(0.3, 0.1),  # throat_chakra_y
            np.random.normal(0.2, 0.1),  # third_eye_y
            np.random.normal(0.1, 0.1),  # crown_chakra_y
            # Advanced geometric features
            np.random.normal(50, 15),  # body_compactness
            np.random.normal(60, 20),  # arm_extension
            np.random.normal(120, 30),  # leg_extension
            np.random.normal(0.1, 0.05),  # spinal_curve
        ]
        
        return features
    
    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """
        Train the asana classifier.
        
        Args:
            X: Feature matrix (optional, will generate if not provided)
            y: Label array (optional, will generate if not provided)
        """
        if X is None or y is None:
            print("Generating training data...")
            X, y = self.generate_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("🧘‍♀️ Training traditional asana classifier...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed! Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
    
    def predict(self, pose_keypoints, top_k: int = 3) -> ClassificationResult:
        """
        Predict asana from pose keypoints.
        
        Args:
            pose_keypoints: PoseKeypoints from pose detector
            top_k: Number of top predictions to return
            
        Returns:
            ClassificationResult with prediction and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        start_time = datetime.now()
        
        # Extract features
        features = self.extract_features(pose_keypoints)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_predictions = [
            (self.asana_labels[i], probabilities[i]) 
            for i in top_indices
        ]
        
        # Best prediction
        predicted_asana = top_predictions[0][0]
        confidence = top_predictions[0][1]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ClassificationResult(
            predicted_asana=predicted_asana,
            confidence=confidence,
            top_predictions=top_predictions,
            features_used=len(features),
            processing_time=processing_time,
            timestamp=datetime.now()
        )
    
    def get_asana_info(self, asana_name: str) -> Optional[AsanaInfo]:
        """Get detailed information about an asana."""
        return self.asana_info.get(asana_name)
    
    def save_model(self, path: str):
        """Save trained model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'asana_labels': self.asana_labels,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from file."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.asana_labels = model_data['asana_labels']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {path}")
    
    def get_supported_asanas(self) -> List[str]:
        """Get list of supported asana names."""
        return self.asana_labels.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        return {
            'is_trained': self.is_trained,
            'num_asanas': len(self.asana_labels),
            'asanas': self.asana_labels,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_type': 'RandomForestClassifier',
            'model_params': self.model.get_params() if self.is_trained else None
        }

# Usage example
if __name__ == "__main__":
    print("🕉️ Initializing Traditional Asana Classifier...")
    
    # Initialize classifier
    classifier = AsanaClassifier()
    
    # Train with synthetic data
    classifier.train()
    
    # Show model info
    info = classifier.get_model_info()
    print(f"\nModel Info:")
    print(f"Trained: {info['is_trained']}")
    print(f"Asanas: {info['num_asanas']}")
    print(f"Features: {info['num_features']}")
    
    # Show supported asanas
    print(f"\nSupported Traditional Asanas:")
    for i, asana in enumerate(classifier.get_supported_asanas(), 1):
        asana_info = classifier.get_asana_info(asana)
        print(f"  {i:2}. {asana} ({asana_info.sanskrit_name}) - {asana_info.difficulty.value}")
    
    print("\n🙏 Traditional wisdom meets modern technology!")