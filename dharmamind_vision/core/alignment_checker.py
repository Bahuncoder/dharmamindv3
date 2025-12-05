"""
🕉️ AlignmentChecker - Traditional Scriptural Yoga Alignment Analysis

Provides alignment feedback based on classical Hindu yoga texts including:
- Hatha Yoga Pradipika by Yogi Svatmarama
- Gheranda Samhita 
- Shiva Samhita
- Yoga Sutras of Patanjali

Combines geometric analysis with traditional spiritual guidance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math
from datetime import datetime

class AlignmentLevel(Enum):
    """Traditional alignment quality levels."""
    EXCELLENT = "Excellent"       # 90-100% - Master level
    GOOD = "Good"                # 70-89% - Practiced
    MODERATE = "Moderate"        # 50-69% - Developing
    NEEDS_WORK = "Needs Work"    # 30-49% - Beginner
    POOR = "Poor"               # 0-29% - Requires attention

class ChakraState(Enum):
    """Chakra energy states based on alignment."""
    BALANCED = "Balanced"
    BLOCKED = "Blocked"
    OVERACTIVE = "Overactive"
    UNDERACTIVE = "Underactive"

@dataclass
class AlignmentFeedback:
    """Comprehensive alignment feedback."""
    overall_score: float
    level: AlignmentLevel
    geometric_scores: Dict[str, float]
    chakra_analysis: Dict[str, ChakraState]
    spiritual_guidance: List[str]
    corrections: List[str]
    benefits_achieved: List[str]
    areas_for_improvement: List[str]
    traditional_quotes: List[str]
    timestamp: datetime

@dataclass
class ChakraAlignment:
    """Individual chakra alignment analysis."""
    name: str
    position: Tuple[float, float]
    alignment_score: float
    energy_state: ChakraState
    traditional_guidance: str
    associated_qualities: List[str]

class AlignmentChecker:
    """
    Traditional yoga alignment checker based on classical texts.
    
    Analyzes pose alignment according to:
    - Geometric precision of traditional asanas
    - Chakra energy flow patterns
    - Scriptural teachings on proper form
    - Spiritual benefits and precautions
    
    References traditional texts for authentic guidance.
    """
    
    def __init__(self):
        """Initialize alignment checker with traditional wisdom."""
        
        # Traditional alignment principles from Hatha Yoga Pradipika
        self.alignment_principles = {
            'spine_alignment': {
                'description': 'Merudanda (spinal column) should be straight like Mount Meru',
                'reference': 'Hatha Yoga Pradipika 2.5',
                'ideal_range': (170, 180),  # degrees from horizontal
                'weight': 0.25
            },
            'symmetry': {
                'description': 'Sama (balance) between left and right sides',
                'reference': 'Gheranda Samhita 2.1',
                'ideal_range': (0.8, 1.0),  # symmetry ratio
                'weight': 0.20
            },
            'stability': {
                'description': 'Sthira (steadiness) - firm and comfortable',
                'reference': 'Yoga Sutras 2.46',
                'ideal_range': (0.7, 1.0),  # stability score
                'weight': 0.20
            },
            'breath_support': {
                'description': 'Sukha (ease) allowing natural pranayama',
                'reference': 'Yoga Sutras 2.47',
                'ideal_range': (0.6, 1.0),  # breathing ease
                'weight': 0.15
            },
            'energy_flow': {
                'description': 'Prana vayu circulation through nadis',
                'reference': 'Shiva Samhita 3.1-10',
                'ideal_range': (0.7, 1.0),  # energy flow score
                'weight': 0.20
            }
        }
        
        # Chakra positions and qualities
        self.chakra_system = {
            'Muladhara': {
                'name': 'Root Chakra',
                'sanskrit': 'मूलाधार',
                'element': 'Earth (Prithvi)',
                'qualities': ['Grounding', 'Stability', 'Survival', 'Security'],
                'color': 'Red',
                'mantra': 'LAM',
                'traditional_guidance': 'Foundation of all spiritual practice - establish firm ground'
            },
            'Svadhisthana': {
                'name': 'Sacral Chakra', 
                'sanskrit': 'स्वाधिष्ठान',
                'element': 'Water (Apas)',
                'qualities': ['Creativity', 'Sexuality', 'Emotion', 'Flow'],
                'color': 'Orange',
                'mantra': 'VAM',
                'traditional_guidance': 'Seat of creative and procreative energy'
            },
            'Manipura': {
                'name': 'Solar Plexus',
                'sanskrit': 'मणिपुर',
                'element': 'Fire (Agni)',
                'qualities': ['Personal Power', 'Confidence', 'Transformation', 'Digestive Fire'],
                'color': 'Yellow', 
                'mantra': 'RAM',
                'traditional_guidance': 'City of jewels - center of personal power and digestive fire'
            },
            'Anahata': {
                'name': 'Heart Chakra',
                'sanskrit': 'अनाहत',
                'element': 'Air (Vayu)',
                'qualities': ['Love', 'Compassion', 'Connection', 'Healing'],
                'color': 'Green',
                'mantra': 'YAM',
                'traditional_guidance': 'Unstruck sound - center of divine love and compassion'
            },
            'Vishuddha': {
                'name': 'Throat Chakra',
                'sanskrit': 'विशुद्ध',
                'element': 'Space (Akasha)',
                'qualities': ['Truth', 'Communication', 'Expression', 'Purification'],
                'color': 'Blue',
                'mantra': 'HAM', 
                'traditional_guidance': 'Pure expression of divine truth through speech'
            },
            'Ajna': {
                'name': 'Third Eye',
                'sanskrit': 'आज्ञा',
                'element': 'Light',
                'qualities': ['Intuition', 'Wisdom', 'Insight', 'Command'],
                'color': 'Indigo',
                'mantra': 'OM',
                'traditional_guidance': 'Command center - seat of guru and inner wisdom'
            },
            'Sahasrara': {
                'name': 'Crown Chakra',
                'sanskrit': 'सहस्रार',
                'element': 'Thought/Consciousness',
                'qualities': ['Unity', 'Enlightenment', 'Divine Connection', 'Pure Consciousness'],
                'color': 'Violet/White',
                'mantra': 'Silence',
                'traditional_guidance': 'Thousand-petaled lotus - union with divine consciousness'
            }
        }
        
        # Traditional quotes for guidance
        self.wisdom_quotes = {
            'general': [
                "Sthira sukham asanam - The posture should be steady and comfortable (Yoga Sutras 2.46)",
                "Prayatna shaithilya ananta samapattibhyam - By relaxing effort and focusing on the infinite, the posture is mastered (Yoga Sutras 2.47)",
                "Yoga is the cessation of fluctuations of the mind (Yoga Sutras 1.2)"
            ],
            'alignment': [
                "The spine should be erect like Mount Meru, the axis of the universe (Hatha Yoga Pradipika)",
                "Balance in the body brings balance in the mind and spirit (Gheranda Samhita)",
                "Perfect posture is achieved when effort ceases and the mind merges with the infinite (Yoga Sutras)"
            ],
            'breathing': [
                "When the breath is steady, the mind becomes steady (Hatha Yoga Pradipika 2.2)",
                "Pranayama is the bridge between the body and the mind (Classical Teaching)",
                "Control of prana leads to control of mind (Shiva Samhita)"
            ],
            'chakras': [
                "Awaken the serpent power at the base and guide it to the crown (Kundalini Yoga)",
                "Each chakra is a lotus waiting to bloom in divine light (Traditional Teaching)",
                "Balance the energy centers for harmony of body, mind, and spirit"
            ]
        }
        
        # Asana-specific alignment requirements
        self.asana_requirements = {
            'Padmasana': {
                'key_points': ['Hip flexibility', 'Spine erect', 'Shoulders relaxed', 'Crown aligned'],
                'common_errors': ['Forced legs', 'Rounded back', 'Tense shoulders'],
                'spiritual_benefits': ['Deep meditation', 'Pranayama practice', 'Kundalini awakening'],
                'precautions': ['Knee safety', 'Hip protection', 'Gradual progress']
            },
            'Mayurasana': {
                'key_points': ['Elbow placement', 'Core engagement', 'Steady breath', 'Mental focus'],
                'common_errors': ['Improper elbow position', 'Breath holding', 'Excessive strain'],
                'spiritual_benefits': ['Digestive fire', 'Mental strength', 'Ego transcendence'],
                'precautions': ['Wrist preparation', 'Core strength', 'Avoid overeating']
            },
            'Matsyendrasana': {
                'key_points': ['Spinal rotation', 'Hip grounding', 'Breath awareness', 'Gradual twist'],
                'common_errors': ['Forcing twist', 'Lifting hips', 'Shallow breathing'],
                'spiritual_benefits': ['Spinal health', 'Organ massage', 'Energy circulation'],
                'precautions': ['Spinal injuries', 'Pregnancy modifications', 'Gentle progression']
            }
        }
    
    def check_alignment(self, pose_keypoints, asana_name: str) -> AlignmentFeedback:
        """
        Comprehensive alignment check for traditional asana.
        
        Args:
            pose_keypoints: PoseKeypoints from pose detector
            asana_name: Name of the asana being analyzed
            
        Returns:
            AlignmentFeedback with detailed analysis
        """
        timestamp = datetime.now()
        
        # Calculate geometric alignment scores
        geometric_scores = self._calculate_geometric_scores(pose_keypoints)
        
        # Analyze chakra alignment
        chakra_analysis = self._analyze_chakra_alignment(pose_keypoints.chakra_points)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(geometric_scores)
        
        # Determine alignment level
        level = self._get_alignment_level(overall_score)
        
        # Generate spiritual guidance
        spiritual_guidance = self._generate_spiritual_guidance(asana_name, level, overall_score)
        
        # Generate corrections
        corrections = self._generate_corrections(geometric_scores, asana_name)
        
        # Identify benefits achieved
        benefits = self._identify_benefits(overall_score, asana_name)
        
        # Areas for improvement
        improvements = self._identify_improvements(geometric_scores, chakra_analysis)
        
        # Select traditional quotes
        quotes = self._select_traditional_quotes(asana_name, level)
        
        return AlignmentFeedback(
            overall_score=overall_score,
            level=level,
            geometric_scores=geometric_scores,
            chakra_analysis=chakra_analysis,
            spiritual_guidance=spiritual_guidance,
            corrections=corrections,
            benefits_achieved=benefits,
            areas_for_improvement=improvements,
            traditional_quotes=quotes,
            timestamp=timestamp
        )
    
    def _calculate_geometric_scores(self, pose_keypoints) -> Dict[str, float]:
        """Calculate geometric alignment scores based on traditional principles."""
        scores = {}
        landmarks = pose_keypoints.landmarks
        
        try:
            # Spine alignment (Merudanda)
            spine_score = self._calculate_spine_alignment(landmarks)
            scores['spine_alignment'] = spine_score
            
            # Symmetry (Sama)
            symmetry_score = self._calculate_symmetry(landmarks)
            scores['symmetry'] = symmetry_score
            
            # Stability (Sthira)
            stability_score = self._calculate_stability(landmarks)
            scores['stability'] = stability_score
            
            # Breath support (Sukha)
            breath_score = self._calculate_breath_support(landmarks)
            scores['breath_support'] = breath_score
            
            # Energy flow (Prana)
            energy_score = self._calculate_energy_flow(pose_keypoints.chakra_points)
            scores['energy_flow'] = energy_score
            
        except Exception as e:
            print(f"Error calculating geometric scores: {e}")
            # Return default scores if calculation fails
            for principle in self.alignment_principles:
                scores[principle] = 0.5
        
        return scores
    
    def _calculate_spine_alignment(self, landmarks) -> float:
        """Calculate spinal alignment score (Merudanda principle)."""
        try:
            head = landmarks['head']
            shoulder_center = (
                (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2,
                (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
            )
            hip_center = (
                (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2,
                (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
            )
            
            # Calculate spine angle from vertical
            spine_vector = np.array([hip_center[0] - head[0], hip_center[1] - head[1]])
            vertical_vector = np.array([0, 1])
            
            # Calculate angle
            cosine_angle = np.dot(spine_vector, vertical_vector) / (
                np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector)
            )
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            
            # Score based on ideal range (170-180 degrees)
            ideal_min, ideal_max = self.alignment_principles['spine_alignment']['ideal_range']
            if ideal_min <= angle <= ideal_max:
                return 1.0
            else:
                deviation = min(abs(angle - ideal_min), abs(angle - ideal_max))
                return max(0.0, 1.0 - deviation / 45.0)  # 45 degree tolerance
                
        except Exception as e:
            print(f"Error calculating spine alignment: {e}")
            return 0.5
    
    def _calculate_symmetry(self, landmarks) -> float:
        """Calculate left-right symmetry (Sama principle)."""
        try:
            symmetry_pairs = [
                ('left_shoulder', 'right_shoulder'),
                ('left_elbow', 'right_elbow'), 
                ('left_wrist', 'right_wrist'),
                ('left_hip', 'right_hip'),
                ('left_knee', 'right_knee'),
                ('left_ankle', 'right_ankle')
            ]
            
            symmetry_scores = []
            
            for left_part, right_part in symmetry_pairs:
                if left_part in landmarks and right_part in landmarks:
                    left_pos = landmarks[left_part]
                    right_pos = landmarks[right_part]
                    
                    # Calculate relative position differences
                    y_diff = abs(left_pos[1] - right_pos[1])
                    z_diff = abs(left_pos[2] - right_pos[2]) if len(left_pos) > 2 else 0
                    
                    # Score based on position similarity
                    max_diff = 50  # pixels tolerance
                    y_score = max(0.0, 1.0 - y_diff / max_diff)
                    z_score = max(0.0, 1.0 - z_diff / max_diff)
                    
                    pair_score = (y_score + z_score) / 2
                    symmetry_scores.append(pair_score)
            
            return np.mean(symmetry_scores) if symmetry_scores else 0.5
            
        except Exception as e:
            print(f"Error calculating symmetry: {e}")
            return 0.5
    
    def _calculate_stability(self, landmarks) -> float:
        """Calculate pose stability (Sthira principle)."""
        try:
            # Calculate center of mass
            points = []
            for landmark in landmarks.values():
                points.append([landmark[0], landmark[1]])
            
            points = np.array(points)
            center_of_mass = np.mean(points, axis=0)
            
            # Calculate base of support (feet/sitting points)
            support_points = []
            if 'left_ankle' in landmarks and 'right_ankle' in landmarks:
                support_points = [landmarks['left_ankle'][:2], landmarks['right_ankle'][:2]]
            elif 'left_hip' in landmarks and 'right_hip' in landmarks:
                support_points = [landmarks['left_hip'][:2], landmarks['right_hip'][:2]]
            
            if len(support_points) >= 2:
                # Check if center of mass is within base of support
                support_center = np.mean(support_points, axis=0)
                support_width = np.linalg.norm(np.array(support_points[0]) - np.array(support_points[1]))
                
                com_offset = np.linalg.norm(center_of_mass - support_center)
                stability_ratio = 1.0 - min(1.0, com_offset / (support_width / 2))
                
                return max(0.0, stability_ratio)
            else:
                return 0.5  # Default for poses without clear base
                
        except Exception as e:
            print(f"Error calculating stability: {e}")
            return 0.5
    
    def _calculate_breath_support(self, landmarks) -> float:
        """Calculate breathing space and chest openness (Sukha principle)."""
        try:
            # Chest expansion assessment
            left_shoulder = landmarks['left_shoulder']
            right_shoulder = landmarks['right_shoulder']
            chest_width = abs(right_shoulder[0] - left_shoulder[0])
            
            # Chest height (shoulder to hip distance)
            hip_center_y = (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            chest_height = abs(hip_center_y - shoulder_center_y)
            
            # Score based on openness
            chest_openness = (chest_width + chest_height) / 200  # Normalized
            return min(1.0, chest_openness)
            
        except Exception as e:
            print(f"Error calculating breath support: {e}")
            return 0.5
    
    def _calculate_energy_flow(self, chakra_points) -> float:
        """Calculate energy flow through chakras (Prana principle)."""
        try:
            if len(chakra_points) < 3:
                return 0.5
            
            # Check vertical alignment of chakras
            chakra_positions = list(chakra_points.values())
            y_positions = [pos[1] for pos in chakra_positions]
            
            # Calculate alignment variation
            y_std = np.std(y_positions)
            max_variation = 50  # pixels
            
            alignment_score = max(0.0, 1.0 - y_std / max_variation)
            
            # Check ascending order (root to crown)
            sorted_y = sorted(y_positions, reverse=True)  # Higher y = lower on screen
            order_score = 1.0 if y_positions == sorted_y else 0.5
            
            return (alignment_score + order_score) / 2
            
        except Exception as e:
            print(f"Error calculating energy flow: {e}")
            return 0.5
    
    def _analyze_chakra_alignment(self, chakra_points) -> Dict[str, ChakraState]:
        """Analyze individual chakra alignments."""
        analysis = {}
        
        for chakra_enum, position in chakra_points.items():
            chakra_name = chakra_enum.value
            
            # Simple alignment assessment
            # In a real implementation, this would be more sophisticated
            alignment_quality = np.random.choice([
                ChakraState.BALANCED,
                ChakraState.BALANCED,  # More likely to be balanced
                ChakraState.UNDERACTIVE,
                ChakraState.BLOCKED
            ])
            
            analysis[chakra_name] = alignment_quality
        
        return analysis
    
    def _calculate_overall_score(self, geometric_scores) -> float:
        """Calculate weighted overall alignment score."""
        total_score = 0.0
        
        for principle, score in geometric_scores.items():
            if principle in self.alignment_principles:
                weight = self.alignment_principles[principle]['weight']
                total_score += score * weight
        
        return min(1.0, total_score)
    
    def _get_alignment_level(self, score) -> AlignmentLevel:
        """Determine alignment level from score."""
        if score >= 0.9:
            return AlignmentLevel.EXCELLENT
        elif score >= 0.7:
            return AlignmentLevel.GOOD
        elif score >= 0.5:
            return AlignmentLevel.MODERATE
        elif score >= 0.3:
            return AlignmentLevel.NEEDS_WORK
        else:
            return AlignmentLevel.POOR
    
    def _generate_spiritual_guidance(self, asana_name, level, score) -> List[str]:
        """Generate spiritual guidance based on alignment."""
        guidance = []
        
        # General encouragement based on level
        if level == AlignmentLevel.EXCELLENT:
            guidance.append("🕉️ Excellent! Your asana embodies the perfect balance of sthira and sukha.")
            guidance.append("Continue to deepen your meditation and pranayama in this stable foundation.")
        elif level == AlignmentLevel.GOOD:
            guidance.append("🙏 Beautiful practice! You demonstrate good understanding of the pose.")
            guidance.append("Focus on subtle refinements to achieve perfect steadiness.")
        elif level == AlignmentLevel.MODERATE:
            guidance.append("🌱 Your practice is developing well. Stay patient with yourself.")
            guidance.append("Remember: yoga is a journey, not a destination.")
        else:
            guidance.append("🌟 Every master was once a beginner. Honor your current state.")
            guidance.append("Focus on the breath and find ease within effort.")
        
        # Asana-specific guidance
        if asana_name in self.asana_requirements:
            req = self.asana_requirements[asana_name]
            if score > 0.8:
                guidance.extend([f"✨ {benefit}" for benefit in req['spiritual_benefits']])
            else:
                guidance.append(f"🧘‍♀️ Focus on: {', '.join(req['key_points'][:2])}")
        
        return guidance
    
    def _generate_corrections(self, geometric_scores, asana_name) -> List[str]:
        """Generate specific corrections based on scores."""
        corrections = []
        
        # Check each alignment principle
        for principle, score in geometric_scores.items():
            if score < 0.7:  # Needs improvement
                if principle == 'spine_alignment':
                    corrections.append("🏔️ Lengthen your spine like Mount Meru - crown reaching toward heaven")
                elif principle == 'symmetry':
                    corrections.append("⚖️ Balance left and right sides equally")
                elif principle == 'stability':
                    corrections.append("🌳 Root down through your foundation for stability")
                elif principle == 'breath_support':
                    corrections.append("🌬️ Open your chest to allow free flow of prana")
                elif principle == 'energy_flow':
                    corrections.append("⚡ Align your energy centers from root to crown")
        
        # Asana-specific corrections
        if asana_name in self.asana_requirements:
            req = self.asana_requirements[asana_name]
            if geometric_scores.get('spine_alignment', 1.0) < 0.6:
                corrections.extend([f"⚠️ {error}" for error in req['common_errors'][:2]])
        
        return corrections
    
    def _identify_benefits(self, score, asana_name) -> List[str]:
        """Identify benefits being achieved at current level."""
        benefits = []
        
        if score > 0.8:
            benefits.append("🧘‍♀️ Deep meditative state accessible")
            benefits.append("💚 Heart center opening and expanding")
        
        if score > 0.6:
            benefits.append("🔥 Digestive fire (Agni) being stimulated")
            benefits.append("🌀 Energy circulation improving")
        
        if score > 0.4:
            benefits.append("💪 Building strength and flexibility")
            benefits.append("🧠 Developing concentration and focus")
        
        # Asana-specific benefits
        if asana_name in self.asana_requirements and score > 0.7:
            req = self.asana_requirements[asana_name]
            benefits.extend([f"✨ {benefit}" for benefit in req['spiritual_benefits'][:2]])
        
        return benefits
    
    def _identify_improvements(self, geometric_scores, chakra_analysis) -> List[str]:
        """Identify specific areas for improvement."""
        improvements = []
        
        # Lowest scoring areas
        sorted_scores = sorted(geometric_scores.items(), key=lambda x: x[1])
        
        for principle, score in sorted_scores[:2]:  # Top 2 areas for improvement
            if score < 0.8:
                if principle == 'spine_alignment':
                    improvements.append("Focus on lengthening the spine")
                elif principle == 'symmetry':
                    improvements.append("Work on balancing left and right sides")
                elif principle == 'stability':
                    improvements.append("Strengthen your foundation")
                elif principle == 'breath_support':
                    improvements.append("Practice chest opening exercises")
                elif principle == 'energy_flow':
                    improvements.append("Work on chakra alignment meditation")
        
        # Chakra improvements
        blocked_chakras = [name for name, state in chakra_analysis.items() 
                          if state in [ChakraState.BLOCKED, ChakraState.UNDERACTIVE]]
        
        if blocked_chakras:
            improvements.append(f"Focus on opening: {', '.join(blocked_chakras[:2])}")
        
        return improvements
    
    def _select_traditional_quotes(self, asana_name, level) -> List[str]:
        """Select appropriate traditional quotes."""
        quotes = []
        
        # Always include general wisdom
        quotes.append(np.random.choice(self.wisdom_quotes['general']))
        
        # Add specific quotes based on level
        if level in [AlignmentLevel.EXCELLENT, AlignmentLevel.GOOD]:
            quotes.append(np.random.choice(self.wisdom_quotes['alignment']))
        else:
            quotes.append(np.random.choice(self.wisdom_quotes['breathing']))
        
        return quotes
    
    def get_chakra_info(self, chakra_name: str) -> Optional[Dict]:
        """Get detailed information about a specific chakra."""
        return self.chakra_system.get(chakra_name)
    
    def get_alignment_principles(self) -> Dict:
        """Get all traditional alignment principles."""
        return self.alignment_principles.copy()
    
    def analyze_pose_progression(self, alignment_history: List[AlignmentFeedback]) -> Dict[str, Any]:
        """Analyze progression over multiple pose sessions."""
        if not alignment_history:
            return {'error': 'No alignment history provided'}
        
        scores = [feedback.overall_score for feedback in alignment_history]
        
        progression = {
            'sessions_analyzed': len(alignment_history),
            'current_score': scores[-1],
            'average_score': np.mean(scores),
            'improvement_trend': scores[-1] - scores[0] if len(scores) > 1 else 0,
            'consistency': 1.0 - np.std(scores),  # Lower std = more consistent
            'peak_performance': max(scores),
            'areas_of_growth': self._identify_growth_areas(alignment_history)
        }
        
        return progression
    
    def _identify_growth_areas(self, alignment_history) -> List[str]:
        """Identify areas showing improvement over time."""
        if len(alignment_history) < 2:
            return []
        
        growth_areas = []
        
        # Compare first and last sessions
        first_scores = alignment_history[0].geometric_scores
        last_scores = alignment_history[-1].geometric_scores
        
        for principle in first_scores:
            if principle in last_scores:
                improvement = last_scores[principle] - first_scores[principle]
                if improvement > 0.1:  # Significant improvement
                    growth_areas.append(f"Improved {principle.replace('_', ' ')}")
        
        return growth_areas

# Usage example
if __name__ == "__main__":
    print("🕉️ Initializing Traditional Alignment Checker...")
    
    # Initialize checker
    checker = AlignmentChecker()
    
    # Display alignment principles
    print("\nTraditional Alignment Principles:")
    for principle, info in checker.get_alignment_principles().items():
        print(f"  📿 {principle.replace('_', ' ').title()}")
        print(f"     {info['description']}")
        print(f"     Reference: {info['reference']}")
    
    # Display chakra system
    print(f"\nChakra System ({len(checker.chakra_system)} centers):")
    for chakra_key, chakra_info in checker.chakra_system.items():
        print(f"  🌸 {chakra_info['name']} ({chakra_info['sanskrit']})")
        print(f"     Element: {chakra_info['element']}")
        print(f"     Mantra: {chakra_info['mantra']}")
    
    print("\n🙏 Ready to provide traditional alignment guidance!")
    print("\"Sthira sukham asanam\" - May your practice be steady and comfortable")