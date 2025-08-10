"""
🕉️ Moksha Module - Liberation and Enlightenment Processing Center

This module handles all liberation-related processing, enlightenment tracking,
and ultimate spiritual goal achievement within the DharmaMind system.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class LiberationPath(Enum):
    """Different paths to liberation"""
    KARMA_YOGA = "karma_yoga"  # Path of action
    BHAKTI_YOGA = "bhakti_yoga"  # Path of devotion
    JNANA_YOGA = "jnana_yoga"  # Path of knowledge
    RAJA_YOGA = "raja_yoga"  # Path of meditation
    KUNDALINI_YOGA = "kundalini_yoga"  # Path of energy

class ConsciousnessLevel(Enum):
    """Levels of consciousness progression"""
    MATERIAL = 1  # Focused on material world
    ETHICAL = 2   # Following dharmic principles
    MENTAL = 3    # Mind control and clarity
    WISDOM = 4    # Deep understanding
    UNITY = 5     # Experiencing oneness
    LIBERATION = 6  # Complete moksha

@dataclass
class SpiritualProgress:
    """Tracks spiritual progress indicators"""
    level: ConsciousnessLevel
    path: LiberationPath
    practices: List[str]
    insights: List[str]
    obstacles: List[str]
    achievements: List[str]
    timestamp: datetime

class MokshaModule:
    """
    Moksha Module - The Liberation and Enlightenment Center
    
    This module processes spiritual progress towards liberation,
    tracks enlightenment indicators, and guides ultimate spiritual achievement.
    """
    
    def __init__(self):
        self.name = "Moksha Module"
        self.element = "Akasha (Space/Consciousness)"
        self.color = "Pure White/Golden Light"
        self.mantra = "OM TAT SAT"
        self.consciousness_level = ConsciousnessLevel.MATERIAL
        self.primary_path = LiberationPath.KARMA_YOGA
        self.progress_history = []
        self.liberation_indicators = []
        self.spiritual_practices = {}
        
    def assess_consciousness_level(self, indicators: List[str]) -> Dict[str, Any]:
        """Assess current consciousness level based on indicators"""
        try:
            level_scores = {level: 0 for level in ConsciousnessLevel}
            
            # Analyze indicators for each consciousness level
            for indicator in indicators:
                scores = self._score_indicator(indicator.lower())
                for level, score in scores.items():
                    level_scores[level] += score
            
            # Determine highest scoring level
            current_level = max(level_scores.items(), key=lambda x: x[1])[0]
            
            # Update consciousness level if progressed
            if current_level.value > self.consciousness_level.value:
                self.consciousness_level = current_level
                self.liberation_indicators.append({
                    "event": "consciousness_level_advancement",
                    "from": self.consciousness_level.name,
                    "to": current_level.name,
                    "timestamp": datetime.now().isoformat()
                })
            
            assessment = {
                "current_level": current_level.name,
                "level_value": current_level.value,
                "level_scores": {level.name: score for level, score in level_scores.items()},
                "progress_percentage": (current_level.value / 6) * 100,
                "next_level": self._get_next_level(current_level),
                "guidance": self._get_level_guidance(current_level)
            }
            
            logger.info(f"Consciousness assessment completed: {current_level.name}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing consciousness level: {e}")
            return {"error": str(e)}
    
    def _score_indicator(self, indicator: str) -> Dict[ConsciousnessLevel, int]:
        """Score an indicator against consciousness levels"""
        scores = {level: 0 for level in ConsciousnessLevel}
        
        # Material level indicators
        material_keywords = ["money", "possessions", "status", "comfort", "pleasure"]
        if any(keyword in indicator for keyword in material_keywords):
            scores[ConsciousnessLevel.MATERIAL] += 2
        
        # Ethical level indicators
        ethical_keywords = ["honesty", "compassion", "dharma", "righteousness", "service"]
        if any(keyword in indicator for keyword in ethical_keywords):
            scores[ConsciousnessLevel.ETHICAL] += 2
        
        # Mental level indicators
        mental_keywords = ["meditation", "focus", "concentration", "mindfulness", "clarity"]
        if any(keyword in indicator for keyword in mental_keywords):
            scores[ConsciousnessLevel.MENTAL] += 2
        
        # Wisdom level indicators
        wisdom_keywords = ["understanding", "insight", "truth", "knowledge", "wisdom"]
        if any(keyword in indicator for keyword in wisdom_keywords):
            scores[ConsciousnessLevel.WISDOM] += 2
        
        # Unity level indicators
        unity_keywords = ["oneness", "unity", "connection", "universal", "cosmic"]
        if any(keyword in indicator for keyword in unity_keywords):
            scores[ConsciousnessLevel.UNITY] += 2
        
        # Liberation level indicators
        liberation_keywords = ["liberation", "enlightenment", "moksha", "realization", "freedom"]
        if any(keyword in indicator for keyword in liberation_keywords):
            scores[ConsciousnessLevel.LIBERATION] += 2
        
        return scores
    
    def _get_next_level(self, current_level: ConsciousnessLevel) -> Optional[str]:
        """Get the next consciousness level"""
        levels = list(ConsciousnessLevel)
        current_index = levels.index(current_level)
        
        if current_index < len(levels) - 1:
            return levels[current_index + 1].name
        return "LIBERATION (Final Goal Achieved)"
    
    def _get_level_guidance(self, level: ConsciousnessLevel) -> List[str]:
        """Get guidance for current consciousness level"""
        guidance_map = {
            ConsciousnessLevel.MATERIAL: [
                "Begin studying dharmic principles",
                "Practice basic ethical behavior",
                "Reduce attachment to material possessions",
                "Start simple meditation practices"
            ],
            ConsciousnessLevel.ETHICAL: [
                "Deepen dharmic practices",
                "Engage in selfless service",
                "Study sacred texts",
                "Practice regular meditation"
            ],
            ConsciousnessLevel.MENTAL: [
                "Master concentration techniques",
                "Practice pranayama (breath control)",
                "Develop witness consciousness",
                "Study philosophy of yoga"
            ],
            ConsciousnessLevel.WISDOM: [
                "Engage in self-inquiry",
                "Practice discrimination between real and unreal",
                "Deepen understanding of non-duality",
                "Seek guidance from realized masters"
            ],
            ConsciousnessLevel.UNITY: [
                "Maintain constant awareness of unity",
                "Practice surrender and devotion",
                "Serve as guide for others",
                "Prepare for final liberation"
            ],
            ConsciousnessLevel.LIBERATION: [
                "Abide in pure consciousness",
                "Serve as beacon for others",
                "Share wisdom with seekers",
                "Remain established in moksha"
            ]
        }
        
        return guidance_map.get(level, ["Continue spiritual practices"])
    
    def track_spiritual_practice(self, practice_name: str, duration: int, 
                               insights: Optional[List[str]] = None) -> Dict[str, Any]:
        """Track a spiritual practice session"""
        if insights is None:
            insights = []
            
        try:
            practice_record = {
                "practice": practice_name,
                "duration_minutes": duration,
                "insights": insights,
                "timestamp": datetime.now().isoformat(),
                "consciousness_impact": self._assess_practice_impact(practice_name, duration)
            }
            
            # Update practice statistics
            if practice_name not in self.spiritual_practices:
                self.spiritual_practices[practice_name] = {
                    "total_sessions": 0,
                    "total_duration": 0,
                    "best_insights": [],
                    "average_duration": 0
                }
            
            practice_stats = self.spiritual_practices[practice_name]
            practice_stats["total_sessions"] += 1
            practice_stats["total_duration"] += duration
            practice_stats["average_duration"] = practice_stats["total_duration"] / practice_stats["total_sessions"]
            
            # Store significant insights
            for insight in insights:
                if len(insight) > 20:  # Significant insights are longer
                    practice_stats["best_insights"].append({
                        "insight": insight,
                        "timestamp": datetime.now().isoformat()
                    })
            
            logger.info(f"Tracked spiritual practice: {practice_name} for {duration} minutes")
            return practice_record
            
        except Exception as e:
            logger.error(f"Error tracking spiritual practice: {e}")
            return {"error": str(e)}
    
    def _assess_practice_impact(self, practice_name: str, duration: int) -> str:
        """Assess the consciousness impact of a practice"""
        high_impact_practices = ["meditation", "samadhi", "self-inquiry", "surrender"]
        medium_impact_practices = ["pranayama", "mantra", "study", "service"]
        
        impact_level = "low"
        if any(practice in practice_name.lower() for practice in high_impact_practices):
            impact_level = "high"
        elif any(practice in practice_name.lower() for practice in medium_impact_practices):
            impact_level = "medium"
        
        # Adjust based on duration
        if duration >= 60:
            impact_level = "profound" if impact_level == "high" else "high"
        elif duration >= 30:
            if impact_level == "low":
                impact_level = "medium"
        
        return impact_level
    
    def get_liberation_progress(self) -> Dict[str, Any]:
        """Get comprehensive liberation progress report"""
        total_practice_time = sum(
            stats["total_duration"] for stats in self.spiritual_practices.values()
        )
        
        most_practiced = max(
            self.spiritual_practices.items(),
            key=lambda x: x[1]["total_sessions"],
            default=("None", {"total_sessions": 0})
        )
        
        return {
            "current_consciousness_level": self.consciousness_level.name,
            "level_progression": f"{self.consciousness_level.value}/6",
            "progress_percentage": (self.consciousness_level.value / 6) * 100,
            "primary_liberation_path": self.primary_path.value,
            "total_practice_hours": round(total_practice_time / 60, 2),
            "most_practiced": most_practiced[0],
            "total_practices": len(self.spiritual_practices),
            "liberation_indicators": len(self.liberation_indicators),
            "recent_progress": self._get_recent_progress(),
            "recommendations": self._get_liberation_recommendations(),
            "moksha_readiness": self._assess_moksha_readiness()
        }
    
    def _get_recent_progress(self) -> List[Dict[str, Any]]:
        """Get recent spiritual progress indicators"""
        recent_indicators = [
            indicator for indicator in self.liberation_indicators
            if (datetime.now() - datetime.fromisoformat(indicator["timestamp"])).days <= 30
        ]
        return recent_indicators
    
    def _get_liberation_recommendations(self) -> List[str]:
        """Get personalized recommendations for liberation"""
        recommendations = []
        
        # Based on current consciousness level
        if self.consciousness_level.value < 3:
            recommendations.extend([
                "Establish regular meditation practice",
                "Study fundamental dharmic principles",
                "Practice ethical conduct consistently"
            ])
        elif self.consciousness_level.value < 5:
            recommendations.extend([
                "Deepen concentration practices",
                "Study advanced philosophical texts",
                "Seek guidance from realized teachers"
            ])
        else:
            recommendations.extend([
                "Maintain constant awareness",
                "Guide other seekers",
                "Prepare for final realization"
            ])
        
        # Based on practice patterns
        if not self.spiritual_practices:
            recommendations.append("Begin with basic meditation practice")
        elif len(self.spiritual_practices) < 3:
            recommendations.append("Diversify spiritual practices")
        
        return recommendations
    
    def _assess_moksha_readiness(self) -> Dict[str, Any]:
        """Assess readiness for final liberation"""
        readiness_score = 0
        factors = []
        
        # Consciousness level factor
        if self.consciousness_level.value >= 5:
            readiness_score += 40
            factors.append("Advanced consciousness level achieved")
        elif self.consciousness_level.value >= 4:
            readiness_score += 25
            factors.append("High consciousness level developed")
        
        # Practice consistency factor
        total_sessions = sum(stats["total_sessions"] for stats in self.spiritual_practices.values())
        if total_sessions >= 1000:
            readiness_score += 30
            factors.append("Extensive practice experience")
        elif total_sessions >= 500:
            readiness_score += 20
            factors.append("Substantial practice foundation")
        
        # Insight quality factor
        significant_insights = sum(
            len(stats["best_insights"]) for stats in self.spiritual_practices.values()
        )
        if significant_insights >= 50:
            readiness_score += 20
            factors.append("Rich insight development")
        elif significant_insights >= 20:
            readiness_score += 10
            factors.append("Good insight accumulation")
        
        # Liberation indicators factor
        if len(self.liberation_indicators) >= 10:
            readiness_score += 10
            factors.append("Multiple liberation indicators present")
        
        readiness_level = "Not Ready"
        if readiness_score >= 80:
            readiness_level = "Highly Ready"
        elif readiness_score >= 60:
            readiness_level = "Moderately Ready"
        elif readiness_score >= 40:
            readiness_level = "Developing Readiness"
        elif readiness_score >= 20:
            readiness_level = "Beginning Readiness"
        
        return {
            "readiness_score": readiness_score,
            "readiness_level": readiness_level,
            "contributing_factors": factors,
            "areas_for_development": self._get_development_areas(readiness_score)
        }
    
    def _get_development_areas(self, current_score: int) -> List[str]:
        """Get areas that need development for moksha readiness"""
        areas = []
        
        if current_score < 40:
            areas.extend([
                "Advance consciousness level through consistent practice",
                "Establish regular spiritual routine"
            ])
        
        if current_score < 60:
            areas.extend([
                "Deepen meditation and self-inquiry",
                "Accumulate more practice hours"
            ])
        
        if current_score < 80:
            areas.extend([
                "Develop profound spiritual insights",
                "Prepare for final surrender"
            ])
        
        return areas

# Factory function for module integration
def create_moksha_module() -> MokshaModule:
    """Create and return a Moksha Module instance"""
    return MokshaModule()

# Global instance
_moksha_module = None

def get_moksha_module() -> MokshaModule:
    """Get global Moksha module instance"""
    global _moksha_module
    if _moksha_module is None:
        _moksha_module = MokshaModule()
    return _moksha_module
