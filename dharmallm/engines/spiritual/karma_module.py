"""
🕉️ Karma Module - Action and Consequence Processing Center

This module handles all action-based processing, karma tracking,
and dharmic decision-making within the DharmaMind system.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions tracked by Karma Module"""
    DHARMIC = "dharmic"  # Righteous actions
    ADHARMIC = "adharmic"  # Unrighteous actions
    NEUTRAL = "neutral"  # Neutral actions
    SPIRITUAL = "spiritual"  # Spiritual practices
    SEVA = "seva"  # Service actions

class KarmaWeight(Enum):
    """Weight/Impact levels of karmic actions"""
    LIGHT = 1
    MODERATE = 3
    SIGNIFICANT = 5
    PROFOUND = 7
    LIFE_CHANGING = 10

@dataclass
class KarmicAction:
    """Represents a karmic action and its properties"""
    action_id: str
    description: str
    action_type: ActionType
    weight: KarmaWeight
    timestamp: datetime
    consequences: List[str]
    dharmic_score: float
    associated_intentions: List[str]

class KarmaModule:
    """
    Karma Module - The Action and Consequence Center
    
    This module processes all actions, tracks karmic patterns,
    and provides guidance on dharmic decision-making.
    """
    
    def __init__(self):
        self.name = "Karma Module"
        self.element = "Agni (Fire)"
        self.color = "Bright Orange"
        self.mantra = "OM KARMA YOGA NAMAHA"
        self.action_history = []
        self.karmic_balance = 0.0
        self.dharmic_patterns = {}
        
    def process_action(self, description: str, action_type: ActionType, 
                      intentions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process and record a karmic action"""
        try:
            if intentions is None:
                intentions = []
                
            # Assess the action
            weight = self._assess_action_weight(description, action_type)
            dharmic_score = self._calculate_dharmic_score(description, action_type, intentions)
            consequences = self._predict_consequences(description, action_type)
            
            # Create karmic action record
            action = KarmicAction(
                action_id=f"karma_{len(self.action_history)}",
                description=description,
                action_type=action_type,
                weight=weight,
                timestamp=datetime.now(),
                consequences=consequences,
                dharmic_score=dharmic_score,
                associated_intentions=intentions
            )
            
            # Update karmic balance
            self.karmic_balance += dharmic_score
            self.action_history.append(action)
            
            # Update patterns
            self._update_dharmic_patterns(action)
            
            logger.info(f"Processed action: {action.action_id} with dharmic score: {dharmic_score}")
            
            return {
                "action_id": action.action_id,
                "dharmic_score": dharmic_score,
                "karmic_weight": weight.value,
                "consequences": consequences,
                "current_balance": self.karmic_balance,
                "guidance": self._provide_guidance(action)
            }
            
        except Exception as e:
            logger.error(f"Error processing action: {e}")
            return {"error": str(e)}
    
    def _assess_action_weight(self, description: str, action_type: ActionType) -> KarmaWeight:
        """Assess the karmic weight of an action"""
        # Check for significant indicators
        high_impact_words = ["harm", "hurt", "destroy", "steal", "lie", "betray"]
        profound_words = ["save", "heal", "enlighten", "liberate", "transform"]
        spiritual_words = ["meditate", "pray", "worship", "surrender", "serve"]
        
        description_lower = description.lower()
        
        if any(word in description_lower for word in profound_words):
            return KarmaWeight.PROFOUND
        elif any(word in description_lower for word in high_impact_words):
            return KarmaWeight.SIGNIFICANT
        elif any(word in description_lower for word in spiritual_words):
            return KarmaWeight.MODERATE
        elif action_type == ActionType.SPIRITUAL:
            return KarmaWeight.MODERATE
        else:
            return KarmaWeight.LIGHT
    
    def _calculate_dharmic_score(self, description: str, action_type: ActionType, 
                               intentions: List[str]) -> float:
        """Calculate the dharmic score of an action"""
        base_scores = {
            ActionType.DHARMIC: 5.0,
            ActionType.SPIRITUAL: 4.0,
            ActionType.SEVA: 4.5,
            ActionType.NEUTRAL: 0.0,
            ActionType.ADHARMIC: -3.0
        }
        
        score = base_scores.get(action_type, 0.0)
        
        # Adjust based on intentions
        positive_intentions = ["compassion", "love", "service", "truth", "liberation"]
        negative_intentions = ["greed", "anger", "pride", "envy", "delusion"]
        
        for intention in intentions:
            if any(pos in intention.lower() for pos in positive_intentions):
                score += 1.0
            elif any(neg in intention.lower() for neg in negative_intentions):
                score -= 1.0
        
        # Adjust based on description content
        dharmic_words = ["help", "serve", "heal", "teach", "protect", "care"]
        adharmic_words = ["harm", "cheat", "steal", "destroy", "abuse"]
        
        description_lower = description.lower()
        for word in dharmic_words:
            if word in description_lower:
                score += 0.5
        
        for word in adharmic_words:
            if word in description_lower:
                score -= 1.0
        
        return round(score, 2)
    
    def _predict_consequences(self, description: str, action_type: ActionType) -> List[str]:
        """Predict potential consequences of an action"""
        consequences = []
        
        consequence_map = {
            ActionType.DHARMIC: [
                "Increased inner peace and contentment",
                "Positive karma accumulation",
                "Spiritual progress and growth"
            ],
            ActionType.SPIRITUAL: [
                "Enhanced spiritual awareness",
                "Deeper connection with the divine",
                "Purification of consciousness"
            ],
            ActionType.SEVA: [
                "Selfless service merit",
                "Community benefit and gratitude",
                "Reduction of ego and pride"
            ],
            ActionType.ADHARMIC: [
                "Negative karmic debt",
                "Inner turmoil and guilt",
                "Obstacles in spiritual path"
            ],
            ActionType.NEUTRAL: [
                "Maintenance of current karmic state",
                "Neither positive nor negative impact"
            ]
        }
        
        consequences.extend(consequence_map.get(action_type, []))
        
        # Add specific consequences based on description
        if "meditation" in description.lower():
            consequences.append("Increased mindfulness and clarity")
        elif "study" in description.lower():
            consequences.append("Expansion of knowledge and wisdom")
        elif "charity" in description.lower():
            consequences.append("Abundance and generosity return")
        
        return consequences
    
    def _update_dharmic_patterns(self, action: KarmicAction) -> None:
        """Update patterns of dharmic behavior"""
        pattern_key = f"{action.action_type.value}_{action.weight.name}"
        
        if pattern_key not in self.dharmic_patterns:
            self.dharmic_patterns[pattern_key] = {
                "count": 0,
                "total_score": 0.0,
                "average_score": 0.0,
                "frequency": 0.0
            }
        
        pattern = self.dharmic_patterns[pattern_key]
        pattern["count"] += 1
        pattern["total_score"] += action.dharmic_score
        pattern["average_score"] = pattern["total_score"] / pattern["count"]
        
        # Calculate frequency (actions per day)
        if len(self.action_history) > 1:
            time_span = (self.action_history[-1].timestamp - self.action_history[0].timestamp).days
            pattern["frequency"] = pattern["count"] / max(time_span, 1)
    
    def _provide_guidance(self, action: KarmicAction) -> str:
        """Provide dharmic guidance based on the action"""
        if action.dharmic_score >= 4.0:
            return "Excellent dharmic action! Continue on this righteous path."
        elif action.dharmic_score >= 2.0:
            return "Good action with positive karmic impact. Maintain these practices."
        elif action.dharmic_score >= 0.0:
            return "Neutral action. Consider how to make future actions more dharmic."
        elif action.dharmic_score >= -2.0:
            return "This action may have negative consequences. Reflect and seek better choices."
        else:
            return "Strong adharmic action detected. Immediate correction and atonement recommended."
    
    def get_karmic_balance(self) -> Dict[str, Any]:
        """Get current karmic balance and statistics"""
        recent_actions = [a for a in self.action_history 
                         if (datetime.now() - a.timestamp).days <= 30]
        
        action_type_counts = {}
        for action in self.action_history:
            action_type = action.action_type.value
            action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
        
        return {
            "current_karmic_balance": self.karmic_balance,
            "total_actions": len(self.action_history),
            "recent_actions_30_days": len(recent_actions),
            "action_type_distribution": action_type_counts,
            "dharmic_patterns": self.dharmic_patterns,
            "balance_interpretation": self._interpret_balance(),
            "recommendations": self._get_recommendations()
        }
    
    def _interpret_balance(self) -> str:
        """Interpret the current karmic balance"""
        if self.karmic_balance >= 50:
            return "Excellent karmic state - High dharmic merit accumulated"
        elif self.karmic_balance >= 20:
            return "Good karmic state - Positive spiritual progress"
        elif self.karmic_balance >= 0:
            return "Balanced karmic state - Maintain dharmic practices"
        elif self.karmic_balance >= -20:
            return "Slight karmic debt - Focus on positive actions"
        else:
            return "Significant karmic debt - Urgent dharmic correction needed"
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations for improving karmic balance"""
        recommendations = []
        
        if self.karmic_balance < 0:
            recommendations.extend([
                "Increase daily meditation and prayer",
                "Engage in selfless service (seva)",
                "Practice truthfulness and non-violence",
                "Seek forgiveness and make amends"
            ])
        elif self.karmic_balance < 20:
            recommendations.extend([
                "Continue dharmic practices consistently",
                "Study sacred texts for guidance",
                "Practice compassion towards all beings"
            ])
        else:
            recommendations.extend([
                "Share your wisdom with others",
                "Guide others on the dharmic path",
                "Maintain humility despite progress"
            ])
        
        return recommendations

# Factory function for module integration
def create_karma_module() -> KarmaModule:
    """Create and return a Karma Module instance"""
    return KarmaModule()

# Global instance
_karma_module = None

def get_karma_module() -> KarmaModule:
    """Get global Karma module instance"""
    global _karma_module
    if _karma_module is None:
        _karma_module = KarmaModule()
    return _karma_module
