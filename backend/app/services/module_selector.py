"""
🕉️ Module Selector Service
===========================

Intelligent module selection service that chooses the most appropriate 
processing module based on user input, context, and system state.

Features:
- Context-aware module selection
- Multi-modal processing support
- Dynamic priority adjustment
- Performance-based routing
- Dharmic content validation
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ModuleType(Enum):
    """Available processing modules"""
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    SPIRITUAL_GUIDANCE = "spiritual_guidance"
    KNOWLEDGE_BASE = "knowledge_base"
    MEDITATION_GUIDE = "meditation_guide"
    DHARMIC_VALIDATION = "dharmic_validation"
    GENERAL_CHAT = "general_chat"

class ModuleSelector:
    """🎯 Intelligent module selection service"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.module_priorities = {}
        self.performance_metrics = {}
        self.selection_history = []
        
    async def initialize(self):
        """Initialize the module selector"""
        try:
            self.logger.info("🌟 Initializing Module Selector...")
            
            # Set default module priorities
            self.module_priorities = {
                ModuleType.EMOTIONAL_INTELLIGENCE: 0.8,
                ModuleType.SPIRITUAL_GUIDANCE: 0.9,
                ModuleType.KNOWLEDGE_BASE: 0.7,
                ModuleType.MEDITATION_GUIDE: 0.6,
                ModuleType.DHARMIC_VALIDATION: 0.8,
                ModuleType.GENERAL_CHAT: 0.5
            }
            
            # Initialize performance metrics
            for module_type in ModuleType:
                self.performance_metrics[module_type] = {
                    "success_rate": 0.8,
                    "avg_response_time": 1.0,
                    "user_satisfaction": 0.7,
                    "last_updated": datetime.now()
                }
            
            self.logger.info("✅ Module Selector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Module Selector: {e}")
    
    async def select_module(self, message: str, context: Dict[str, Any] = None) -> Tuple[ModuleType, float]:
        """Select the most appropriate module for processing"""
        try:
            context = context or {}
            
            # Analyze message content
            content_scores = self._analyze_content(message)
            
            # Consider context factors
            context_scores = self._analyze_context(context)
            
            # Apply performance metrics
            performance_scores = self._apply_performance_metrics()
            
            # Calculate final scores
            final_scores = {}
            for module_type in ModuleType:
                final_score = (
                    content_scores.get(module_type, 0.0) * 0.5 +
                    context_scores.get(module_type, 0.0) * 0.3 +
                    performance_scores.get(module_type, 0.0) * 0.2
                )
                final_scores[module_type] = final_score
            
            # Select best module
            best_module = max(final_scores.items(), key=lambda x: x[1])
            selected_module, confidence = best_module
            
            # Record selection
            self.selection_history.append({
                "module": selected_module,
                "confidence": confidence,
                "message": message[:100],  # First 100 chars
                "timestamp": datetime.now()
            })
            
            self.logger.info(f"📌 Selected {selected_module.value} with confidence {confidence:.2f}")
            
            return selected_module, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Module selection failed: {e}")
            return ModuleType.GENERAL_CHAT, 0.5
    
    def _analyze_content(self, message: str) -> Dict[ModuleType, float]:
        """Analyze message content for module relevance"""
        scores = {}
        message_lower = message.lower()
        
        # Emotional intelligence keywords
        emotional_keywords = ["feel", "emotion", "sad", "happy", "angry", "anxious", "peaceful", "depressed", "joyful"]
        emotional_score = sum(1 for word in emotional_keywords if word in message_lower) / len(emotional_keywords)
        scores[ModuleType.EMOTIONAL_INTELLIGENCE] = min(emotional_score * 2, 1.0)
        
        # Spiritual guidance keywords
        spiritual_keywords = ["dharma", "karma", "moksha", "enlightenment", "wisdom", "spiritual", "divine", "sacred"]
        spiritual_score = sum(1 for word in spiritual_keywords if word in message_lower) / len(spiritual_keywords)
        scores[ModuleType.SPIRITUAL_GUIDANCE] = min(spiritual_score * 2, 1.0)
        
        # Knowledge base keywords
        knowledge_keywords = ["what", "how", "why", "explain", "tell me", "definition", "meaning", "understand"]
        knowledge_score = sum(1 for word in knowledge_keywords if word in message_lower) / len(knowledge_keywords)
        scores[ModuleType.KNOWLEDGE_BASE] = min(knowledge_score * 2, 1.0)
        
        # Meditation guide keywords
        meditation_keywords = ["meditate", "meditation", "breathe", "mindfulness", "awareness", "concentration", "focus"]
        meditation_score = sum(1 for word in meditation_keywords if word in message_lower) / len(meditation_keywords)
        scores[ModuleType.MEDITATION_GUIDE] = min(meditation_score * 2, 1.0)
        
        # Dharmic validation keywords
        dharmic_keywords = ["right", "wrong", "ethical", "moral", "righteous", "dharmic", "virtue", "sin"]
        dharmic_score = sum(1 for word in dharmic_keywords if word in message_lower) / len(dharmic_keywords)
        scores[ModuleType.DHARMIC_VALIDATION] = min(dharmic_score * 2, 1.0)
        
        # General chat (default baseline)
        scores[ModuleType.GENERAL_CHAT] = 0.3
        
        return scores
    
    def _analyze_context(self, context: Dict[str, Any]) -> Dict[ModuleType, float]:
        """Analyze context for module relevance"""
        scores = {module_type: 0.0 for module_type in ModuleType}
        
        # User emotional state
        if context.get("emotional_state"):
            scores[ModuleType.EMOTIONAL_INTELLIGENCE] += 0.5
        
        # Session type
        session_type = context.get("session_type", "")
        if session_type == "spiritual_guidance":
            scores[ModuleType.SPIRITUAL_GUIDANCE] += 0.6
        elif session_type == "meditation":
            scores[ModuleType.MEDITATION_GUIDE] += 0.6
        elif session_type == "learning":
            scores[ModuleType.KNOWLEDGE_BASE] += 0.6
        
        # User preferences
        preferences = context.get("user_preferences", {})
        if preferences.get("prefer_emotional_support"):
            scores[ModuleType.EMOTIONAL_INTELLIGENCE] += 0.3
        if preferences.get("prefer_spiritual_guidance"):
            scores[ModuleType.SPIRITUAL_GUIDANCE] += 0.3
        
        return scores
    
    def _apply_performance_metrics(self) -> Dict[ModuleType, float]:
        """Apply performance-based scoring"""
        scores = {}
        
        for module_type, metrics in self.performance_metrics.items():
            # Combine success rate and user satisfaction
            performance_score = (
                metrics["success_rate"] * 0.6 +
                metrics["user_satisfaction"] * 0.4
            )
            scores[module_type] = performance_score
        
        return scores
    
    async def update_performance(self, module_type: ModuleType, success: bool, response_time: float, user_rating: Optional[float] = None):
        """Update performance metrics for a module"""
        try:
            metrics = self.performance_metrics[module_type]
            
            # Update success rate (exponential moving average)
            current_success = 1.0 if success else 0.0
            metrics["success_rate"] = 0.9 * metrics["success_rate"] + 0.1 * current_success
            
            # Update response time
            metrics["avg_response_time"] = 0.9 * metrics["avg_response_time"] + 0.1 * response_time
            
            # Update user satisfaction if provided
            if user_rating is not None:
                metrics["user_satisfaction"] = 0.9 * metrics["user_satisfaction"] + 0.1 * user_rating
            
            metrics["last_updated"] = datetime.now()
            
            self.logger.debug(f"📊 Updated metrics for {module_type.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update performance for {module_type}: {e}")
    
    async def get_module_stats(self) -> Dict[str, Any]:
        """Get module selection statistics"""
        try:
            # Calculate selection frequency
            total_selections = len(self.selection_history)
            if total_selections == 0:
                return {"message": "No selections recorded yet"}
            
            frequency = {}
            for record in self.selection_history:
                module_name = record["module"].value
                frequency[module_name] = frequency.get(module_name, 0) + 1
            
            # Calculate percentages
            percentages = {k: (v / total_selections) * 100 for k, v in frequency.items()}
            
            return {
                "total_selections": total_selections,
                "frequency": frequency,
                "percentages": percentages,
                "performance_metrics": {
                    module_type.value: metrics 
                    for module_type, metrics in self.performance_metrics.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get module stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if module selector is healthy"""
        try:
            return len(self.module_priorities) > 0
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

# Global instance
_module_selector: Optional[ModuleSelector] = None

async def get_module_selector() -> ModuleSelector:
    """Get global module selector instance"""
    global _module_selector
    if _module_selector is None:
        _module_selector = ModuleSelector()
        await _module_selector.initialize()
    return _module_selector
