"""
🕉️ Response Evaluator Service
==============================

Advanced response evaluation service that assesses the quality, relevance,
and dharmic alignment of generated responses.

Features:
- Multi-dimensional response scoring
- Dharmic compliance validation
- Emotional appropriateness assessment
- Content quality analysis
- User satisfaction prediction
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import re

logger = logging.getLogger(__name__)

class EvaluationDimension(Enum):
    """Response evaluation dimensions"""
    RELEVANCE = "relevance"
    DHARMIC_ALIGNMENT = "dharmic_alignment"
    EMOTIONAL_APPROPRIATENESS = "emotional_appropriateness"
    CONTENT_QUALITY = "content_quality"
    HELPFULNESS = "helpfulness"
    COMPASSION = "compassion"

class ResponseEvaluator:
    """📊 Advanced response evaluation service"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.evaluation_criteria = {}
        self.dharmic_principles = []
        self.evaluation_history = []
        
    async def initialize(self):
        """Initialize the response evaluator"""
        try:
            self.logger.info("🌟 Initializing Response Evaluator...")
            
            # Set up evaluation criteria
            self._setup_evaluation_criteria()
            
            # Set up dharmic principles
            self._setup_dharmic_principles()
            
            self.logger.info("✅ Response Evaluator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Response Evaluator: {e}")
    
    def _setup_evaluation_criteria(self):
        """Set up evaluation criteria for each dimension"""
        self.evaluation_criteria = {
            EvaluationDimension.RELEVANCE: {
                "keywords_match": 0.3,
                "context_awareness": 0.3,
                "topic_consistency": 0.4
            },
            EvaluationDimension.DHARMIC_ALIGNMENT: {
                "non_violence": 0.3,
                "truthfulness": 0.3,
                "compassion": 0.4
            },
            EvaluationDimension.EMOTIONAL_APPROPRIATENESS: {
                "tone_matching": 0.4,
                "empathy_level": 0.3,
                "emotional_intelligence": 0.3
            },
            EvaluationDimension.CONTENT_QUALITY: {
                "clarity": 0.3,
                "coherence": 0.3,
                "wisdom_depth": 0.4
            },
            EvaluationDimension.HELPFULNESS: {
                "actionable_guidance": 0.4,
                "practical_wisdom": 0.3,
                "solution_orientation": 0.3
            },
            EvaluationDimension.COMPASSION: {
                "empathetic_language": 0.4,
                "non_judgmental": 0.3,
                "healing_intent": 0.3
            }
        }
    
    def _setup_dharmic_principles(self):
        """Set up dharmic principles for validation"""
        self.dharmic_principles = [
            "ahimsa",  # Non-violence
            "satya",   # Truthfulness
            "karuna",  # Compassion
            "daya",    # Mercy
            "shanti",  # Peace
            "seva",    # Service
            "dharma"   # Righteousness
        ]
    
    async def evaluate_response(
        self, 
        response: str, 
        user_message: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Evaluate a response across all dimensions"""
        try:
            context = context or {}
            
            evaluation_results = {}
            overall_scores = []
            
            # Evaluate each dimension
            for dimension in EvaluationDimension:
                score = await self._evaluate_dimension(dimension, response, user_message, context)
                evaluation_results[dimension.value] = score
                overall_scores.append(score)
            
            # Calculate overall score
            overall_score = sum(overall_scores) / len(overall_scores)
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(evaluation_results)
            
            # Record evaluation
            evaluation_record = {
                "timestamp": datetime.now(),
                "response": response[:200],  # First 200 chars
                "user_message": user_message[:100],  # First 100 chars
                "scores": evaluation_results,
                "overall_score": overall_score,
                "quality_level": quality_level
            }
            self.evaluation_history.append(evaluation_record)
            
            self.logger.info(f"📊 Response evaluated: {quality_level} (score: {overall_score:.2f})")
            
            return {
                "overall_score": overall_score,
                "quality_level": quality_level,
                "dimension_scores": evaluation_results,
                "recommendations": recommendations,
                "dharmic_compliance": evaluation_results[EvaluationDimension.DHARMIC_ALIGNMENT.value],
                "evaluation_id": len(self.evaluation_history)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Response evaluation failed: {e}")
            return {
                "overall_score": 0.5,
                "quality_level": "unknown",
                "dimension_scores": {},
                "recommendations": ["Unable to evaluate response"],
                "error": str(e)
            }
    
    async def _evaluate_dimension(
        self, 
        dimension: EvaluationDimension, 
        response: str, 
        user_message: str, 
        context: Dict[str, Any]
    ) -> float:
        """Evaluate a specific dimension"""
        try:
            if dimension == EvaluationDimension.RELEVANCE:
                return self._evaluate_relevance(response, user_message, context)
            elif dimension == EvaluationDimension.DHARMIC_ALIGNMENT:
                return self._evaluate_dharmic_alignment(response)
            elif dimension == EvaluationDimension.EMOTIONAL_APPROPRIATENESS:
                return self._evaluate_emotional_appropriateness(response, context)
            elif dimension == EvaluationDimension.CONTENT_QUALITY:
                return self._evaluate_content_quality(response)
            elif dimension == EvaluationDimension.HELPFULNESS:
                return self._evaluate_helpfulness(response, user_message)
            elif dimension == EvaluationDimension.COMPASSION:
                return self._evaluate_compassion(response)
            else:
                return 0.5  # Default score
                
        except Exception as e:
            self.logger.error(f"❌ Failed to evaluate {dimension}: {e}")
            return 0.5
    
    def _evaluate_relevance(self, response: str, user_message: str, context: Dict[str, Any]) -> float:
        """Evaluate response relevance"""
        # Simple keyword matching
        user_words = set(user_message.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate word overlap
        overlap = len(user_words.intersection(response_words))
        relevance_score = min(overlap / max(len(user_words), 1), 1.0)
        
        # Boost for context awareness
        if context.get("emotional_state") and any(emotion in response.lower() for emotion in ["feel", "emotion", "understand"]):
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)
    
    def _evaluate_dharmic_alignment(self, response: str) -> float:
        """Evaluate dharmic alignment"""
        response_lower = response.lower()
        
        # Check for positive dharmic indicators
        positive_indicators = [
            "compassion", "kindness", "wisdom", "peace", "love", "understanding",
            "namaste", "dharma", "service", "truth", "non-violence", "harmony"
        ]
        positive_score = sum(1 for indicator in positive_indicators if indicator in response_lower)
        
        # Check for negative indicators
        negative_indicators = [
            "violence", "hatred", "anger", "revenge", "harm", "hurt", "destroy"
        ]
        negative_score = sum(1 for indicator in negative_indicators if indicator in response_lower)
        
        # Calculate dharmic score
        dharmic_score = min((positive_score - negative_score) / max(len(positive_indicators), 1), 1.0)
        return max(dharmic_score, 0.0)
    
    def _evaluate_emotional_appropriateness(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate emotional appropriateness"""
        response_lower = response.lower()
        
        # Check for empathetic language
        empathy_indicators = [
            "understand", "feel", "sorry", "support", "here for you", 
            "i hear you", "validate", "acknowledge"
        ]
        empathy_score = sum(1 for indicator in empathy_indicators if indicator in response_lower)
        
        # Normalize score
        emotional_score = min(empathy_score / max(len(empathy_indicators), 1), 1.0)
        
        # Consider emotional context if available
        if context.get("emotional_state") == "sadness" and any(comfort in response_lower for comfort in ["comfort", "peace", "healing"]):
            emotional_score += 0.3
        
        return min(emotional_score, 1.0)
    
    def _evaluate_content_quality(self, response: str) -> float:
        """Evaluate content quality"""
        # Length appropriateness (not too short, not too long)
        length_score = 1.0 if 50 <= len(response) <= 500 else 0.7
        
        # Sentence structure
        sentences = response.split('.')
        structure_score = 1.0 if 2 <= len(sentences) <= 8 else 0.7
        
        # Wisdom indicators
        wisdom_indicators = ["wisdom", "teaching", "guidance", "insight", "understanding", "enlightenment"]
        wisdom_score = min(sum(1 for indicator in wisdom_indicators if indicator in response.lower()) / 3, 1.0)
        
        return (length_score + structure_score + wisdom_score) / 3
    
    def _evaluate_helpfulness(self, response: str, user_message: str) -> float:
        """Evaluate helpfulness"""
        response_lower = response.lower()
        
        # Check for actionable guidance
        action_indicators = [
            "try", "practice", "consider", "begin", "start", "focus on",
            "breathe", "meditate", "reflect", "journal", "explore"
        ]
        action_score = sum(1 for indicator in action_indicators if indicator in response_lower)
        
        # Check for solution orientation
        solution_indicators = [
            "solution", "way", "path", "approach", "method", "technique",
            "practice", "exercise", "step"
        ]
        solution_score = sum(1 for indicator in solution_indicators if indicator in response_lower)
        
        helpfulness_score = (action_score + solution_score) / max(len(action_indicators + solution_indicators), 1)
        return min(helpfulness_score, 1.0)
    
    def _evaluate_compassion(self, response: str) -> float:
        """Evaluate compassion level"""
        response_lower = response.lower()
        
        # Compassionate language indicators
        compassion_indicators = [
            "love", "care", "gentle", "kind", "warm", "embrace",
            "support", "heal", "comfort", "peace", "blessing"
        ]
        compassion_score = sum(1 for indicator in compassion_indicators if indicator in response_lower)
        
        # Non-judgmental language
        non_judgmental = not any(word in response_lower for word in ["wrong", "bad", "should not", "never"])
        
        compassion_level = compassion_score / max(len(compassion_indicators), 1)
        if non_judgmental:
            compassion_level += 0.2
        
        return min(compassion_level, 1.0)
    
    def _determine_quality_level(self, overall_score: float) -> str:
        """Determine quality level based on overall score"""
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.8:
            return "very_good"
        elif overall_score >= 0.7:
            return "good"
        elif overall_score >= 0.6:
            return "fair"
        elif overall_score >= 0.5:
            return "needs_improvement"
        else:
            return "poor"
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        for dimension, score in scores.items():
            if score < 0.6:
                if dimension == "relevance":
                    recommendations.append("Improve response relevance to user query")
                elif dimension == "dharmic_alignment":
                    recommendations.append("Strengthen dharmic principles in response")
                elif dimension == "emotional_appropriateness":
                    recommendations.append("Enhance emotional intelligence and empathy")
                elif dimension == "content_quality":
                    recommendations.append("Improve content clarity and wisdom depth")
                elif dimension == "helpfulness":
                    recommendations.append("Provide more actionable guidance")
                elif dimension == "compassion":
                    recommendations.append("Increase compassionate and loving language")
        
        if not recommendations:
            recommendations.append("Response quality is good - maintain current approach")
        
        return recommendations
    
    async def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        try:
            if not self.evaluation_history:
                return {"message": "No evaluations recorded yet"}
            
            # Calculate average scores
            total_evaluations = len(self.evaluation_history)
            avg_overall_score = sum(record["overall_score"] for record in self.evaluation_history) / total_evaluations
            
            # Calculate quality distribution
            quality_distribution = {}
            for record in self.evaluation_history:
                level = record["quality_level"]
                quality_distribution[level] = quality_distribution.get(level, 0) + 1
            
            return {
                "total_evaluations": total_evaluations,
                "average_overall_score": avg_overall_score,
                "quality_distribution": quality_distribution,
                "recent_evaluations": self.evaluation_history[-5:]  # Last 5 evaluations
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get evaluation stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if evaluator is healthy"""
        try:
            return len(self.evaluation_criteria) > 0
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

# Global instance
_response_evaluator: Optional[ResponseEvaluator] = None

async def get_response_evaluator() -> ResponseEvaluator:
    """Get global response evaluator instance"""
    global _response_evaluator
    if _response_evaluator is None:
        _response_evaluator = ResponseEvaluator()
        await _response_evaluator.initialize()
    return _response_evaluator
