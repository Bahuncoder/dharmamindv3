"""
Response Evaluator Service

Evaluates AI responses for quality, dharmic alignment, and relevance.
Provides confidence scores and suggestions for improvement.

Ensures responses align with dharmic principles and provide value to users.
"""

import asyncio
import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..models.chat import ModuleInfo, EvaluationResult
from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for response evaluation"""
    relevance: float
    completeness: float
    clarity: float
    helpfulness: float
    dharmic_alignment: float
    accuracy: float

class ResponseEvaluator:
    """Service for evaluating AI response quality and dharmic alignment"""
    
    def __init__(self):
        self.dharmic_principles = [
            "ahimsa",      # non-violence
            "satya",       # truthfulness  
            "asteya",      # non-stealing
            "brahmacharya", # continence
            "aparigraha",  # non-possessiveness
            "compassion",
            "wisdom",
            "peace",
            "dharma",
            "karma"
        ]
        self.negative_indicators = [
            "violence", "hatred", "discrimination", "harm", "anger",
            "greed", "attachment", "ego", "pride", "jealousy"
        ]
        
    async def initialize(self):
        """Initialize response evaluator"""
        logger.info("Initializing Response Evaluator...")
        
        try:
            # Load evaluation models/criteria
            await self._load_evaluation_criteria()
            logger.info("Response Evaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Response Evaluator: {e}")
            raise
    
    async def _load_evaluation_criteria(self):
        """Load evaluation criteria and models"""
        # This would typically load ML models for quality assessment
        # For now, we'll use rule-based evaluation
        pass
    
    async def evaluate_response(
        self,
        question: str,
        response: str,
        modules: List[ModuleInfo],
        context: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate a response comprehensively"""
        
        try:
            logger.debug("Evaluating response quality and alignment...")
            
            # Calculate quality metrics
            metrics = await self._calculate_quality_metrics(question, response, modules, context)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(question, response, metrics)
            
            # Extract sources/references
            sources = self._extract_sources(response)
            
            # Calculate overall scores
            confidence_score = self._calculate_confidence_score(metrics)
            dharmic_alignment = metrics.dharmic_alignment
            relevance_score = metrics.relevance
            
            return EvaluationResult(
                confidence_score=confidence_score,
                dharmic_alignment=dharmic_alignment,
                relevance_score=relevance_score,
                sources=sources,
                suggestions=suggestions,
                explanation=f"Response evaluated with {confidence_score:.2f} confidence"
            )
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            # Return default evaluation if evaluation fails
            return EvaluationResult(
                confidence_score=0.5,
                dharmic_alignment=0.7,
                relevance_score=0.6,
                sources=[],
                suggestions=["Consider seeking guidance from multiple perspectives"],
                explanation="Evaluation service encountered an error"
            )
    
    async def _calculate_quality_metrics(
        self,
        question: str,
        response: str,
        modules: List[ModuleInfo],
        context: Optional[str] = None
    ) -> QualityMetrics:
        """Calculate detailed quality metrics"""
        
        # Relevance: How well does response address the question
        relevance = await self._assess_relevance(question, response, context)
        
        # Completeness: How complete is the response
        completeness = self._assess_completeness(question, response)
        
        # Clarity: How clear and understandable is the response
        clarity = self._assess_clarity(response)
        
        # Helpfulness: How helpful is the response to the user
        helpfulness = self._assess_helpfulness(response, modules)
        
        # Dharmic alignment: How well does it align with dharmic principles
        dharmic_alignment = self._assess_dharmic_alignment(response)
        
        # Accuracy: How accurate is the information (basic checks)
        accuracy = self._assess_accuracy(response)
        
        return QualityMetrics(
            relevance=relevance,
            completeness=completeness,
            clarity=clarity,
            helpfulness=helpfulness,
            dharmic_alignment=dharmic_alignment,
            accuracy=accuracy
        )
    
    async def _assess_relevance(self, question: str, response: str, context: Optional[str] = None) -> float:
        """Assess how relevant the response is to the question"""
        
        # Extract key terms from question
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate word overlap
        common_words = question_words.intersection(response_words)
        relevance_score = len(common_words) / max(len(question_words), 1)
        
        # Check for direct question answering patterns
        if "?" in question:
            # Look for answer patterns
            answer_patterns = [
                "the answer is", "yes,", "no,", "because", "due to",
                "this is", "it means", "you can", "you should"
            ]
            
            if any(pattern in response.lower() for pattern in answer_patterns):
                relevance_score += 0.2
        
        # Check for contextual relevance if context provided
        if context:
            context_words = set(context.lower().split())
            context_overlap = context_words.intersection(response_words)
            context_relevance = len(context_overlap) / max(len(context_words), 1)
            relevance_score = (relevance_score + context_relevance) / 2
        
        return min(relevance_score, 1.0)
    
    def _assess_completeness(self, question: str, response: str) -> float:
        """Assess how complete the response is"""
        
        # Basic length-based assessment
        response_length = len(response.split())
        
        # Consider question complexity
        question_complexity = 1.0
        if "?" in question:
            question_complexity += 0.2
        if any(word in question.lower() for word in ["how", "why", "explain", "describe"]):
            question_complexity += 0.3
        if any(word in question.lower() for word in ["multiple", "various", "different", "compare"]):
            question_complexity += 0.4
        
        # Expected length based on complexity
        expected_length = 50 * question_complexity
        
        # Calculate completeness score
        if response_length >= expected_length:
            completeness = 1.0
        else:
            completeness = response_length / expected_length
        
        # Bonus for structured responses
        if any(marker in response for marker in ["1.", "2.", "•", "-", "First", "Second"]):
            completeness += 0.1
        
        return min(completeness, 1.0)
    
    def _assess_clarity(self, response: str) -> float:
        """Assess how clear and readable the response is"""
        
        clarity_score = 0.8  # Base score
        
        # Check sentence structure
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Prefer moderate sentence length (10-25 words)
        if 10 <= avg_sentence_length <= 25:
            clarity_score += 0.1
        elif avg_sentence_length > 35:
            clarity_score -= 0.2
        
        # Check for clear structure
        structure_indicators = ["first", "second", "third", "finally", "however", "therefore", "because"]
        structure_count = sum(1 for indicator in structure_indicators if indicator in response.lower())
        clarity_score += min(structure_count * 0.05, 0.2)
        
        # Penalize excessive jargon (very basic check)
        complex_words = ["aforementioned", "subsequently", "furthermore", "nevertheless", "notwithstanding"]
        jargon_count = sum(1 for word in complex_words if word in response.lower())
        clarity_score -= min(jargon_count * 0.1, 0.3)
        
        return max(min(clarity_score, 1.0), 0.1)
    
    def _assess_helpfulness(self, response: str, modules: List[ModuleInfo]) -> float:
        """Assess how helpful the response is"""
        
        helpfulness = 0.5  # Base score
        
        # Check for actionable advice
        action_words = ["try", "practice", "consider", "meditate", "reflect", "focus", "breathe"]
        action_count = sum(1 for word in action_words if word in response.lower())
        helpfulness += min(action_count * 0.1, 0.3)
        
        # Check for encouragement
        encouraging_words = ["can", "will", "possible", "achieve", "overcome", "grow", "learn"]
        encouragement_count = sum(1 for word in encouraging_words if word in response.lower())
        helpfulness += min(encouragement_count * 0.05, 0.2)
        
        # Bonus for using module expertise
        module_terms = []
        for module in modules:
            module_terms.extend(module.expertise_areas)
        
        module_usage = sum(1 for term in module_terms if term.replace("_", " ") in response.lower())
        helpfulness += min(module_usage * 0.05, 0.2)
        
        # Check for practical guidance
        if any(phrase in response.lower() for phrase in ["you can", "try this", "practice", "daily", "routine"]):
            helpfulness += 0.1
        
        return min(helpfulness, 1.0)
    
    def _assess_dharmic_alignment(self, response: str) -> float:
        """Assess how well the response aligns with dharmic principles"""
        
        alignment_score = 0.7  # Base score (assume generally positive)
        
        # Check for dharmic principles
        dharmic_mentions = 0
        for principle in self.dharmic_principles:
            if principle in response.lower():
                dharmic_mentions += 1
        
        alignment_score += min(dharmic_mentions * 0.05, 0.2)
        
        # Check for positive values
        positive_values = [
            "compassion", "kindness", "love", "peace", "harmony", "wisdom",
            "understanding", "patience", "forgiveness", "gratitude", "mindfulness"
        ]
        
        positive_count = sum(1 for value in positive_values if value in response.lower())
        alignment_score += min(positive_count * 0.03, 0.15)
        
        # Penalize negative indicators
        negative_count = sum(1 for indicator in self.negative_indicators if indicator in response.lower())
        alignment_score -= min(negative_count * 0.1, 0.3)
        
        # Bonus for wisdom quotes or references
        if any(indicator in response for indicator in ["scripture", "vedas", "upanishads", "gita", "puranas", "ancient wisdom"]):
            alignment_score += 0.1
        
        # Check for non-judgmental language
        judgmental_words = ["should always", "must", "never", "wrong", "bad", "evil"]
        judgmental_count = sum(1 for word in judgmental_words if word in response.lower())
        alignment_score -= min(judgmental_count * 0.05, 0.2)
        
        return max(min(alignment_score, 1.0), 0.0)
    
    def _assess_accuracy(self, response: str) -> float:
        """Basic accuracy assessment (fact-checking would require external sources)"""
        
        accuracy = 0.8  # Base assumption of accuracy
        
        # Check for cautious language (indicates awareness of limitations)
        cautious_phrases = ["may", "might", "possibly", "generally", "often", "typically", "consider"]
        cautious_count = sum(1 for phrase in cautious_phrases if phrase in response.lower())
        if cautious_count > 0:
            accuracy += 0.1
        
        # Penalize absolute statements without qualification
        absolute_words = ["always", "never", "all", "none", "every", "impossible"]
        absolute_count = sum(1 for word in absolute_words if word in response.lower())
        accuracy -= min(absolute_count * 0.05, 0.2)
        
        # Bonus for citing limitations or encouraging further exploration
        if any(phrase in response.lower() for phrase in ["learn more", "consult", "personal experience", "individual"]):
            accuracy += 0.1
        
        return max(min(accuracy, 1.0), 0.1)
    
    def _calculate_confidence_score(self, metrics: QualityMetrics) -> float:
        """Calculate overall confidence score from metrics"""
        
        # Weighted average of all metrics
        weights = {
            "relevance": 0.25,
            "completeness": 0.15,
            "clarity": 0.15,
            "helpfulness": 0.20,
            "dharmic_alignment": 0.15,
            "accuracy": 0.10
        }
        
        confidence = (
            metrics.relevance * weights["relevance"] +
            metrics.completeness * weights["completeness"] +
            metrics.clarity * weights["clarity"] +
            metrics.helpfulness * weights["helpfulness"] +
            metrics.dharmic_alignment * weights["dharmic_alignment"] +
            metrics.accuracy * weights["accuracy"]
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    async def _generate_suggestions(
        self,
        question: str,
        response: str,
        metrics: QualityMetrics
    ) -> List[str]:
        """Generate suggestions for improvement or follow-up"""
        
        suggestions = []
        
        # Suggestions based on quality metrics
        if metrics.relevance < 0.7:
            suggestions.append("Consider asking a more specific question for a more targeted response")
        
        if metrics.completeness < 0.6:
            suggestions.append("Ask follow-up questions to explore this topic more deeply")
        
        if metrics.dharmic_alignment < 0.8:
            suggestions.append("Explore how dharmic principles might apply to your situation")
        
        if metrics.helpfulness < 0.7:
            suggestions.append("Consider how you might practically apply this guidance in daily life")
        
        # Content-based suggestions
        if "meditation" in response.lower():
            suggestions.append("Try incorporating a brief meditation practice to experience these insights")
        
        if "compassion" in response.lower():
            suggestions.append("Practice loving-kindness meditation to cultivate compassion")
        
        if "wisdom" in response.lower():
            suggestions.append("Reflect on how this wisdom applies to your current circumstances")
        
        # Question-based suggestions
        if "?" in question and "how" in question.lower():
            suggestions.append("Practice the suggested approaches gradually and mindfully")
        
        if any(word in question.lower() for word in ["problem", "difficulty", "struggle"]):
            suggestions.append("Remember that challenges are opportunities for growth and learning")
        
        # Limit to most relevant suggestions
        return suggestions[:3]
    
    def _extract_sources(self, response: str) -> List[str]:
        """Extract potential sources or references from response"""
        
        sources = []
        
        # Look for scripture references
        scripture_patterns = [
            r"Bhagavad Gita (\d+\.\d+)",
            r"Upanishads?",
            r"Vedas?",
            r"Yoga Sutras? (\d+\.\d+)",
            r"Puranic texts?",
            r"ancient wisdom"
        ]
        
        for pattern in scripture_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            sources.extend(matches)
        
        # Look for general source indicators
        source_indicators = [
            "according to", "as mentioned in", "traditional wisdom",
            "ancient teachings", "scriptural guidance", "dharmic principles"
        ]
        
        for indicator in source_indicators:
            if indicator in response.lower():
                sources.append(indicator)
        
        # Remove duplicates and limit
        return list(set(sources))[:5]
    
    async def evaluate_wisdom_response(
        self,
        question: str,
        response: str,
        modules: List[ModuleInfo]
    ) -> EvaluationResult:
        """Specialized evaluation for wisdom responses"""
        
        # Use standard evaluation but with higher weight on dharmic alignment
        result = await self.evaluate_response(question, response, modules)
        
        # Adjust scores for wisdom content
        if result.dharmic_alignment > 0.8:
            result.confidence_score = min(result.confidence_score + 0.1, 1.0)
        
        # Add wisdom-specific suggestions
        wisdom_suggestions = [
            "Contemplate how this wisdom relates to your personal journey",
            "Consider sharing this insight with others who might benefit",
            "Practice integrating this understanding into daily life"
        ]
        
        result.suggestions.extend(wisdom_suggestions[:1])
        result.suggestions = result.suggestions[:3]  # Limit total suggestions
        
        return result
    
    async def health_check(self) -> bool:
        """Check if evaluator is healthy"""
        return True  # Simple health check


# Dependency injection function for FastAPI
_response_evaluator_instance = None


def get_response_evaluator() -> ResponseEvaluator:
    """Get the response evaluator instance (singleton pattern)"""
    global _response_evaluator_instance
    if _response_evaluator_instance is None:
        _response_evaluator_instance = ResponseEvaluator()
    return _response_evaluator_instance
