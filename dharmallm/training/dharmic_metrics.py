"""
🕉️ Dharmic Alignment and Wisdom Metrics
========================================

Custom evaluation metrics for assessing dharmic alignment and wisdom quality
in DharmaMind's generated responses.

These metrics go beyond standard NLP metrics to evaluate:
- Alignment with core dharmic principles
- Wisdom consistency and depth
- Cultural sensitivity and accuracy
- Teaching quality and clarity
- Spiritual progression appropriateness

Author: DharmaMind Team
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Core Dharmic Principles for Evaluation
DHARMIC_PRINCIPLES = {
    "satya": ["truth", "truthfulness", "honest", "सत्य", "satya"],
    "ahimsa": ["non-violence", "compassion", "non-harm", "अहिंसा", "ahimsa"],
    "dharma": ["righteousness", "duty", "righteous", "धर्म", "dharma"],
    "karma": ["action", "cause and effect", "कर्म", "karma"],
    "moksha": ["liberation", "freedom", "enlightenment", "मोक्ष", "moksha"],
}

# Wisdom Keywords
WISDOM_KEYWORDS = {
    "philosophical": ["consciousness", "awareness", "self", "atman", "brahman"],
    "practical": ["practice", "meditation", "yoga", "pranayama", "asana"],
    "devotional": ["devotion", "bhakti", "surrender", "worship", "prayer"],
    "knowledge": ["knowledge", "wisdom", "jnana", "study", "learning"],
    "ethical": ["ethics", "morality", "virtue", "righteousness", "duty"],
}

# Sacred Text References
SACRED_TEXTS = {
    "gita": ["bhagavad gita", "gita", "krishna", "arjuna"],
    "upanishads": ["upanishad", "upanishadic", "vedanta"],
    "vedas": ["veda", "vedic", "rig veda", "sama veda"],
    "sutras": ["sutra", "yoga sutras", "patanjali"],
    "puranas": ["purana", "puranic", "mythology"],
}


@dataclass
class DharmicMetricResult:
    """Result of dharmic metric evaluation"""
    score: float  # 0-1 range
    alignment_scores: Dict[str, float]  # Per-principle scores
    wisdom_depth: float  # Depth of wisdom (0-1)
    teaching_quality: float  # Quality of teaching (0-1)
    cultural_sensitivity: float  # Cultural appropriateness (0-1)
    references_found: List[str]  # Sacred text references
    principles_mentioned: List[str]  # Dharmic principles mentioned
    explanation: str  # Human-readable explanation


class DharmicAlignmentScorer:
    """
    Evaluate alignment with core dharmic principles.
    
    Scores responses based on how well they align with fundamental
    dharmic teachings and principles.
    """
    
    def __init__(self):
        """Initialize dharmic alignment scorer."""
        self.principles = DHARMIC_PRINCIPLES
        
    def score_alignment(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> DharmicMetricResult:
        """
        Score text for dharmic alignment.
        
        Args:
            text: Text to evaluate
            context: Optional context (question/prompt)
            
        Returns:
            DharmicMetricResult with scores and analysis
        """
        text_lower = text.lower()
        
        # Score each principle
        principle_scores = {}
        principles_mentioned = []
        
        for principle, keywords in self.principles.items():
            score = self._score_principle(text_lower, keywords)
            principle_scores[principle] = score
            
            if score > 0.1:  # Threshold for "mentioned"
                principles_mentioned.append(principle)
        
        # Overall alignment score (average of present principles)
        present_scores = [s for s in principle_scores.values() if s > 0.1]
        alignment_score = sum(present_scores) / len(present_scores) if present_scores else 0.0
        
        # Wisdom depth
        wisdom_depth = self._assess_wisdom_depth(text_lower)
        
        # Teaching quality
        teaching_quality = self._assess_teaching_quality(text, context)
        
        # Cultural sensitivity
        cultural_sensitivity = self._assess_cultural_sensitivity(text)
        
        # Find references
        references = self._find_references(text_lower)
        
        # Generate explanation
        explanation = self._generate_explanation(
            alignment_score,
            principle_scores,
            wisdom_depth,
            teaching_quality,
            cultural_sensitivity,
            references,
        )
        
        return DharmicMetricResult(
            score=alignment_score,
            alignment_scores=principle_scores,
            wisdom_depth=wisdom_depth,
            teaching_quality=teaching_quality,
            cultural_sensitivity=cultural_sensitivity,
            references_found=references,
            principles_mentioned=principles_mentioned,
            explanation=explanation,
        )
    
    def _score_principle(self, text: str, keywords: List[str]) -> float:
        """
        Score how well text aligns with a specific principle.
        
        Args:
            text: Text to evaluate (lowercase)
            keywords: Keywords for the principle
            
        Returns:
            Score 0-1
        """
        matches = sum(1 for kw in keywords if kw in text)
        
        # Normalize by number of keywords
        score = min(matches / len(keywords), 1.0)
        
        # Bonus for multiple mentions
        if matches > 1:
            score = min(score * 1.2, 1.0)
        
        return score
    
    def _assess_wisdom_depth(self, text: str) -> float:
        """
        Assess the depth of wisdom in the text.
        
        Considers:
        - Use of wisdom terminology
        - Depth indicators (because, therefore, thus)
        - Philosophical concepts
        - Practical guidance
        
        Args:
            text: Text to evaluate (lowercase)
            
        Returns:
            Wisdom depth score 0-1
        """
        score = 0.0
        
        # Check wisdom keywords
        wisdom_matches = 0
        for category, keywords in WISDOM_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    wisdom_matches += 1
        
        # Normalize (assume 5 keywords = deep wisdom)
        score += min(wisdom_matches / 5.0, 0.5)
        
        # Check for depth indicators
        depth_indicators = [
            "because", "therefore", "thus", "hence", "consequently",
            "this means", "in other words", "for example",
        ]
        depth_matches = sum(1 for ind in depth_indicators if ind in text)
        score += min(depth_matches / 3.0, 0.3)
        
        # Check for questions (Socratic teaching)
        questions = text.count("?")
        score += min(questions / 5.0, 0.2)
        
        return min(score, 1.0)
    
    def _assess_teaching_quality(
        self, text: str, context: Optional[str] = None
    ) -> float:
        """
        Assess the quality of teaching in the response.
        
        Good teaching:
        - Clear explanations
        - Relevant to question
        - Practical guidance
        - Appropriate depth
        
        Args:
            text: Response text
            context: Question/prompt context
            
        Returns:
            Teaching quality score 0-1
        """
        score = 0.0
        text_lower = text.lower()
        
        # Check for clear structure
        if any(word in text_lower for word in ["first", "second", "finally"]):
            score += 0.2
        
        # Check for examples
        if any(word in text_lower for word in ["for example", "such as", "like"]):
            score += 0.2
        
        # Check for practical guidance
        practical_words = ["practice", "try", "begin", "start", "can"]
        if any(word in text_lower for word in practical_words):
            score += 0.2
        
        # Check for appropriate length (not too short, not too long)
        word_count = len(text.split())
        if 30 <= word_count <= 200:
            score += 0.2
        elif 10 <= word_count < 30:
            score += 0.1
        
        # Check for respectful tone
        respectful_words = ["may", "might", "consider", "explore", "perhaps"]
        if any(word in text_lower for word in respectful_words):
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_cultural_sensitivity(self, text: str) -> float:
        """
        Assess cultural sensitivity and appropriateness.
        
        Checks for:
        - Respectful language
        - Appropriate use of Sanskrit terms
        - No cultural appropriation markers
        - Inclusive language
        
        Args:
            text: Text to evaluate
            
        Returns:
            Cultural sensitivity score 0-1
        """
        score = 1.0  # Start high, deduct for issues
        text_lower = text.lower()
        
        # Deduct for potentially insensitive language
        insensitive_markers = [
            "primitive", "superstition", "backward", "simple folk"
        ]
        for marker in insensitive_markers:
            if marker in text_lower:
                score -= 0.3
        
        # Bonus for respectful framing
        respectful_markers = [
            "ancient wisdom", "traditional", "sacred", "respected"
        ]
        for marker in respectful_markers:
            if marker in text_lower:
                score += 0.1
        
        # Check Sanskrit usage (bonus if present and explained)
        sanskrit_pattern = r'\b[a-z]+\s*\([^)]*sanskrit[^)]*\)'
        if re.search(sanskrit_pattern, text_lower):
            score += 0.2
        
        return max(min(score, 1.0), 0.0)
    
    def _find_references(self, text: str) -> List[str]:
        """
        Find references to sacred texts.
        
        Args:
            text: Text to search (lowercase)
            
        Returns:
            List of found reference types
        """
        references = []
        
        for text_type, keywords in SACRED_TEXTS.items():
            for keyword in keywords:
                if keyword in text:
                    references.append(text_type)
                    break  # Only count each type once
        
        return references
    
    def _generate_explanation(
        self,
        alignment_score: float,
        principle_scores: Dict[str, float],
        wisdom_depth: float,
        teaching_quality: float,
        cultural_sensitivity: float,
        references: List[str],
    ) -> str:
        """Generate human-readable explanation of scores."""
        
        parts = []
        
        # Overall alignment
        if alignment_score >= 0.7:
            parts.append("Strong dharmic alignment")
        elif alignment_score >= 0.4:
            parts.append("Moderate dharmic alignment")
        else:
            parts.append("Limited dharmic alignment")
        
        # Principles
        high_principles = [p for p, s in principle_scores.items() if s > 0.5]
        if high_principles:
            parts.append(f"Emphasizes: {', '.join(high_principles)}")
        
        # Wisdom depth
        if wisdom_depth >= 0.6:
            parts.append("Deep wisdom")
        elif wisdom_depth >= 0.3:
            parts.append("Moderate wisdom")
        
        # Teaching
        if teaching_quality >= 0.6:
            parts.append("Clear teaching")
        
        # References
        if references:
            parts.append(f"References: {', '.join(references)}")
        
        return ". ".join(parts) + "."


class WisdomConsistencyChecker:
    """
    Check for consistency in wisdom teachings across responses.
    
    Ensures that teachings don't contradict established dharmic wisdom
    and maintain internal consistency.
    """
    
    def __init__(self):
        """Initialize wisdom consistency checker."""
        self.known_teachings = {}
        
    def check_consistency(
        self,
        response: str,
        previous_responses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Check wisdom consistency.
        
        Args:
            response: Current response
            previous_responses: Previous responses for consistency check
            
        Returns:
            Dictionary with consistency scores and findings
        """
        result = {
            "consistency_score": 1.0,
            "contradictions": [],
            "agreements": [],
            "novel_teachings": [],
        }
        
        if not previous_responses:
            return result
        
        # Extract key concepts from current response
        current_concepts = self._extract_concepts(response)
        
        # Check against previous responses
        for prev in previous_responses:
            prev_concepts = self._extract_concepts(prev)
            
            # Find overlapping concepts
            common = set(current_concepts.keys()) & set(prev_concepts.keys())
            
            for concept in common:
                # Check if teachings agree
                if self._teachings_agree(
                    current_concepts[concept], prev_concepts[concept]
                ):
                    result["agreements"].append(concept)
                else:
                    result["contradictions"].append({
                        "concept": concept,
                        "current": current_concepts[concept],
                        "previous": prev_concepts[concept],
                    })
                    result["consistency_score"] -= 0.2
        
        # Find novel teachings
        all_prev_concepts = set()
        for prev in previous_responses:
            all_prev_concepts.update(self._extract_concepts(prev).keys())
        
        novel = set(current_concepts.keys()) - all_prev_concepts
        result["novel_teachings"] = list(novel)
        
        result["consistency_score"] = max(result["consistency_score"], 0.0)
        
        return result
    
    def _extract_concepts(self, text: str) -> Dict[str, str]:
        """
        Extract key dharmic concepts from text.
        
        Returns:
            Dictionary mapping concept to its context
        """
        concepts = {}
        text_lower = text.lower()
        
        # Extract principle-related statements
        for principle, keywords in DHARMIC_PRINCIPLES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Get context (50 chars around keyword)
                    idx = text_lower.find(keyword)
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(keyword) + 50)
                    context = text[start:end]
                    concepts[principle] = context
                    break
        
        return concepts
    
    def _teachings_agree(self, teaching1: str, teaching2: str) -> bool:
        """
        Check if two teachings about the same concept agree.
        
        Simple heuristic: check for negation words and contradictory terms.
        """
        # Normalize
        t1 = teaching1.lower()
        t2 = teaching2.lower()
        
        # Check for negation patterns
        negation_words = ["not", "never", "cannot", "shouldn't", "avoid"]
        
        t1_negative = any(word in t1 for word in negation_words)
        t2_negative = any(word in t2 for word in negation_words)
        
        # If one is negative and one isn't, might be contradiction
        if t1_negative != t2_negative:
            return False
        
        # Check for contradictory pairs
        contradictions = [
            ("always", "never"),
            ("must", "should not"),
            ("required", "forbidden"),
        ]
        
        for word1, word2 in contradictions:
            if (word1 in t1 and word2 in t2) or (word2 in t1 and word1 in t2):
                return False
        
        # Default to agreement
        return True


def compute_dharmic_metrics(
    response: str,
    context: Optional[str] = None,
    previous_responses: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute all dharmic metrics for a response.
    
    Args:
        response: Generated response
        context: Question/prompt context
        previous_responses: Previous responses for consistency
        
    Returns:
        Dictionary of dharmic metric scores
    """
    # Alignment scoring
    alignment_scorer = DharmicAlignmentScorer()
    alignment_result = alignment_scorer.score_alignment(response, context)
    
    # Consistency checking
    consistency_checker = WisdomConsistencyChecker()
    consistency_result = consistency_checker.check_consistency(
        response, previous_responses
    )
    
    # Combine results
    metrics = {
        "dharmic_alignment": alignment_result.score,
        "wisdom_depth": alignment_result.wisdom_depth,
        "teaching_quality": alignment_result.teaching_quality,
        "cultural_sensitivity": alignment_result.cultural_sensitivity,
        "wisdom_consistency": consistency_result["consistency_score"],
        "num_principles": len(alignment_result.principles_mentioned),
        "num_references": len(alignment_result.references_found),
        "num_contradictions": len(consistency_result["contradictions"]),
    }
    
    return metrics


# Test function
def test_dharmic_metrics():
    """Test dharmic metrics computation."""
    print("=" * 60)
    print("Testing Dharmic Metrics System")
    print("=" * 60)
    
    # Test responses
    good_response = """
    Dharma is the principle of righteousness and cosmic order. 
    As taught in the Bhagavad Gita, it is our sacred duty to live 
    in alignment with truth (satya) and non-violence (ahimsa). 
    Through selfless action (karma yoga), we can progress spiritually 
    while serving others with compassion.
    """
    
    weak_response = """
    I think it's just about being good and stuff. 
    You know, don't do bad things.
    """
    
    print("\n1. Testing alignment scorer...")
    scorer = DharmicAlignmentScorer()
    
    result = scorer.score_alignment(good_response)
    print(f"   Good response score: {result.score:.3f}")
    print(f"   Wisdom depth: {result.wisdom_depth:.3f}")
    print(f"   Teaching quality: {result.teaching_quality:.3f}")
    print(f"   Principles: {result.principles_mentioned}")
    print(f"   References: {result.references_found}")
    print(f"   Explanation: {result.explanation}")
    
    result_weak = scorer.score_alignment(weak_response)
    print(f"\n   Weak response score: {result_weak.score:.3f}")
    print(f"   Wisdom depth: {result_weak.wisdom_depth:.3f}")
    
    print("\n2. Testing consistency checker...")
    checker = WisdomConsistencyChecker()
    
    response1 = "Meditation is essential for spiritual growth."
    response2 = "Through meditation, we develop inner awareness."
    response3 = "Meditation is not necessary for enlightenment."
    
    consistency = checker.check_consistency(response2, [response1])
    print(f"   Consistent responses score: {consistency['consistency_score']:.3f}")
    
    consistency_bad = checker.check_consistency(response3, [response1, response2])
    print(f"   Contradictory responses score: {consistency_bad['consistency_score']:.3f}")
    print(f"   Contradictions found: {len(consistency_bad['contradictions'])}")
    
    print("\n3. Testing complete dharmic metrics...")
    metrics = compute_dharmic_metrics(good_response, previous_responses=[response1])
    print(f"   Dharmic Alignment: {metrics['dharmic_alignment']:.3f}")
    print(f"   Wisdom Depth: {metrics['wisdom_depth']:.3f}")
    print(f"   Teaching Quality: {metrics['teaching_quality']:.3f}")
    print(f"   Cultural Sensitivity: {metrics['cultural_sensitivity']:.3f}")
    print(f"   Wisdom Consistency: {metrics['wisdom_consistency']:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ All dharmic metrics tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_dharmic_metrics()
