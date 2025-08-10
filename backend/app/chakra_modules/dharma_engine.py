"""
Dharma Engine - Integrated into DharmaMind Backend
=================================================

This module implements the core dharma processing engine that ensures
all AI operations align with dharmic principles and righteous conduct.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DharmaViolationType(Enum):
    """Types of dharma violations"""
    AHIMSA_VIOLATION = "ahimsa_violation"  # Non-violence
    SATYA_VIOLATION = "satya_violation"    # Truthfulness
    ASTEYA_VIOLATION = "asteya_violation"  # Non-stealing
    CULTURAL_INSENSITIVITY = "cultural_insensitivity"
    SACRED_DISRESPECT = "sacred_disrespect"
    HARMFUL_CONTENT = "harmful_content"

class DharmaLevel(Enum):
    """Levels of dharmic conduct"""
    ADHARMIC = -1    # Against dharma
    NEUTRAL = 0      # Neither dharmic nor adharmic
    DHARMIC = 1      # Aligned with dharma
    HIGHLY_DHARMIC = 2  # Highly righteous

@dataclass
class DharmaViolation:
    """Represents a dharma violation"""
    violation_type: DharmaViolationType
    severity: float  # 0.0 to 1.0
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None

@dataclass
class DharmaAssessment:
    """Complete dharma assessment result"""
    overall_level: DharmaLevel
    dharma_score: float  # -1.0 to 2.0
    violations: List[DharmaViolation]
    positive_aspects: List[str]
    recommendations: List[str]
    cultural_sensitivity_score: float
    sacred_respect_score: float

class DharmaEngine:
    """
    Core Dharma Engine for ensuring righteous AI conduct
    
    This engine processes all content and actions through dharmic principles
    to ensure alignment with universal righteousness and wisdom traditions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dharma_principles = self._initialize_dharma_principles()
        self.cultural_guidelines = self._load_cultural_guidelines()
        self.sacred_terms = self._load_sacred_terms()
        self.violation_patterns = self._initialize_violation_patterns()
        
        # Dharma processing metrics
        self.assessments_count = 0
        self.violations_prevented = 0
        self.dharma_alignment_score = 1.0
        
        self.logger.info("Dharma Engine initialized with righteous principles")
    
    def _initialize_dharma_principles(self) -> Dict[str, Any]:
        """Initialize core dharmic principles from diverse traditions"""
        
        return {
            "yamas": {  # Hindu restraints
                "ahimsa": {
                    "description": "Non-violence in thought, word, and deed",
                    "keywords": ["violence", "harm", "hurt", "kill", "destroy"],
                    "weight": 1.0,
                    "tradition": "Hindu"
                },
                "satya": {
                    "description": "Truthfulness and honesty",
                    "keywords": ["lie", "false", "deception", "fraud", "cheat"],
                    "weight": 0.9,
                    "tradition": "Hindu"
                },
                "asteya": {
                    "description": "Non-stealing",
                    "keywords": ["steal", "theft", "piracy", "plagiarism"],
                    "weight": 0.8,
                    "tradition": "Hindu"
                },
                "brahmacharya": {
                    "description": "Energy conservation and purity",
                    "keywords": ["inappropriate", "vulgar", "crude"],
                    "weight": 0.7,
                    "tradition": "Hindu"
                },
                "aparigraha": {
                    "description": "Non-possessiveness",
                    "keywords": ["greed", "hoarding", "excessive"],
                    "weight": 0.6,
                    "tradition": "Hindu"
                }
            },
            
            "niyamas": {  # Hindu observances
                "saucha": {
                    "description": "Cleanliness and purity",
                    "positive_keywords": ["clean", "pure", "clear", "organized"],
                    "weight": 0.6,
                    "tradition": "Hindu"
                },
                "santosha": {
                    "description": "Contentment",
                    "positive_keywords": ["contentment", "gratitude", "satisfaction"],
                    "weight": 0.7,
                    "tradition": "Hindu"
                },
                "tapas": {
                    "description": "Discipline and austerity",
                    "positive_keywords": ["discipline", "practice", "dedication"],
                    "weight": 0.8,
                    "tradition": "Hindu"
                },
                "svadhyaya": {
                    "description": "Self-study and learning",
                    "positive_keywords": ["study", "learning", "wisdom", "knowledge"],
                    "weight": 0.9,
                    "tradition": "Hindu"
                },
                "ishvara_pranidhana": {
                    "description": "Surrender to the Divine",
                    "positive_keywords": ["devotion", "surrender", "divine", "sacred"],
                    "weight": 1.0,
                    "tradition": "Hindu"
                }
            },
            
            "buddhist_principles": {
                "right_speech": {
                    "description": "Truthful, kind, and helpful communication",
                    "positive_keywords": ["truth", "kindness", "helpful", "honest"],
                    "keywords": ["lie", "harsh", "gossip", "slander"],
                    "weight": 0.9,
                    "tradition": "Buddhist"
                },
                "right_action": {
                    "description": "Ethical conduct that doesn't harm others",
                    "positive_keywords": ["ethical", "moral", "righteous", "proper"],
                    "keywords": ["unethical", "immoral", "wrong", "harmful"],
                    "weight": 0.9,
                    "tradition": "Buddhist"
                },
                "compassion": {
                    "description": "Universal loving-kindness for all beings",
                    "positive_keywords": ["compassion", "loving-kindness", "metta", "karuna"],
                    "weight": 1.0,
                    "tradition": "Buddhist"
                }
            },
            
            "universal_values": {
                "compassion": {
                    "description": "Universal compassion for all beings",
                    "positive_keywords": ["compassion", "kindness", "empathy", "care"],
                    "weight": 1.0,
                    "tradition": "Universal"
                },
                "wisdom": {
                    "description": "Pursuit of higher wisdom and understanding",
                    "positive_keywords": ["wisdom", "understanding", "insight", "enlightenment"],
                    "weight": 0.9,
                    "tradition": "Universal"
                },
                "service": {
                    "description": "Selfless service to others",
                    "positive_keywords": ["service", "help", "support", "seva"],
                    "weight": 0.8,
                    "tradition": "Universal"
                },
                "love": {
                    "description": "Universal love and connection",
                    "positive_keywords": ["love", "unity", "oneness", "connection"],
                    "weight": 1.0,
                    "tradition": "Universal"
                },
                "peace": {
                    "description": "Inner and outer peace",
                    "positive_keywords": ["peace", "harmony", "tranquility", "serenity"],
                    "weight": 0.9,
                    "tradition": "Universal"
                }
            }
        }
    
    def _load_cultural_guidelines(self) -> Dict[str, Any]:
        """Load cultural sensitivity guidelines for diverse traditions"""
        
        return {
            "respectful_language": {
                "deity_references": [
                    "Always use respectful titles (Lord, Goddess, Divine, Buddha, etc.)",
                    "Avoid casual or dismissive language about deities",
                    "Acknowledge the sacred nature of divine forms"
                ],
                "scriptural_references": [
                    "Cite sources accurately and with respect",
                    "Provide proper context for teachings",
                    "Acknowledge the sacred nature of texts"
                ],
                "cultural_practices": [
                    "Describe practices with respect and understanding",
                    "Avoid judgmental or dismissive language",
                    "Recognize the deep meaning behind traditions"
                ]
            },
            
            "sensitive_topics": {
                "religious_practices": "approach with reverence and understanding",
                "interfaith_dialogue": "promote understanding and harmony",
                "conversion": "respect individual spiritual journeys",
                "sacred_sites": "acknowledge holiness and significance",
                "spiritual_teachers": "show appropriate respect for wisdom lineages"
            },
            
            "inclusive_language": {
                "tradition_neutrality": "Avoid claiming one tradition is superior",
                "universal_wisdom": "Recognize wisdom in all authentic traditions",
                "respectful_comparison": "Compare traditions respectfully when needed",
                "cultural_context": "Acknowledge cultural contexts of practices"
            }
        }
    
    def _load_sacred_terms(self) -> Dict[str, str]:
        """Load sacred terms from diverse traditions and their proper usage"""
        
        return {
            # Hindu terms
            "Om": "Sacred primordial sound - use with reverence",
            "Aum": "Variant of Om - equally sacred",
            "Brahman": "Ultimate reality - highest philosophical concept",
            "Atman": "Individual soul - treat with respect",
            "Guru": "Spiritual teacher - acknowledge sacred relationship",
            "Mantra": "Sacred sound - recognize spiritual power",
            "Dharma": "Righteous duty - central concept",
            "Karma": "Action and consequence - fundamental principle",
            "Moksha": "Liberation - ultimate goal",
            "Ahimsa": "Non-violence - core principle",
            "Yoga": "Union - sacred practice",
            "Vedas": "Sacred scriptures - highest reverence",
            "Bhagavad Gita": "Sacred dialogue - honor teaching",
            
            # Buddhist terms
            "Buddha": "Awakened One - show deep respect",
            "Dharma": "Buddhist teaching - sacred wisdom",
            "Sangha": "Spiritual community - honor fellowship",
            "Nirvana": "Liberation from suffering - ultimate goal",
            "Metta": "Loving-kindness - compassionate practice",
            "Karuna": "Compassion - fundamental virtue",
            "Mindfulness": "Present awareness - sacred practice",
            "Bodhisattva": "Enlightened being - deep reverence",
            
            # Christian terms
            "Christ": "The Anointed One - show reverence",
            "God": "Divine Creator - acknowledge sacredness",
            "Holy Spirit": "Divine presence - treat with respect",
            "Prayer": "Sacred communion - honor practice",
            "Gospel": "Sacred teaching - acknowledge holiness",
            
            # Islamic terms
            "Allah": "The Divine - show utmost respect",
            "Prophet": "Divine messenger - acknowledge sacred role",
            "Quran": "Sacred scripture - highest reverence",
            "Salah": "Sacred prayer - honor practice",
            
            # Universal spiritual terms
            "Sacred": "Holy and divine - treat with reverence",
            "Divine": "Of God or ultimate reality - acknowledge sacredness",
            "Prayer": "Spiritual communication - honor practice",
            "Meditation": "Contemplative practice - recognize depth",
            "Wisdom": "Spiritual understanding - acknowledge value",
            "Love": "Divine force - recognize sacred nature"
        }
    
    def _initialize_violation_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns that indicate dharma violations"""
        
        return {
            "violence_patterns": [
                r"\b(kill|murder|destroy|annihilate|eliminate)\b",
                r"\b(violence|violent|aggressive|attack|assault)\b",
                r"\b(harm|hurt|damage|injure|wound)\b",
                r"\b(fight|battle|war|combat)\s+(?!ignorance|suffering|negativity)\b"
            ],
            
            "disrespect_patterns": [
                r"\b(mock|ridicule|laugh at|make fun of)\s+.*(god|deity|divine|sacred|holy)\b",
                r"\b(stupid|dumb|nonsense|ridiculous|absurd)\s+.*(religion|belief|faith|tradition)\b",
                r"\b(primitive|backward|outdated|superstitious)\s+.*(tradition|practice|ritual|belief)\b",
                r"\b(fake|false|made-up)\s+.*(religion|god|deity|scripture)\b"
            ],
            
            "cultural_insensitivity": [
                r"\b(weird|strange|bizarre|crazy)\s+.*(tradition|custom|practice|ritual)\b",
                r"\b(superstition|myth|legend)\s+.*(belief|practice|teaching)\b",
                r"\b(all (hindus|buddhists|christians|muslims))\s+(are|believe|do)\b"
            ],
            
            "sacred_misuse": [
                r"\bom\b.*\b(joke|funny|casual|random)\b",
                r"\b(mantra|prayer|sacred|holy)\b.*\b(entertainment|fun|game|casual)\b",
                r"\b(god|divine|sacred)\b.*\b(whatever|random|casual)\b"
            ],
            
            "harmful_intent": [
                r"\b(cause (harm|pain|suffering))\b",
                r"\b(make (someone )?suffer)\b",
                r"\b(bring (misery|pain|harm))\b",
                r"\b(inflict (pain|harm|suffering))\b"
            ]
        }
    
    async def assess_dharma_compliance(self, content: str, context: Optional[Dict] = None) -> DharmaAssessment:
        """Assess content for dharma compliance across traditions"""
        
        try:
            self.logger.debug(f"Assessing dharma compliance for content")
            
            # Initialize assessment
            violations = []
            positive_aspects = []
            recommendations = []
            
            # Check for violations
            violence_violations = self._check_ahimsa_violations(content)
            violations.extend(violence_violations)
            
            disrespect_violations = self._check_sacred_disrespect(content)
            violations.extend(disrespect_violations)
            
            cultural_violations = self._check_cultural_sensitivity(content)
            violations.extend(cultural_violations)
            
            harmful_violations = self._check_harmful_intent(content)
            violations.extend(harmful_violations)
            
            # Check for positive dharmic aspects
            positive_aspects = self._identify_positive_aspects(content)
            
            # Calculate scores
            dharma_score = self._calculate_dharma_score(violations, positive_aspects)
            cultural_score = self._calculate_cultural_sensitivity_score(content)
            sacred_score = self._calculate_sacred_respect_score(content)
            
            # Determine overall level
            overall_level = self._determine_dharma_level(dharma_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(violations, positive_aspects)
            
            # Create assessment
            assessment = DharmaAssessment(
                overall_level=overall_level,
                dharma_score=dharma_score,
                violations=violations,
                positive_aspects=positive_aspects,
                recommendations=recommendations,
                cultural_sensitivity_score=cultural_score,
                sacred_respect_score=sacred_score
            )
            
            # Update metrics
            self.assessments_count += 1
            if violations:
                self.violations_prevented += len(violations)
            
            self.logger.debug(f"Dharma assessment completed: Level {overall_level.name}")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in dharma assessment: {str(e)}")
            raise
    
    def _check_ahimsa_violations(self, content: str) -> List[DharmaViolation]:
        """Check for violations of ahimsa (non-violence)"""
        
        violations = []
        content_lower = content.lower()
        
        # Check for violent language
        for pattern in self.violation_patterns["violence_patterns"]:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                violation = DharmaViolation(
                    violation_type=DharmaViolationType.AHIMSA_VIOLATION,
                    severity=0.8,
                    description="Content contains violent language or concepts",
                    location=f"Position {match.start()}-{match.end()}",
                    suggestion="Consider using peaceful, non-violent language that promotes healing"
                )
                violations.append(violation)
        
        return violations
    
    def _check_sacred_disrespect(self, content: str) -> List[DharmaViolation]:
        """Check for disrespect towards sacred concepts across traditions"""
        
        violations = []
        content_lower = content.lower()
        
        # Check disrespect patterns
        for pattern in self.violation_patterns["disrespect_patterns"]:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                violation = DharmaViolation(
                    violation_type=DharmaViolationType.SACRED_DISRESPECT,
                    severity=1.0,
                    description="Content shows disrespect towards sacred concepts",
                    location=f"Position {match.start()}-{match.end()}",
                    suggestion="Use reverent language when discussing sacred topics from any tradition"
                )
                violations.append(violation)
        
        # Check sacred term misuse
        for sacred_term, guidance in self.sacred_terms.items():
            if sacred_term.lower() in content_lower:
                # Check for inappropriate context
                context_words = self._get_context_words(content_lower, sacred_term.lower())
                inappropriate_context = [
                    "joke", "funny", "amusing", "entertainment", "casual", "random", "whatever"
                ]
                
                if any(word in context_words for word in inappropriate_context):
                    violation = DharmaViolation(
                        violation_type=DharmaViolationType.SACRED_DISRESPECT,
                        severity=0.7,
                        description=f"Sacred term '{sacred_term}' used inappropriately",
                        suggestion=f"Sacred guidance: {guidance}"
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_cultural_sensitivity(self, content: str) -> List[DharmaViolation]:
        """Check for cultural insensitivity across traditions"""
        
        violations = []
        content_lower = content.lower()
        
        # Check cultural insensitivity patterns
        for pattern in self.violation_patterns["cultural_insensitivity"]:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                violation = DharmaViolation(
                    violation_type=DharmaViolationType.CULTURAL_INSENSITIVITY,
                    severity=0.6,
                    description="Content shows cultural insensitivity",
                    location=f"Position {match.start()}-{match.end()}",
                    suggestion="Use respectful language when discussing cultural and religious practices"
                )
                violations.append(violation)
        
        # Check for overgeneralization
        tradition_generalizations = [
            "all hindus", "all buddhists", "all christians", "all muslims",
            "hinduism is just", "buddhism is only", "christianity teaches that all"
        ]
        
        for pattern in tradition_generalizations:
            if pattern in content_lower:
                violation = DharmaViolation(
                    violation_type=DharmaViolationType.CULTURAL_INSENSITIVITY,
                    severity=0.8,
                    description="Content contains overgeneralization about religious groups",
                    suggestion="Avoid generalizations and acknowledge diversity within all traditions"
                )
                violations.append(violation)
        
        return violations
    
    def _check_harmful_intent(self, content: str) -> List[DharmaViolation]:
        """Check for harmful intent in content"""
        
        violations = []
        content_lower = content.lower()
        
        # Check for harmful intent patterns
        for pattern in self.violation_patterns["harmful_intent"]:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                violation = DharmaViolation(
                    violation_type=DharmaViolationType.HARMFUL_CONTENT,
                    severity=0.9,
                    description="Content expresses harmful intent",
                    location=f"Position {match.start()}-{match.end()}",
                    suggestion="Focus on healing, helping, and positive outcomes for all beings"
                )
                violations.append(violation)
        
        return violations
    
    def _get_context_words(self, content: str, term: str) -> List[str]:
        """Get words around a specific term for context analysis"""
        
        words = content.split()
        context_words = []
        
        for i, word in enumerate(words):
            if term in word.lower():
                # Get 3 words before and after
                start = max(0, i - 3)
                end = min(len(words), i + 4)
                context_words.extend(words[start:end])
        
        return [word.lower() for word in context_words]
    
    def _identify_positive_aspects(self, content: str) -> List[str]:
        """Identify positive dharmic aspects in content"""
        
        positive_aspects = []
        content_lower = content.lower()
        
        # Check for dharmic principles across traditions
        for category, principles in self.dharma_principles.items():
            for principle, config in principles.items():
                positive_keywords = config.get("positive_keywords", [])
                tradition = config.get("tradition", "Universal")
                
                for keyword in positive_keywords:
                    if keyword in content_lower:
                        positive_aspects.append(f"Promotes {principle} ({tradition}): {config['description']}")
        
        # Check for sacred term reverent usage
        for sacred_term, guidance in self.sacred_terms.items():
            if sacred_term.lower() in content_lower:
                context_words = self._get_context_words(content_lower, sacred_term.lower())
                reverent_context = [
                    "sacred", "divine", "holy", "blessed", "reverent", "respectful", "honor"
                ]
                
                if any(word in context_words for word in reverent_context):
                    positive_aspects.append(f"Reverent use of sacred term: {sacred_term}")
        
        # Check for universal spiritual themes
        spiritual_themes = {
            "spiritual growth": ["growth", "development", "evolution", "progress", "transformation"],
            "inner peace": ["peace", "calm", "tranquil", "serene", "stillness"],
            "wisdom seeking": ["wisdom", "knowledge", "understanding", "truth", "insight"],
            "compassion": ["compassion", "kindness", "love", "empathy", "caring"],
            "service": ["service", "help", "support", "seva", "volunteer"],
            "unity": ["unity", "oneness", "connection", "harmony", "togetherness"],
            "forgiveness": ["forgiveness", "mercy", "pardon", "release", "letting go"],
            "gratitude": ["gratitude", "thankfulness", "appreciation", "blessing"]
        }
        
        for theme, keywords in spiritual_themes.items():
            if any(keyword in content_lower for keyword in keywords):
                positive_aspects.append(f"Promotes {theme}")
        
        return positive_aspects
    
    def _calculate_dharma_score(self, violations: List[DharmaViolation], 
                               positive_aspects: List[str]) -> float:
        """Calculate overall dharma score"""
        
        # Start with neutral score
        score = 0.0
        
        # Subtract for violations
        for violation in violations:
            penalty = violation.severity
            if violation.violation_type == DharmaViolationType.SACRED_DISRESPECT:
                penalty *= 1.5  # Sacred disrespect is more serious
            elif violation.violation_type == DharmaViolationType.HARMFUL_CONTENT:
                penalty *= 1.3  # Harmful content is very serious
            score -= penalty
        
        # Add for positive aspects
        score += len(positive_aspects) * 0.2
        
        # Bonus for multiple positive aspects
        if len(positive_aspects) >= 3:
            score += 0.5
        elif len(positive_aspects) >= 5:
            score += 0.8
        
        # Clamp to valid range
        return max(-1.0, min(2.0, score))
    
    def _calculate_cultural_sensitivity_score(self, content: str) -> float:
        """Calculate cultural sensitivity score"""
        
        score = 1.0  # Start with neutral
        content_lower = content.lower()
        
        # Check for cultural terms mentioned respectfully
        cultural_terms = [
            "hinduism", "buddhism", "christianity", "islam", "judaism",
            "vedic", "sanskrit", "pali", "hebrew", "arabic",
            "yoga", "meditation", "prayer", "worship", "ritual"
        ]
        
        respectful_context = [
            "ancient", "traditional", "sacred", "wise", "profound", "deep",
            "beautiful", "meaningful", "spiritual", "divine"
        ]
        
        for term in cultural_terms:
            if term in content_lower:
                context_words = self._get_context_words(content_lower, term)
                if any(word in context_words for word in respectful_context):
                    score += 0.1
        
        return min(2.0, score)
    
    def _calculate_sacred_respect_score(self, content: str) -> float:
        """Calculate sacred respect score"""
        
        score = 1.0  # Start with neutral
        content_lower = content.lower()
        
        # Check for sacred terms
        sacred_count = sum(1 for term in self.sacred_terms.keys() 
                          if term.lower() in content_lower)
        
        if sacred_count > 0:
            # Check overall tone
            reverent_words = [
                "sacred", "holy", "divine", "blessed", "revered", "honored",
                "profound", "deep", "meaningful", "spiritual"
            ]
            reverent_count = sum(1 for word in reverent_words if word in content_lower)
            
            if reverent_count > 0:
                score += 0.3 * (reverent_count / sacred_count)
        
        return min(2.0, score)
    
    def _determine_dharma_level(self, dharma_score: float) -> DharmaLevel:
        """Determine dharma level from score"""
        
        if dharma_score >= 1.5:
            return DharmaLevel.HIGHLY_DHARMIC
        elif dharma_score >= 0.5:
            return DharmaLevel.DHARMIC
        elif dharma_score >= -0.2:
            return DharmaLevel.NEUTRAL
        else:
            return DharmaLevel.ADHARMIC
    
    def _generate_recommendations(self, violations: List[DharmaViolation], 
                                positive_aspects: List[str]) -> List[str]:
        """Generate recommendations for improvement"""
        
        recommendations = []
        
        # Recommendations based on violations
        violation_types = {v.violation_type for v in violations}
        
        if DharmaViolationType.AHIMSA_VIOLATION in violation_types:
            recommendations.append("Practice ahimsa: Use non-violent language and promote peace and healing")
        
        if DharmaViolationType.SACRED_DISRESPECT in violation_types:
            recommendations.append("Honor the sacred: Approach divine concepts from all traditions with reverence")
        
        if DharmaViolationType.CULTURAL_INSENSITIVITY in violation_types:
            recommendations.append("Cultivate cultural sensitivity: Respect diverse wisdom traditions")
        
        if DharmaViolationType.HARMFUL_CONTENT in violation_types:
            recommendations.append("Focus on benefit: Ensure all content serves the highest good of all beings")
        
        # Positive reinforcement
        if len(positive_aspects) >= 3:
            recommendations.append("Continue promoting dharmic values - your content shows excellent alignment")
        elif len(positive_aspects) >= 1:
            recommendations.append("Build on positive aspects: Further develop dharmic and spiritual themes")
        else:
            recommendations.append("Consider incorporating universal principles: compassion, truth, wisdom, service")
        
        # General dharmic guidance
        if not violations:
            recommendations.append("Maintain dharmic excellence: Your content aligns beautifully with righteous principles")
        
        return recommendations
    
    def get_dharma_metrics(self) -> Dict[str, Any]:
        """Get dharma engine metrics"""
        
        return {
            "assessments_completed": self.assessments_count,
            "violations_prevented": self.violations_prevented,
            "dharma_alignment_score": self.dharma_alignment_score,
            "prevention_rate": (self.violations_prevented / max(1, self.assessments_count)) * 100,
            "traditions_supported": ["Hindu", "Buddhist", "Christian", "Islamic", "Universal"]
        }
    
    async def purify_content(self, content: str) -> Tuple[str, DharmaAssessment]:
        """
        Purify content to align with dharmic principles
        
        Args:
            content: Content to purify
            
        Returns:
            Tuple of (purified_content, assessment)
        """
        
        self.logger.info("🔱 Purifying content through dharmic principles...")
        
        # Assess current content
        assessment = await self.assess_dharma_compliance(content)
        
        purified_content = content
        
        # Apply purification based on violations
        for violation in assessment.violations:
            purified_content = await self._apply_purification(purified_content, violation)
        
        # Re-assess purified content
        final_assessment = await self.assess_dharma_compliance(purified_content)
        
        self.logger.info(f"✨ Content purified: {final_assessment.overall_level.name}")
        return purified_content, final_assessment
    
    async def _apply_purification(self, content: str, violation: DharmaViolation) -> str:
        """Apply specific purification for a violation"""
        
        if violation.violation_type == DharmaViolationType.AHIMSA_VIOLATION:
            # Replace violent language with peaceful alternatives
            peaceful_replacements = {
                "kill": "transform",
                "destroy": "transcend", 
                "attack": "address",
                "fight": "overcome",
                "eliminate": "resolve",
                "battle": "work with",
                "defeat": "transform",
                "crush": "dissolve"
            }
            
            for violent_word, peaceful_word in peaceful_replacements.items():
                content = re.sub(r'\b' + violent_word + r'\b', peaceful_word, content, flags=re.IGNORECASE)
        
        elif violation.violation_type == DharmaViolationType.SACRED_DISRESPECT:
            # Add respectful qualifiers
            respectful_replacements = {
                "god": "the Divine",
                "gods": "divine beings",
                "deity": "divine presence"
            }
            
            for disrespectful, respectful in respectful_replacements.items():
                content = re.sub(r'\b' + disrespectful + r'\b', respectful, content, flags=re.IGNORECASE)
        
        elif violation.violation_type == DharmaViolationType.CULTURAL_INSENSITIVITY:
            # Replace insensitive language
            sensitive_replacements = {
                "weird": "unique",
                "strange": "different",
                "bizarre": "unfamiliar",
                "primitive": "traditional",
                "backward": "traditional"
            }
            
            for insensitive, sensitive in sensitive_replacements.items():
                content = re.sub(r'\b' + insensitive + r'\b', sensitive, content, flags=re.IGNORECASE)
        
        return content

# Global dharma engine instance
_dharma_engine = None

def get_dharma_engine() -> DharmaEngine:
    """Get global dharma engine instance"""
    global _dharma_engine
    if _dharma_engine is None:
        _dharma_engine = DharmaEngine()
    return _dharma_engine

# Export the main class
__all__ = ["DharmaEngine", "get_dharma_engine", "DharmaAssessment", "DharmaLevel"]
