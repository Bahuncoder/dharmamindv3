"""
🕉️ DharmaLLM Advanced Data Pipeline - Comprehensive Preprocessing System

Enterprise-grade data processing for dharmic AI training featuring:

Data Processing Stages:
- Raw text ingestion and validation
- Dharmic content scoring and filtering
- Cultural sensitivity analysis and adaptation
- Wisdom tradition classification and tagging
- Quality assessment and enhancement
- Multi-modal data integration
- Augmentation and diversification

Advanced Features:
- Automated dharmic principle detection
- Scripture and wisdom text integration
- Cultural context preservation
- Language normalization and cleaning
- Bias detection and mitigation
- Data quality scoring and filtering
- Real-time processing capabilities

Data Sources:
- Sacred texts and scriptures
- Dharmic teachings and commentaries
- Cultural wisdom traditions
- Modern dharmic discussions
- Meditation and spiritual guidance
- Ethical philosophy texts

May this pipeline create pure and wise training data 📚
"""

import os
import re
import json
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from langdetect import detect, DetectorFactory
import requests
from urllib.parse import urljoin

from ..config.advanced_config import (
    DharmaLLMAdvancedConfig, WisdomTradition, 
    DharmicPrinciple
)

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('punkt')
    nltk.data.find('stopwords')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# ===============================
# DATA QUALITY CLASSES
# ===============================

@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment"""
    
    # Basic quality metrics
    text_length: int
    sentence_count: int
    word_count: int
    unique_words: int
    readability_score: float
    
    # Dharmic quality metrics
    dharmic_score: float
    wisdom_score: float
    cultural_sensitivity_score: float
    principle_scores: Dict[str, float]
    
    # Content analysis
    language: str
    detected_traditions: List[str]
    key_concepts: List[str]
    
    # Quality flags
    is_duplicate: bool = False
    is_toxic: bool = False
    is_biased: bool = False
    is_appropriate: bool = True
    
    # Processing metadata
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    quality_threshold_met: bool = True

@dataclass
class ProcessedDataSample:
    """Single processed data sample"""
    
    # Core content
    text: str
    original_text: str
    source: str
    
    # Tokenized content
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    
    # Dharmic annotations
    dharmic_scores: Dict[str, float]
    wisdom_embeddings: Optional[List[float]] = None
    cultural_context: Optional[Dict[str, Any]] = None
    
    # Metadata
    tradition: Optional[WisdomTradition] = None
    quality_metrics: Optional[DataQualityMetrics] = None
    
    # Processing information
    sample_id: str = field(default_factory=lambda: hashlib.md5(
        datetime.now().isoformat().encode()
    ).hexdigest()[:8])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for dataset creation"""
        return {
            "text": self.text,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "dharmic_scores": self.dharmic_scores,
            "wisdom_embeddings": self.wisdom_embeddings,
            "cultural_context": self.cultural_context,
            "source": self.source,
            "tradition": self.tradition.value if self.tradition else None,
            "sample_id": self.sample_id
        }

# ===============================
# TEXT CLEANING AND NORMALIZATION
# ===============================

class TextCleaner:
    """Advanced text cleaning and normalization"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spacy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Define cleaning patterns
        self.cleaning_patterns = [
            (r'http\S+|www\S+|https\S+', ''),  # URLs
            (r'<[^>]+>', ''),  # HTML tags
            (r'\[.*?\]', ''),  # Text in brackets
            (r'\(.*?\)', ''),  # Text in parentheses (optional)
            (r'[^\w\s\.\?\!\,\;\:\'\"]', ' '),  # Special characters except punctuation
            (r'\s+', ' '),  # Multiple spaces
        ]
        
        # Sacred text markers that should be preserved
        self.sacred_markers = [
            'om', 'aum', 'namaste', 'dharma', 'karma', 'moksha',
            'nirvana', 'buddha', 'krishna', 'rama', 'shiva',
            'verse', 'chapter', 'sutra', 'mantra'
        ]
    
    def clean_text(self, text: str, preserve_sacred: bool = True) -> str:
        """Clean and normalize text while preserving sacred content"""
        
        if not text or not isinstance(text, str):
            return ""
        
        original_text = text
        
        # Basic normalization
        text = text.strip()
        text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines
        text = re.sub(r'\t+', ' ', text)  # Replace tabs
        
        # Apply cleaning patterns (except for sacred text markers)
        for pattern, replacement in self.cleaning_patterns:
            if preserve_sacred:
                # Temporarily replace sacred markers
                sacred_placeholders = {}
                for i, marker in enumerate(self.sacred_markers):
                    placeholder = f"__SACRED_{i}__"
                    if marker.lower() in text.lower():
                        text = re.sub(
                            re.escape(marker), 
                            placeholder, 
                            text, 
                            flags=re.IGNORECASE
                        )
                        sacred_placeholders[placeholder] = marker
                
                # Apply cleaning
                text = re.sub(pattern, replacement, text)
                
                # Restore sacred markers
                for placeholder, marker in sacred_placeholders.items():
                    text = text.replace(placeholder, marker)
            else:
                text = re.sub(pattern, replacement, text)
        
        # Final cleanup
        text = ' '.join(text.split())  # Normalize whitespace
        text = text.strip()
        
        return text if text else original_text
    
    def normalize_dharmic_terms(self, text: str) -> str:
        """Normalize dharmic terminology for consistency"""
        
        # Dictionary of term normalizations
        term_mappings = {
            'dharma': ['dhamma', 'dharma'],
            'karma': ['kamma', 'karma'],
            'nirvana': ['nibbana', 'nirvana'],
            'buddha': ['buddha', 'buddah'],
            'krishna': ['krishna', 'krsna', 'kṛṣṇa'],
            'moksha': ['moksa', 'moksha', 'mokṣa'],
            'meditation': ['dhyana', 'jhana', 'meditation']
        }
        
        normalized_text = text.lower()
        
        for standard_term, variations in term_mappings.items():
            for variation in variations:
                normalized_text = re.sub(
                    r'\b' + re.escape(variation) + r'\b',
                    standard_term,
                    normalized_text,
                    flags=re.IGNORECASE
                )
        
        return normalized_text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract and clean individual sentences"""
        
        sentences = sent_tokenize(text)
        
        # Filter and clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = self.clean_text(sentence)
            if len(cleaned.split()) >= 3:  # Minimum word count
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences
    
    def detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            return detect(text)
        except Exception:
            return "unknown"

# ===============================
# DHARMIC CONTENT ANALYZER
# ===============================

class DharmicContentAnalyzer:
    """Analyzes content for dharmic principles and wisdom"""
    
    def __init__(self):
        
        # Dharmic principle keywords
        self.principle_keywords = {
            DharmicPrinciple.AHIMSA: [
                'non-violence', 'non-harm', 'peace', 'compassion', 'kindness',
                'gentle', 'protect', 'care', 'love', 'harmony'
            ],
            DharmicPrinciple.SATYA: [
                'truth', 'truthfulness', 'honest', 'authentic', 'genuine',
                'sincere', 'transparent', 'accurate', 'factual'
            ],
            DharmicPrinciple.ASTEYA: [
                'respect', 'honor', 'acknowledge', 'give credit', 'grateful',
                'appreciate', 'recognize', 'value', 'thank'
            ],
            DharmicPrinciple.BRAHMACHARYA: [
                'moderation', 'balance', 'self-control', 'discipline', 'restraint',
                'mindful', 'conscious', 'appropriate', 'measured'
            ],
            DharmicPrinciple.APARIGRAHA: [
                'non-possessive', 'generous', 'sharing', 'selfless', 'giving',
                'humble', 'community', 'collective', 'common good'
            ]
        }
        
        # Wisdom tradition indicators
        self.tradition_indicators = {
            WisdomTradition.VEDANTIC: [
                'vedanta', 'upanishad', 'brahman', 'atman', 'maya', 'moksha',
                'advaita', 'self-realization', 'consciousness'
            ],
            WisdomTradition.BUDDHIST: [
                'buddha', 'dharma', 'sangha', 'four noble truths', 'eightfold path',
                'mindfulness', 'meditation', 'compassion', 'wisdom', 'nirvana'
            ],
            WisdomTradition.HINDU: [
                'hindu', 'hinduism', 'krishna', 'rama', 'shiva', 'vishnu',
                'bhagavad gita', 'ramayana', 'mahabharata', 'yoga'
            ],
            WisdomTradition.UNIVERSAL: [
                'universal love', 'cosmic consciousness', 'divine', 'sacred',
                'spiritual', 'transcendent', 'eternal', 'infinite'
            ]
        }
        
        # Wisdom concepts
        self.wisdom_concepts = [
            'wisdom', 'enlightenment', 'awakening', 'realization', 'insight',
            'understanding', 'knowledge', 'learning', 'growth', 'transformation',
            'consciousness', 'awareness', 'mindfulness', 'presence'
        ]
    
    def analyze_dharmic_content(self, text: str) -> Dict[str, float]:
        """Analyze text for dharmic principle adherence"""
        
        text_lower = text.lower()
        principle_scores = {}
        
        for principle, keywords in self.principle_keywords.items():
            # Count keyword occurrences
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            
            # Calculate score based on frequency and text length
            text_word_count = len(text.split())
            normalized_score = keyword_count / max(text_word_count / 50, 1)  # Normalize by text length
            
            # Apply sigmoid function to bound score between 0 and 1
            score = 1 / (1 + np.exp(-normalized_score))
            principle_scores[principle.value] = min(score, 1.0)
        
        return principle_scores
    
    def detect_wisdom_tradition(self, text: str) -> Dict[WisdomTradition, float]:
        """Detect wisdom tradition alignment"""
        
        text_lower = text.lower()
        tradition_scores = {}
        
        for tradition, indicators in self.tradition_indicators.items():
            indicator_count = sum(1 for indicator in indicators if indicator in text_lower)
            
            # Calculate confidence score
            score = min(indicator_count * 0.2, 1.0)
            tradition_scores[tradition] = score
        
        return tradition_scores
    
    def calculate_wisdom_score(self, text: str) -> float:
        """Calculate overall wisdom score"""
        
        text_lower = text.lower()
        
        # Count wisdom concepts
        wisdom_count = sum(1 for concept in self.wisdom_concepts if concept in text_lower)
        
        # Calculate depth indicators
        depth_indicators = [
            'profound', 'deep', 'fundamental', 'essence', 'core',
            'ultimate', 'absolute', 'eternal', 'timeless'
        ]
        depth_count = sum(1 for indicator in depth_indicators if indicator in text_lower)
        
        # Practical wisdom indicators
        practical_indicators = [
            'practical', 'apply', 'use', 'practice', 'daily life',
            'everyday', 'real world', 'concrete'
        ]
        practical_count = sum(1 for indicator in practical_indicators if indicator in text_lower)
        
        # Combine scores
        text_length = len(text.split())
        normalized_wisdom = wisdom_count / max(text_length / 30, 1)
        normalized_depth = depth_count / max(text_length / 50, 1)
        normalized_practical = practical_count / max(text_length / 40, 1)
        
        # Weighted combination
        wisdom_score = (
            0.4 * normalized_wisdom +
            0.3 * normalized_depth +
            0.3 * normalized_practical
        )
        
        return min(wisdom_score, 1.0)
    
    def extract_key_concepts(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key dharmic and wisdom concepts"""
        
        text_lower = text.lower()
        concept_counts = Counter()
        
        # Count all dharmic keywords
        all_keywords = []
        for keywords in self.principle_keywords.values():
            all_keywords.extend(keywords)
        
        # Add wisdom concepts
        all_keywords.extend(self.wisdom_concepts)
        
        # Add tradition indicators
        for indicators in self.tradition_indicators.values():
            all_keywords.extend(indicators)
        
        # Count occurrences
        for keyword in all_keywords:
            if keyword in text_lower:
                concept_counts[keyword] += text_lower.count(keyword)
        
        # Return top concepts
        return [concept for concept, count in concept_counts.most_common(top_k)]

# ===============================
# CULTURAL SENSITIVITY ANALYZER
# ===============================

class CulturalSensitivityAnalyzer:
    """Analyzes and ensures cultural sensitivity"""
    
    def __init__(self):
        
        # Cultural sensitivity indicators
        self.respectful_terms = [
            'respect', 'honor', 'revere', 'sacred', 'holy', 'blessed',
            'tradition', 'heritage', 'culture', 'wisdom', 'teaching'
        ]
        
        self.problematic_terms = [
            'exotic', 'primitive', 'backward', 'superstitious', 'weird',
            'strange', 'cult', 'fanatical', 'extremist'
        ]
        
        # Cultural context markers
        self.cultural_contexts = {
            'indian': ['india', 'indian', 'hindi', 'sanskrit', 'bharata'],
            'tibetan': ['tibet', 'tibetan', 'dalai lama', 'lhasa'],
            'thai': ['thailand', 'thai', 'theravada'],
            'japanese': ['japan', 'japanese', 'zen'],
            'chinese': ['china', 'chinese', 'tao', 'confucius']
        }
    
    def analyze_cultural_sensitivity(self, text: str) -> Tuple[float, List[str]]:
        """Analyze cultural sensitivity and identify issues"""
        
        text_lower = text.lower()
        
        # Count respectful vs problematic terms
        respectful_count = sum(1 for term in self.respectful_terms if term in text_lower)
        problematic_count = sum(1 for term in self.problematic_terms if term in text_lower)
        
        # Calculate sensitivity score
        if problematic_count > 0:
            sensitivity_score = max(0.0, 0.5 - (problematic_count * 0.2))
        else:
            sensitivity_score = min(1.0, 0.7 + (respectful_count * 0.1))
        
        # Identify potential issues
        issues = []
        for term in self.problematic_terms:
            if term in text_lower:
                issues.append(f"Contains potentially insensitive term: '{term}'")
        
        # Check for cultural context appropriateness
        detected_contexts = []
        for culture, markers in self.cultural_contexts.items():
            if any(marker in text_lower for marker in markers):
                detected_contexts.append(culture)
        
        if detected_contexts and respectful_count == 0:
            issues.append("Cultural content lacks explicit respectful language")
        
        return sensitivity_score, issues
    
    def suggest_improvements(self, text: str, issues: List[str]) -> List[str]:
        """Suggest improvements for cultural sensitivity"""
        
        suggestions = []
        
        for issue in issues:
            if "insensitive term" in issue:
                suggestions.append("Replace insensitive terms with respectful alternatives")
            elif "lacks explicit respectful language" in issue:
                suggestions.append("Add respectful acknowledgment of cultural traditions")
        
        return suggestions

# ===============================
# DATA QUALITY ASSESSOR
# ===============================

class DataQualityAssessor:
    """Comprehensive data quality assessment"""
    
    def __init__(self, config: DharmaLLMAdvancedConfig):
        self.config = config
        self.text_cleaner = TextCleaner()
        self.dharmic_analyzer = DharmicContentAnalyzer()
        self.cultural_analyzer = CulturalSensitivityAnalyzer()
        
        # Quality thresholds
        self.min_length = 10  # Minimum characters
        self.max_length = 5000  # Maximum characters
        self.min_words = 3  # Minimum words
        self.min_sentences = 1  # Minimum sentences
        
    def assess_quality(self, text: str, source: str = "unknown") -> DataQualityMetrics:
        """Comprehensive quality assessment"""
        
        # Basic metrics
        text_length = len(text)
        sentences = self.text_cleaner.extract_sentences(text)
        sentence_count = len(sentences)
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        
        # Language detection
        language = self.text_cleaner.detect_language(text)
        
        # Readability (simple approximation)
        readability_score = self._calculate_readability(text)
        
        # Dharmic content analysis
        principle_scores = self.dharmic_analyzer.analyze_dharmic_content(text)
        dharmic_score = np.mean(list(principle_scores.values()))
        wisdom_score = self.dharmic_analyzer.calculate_wisdom_score(text)
        
        # Cultural sensitivity
        cultural_sensitivity_score, cultural_issues = self.cultural_analyzer.analyze_cultural_sensitivity(text)
        
        # Tradition detection
        tradition_scores = self.dharmic_analyzer.detect_wisdom_tradition(text)
        detected_traditions = [
            tradition.value for tradition, score in tradition_scores.items() 
            if score > 0.3
        ]
        
        # Key concepts
        key_concepts = self.dharmic_analyzer.extract_key_concepts(text)
        
        # Quality flags
        is_appropriate = self._check_appropriateness(text)
        is_duplicate = False  # Would be checked against existing data
        is_toxic = self._check_toxicity(text)
        is_biased = self._check_bias(text)
        
        # Overall quality assessment
        quality_threshold_met = (
            text_length >= self.min_length and
            text_length <= self.max_length and
            word_count >= self.min_words and
            sentence_count >= self.min_sentences and
            dharmic_score >= self.config.data.min_dharmic_score and
            wisdom_score >= self.config.data.min_wisdom_score and
            cultural_sensitivity_score >= self.config.data.min_cultural_sensitivity and
            not is_toxic and
            is_appropriate
        )
        
        return DataQualityMetrics(
            text_length=text_length,
            sentence_count=sentence_count,
            word_count=word_count,
            unique_words=unique_words,
            readability_score=readability_score,
            dharmic_score=dharmic_score,
            wisdom_score=wisdom_score,
            cultural_sensitivity_score=cultural_sensitivity_score,
            principle_scores=principle_scores,
            language=language,
            detected_traditions=detected_traditions,
            key_concepts=key_concepts,
            is_duplicate=is_duplicate,
            is_toxic=is_toxic,
            is_biased=is_biased,
            is_appropriate=is_appropriate,
            quality_threshold_met=quality_threshold_met
        )
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified)"""
        sentences = self.text_cleaner.extract_sentences(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simplified readability score (0-1, higher is more readable)
        if avg_sentence_length < 15:
            return 1.0
        elif avg_sentence_length < 25:
            return 0.8
        elif avg_sentence_length < 35:
            return 0.6
        else:
            return 0.4
    
    def _check_appropriateness(self, text: str) -> bool:
        """Check if content is appropriate for dharmic training"""
        
        inappropriate_keywords = [
            'violence', 'hate', 'anger', 'revenge', 'harm', 'kill',
            'destroy', 'attack', 'fight', 'war', 'weapon'
        ]
        
        text_lower = text.lower()
        inappropriate_count = sum(1 for keyword in inappropriate_keywords if keyword in text_lower)
        
        return inappropriate_count == 0
    
    def _check_toxicity(self, text: str) -> bool:
        """Check for toxic content (simplified implementation)"""
        
        toxic_keywords = [
            'racist', 'sexist', 'homophobic', 'discriminate', 'prejudice',
            'hate', 'toxic', 'abuse', 'harassment', 'bullying'
        ]
        
        text_lower = text.lower()
        toxic_count = sum(1 for keyword in toxic_keywords if keyword in text_lower)
        
        return toxic_count > 0
    
    def _check_bias(self, text: str) -> bool:
        """Check for biased content (simplified implementation)"""
        
        bias_indicators = [
            'stereotype', 'assumption', 'generalization', 'always', 'never',
            'all women', 'all men', 'all people', 'everyone knows'
        ]
        
        text_lower = text.lower()
        bias_count = sum(1 for indicator in bias_indicators if indicator in text_lower)
        
        return bias_count > 2  # Allow some generalizations in dharmic context

# ===============================
# DATA AUGMENTATION ENGINE
# ===============================

class DataAugmentationEngine:
    """Advanced data augmentation for dharmic content"""
    
    def __init__(self):
        
        # Paraphrasing templates for dharmic concepts
        self.paraphrase_templates = {
            'compassion': [
                "showing compassion",
                "expressing loving-kindness",
                "demonstrating empathy",
                "embodying care and understanding"
            ],
            'wisdom': [
                "ancient wisdom",
                "timeless knowledge",
                "profound understanding",
                "deep insight"
            ],
            'meditation': [
                "contemplative practice",
                "mindful reflection",
                "inner stillness",
                "focused awareness"
            ]
        }
        
        # Cultural translations
        self.cultural_translations = {
            'peace': {
                'sanskrit': 'shanti',
                'pali': 'santi',
                'tibetan': 'zhi wa'
            },
            'wisdom': {
                'sanskrit': 'jnana',
                'pali': 'panna',
                'tibetan': 'ye shes'
            }
        }
    
    def augment_data_sample(self, text: str, augmentation_type: str) -> List[str]:
        """Generate augmented versions of text"""
        
        augmented_samples = []
        
        if augmentation_type == "paraphrasing":
            augmented_samples.extend(self._paraphrase_dharmic_content(text))
        
        elif augmentation_type == "cultural_translation":
            augmented_samples.extend(self._add_cultural_translations(text))
        
        elif augmentation_type == "wisdom_expansion":
            augmented_samples.extend(self._expand_wisdom_concepts(text))
        
        elif augmentation_type == "principle_alignment":
            augmented_samples.extend(self._align_with_principles(text))
        
        return augmented_samples
    
    def _paraphrase_dharmic_content(self, text: str) -> List[str]:
        """Paraphrase dharmic content"""
        
        paraphrased = []
        
        for concept, alternatives in self.paraphrase_templates.items():
            if concept in text.lower():
                for alternative in alternatives:
                    new_text = text.replace(concept, alternative)
                    if new_text != text:
                        paraphrased.append(new_text)
        
        return paraphrased[:3]  # Limit to 3 paraphrases
    
    def _add_cultural_translations(self, text: str) -> List[str]:
        """Add cultural term translations"""
        
        translated = []
        
        for english_term, translations in self.cultural_translations.items():
            if english_term in text.lower():
                for language, translation in translations.items():
                    new_text = text + f" (In {language}: {translation})"
                    translated.append(new_text)
        
        return translated
    
    def _expand_wisdom_concepts(self, text: str) -> List[str]:
        """Expand on wisdom concepts"""
        
        expanded = []
        
        wisdom_expansions = {
            'meditation': " - the practice of cultivating inner awareness and peace",
            'dharma': " - the path of righteousness and cosmic law",
            'karma': " - the universal principle of cause and effect",
            'compassion': " - the deep wish for all beings to be free from suffering"
        }
        
        for concept, expansion in wisdom_expansions.items():
            if concept in text.lower():
                new_text = text.replace(concept, concept + expansion)
                expanded.append(new_text)
        
        return expanded
    
    def _align_with_principles(self, text: str) -> List[str]:
        """Align content with dharmic principles"""
        
        aligned = []
        
        principle_alignments = {
            'help': " in the spirit of ahimsa (non-harm)",
            'truth': " following the principle of satya (truthfulness)",
            'respect': " embodying asteya (respect for others)",
            'balance': " practicing brahmacharya (moderation)",
            'share': " expressing aparigraha (non-possessiveness)"
        }
        
        for term, alignment in principle_alignments.items():
            if term in text.lower():
                new_text = text.replace(term, term + alignment)
                aligned.append(new_text)
        
        return aligned

# ===============================
# MAIN DATA PROCESSOR
# ===============================

class DharmaLLMDataProcessor:
    """Main data processing pipeline for DharmaLLM"""
    
    def __init__(self, config: DharmaLLMAdvancedConfig):
        self.config = config
        
        # Initialize components
        self.text_cleaner = TextCleaner()
        self.quality_assessor = DataQualityAssessor(config)
        self.augmentation_engine = DataAugmentationEngine()
        
        # Initialize tokenizer
        self.tokenizer = None
        self._initialize_tokenizer()
        
        # Setup logging
        self.setup_logging()
        
        # Processing statistics
        self.processing_stats = {
            'total_samples': 0,
            'accepted_samples': 0,
            'rejected_samples': 0,
            'augmented_samples': 0
        }
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for the model"""
        try:
            model_name = self.config.model.base_model_path or "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    def setup_logging(self):
        """Setup processing logging"""
        log_dir = Path(self.config.log_dir) / "data_processing"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.processing_logger = logging.getLogger("dharma_data_processing")
        self.processing_logger.setLevel(logging.INFO)
        
        log_file = log_dir / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.processing_logger.addHandler(file_handler)
    
    def process_raw_data(
        self, 
        data_sources: List[Dict[str, Any]],
        output_path: str
    ) -> DatasetDict:
        """Process raw data sources into training-ready dataset"""
        
        self.processing_logger.info("Starting data processing pipeline...")
        
        # Collect and process all samples
        all_samples = []
        
        for source_info in data_sources:
            source_name = source_info['name']
            source_path = source_info['path']
            source_type = source_info.get('type', 'text')
            
            self.processing_logger.info(f"Processing source: {source_name}")
            
            # Load data from source
            raw_data = self._load_data_source(source_path, source_type)
            
            # Process each text sample
            for text_sample in raw_data:
                processed_samples = self._process_text_sample(text_sample, source_name)
                all_samples.extend(processed_samples)
        
        # Split into train/validation/test
        dataset_dict = self._create_dataset_splits(all_samples)
        
        # Save processed dataset
        self._save_dataset(dataset_dict, output_path)
        
        # Log processing statistics
        self._log_processing_stats()
        
        return dataset_dict
    
    def _load_data_source(self, source_path: str, source_type: str) -> List[str]:
        """Load data from various source types"""
        
        data = []
        
        if source_type == "text":
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split into chunks or paragraphs
                data = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        
        elif source_type == "json":
            with open(source_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    data = [item.get('text', str(item)) for item in json_data]
                else:
                    data = [json_data.get('text', str(json_data))]
        
        elif source_type == "csv":
            df = pd.read_csv(source_path)
            text_column = 'text' if 'text' in df.columns else df.columns[0]
            data = df[text_column].dropna().tolist()
        
        elif source_type == "huggingface":
            # Load from Hugging Face datasets
            try:
                dataset = load_dataset(source_path)
                if 'train' in dataset:
                    data = dataset['train']['text']
                else:
                    data = list(dataset.values())[0]['text']
            except Exception as e:
                self.processing_logger.error(f"Failed to load HF dataset {source_path}: {e}")
                data = []
        
        self.processing_logger.info(f"Loaded {len(data)} samples from {source_path}")
        return data
    
    def _process_text_sample(self, text: str, source: str) -> List[ProcessedDataSample]:
        """Process a single text sample"""
        
        self.processing_stats['total_samples'] += 1
        
        # Clean text
        cleaned_text = self.text_cleaner.clean_text(text)
        
        if not cleaned_text:
            self.processing_stats['rejected_samples'] += 1
            return []
        
        # Assess quality
        quality_metrics = self.quality_assessor.assess_quality(cleaned_text, source)
        
        if not quality_metrics.quality_threshold_met:
            self.processing_stats['rejected_samples'] += 1
            return []
        
        # Tokenize
        if self.tokenizer is None:
            self.processing_stats['rejected_samples'] += 1
            return []
        
        # Create base sample
        base_sample = self._create_processed_sample(
            cleaned_text, text, source, quality_metrics
        )
        
        if base_sample is None:
            self.processing_stats['rejected_samples'] += 1
            return []
        
        self.processing_stats['accepted_samples'] += 1
        processed_samples = [base_sample]
        
        # Data augmentation
        if self.config.data.enable_data_augmentation:
            augmented_samples = self._augment_sample(cleaned_text, source, quality_metrics)
            processed_samples.extend(augmented_samples)
            self.processing_stats['augmented_samples'] += len(augmented_samples)
        
        return processed_samples
    
    def _create_processed_sample(
        self, 
        text: str, 
        original_text: str, 
        source: str, 
        quality_metrics: DataQualityMetrics
    ) -> Optional[ProcessedDataSample]:
        """Create a processed data sample"""
        
        try:
            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.config.data.max_sequence_length,
                return_tensors="pt"
            )
            
            input_ids = tokenized["input_ids"].squeeze().tolist()
            attention_mask = tokenized["attention_mask"].squeeze().tolist()
            
            # For language modeling, labels are the same as input_ids
            labels = input_ids.copy()
            
            # Create dharmic scores dictionary
            dharmic_scores = quality_metrics.principle_scores
            
            # Detect tradition
            tradition_scores = self.quality_assessor.dharmic_analyzer.detect_wisdom_tradition(text)
            tradition = max(tradition_scores.items(), key=lambda x: x[1])[0] if tradition_scores else None
            
            # Create cultural context
            cultural_context = {
                "sensitivity_score": quality_metrics.cultural_sensitivity_score,
                "detected_traditions": quality_metrics.detected_traditions,
                "key_concepts": quality_metrics.key_concepts
            }
            
            return ProcessedDataSample(
                text=text,
                original_text=original_text,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                dharmic_scores=dharmic_scores,
                cultural_context=cultural_context,
                source=source,
                tradition=tradition,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            self.processing_logger.error(f"Failed to create processed sample: {e}")
            return None
    
    def _augment_sample(
        self, 
        text: str, 
        source: str, 
        quality_metrics: DataQualityMetrics
    ) -> List[ProcessedDataSample]:
        """Augment a sample with various techniques"""
        
        augmented_samples = []
        
        for strategy in self.config.data.augmentation_strategies:
            augmented_texts = self.augmentation_engine.augment_data_sample(text, strategy)
            
            for aug_text in augmented_texts:
                aug_sample = self._create_processed_sample(
                    aug_text, text, f"{source}_augmented_{strategy}", quality_metrics
                )
                if aug_sample:
                    augmented_samples.append(aug_sample)
        
        return augmented_samples
    
    def _create_dataset_splits(self, samples: List[ProcessedDataSample]) -> DatasetDict:
        """Create train/validation/test splits"""
        
        # Convert samples to dictionaries
        sample_dicts = [sample.to_dict() for sample in samples]
        
        # Create dataset
        full_dataset = Dataset.from_list(sample_dicts)
        
        # Split ratios
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        # Create splits
        train_test_split = full_dataset.train_test_split(train_size=train_size)
        train_dataset = train_test_split['train']
        temp_dataset = train_test_split['test']
        
        val_test_split = temp_dataset.train_test_split(train_size=val_size)
        val_dataset = val_test_split['train']
        test_dataset = val_test_split['test']
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        self.processing_logger.info(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return dataset_dict
    
    def _save_dataset(self, dataset_dict: DatasetDict, output_path: str):
        """Save processed dataset"""
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        dataset_dict.save_to_disk(str(output_dir))
        
        # Save processing configuration and statistics
        config_path = output_dir / "processing_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "config": self.config.__dict__,
                "processing_stats": self.processing_stats,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        self.processing_logger.info(f"Dataset saved to {output_path}")
    
    def _log_processing_stats(self):
        """Log final processing statistics"""
        
        stats = self.processing_stats
        acceptance_rate = stats['accepted_samples'] / stats['total_samples'] * 100
        augmentation_rate = stats['augmented_samples'] / stats['accepted_samples'] * 100 if stats['accepted_samples'] > 0 else 0
        
        self.processing_logger.info("Processing completed!")
        self.processing_logger.info(f"Total samples processed: {stats['total_samples']}")
        self.processing_logger.info(f"Accepted samples: {stats['accepted_samples']} ({acceptance_rate:.1f}%)")
        self.processing_logger.info(f"Rejected samples: {stats['rejected_samples']}")
        self.processing_logger.info(f"Augmented samples: {stats['augmented_samples']} ({augmentation_rate:.1f}%)")


# ===============================
# EXAMPLE USAGE
# ===============================

if __name__ == "__main__":
    from ..config.advanced_config import DharmaLLMConfigFactory
    
    # Create configuration
    config = DharmaLLMConfigFactory.create_config("development")
    
    # Initialize processor
    processor = DharmaLLMDataProcessor(config)
    
    # Example data sources
    data_sources = [
        {
            "name": "bhagavad_gita",
            "path": "data/raw/bhagavad_gita.txt",
            "type": "text"
        },
        {
            "name": "buddhist_teachings",
            "path": "data/raw/buddhist_teachings.json",
            "type": "json"
        }
    ]
    
    # Process data
    dataset = processor.process_raw_data(
        data_sources=data_sources,
        output_path="data/processed/dharma_dataset"
    )
    
    print("Data processing completed successfully!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
