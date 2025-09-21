"""
🕉️ AUTHENTIC SOURCE AUTHENTICATION SYSTEM
=============================================

Advanced system to ensure only genuine dharmic sources and prevent Western misconceptions
from contaminating the knowledge base. This system validates, filters, and enhances
all knowledge entries with traditional authenticity verification.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SourceAuthority(Enum):
    """Classification of source authority levels"""
    PRIMARY_SCRIPTURE = "primary_scripture"  # Vedas, Upanishads, Bhagavad Gita
    CLASSICAL_TEXT = "classical_text"       # Yoga Sutras, Ayurvedic texts
    TRADITIONAL_COMMENTARY = "traditional_commentary"  # Shankara, etc.
    MODERN_SAINT = "modern_saint"           # Ramana, Vivekananda
    ACADEMIC_SCHOLARLY = "academic_scholarly"  # University research
    COMMUNITY_ELDER = "community_elder"     # Traditional practitioners
    UNVERIFIED = "unverified"              # Needs verification
    CONTAMINATED = "contaminated"          # Western misconceptions

class CulturalAuthenticity(Enum):
    """Cultural integrity assessment"""
    PURE_TRADITIONAL = "pure_traditional"
    CULTURALLY_SENSITIVE = "culturally_sensitive"
    NEEDS_CONTEXT = "needs_context"
    WESTERN_INFLUENCE = "western_influence"
    APPROPRIATED = "appropriated"
    COMMERCIALIZED = "commercialized"

@dataclass
class AuthenticityScore:
    """Comprehensive authenticity assessment"""
    source_authority: SourceAuthority
    cultural_integrity: CulturalAuthenticity
    sanskrit_accuracy: float  # 0-1 score
    traditional_alignment: float  # 0-1 score
    misconception_risk: float  # 0-1 risk score
    certification_level: str  # GOLD, SILVER, BRONZE, NEEDS_REVIEW
    flags: List[str]
    recommendations: List[str]

class AuthenticSourceValidator:
    """Advanced authentication system for dharmic knowledge"""
    
    def __init__(self):
        self.authentic_sources = self._load_authentic_sources()
        self.contamination_patterns = self._load_contamination_patterns()
        self.sanskrit_terms = self._load_sanskrit_dictionary()
        self.cultural_markers = self._load_cultural_markers()
        
    def _load_authentic_sources(self) -> Dict[str, List[str]]:
        """Load database of authentic source texts"""
        return {
            "primary_scriptures": [
                "Rigveda", "Samaveda", "Yajurveda", "Atharvaveda",
                "Isha Upanishad", "Kena Upanishad", "Katha Upanishad",
                "Prashna Upanishad", "Mundaka Upanishad", "Mandukya Upanishad",
                "Taittiriya Upanishad", "Aitareya Upanishad", "Chandogya Upanishad",
                "Brihadaranyaka Upanishad", "Svetasvatara Upanishad",
                "Bhagavad Gita", "Brahma Sutras"
            ],
            "classical_texts": [
                "Yoga Sutras of Patanjali", "Hatha Yoga Pradipika",
                "Gheranda Samhita", "Shiva Samhita",
                "Charaka Samhita", "Sushruta Samhita", "Ashtanga Hridaya",
                "Ramayana", "Mahabharata", "Puranas"
            ],
            "traditional_commentaries": [
                "Adi Shankara", "Ramanuja", "Madhva", "Nimbarka",
                "Vallabha", "Chaitanya", "Sayana", "Uvata"
            ],
            "modern_saints": [
                "Ramana Maharshi", "Swami Vivekananda", "Sri Aurobindo",
                "Swami Sivananda", "Paramahansa Yogananda", "Nisargadatta Maharaj"
            ]
        }
    
    def _load_contamination_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that indicate Western contamination"""
        return {
            "new_age_terms": [
                "chakra balancing", "crystal healing", "energy work",
                "manifestation", "law of attraction", "twin flame",
                "lightworker", "starseed", "ascension"
            ],
            "westernized_concepts": [
                "yoga for weight loss", "meditation for success",
                "karma as fate", "yoga as exercise only",
                "hinduism polytheistic", "many gods worship"
            ],
            "commercial_spirituality": [
                "certified yoga teacher", "spiritual coach",
                "wellness guru", "mindfulness expert",
                "meditation app", "spiritual retreat business"
            ],
            "cultural_appropriation": [
                "namaste means hello", "om symbol decoration",
                "hindu mythology", "exotic eastern wisdom",
                "ancient secrets", "mystical powers"
            ],
            "non_hindu_mixing": [
                "external meditation in hinduism", "foreign practices as dharmic",
                "non-vedic influences", "imported concepts", "diluted traditions"
            ]
        }
    
    def _load_sanskrit_dictionary(self) -> Dict[str, Dict[str, str]]:
        """Load accurate Sanskrit terms with proper meanings"""
        return {
            "dharma": {
                "devanagari": "धर्म",
                "transliteration": "dharma",
                "meaning": "Natural law, righteous duty, cosmic order",
                "wrong_meaning": "religion",
                "context": "Individual and universal harmony principle"
            },
            "yoga": {
                "devanagari": "योग",
                "transliteration": "yoga", 
                "meaning": "Union of individual consciousness with universal consciousness",
                "wrong_meaning": "physical exercise",
                "context": "Complete spiritual science for Self-realization"
            },
            "karma": {
                "devanagari": "कर्म",
                "transliteration": "karma",
                "meaning": "Action and its inevitable consequence; law of cause and effect",
                "wrong_meaning": "fate or destiny",
                "context": "Free will operating within cosmic justice"
            },
            "moksha": {
                "devanagari": "मोक्ष",
                "transliteration": "moksha",
                "meaning": "Liberation from cycle of birth and death; Self-realization",
                "wrong_meaning": "enlightenment",
                "context": "Recognition of one's true nature as Brahman"
            }
        }
    
    def _load_cultural_markers(self) -> Dict[str, List[str]]:
        """Load cultural authenticity markers"""
        return {
            "authentic_markers": [
                "guru-disciple lineage", "traditional sampradaya",
                "sanskrit etymology", "scriptural reference",
                "cultural context", "traditional practice method",
                "acharya commentary", "sampradayic teaching"
            ],
            "respect_indicators": [
                "pranama to gurus", "acknowledgment of sources",
                "cultural sensitivity", "traditional attribution",
                "sanskrit pronunciation guide", "proper context"
            ],
            "contamination_signs": [
                "appropriated terms", "decontextualized practices",
                "commercial branding", "western reinterpretation",
                "mixed traditions", "cultural insensitivity"
            ]
        }
    
    def validate_knowledge_entry(self, knowledge_item: Dict[str, Any]) -> AuthenticityScore:
        """Comprehensive validation of knowledge entry authenticity"""
        
        text = knowledge_item.get("text", "")
        source = knowledge_item.get("source", "")
        title = knowledge_item.get("title", "")
        
        # Validate source authority
        source_authority = self._assess_source_authority(source)
        
        # Check cultural integrity  
        cultural_integrity = self._assess_cultural_integrity(text, title)
        
        # Validate Sanskrit accuracy
        sanskrit_accuracy = self._validate_sanskrit_terms(text)
        
        # Check traditional alignment
        traditional_alignment = self._assess_traditional_alignment(text, source)
        
        # Assess misconception risk
        misconception_risk = self._assess_misconception_risk(text)
        
        # Generate certification level
        certification_level = self._determine_certification_level(
            source_authority, cultural_integrity, sanskrit_accuracy, 
            traditional_alignment, misconception_risk
        )
        
        # Collect flags and recommendations
        flags = self._generate_flags(text, source, cultural_integrity, misconception_risk)
        recommendations = self._generate_recommendations(flags, certification_level)
        
        return AuthenticityScore(
            source_authority=source_authority,
            cultural_integrity=cultural_integrity,
            sanskrit_accuracy=sanskrit_accuracy,
            traditional_alignment=traditional_alignment,
            misconception_risk=misconception_risk,
            certification_level=certification_level,
            flags=flags,
            recommendations=recommendations
        )
    
    def _assess_source_authority(self, source: str) -> SourceAuthority:
        """Assess the authority level of the source"""
        source_lower = source.lower()
        
        # Check primary scriptures
        for scripture in self.authentic_sources["primary_scriptures"]:
            if scripture.lower() in source_lower:
                return SourceAuthority.PRIMARY_SCRIPTURE
        
        # Check classical texts
        for text in self.authentic_sources["classical_texts"]:
            if text.lower() in source_lower:
                return SourceAuthority.CLASSICAL_TEXT
        
        # Check traditional commentaries
        for commentator in self.authentic_sources["traditional_commentaries"]:
            if commentator.lower() in source_lower:
                return SourceAuthority.TRADITIONAL_COMMENTARY
        
        # Check modern saints
        for saint in self.authentic_sources["modern_saints"]:
            if saint.lower() in source_lower:
                return SourceAuthority.MODERN_SAINT
        
        # Default classifications
        if any(term in source_lower for term in ["university", "academic", "research"]):
            return SourceAuthority.ACADEMIC_SCHOLARLY
        elif any(term in source_lower for term in ["traditional", "elder", "guru"]):
            return SourceAuthority.COMMUNITY_ELDER
        else:
            return SourceAuthority.UNVERIFIED
    
    def _assess_cultural_integrity(self, text: str, title: str) -> CulturalAuthenticity:
        """Assess cultural authenticity and sensitivity"""
        combined_text = (text + " " + title).lower()
        
        # Check for contamination patterns
        contamination_score = 0
        for category, patterns in self.contamination_patterns.items():
            for pattern in patterns:
                if pattern.lower() in combined_text:
                    contamination_score += 1
        
        # Check for authentic markers
        authentic_score = 0
        for marker in self.cultural_markers["authentic_markers"]:
            if marker.lower() in combined_text:
                authentic_score += 1
        
        # Determine cultural authenticity
        if contamination_score > 3:
            return CulturalAuthenticity.APPROPRIATED
        elif contamination_score > 1:
            return CulturalAuthenticity.WESTERN_INFLUENCE
        elif authentic_score > 2:
            return CulturalAuthenticity.PURE_TRADITIONAL
        elif authentic_score > 0:
            return CulturalAuthenticity.CULTURALLY_SENSITIVE
        else:
            return CulturalAuthenticity.NEEDS_CONTEXT
    
    def _validate_sanskrit_terms(self, text: str) -> float:
        """Validate accuracy of Sanskrit terms used"""
        total_terms = 0
        accurate_terms = 0
        
        for term, details in self.sanskrit_terms.items():
            if term.lower() in text.lower():
                total_terms += 1
                # Check if used in proper context
                if details["wrong_meaning"].lower() not in text.lower():
                    accurate_terms += 1
        
        return accurate_terms / total_terms if total_terms > 0 else 1.0
    
    def _assess_traditional_alignment(self, text: str, source: str) -> float:
        """Assess alignment with traditional teachings"""
        alignment_score = 0.5  # Base score
        
        # Positive indicators
        if any(marker in text.lower() for marker in self.cultural_markers["authentic_markers"]):
            alignment_score += 0.3
        
        if any(marker in text.lower() for marker in self.cultural_markers["respect_indicators"]):
            alignment_score += 0.2
        
        # Negative indicators  
        if any(sign in text.lower() for sign in self.cultural_markers["contamination_signs"]):
            alignment_score -= 0.4
        
        return max(0.0, min(1.0, alignment_score))
    
    def _assess_misconception_risk(self, text: str) -> float:
        """Assess risk of perpetuating Western misconceptions"""
        risk_score = 0.0
        text_lower = text.lower()
        
        # Check each contamination category
        for category, patterns in self.contamination_patterns.items():
            category_risks = sum(1 for pattern in patterns if pattern.lower() in text_lower)
            risk_score += category_risks * 0.2
        
        return min(1.0, risk_score)
    
    def _determine_certification_level(self, source_authority: SourceAuthority,
                                     cultural_integrity: CulturalAuthenticity,
                                     sanskrit_accuracy: float,
                                     traditional_alignment: float,
                                     misconception_risk: float) -> str:
        """Determine certification level based on all factors"""
        
        # GOLD: Highest authenticity
        if (source_authority == SourceAuthority.PRIMARY_SCRIPTURE and
            cultural_integrity == CulturalAuthenticity.PURE_TRADITIONAL and
            sanskrit_accuracy >= 0.9 and traditional_alignment >= 0.8 and
            misconception_risk <= 0.1):
            return "GOLD"
        
        # SILVER: High authenticity
        elif (source_authority in [SourceAuthority.PRIMARY_SCRIPTURE, SourceAuthority.CLASSICAL_TEXT] and
              cultural_integrity in [CulturalAuthenticity.PURE_TRADITIONAL, CulturalAuthenticity.CULTURALLY_SENSITIVE] and
              sanskrit_accuracy >= 0.7 and traditional_alignment >= 0.6 and
              misconception_risk <= 0.3):
            return "SILVER"
        
        # BRONZE: Good authenticity
        elif (source_authority != SourceAuthority.CONTAMINATED and
              cultural_integrity != CulturalAuthenticity.APPROPRIATED and
              sanskrit_accuracy >= 0.5 and traditional_alignment >= 0.4 and
              misconception_risk <= 0.5):
            return "BRONZE"
        
        # NEEDS_REVIEW: Requires attention
        else:
            return "NEEDS_REVIEW"
    
    def _generate_flags(self, text: str, source: str, 
                       cultural_integrity: CulturalAuthenticity,
                       misconception_risk: float) -> List[str]:
        """Generate warning flags for potential issues"""
        flags = []
        text_lower = text.lower()
        
        # Cultural integrity flags
        if cultural_integrity == CulturalAuthenticity.APPROPRIATED:
            flags.append("CULTURAL_APPROPRIATION")
        elif cultural_integrity == CulturalAuthenticity.WESTERN_INFLUENCE:
            flags.append("WESTERN_CONTAMINATION")
        
        # Misconception risk flags
        if misconception_risk > 0.7:
            flags.append("HIGH_MISCONCEPTION_RISK")
        elif misconception_risk > 0.4:
            flags.append("MODERATE_MISCONCEPTION_RISK")
        
        # Specific pattern flags
        if any(pattern in text_lower for pattern in self.contamination_patterns["new_age_terms"]):
            flags.append("NEW_AGE_CONTAMINATION")
        
        if any(pattern in text_lower for pattern in self.contamination_patterns["commercial_spirituality"]):
            flags.append("COMMERCIALIZATION")
        
        if any(pattern in text_lower for pattern in self.contamination_patterns["non_hindu_mixing"]):
            flags.append("TRADITION_MIXING")
        
        return flags
    
    def _generate_recommendations(self, flags: List[str], certification_level: str) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if certification_level == "NEEDS_REVIEW":
            recommendations.append("Requires comprehensive review by traditional scholars")
        
        if "CULTURAL_APPROPRIATION" in flags:
            recommendations.append("Add proper cultural context and attribution")
            recommendations.append("Consult with traditional community elders")
        
        if "WESTERN_CONTAMINATION" in flags:
            recommendations.append("Remove Western interpretations and restore traditional meaning")
            recommendations.append("Add Sanskrit etymology and traditional context")
        
        if "NEW_AGE_CONTAMINATION" in flags:
            recommendations.append("Replace New Age terminology with authentic Sanskrit terms")
        
        if "COMMERCIALIZATION" in flags:
            recommendations.append("Remove commercial aspects and focus on traditional transmission")
        
        if "TRADITION_MIXING" in flags:
            recommendations.append("Separate traditions and maintain pure dharmic content")
        
        return recommendations

class KnowledgeBaseAuthenticator:
    """Main system for authenticating and enhancing knowledge base"""
    
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.validator = AuthenticSourceValidator()
        self.authentication_results = {}
    
    def authenticate_entire_knowledge_base(self) -> Dict[str, Any]:
        """Authenticate all entries in the knowledge base"""
        logger.info("Starting comprehensive knowledge base authentication...")
        
        results = {
            "total_entries": 0,
            "certification_levels": {"GOLD": 0, "SILVER": 0, "BRONZE": 0, "NEEDS_REVIEW": 0},
            "flags_summary": {},
            "overall_authenticity": 0.0,
            "detailed_results": {}
        }
        
        # Process specific knowledge files 
        knowledge_files = ["sanatan_wisdom.json", "sanatan_practices.json", "sanatan_guidance.json"]
        
        for filename in knowledge_files:
            json_file = self.knowledge_base_path / filename
            if json_file.exists():
                logger.info(f"Authenticating {filename}...")
                file_results = self._authenticate_file(json_file)
                results["detailed_results"][filename] = file_results
                
                # Update summary statistics
                for entry_result in file_results["entries"].values():
                    results["total_entries"] += 1
                    cert_level = entry_result["certification_level"]
                    results["certification_levels"][cert_level] += 1
                    
                    for flag in entry_result["flags"]:
                        results["flags_summary"][flag] = results["flags_summary"].get(flag, 0) + 1
            else:
                logger.warning(f"File {filename} not found")
        
        # Calculate overall authenticity score
        total = results["total_entries"]
        if total > 0:
            authenticity_score = (
                results["certification_levels"]["GOLD"] * 1.0 +
                results["certification_levels"]["SILVER"] * 0.8 +
                results["certification_levels"]["BRONZE"] * 0.6 +
                results["certification_levels"]["NEEDS_REVIEW"] * 0.2
            ) / total
            results["overall_authenticity"] = authenticity_score
        
        logger.info(f"Authentication complete. Overall authenticity: {results['overall_authenticity']:.2f}")
        return results
    
    def _authenticate_file(self, json_file: Path) -> Dict[str, Any]:
        """Authenticate all entries in a single JSON file"""
        file_results = {"entries": {}, "file_summary": {}}
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Processing {json_file.name} with {len(data)} categories")
            
            # Process nested structure
            for category_key, category_data in data.items():
                if isinstance(category_data, dict):
                    logger.info(f"Processing category '{category_key}' with {len(category_data)} items")
                    for item_id, item_data in category_data.items():
                        if isinstance(item_data, dict):
                            authenticity_score = self.validator.validate_knowledge_entry(item_data)
                            
                            file_results["entries"][item_id] = {
                                "source_authority": authenticity_score.source_authority.value,
                                "cultural_integrity": authenticity_score.cultural_integrity.value,
                                "sanskrit_accuracy": authenticity_score.sanskrit_accuracy,
                                "traditional_alignment": authenticity_score.traditional_alignment,
                                "misconception_risk": authenticity_score.misconception_risk,
                                "certification_level": authenticity_score.certification_level,
                                "flags": authenticity_score.flags,
                                "recommendations": authenticity_score.recommendations
                            }
                        else:
                            logger.warning(f"Skipping non-dict item: {item_id}")
                else:
                    logger.warning(f"Skipping non-dict category: {category_key}")
        
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            file_results["error"] = str(e)
        
        return file_results
    
    def generate_authenticity_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive authenticity report"""
        report = f"""
🕉️ KNOWLEDGE BASE AUTHENTICITY REPORT
=====================================

📊 OVERALL STATISTICS:
- Total Entries: {results['total_entries']}
- Overall Authenticity Score: {results['overall_authenticity']:.1%}

🏆 CERTIFICATION LEVELS:
- 🥇 GOLD (Highest): {results['certification_levels']['GOLD']} entries
- 🥈 SILVER (High): {results['certification_levels']['SILVER']} entries  
- 🥉 BRONZE (Good): {results['certification_levels']['BRONZE']} entries
- ⚠️ NEEDS REVIEW: {results['certification_levels']['NEEDS_REVIEW']} entries

🚩 FLAGS SUMMARY:
"""
        
        for flag, count in results['flags_summary'].items():
            report += f"- {flag}: {count} occurrences\n"
        
        report += f"""
📋 DETAILED ANALYSIS:
"""
        
        for filename, file_data in results['detailed_results'].items():
            report += f"\n📁 {filename}:\n"
            
            cert_counts = {"GOLD": 0, "SILVER": 0, "BRONZE": 0, "NEEDS_REVIEW": 0}
            for entry_data in file_data["entries"].values():
                cert_counts[entry_data["certification_level"]] += 1
            
            report += f"  🥇 GOLD: {cert_counts['GOLD']}, 🥈 SILVER: {cert_counts['SILVER']}, 🥉 BRONZE: {cert_counts['BRONZE']}, ⚠️ NEEDS REVIEW: {cert_counts['NEEDS_REVIEW']}\n"
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize authenticator with current directory
    authenticator = KnowledgeBaseAuthenticator(".")
    
    # Run comprehensive authentication
    results = authenticator.authenticate_entire_knowledge_base()
    
    # Generate report
    report = authenticator.generate_authenticity_report(results)
    print(report)
    
    # Save results
    with open("knowledge_base_authenticity_report.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open("KNOWLEDGE_BASE_AUTHENTICITY_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("Authentication complete. Reports saved.")
