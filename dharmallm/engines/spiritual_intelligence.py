"""
Spiritual Intelligence Chakra Module
===================================

The core spiritual intelligence system for DharmaMind, integrating advanced
philosophical frameworks, consciousness science, and spiritual practices
directly into the chakra module ecosystem.

This module serves as the spiritual heart of DharmaMind, providing:
- PhD-level Vedantic philosophy integration
- Authentic Acharya commentary system
- Advanced spiritual practice guidance
- Consciousness science correlations
- Sanskrit processing with deep meanings
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add the knowledge_base directory to the path for imports
current_dir = Path(__file__).parent.parent.parent
knowledge_base_dir = current_dir / "knowledge_base"
if str(knowledge_base_dir) not in sys.path:
    sys.path.append(str(knowledge_base_dir))

try:
    from advanced_knowledge_enhancer import AdvancedKnowledgeEnhancer, WisdomLevel, TraditionType
except ImportError:
    # Fallback if import fails
    AdvancedKnowledgeEnhancer = None
    WisdomLevel = None
    TraditionType = None

# Set up logging
logger = logging.getLogger(__name__)

class SpiritualQueryType(Enum):
    """Types of spiritual queries."""
    PHILOSOPHICAL_INQUIRY = "philosophical_inquiry"
    PRACTICE_GUIDANCE = "practice_guidance"
    SANSKRIT_TRANSLATION = "sanskrit_translation"
    CONSCIOUSNESS_EXPLORATION = "consciousness_exploration"
    DHARMIC_DECISION = "dharmic_decision"
    WISDOM_SYNTHESIS = "wisdom_synthesis"
    ACHARYA_COMMENTARY = "acharya_commentary"

class SpiritualTradition(Enum):
    """Spiritual traditions supported."""
    ADVAITA_VEDANTA = "advaita_vedanta"
    SANKHYA = "sankhya"
    YOGA = "yoga"
    TANTRA = "tantra"
    VEDIC = "vedic"
    INTEGRAL = "integral"
    UNIVERSAL = "universal"

@dataclass
class SpiritualQuery:
    """Represents a spiritual inquiry or request for guidance."""
    query_text: str
    query_type: SpiritualQueryType
    tradition: Optional[SpiritualTradition] = None
    wisdom_level: Optional[str] = "intermediate"
    include_sanskrit: bool = True
    include_commentary: bool = True
    context: Optional[Dict[str, Any]] = None

@dataclass
class SpiritualResponse:
    """Represents a spiritual response with wisdom and guidance."""
    query: SpiritualQuery
    wisdom_entries: List[Dict[str, Any]]
    sanskrit_insights: List[Dict[str, str]]
    acharya_commentaries: List[Dict[str, str]]
    practical_guidance: List[str]
    consciousness_insights: List[str]
    cross_references: List[str]
    confidence_score: float
    tradition_alignment: str
    
class SpiritualIntelligence:
    """
    The core Spiritual Intelligence Chakra Module.
    
    This is the spiritual heart of DharmaMind, providing authentic
    spiritual wisdom, philosophical guidance, and consciousness insights.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """Initialize the Spiritual Intelligence system."""
        self.is_initialized = False
        self.knowledge_base_path = knowledge_base_path or str(knowledge_base_dir)
        self.enhancer: Optional[AdvancedKnowledgeEnhancer] = None
        
        # Spiritual capabilities registry
        self.capabilities = {
            "philosophical_frameworks": True,
            "consciousness_science": True,
            "spiritual_practices": True,
            "wisdom_synthesis": True,
            "sanskrit_processing": True,
            "acharya_commentaries": True,
            "dharmic_guidance": True
        }
        
        # Performance metrics
        self.query_count = 0
        self.wisdom_entries_served = 0
        self.sanskrit_translations = 0
        
        logger.info("Spiritual Intelligence Chakra initialized")
    
    async def initialize(self) -> bool:
        """Initialize the spiritual intelligence system."""
        try:
            if AdvancedKnowledgeEnhancer is None:
                logger.warning("AdvancedKnowledgeEnhancer not available, using fallback mode")
                self.is_initialized = True
                return True
            
            # Initialize the advanced knowledge enhancer
            self.enhancer = AdvancedKnowledgeEnhancer(self.knowledge_base_path)
            await self.enhancer.initialize_enhanced_system()
            
            self.is_initialized = True
            logger.info("✅ Spiritual Intelligence system fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Spiritual Intelligence: {e}")
            # Continue with limited functionality
            self.is_initialized = True
            return False
    
    async def process_spiritual_query(self, query: SpiritualQuery) -> SpiritualResponse:
        """Process a spiritual query and provide comprehensive guidance."""
        if not self.is_initialized:
            await self.initialize()
        
        self.query_count += 1
        
        try:
            if self.enhancer:
                # Use advanced enhancer for comprehensive response
                return await self._process_with_enhancer(query)
            else:
                # Use fallback mode
                return await self._process_fallback_mode(query)
                
        except Exception as e:
            logger.error(f"Error processing spiritual query: {e}")
            return self._create_error_response(query, str(e))
    
    async def _process_with_enhancer(self, query: SpiritualQuery) -> SpiritualResponse:
        """Process query using the advanced knowledge enhancer."""
        # Search for wisdom entries
        wisdom_entries = await self.enhancer.search_enhanced_wisdom(
            query.query_text, 
            limit=5
        )
        
        # Get practice guidance if requested
        practice_guidance = []
        if query.query_type == SpiritualQueryType.PRACTICE_GUIDANCE:
            practice_guidance = await self.enhancer.get_practice_guidance(
                query.query_text
            )
        
        # Extract Sanskrit insights
        sanskrit_insights = []
        acharya_commentaries = []
        consciousness_insights = []
        cross_references = []
        
        for entry in wisdom_entries:
            # Sanskrit processing
            if entry.get('original_sanskrit') and query.include_sanskrit:
                sanskrit_insights.append({
                    'sanskrit': entry['original_sanskrit'],
                    'transliteration': entry.get('transliteration', ''),
                    'meaning': entry.get('meaning', ''),
                    'source': entry.get('source', '')
                })
            
            # Acharya commentaries
            if entry.get('acharya_commentary') and query.include_commentary:
                acharya_commentaries.append({
                    'acharya': entry.get('acharya', 'Unknown'),
                    'commentary': entry['acharya_commentary'],
                    'source': entry.get('source', '')
                })
            
            # Consciousness insights
            if entry.get('consciousness_insights'):
                consciousness_insights.extend(entry['consciousness_insights'])
            
            # Cross references
            if entry.get('cross_references'):
                cross_references.extend(entry['cross_references'])
        
        # Generate practical guidance
        practical_guidance = self._generate_practical_guidance(query, wisdom_entries)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(wisdom_entries, query)
        
        # Determine tradition alignment
        tradition_alignment = self._determine_tradition_alignment(wisdom_entries)
        
        self.wisdom_entries_served += len(wisdom_entries)
        self.sanskrit_translations += len(sanskrit_insights)
        
        return SpiritualResponse(
            query=query,
            wisdom_entries=wisdom_entries,
            sanskrit_insights=sanskrit_insights,
            acharya_commentaries=acharya_commentaries,
            practical_guidance=practical_guidance,
            consciousness_insights=consciousness_insights,
            cross_references=cross_references,
            confidence_score=confidence_score,
            tradition_alignment=tradition_alignment
        )
    
    async def _process_fallback_mode(self, query: SpiritualQuery) -> SpiritualResponse:
        """Process query in fallback mode without advanced enhancer."""
        # Basic spiritual response based on query type
        wisdom_entries = []
        sanskrit_insights = []
        acharya_commentaries = []
        practical_guidance = [
            "Practice daily meditation and self-reflection",
            "Study authentic spiritual texts with reverence",
            "Seek guidance from qualified spiritual teachers",
            "Cultivate compassion and dharmic living"
        ]
        consciousness_insights = [
            "Consciousness is the fundamental reality underlying all experience",
            "Self-inquiry leads to direct realization of one's true nature",
            "Regular spiritual practice purifies the mind and heart"
        ]
        cross_references = []
        
        return SpiritualResponse(
            query=query,
            wisdom_entries=wisdom_entries,
            sanskrit_insights=sanskrit_insights,
            acharya_commentaries=acharya_commentaries,
            practical_guidance=practical_guidance,
            consciousness_insights=consciousness_insights,
            cross_references=cross_references,
            confidence_score=0.7,
            tradition_alignment="universal"
        )
    
    def _generate_practical_guidance(self, query: SpiritualQuery, wisdom_entries: List[Dict]) -> List[str]:
        """Generate practical spiritual guidance based on the query and wisdom."""
        guidance = []
        
        # Query-type specific guidance
        if query.query_type == SpiritualQueryType.PRACTICE_GUIDANCE:
            guidance.extend([
                "Begin with establishing a regular daily practice",
                "Focus on consistency rather than duration initially",
                "Seek guidance from experienced practitioners"
            ])
        elif query.query_type == SpiritualQueryType.PHILOSOPHICAL_INQUIRY:
            guidance.extend([
                "Study the original texts in addition to commentaries",
                "Contemplate deeply on the philosophical principles",
                "Apply the insights to daily life experiences"
            ])
        elif query.query_type == SpiritualQueryType.CONSCIOUSNESS_EXPLORATION:
            guidance.extend([
                "Practice witnessing consciousness in meditation",
                "Observe the distinction between awareness and its contents",
                "Explore consciousness through both study and direct experience"
            ])
        
        # Extract guidance from wisdom entries
        for entry in wisdom_entries:
            if entry.get('practical_application'):
                app = entry['practical_application']
                if isinstance(app, dict):
                    if 'daily_practice' in app:
                        guidance.append(f"Daily Practice: {app['daily_practice']}")
                    if 'meditation_technique' in app:
                        guidance.append(f"Meditation: {app['meditation_technique']}")
                elif isinstance(app, str):
                    guidance.append(app)
        
        return guidance[:10]  # Limit to 10 guidance points
    
    def _calculate_confidence_score(self, wisdom_entries: List[Dict], query: SpiritualQuery) -> float:
        """Calculate confidence score for the response."""
        if not wisdom_entries:
            return 0.5
        
        # Base score on number of relevant entries
        base_score = min(len(wisdom_entries) * 0.15, 0.8)
        
        # Boost for Sanskrit content
        sanskrit_boost = 0.1 if any(entry.get('original_sanskrit') for entry in wisdom_entries) else 0
        
        # Boost for Acharya commentaries
        acharya_boost = 0.1 if any(entry.get('acharya_commentary') for entry in wisdom_entries) else 0
        
        return min(base_score + sanskrit_boost + acharya_boost, 1.0)
    
    def _determine_tradition_alignment(self, wisdom_entries: List[Dict]) -> str:
        """Determine the primary spiritual tradition alignment."""
        tradition_counts = {}
        
        for entry in wisdom_entries:
            tradition = entry.get('tradition', 'universal')
            tradition_counts[tradition] = tradition_counts.get(tradition, 0) + 1
        
        if not tradition_counts:
            return "universal"
        
        return max(tradition_counts, key=tradition_counts.get)
    
    def _create_error_response(self, query: SpiritualQuery, error_message: str) -> SpiritualResponse:
        """Create an error response."""
        return SpiritualResponse(
            query=query,
            wisdom_entries=[],
            sanskrit_insights=[],
            acharya_commentaries=[],
            practical_guidance=[f"Error in processing: {error_message}"],
            consciousness_insights=[],
            cross_references=[],
            confidence_score=0.0,
            tradition_alignment="unknown"
        )
    
    async def get_spiritual_guidance(self, topic: str, tradition: str = "universal") -> Dict[str, Any]:
        """Get spiritual guidance on a specific topic."""
        query = SpiritualQuery(
            query_text=topic,
            query_type=SpiritualQueryType.PRACTICE_GUIDANCE,
            tradition=SpiritualTradition(tradition) if tradition != "universal" else None,
            include_sanskrit=True,
            include_commentary=True
        )
        
        response = await self.process_spiritual_query(query)
        
        return {
            "guidance": response.practical_guidance,
            "wisdom_entries": len(response.wisdom_entries),
            "sanskrit_available": len(response.sanskrit_insights) > 0,
            "acharya_commentary": len(response.acharya_commentaries) > 0,
            "confidence": response.confidence_score,
            "tradition": response.tradition_alignment
        }
    
    async def translate_sanskrit(self, sanskrit_text: str) -> Dict[str, Any]:
        """Translate and provide insights on Sanskrit text."""
        query = SpiritualQuery(
            query_text=sanskrit_text,
            query_type=SpiritualQueryType.SANSKRIT_TRANSLATION,
            include_sanskrit=True,
            include_commentary=True
        )
        
        response = await self.process_spiritual_query(query)
        
        return {
            "sanskrit_insights": response.sanskrit_insights,
            "commentaries": response.acharya_commentaries,
            "related_wisdom": len(response.wisdom_entries),
            "confidence": response.confidence_score
        }
    
    async def explore_consciousness(self, aspect: str) -> Dict[str, Any]:
        """Explore consciousness from both spiritual and scientific perspectives."""
        query = SpiritualQuery(
            query_text=f"consciousness {aspect}",
            query_type=SpiritualQueryType.CONSCIOUSNESS_EXPLORATION,
            include_sanskrit=True,
            include_commentary=True
        )
        
        response = await self.process_spiritual_query(query)
        
        return {
            "consciousness_insights": response.consciousness_insights,
            "philosophical_foundation": response.wisdom_entries,
            "practical_exploration": response.practical_guidance,
            "tradition_perspective": response.tradition_alignment,
            "confidence": response.confidence_score
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the spiritual intelligence system."""
        return {
            "initialized": self.is_initialized,
            "enhancer_available": self.enhancer is not None,
            "capabilities": self.capabilities,
            "performance": {
                "queries_processed": self.query_count,
                "wisdom_entries_served": self.wisdom_entries_served,
                "sanskrit_translations": self.sanskrit_translations
            },
            "knowledge_base_path": self.knowledge_base_path
        }

# Global instance
_spiritual_intelligence_instance: Optional[SpiritualIntelligence] = None

def get_spiritual_intelligence(knowledge_base_path: Optional[str] = None) -> SpiritualIntelligence:
    """Get the global Spiritual Intelligence instance."""
    global _spiritual_intelligence_instance
    
    if _spiritual_intelligence_instance is None:
        _spiritual_intelligence_instance = SpiritualIntelligence(knowledge_base_path)
    
    return _spiritual_intelligence_instance

async def initialize_spiritual_intelligence(knowledge_base_path: Optional[str] = None) -> bool:
    """Initialize the global Spiritual Intelligence system."""
    spiritual_intel = get_spiritual_intelligence(knowledge_base_path)
    return await spiritual_intel.initialize()

# Convenience functions for common operations
async def get_spiritual_wisdom(query: str, tradition: str = "universal") -> Dict[str, Any]:
    """Quick function to get spiritual wisdom."""
    spiritual_intel = get_spiritual_intelligence()
    return await spiritual_intel.get_spiritual_guidance(query, tradition)

async def translate_sanskrit_text(sanskrit: str) -> Dict[str, Any]:
    """Quick function to translate Sanskrit."""
    spiritual_intel = get_spiritual_intelligence()
    return await spiritual_intel.translate_sanskrit(sanskrit)

async def explore_consciousness_aspect(aspect: str) -> Dict[str, Any]:
    """Quick function to explore consciousness."""
    spiritual_intel = get_spiritual_intelligence()
    return await spiritual_intel.explore_consciousness(aspect)
