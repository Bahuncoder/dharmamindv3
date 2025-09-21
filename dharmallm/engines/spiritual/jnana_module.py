"""
🕉️ Jnana Module - Knowledge and Wisdom Processing Center

This module handles all knowledge acquisition, wisdom processing,
and intellectual understanding within the DharmaMind system.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """Types of knowledge processed by Jnana Module"""
    SCRIPTURAL = "scriptural"  # Vedas, Upanishads, etc.
    PHILOSOPHICAL = "philosophical"  # Darshana systems
    PRACTICAL = "practical"  # Daily dharmic practices
    EXPERIENTIAL = "experiential"  # Personal spiritual insights
    HISTORICAL = "historical"  # Hindu history and traditions

class WisdomLevel(Enum):
    """Levels of wisdom understanding"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    REALIZED = "realized"

class JnanaModule:
    """
    Jnana Module - The Knowledge and Wisdom Center
    
    This module processes all forms of knowledge acquisition,
    understanding, and wisdom integration within the DharmaMind system.
    """
    
    def __init__(self):
        self.name = "Jnana Module"
        self.element = "Akasha (Space)"
        self.color = "Deep Indigo"
        self.mantra = "OM GYAN NAMAHA"
        self.knowledge_base = {}
        self.wisdom_insights = []
        
    def process_knowledge(self, content: str, knowledge_type: KnowledgeType) -> Dict[str, Any]:
        """Process and categorize knowledge input"""
        try:
            processed_knowledge = {
                "content": content,
                "type": knowledge_type.value,
                "timestamp": datetime.now().isoformat(),
                "insights": self._extract_insights(content),
                "references": self._find_references(content),
                "wisdom_level": self._assess_wisdom_level(content)
            }
            
            # Store in knowledge base
            knowledge_id = f"jnana_{len(self.knowledge_base)}"
            self.knowledge_base[knowledge_id] = processed_knowledge
            
            logger.info(f"Processed {knowledge_type.value} knowledge: {knowledge_id}")
            return processed_knowledge
            
        except Exception as e:
            logger.error(f"Error processing knowledge: {e}")
            return {"error": str(e)}
    
    def _extract_insights(self, content: str) -> List[str]:
        """Extract key insights from content"""
        # Simplified insight extraction
        insights = []
        keywords = ["dharma", "karma", "moksha", "atman", "brahman", "yoga", "meditation"]
        
        for keyword in keywords:
            if keyword.lower() in content.lower():
                insights.append(f"Contains teachings about {keyword}")
        
        return insights
    
    def _find_references(self, content: str) -> List[str]:
        """Find scriptural and philosophical references"""
        references = []
        scriptures = ["Bhagavad Gita", "Upanishads", "Vedas", "Ramayana", "Mahabharata"]
        
        for scripture in scriptures:
            if scripture.lower() in content.lower():
                references.append(scripture)
        
        return references
    
    def _assess_wisdom_level(self, content: str) -> str:
        """Assess the wisdom level of the content"""
        # Simplified assessment based on complexity indicators
        complexity_indicators = [
            "consciousness", "self-realization", "non-duality", 
            "transcendence", "enlightenment", "samadhi"
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators 
                            if indicator.lower() in content.lower())
        
        if indicator_count >= 3:
            return WisdomLevel.REALIZED.value
        elif indicator_count >= 2:
            return WisdomLevel.ADVANCED.value
        elif indicator_count >= 1:
            return WisdomLevel.INTERMEDIATE.value
        else:
            return WisdomLevel.BASIC.value
    
    def seek_knowledge(self, query: str) -> Dict[str, Any]:
        """Search for knowledge based on query"""
        results = []
        
        for knowledge_id, knowledge in self.knowledge_base.items():
            if query.lower() in knowledge["content"].lower():
                results.append({
                    "id": knowledge_id,
                    "content": knowledge["content"][:200] + "...",
                    "type": knowledge["type"],
                    "wisdom_level": knowledge["wisdom_level"],
                    "relevance_score": self._calculate_relevance(query, knowledge["content"])
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "query": query,
            "results": results[:5],  # Top 5 results
            "total_found": len(results)
        }
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_wisdom_summary(self) -> Dict[str, Any]:
        """Get summary of accumulated wisdom"""
        wisdom_counts = {}
        knowledge_counts = {}
        
        for knowledge in self.knowledge_base.values():
            # Count by wisdom level
            level = knowledge["wisdom_level"]
            wisdom_counts[level] = wisdom_counts.get(level, 0) + 1
            
            # Count by knowledge type
            k_type = knowledge["type"]
            knowledge_counts[k_type] = knowledge_counts.get(k_type, 0) + 1
        
        return {
            "total_knowledge_entries": len(self.knowledge_base),
            "wisdom_level_distribution": wisdom_counts,
            "knowledge_type_distribution": knowledge_counts,
            "chakra_status": "Active and Processing",
            "last_update": datetime.now().isoformat()
        }

# Factory function for module integration
def create_jnana_module() -> JnanaModule:
    """Create and return a Jnana Module instance"""
    return JnanaModule()

# Global instance
_jnana_module = None

def get_jnana_module() -> JnanaModule:
    """Get global Jnana module instance"""
    global _jnana_module
    if _jnana_module is None:
        _jnana_module = JnanaModule()
    return _jnana_module
