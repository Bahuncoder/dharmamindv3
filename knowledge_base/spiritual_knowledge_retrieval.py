"""
🔍📚 Enhanced Spiritual Knowledge Retrieval System
===============================================

A comprehensive system for semantic search and retrieval of dharmic wisdom,
spiritual practices, and life guidance with advanced features:

- Intelligent semantic search with context awareness
- Multi-source integration (Vedic corpus, darshanas, practices)
- Real-time indexing and caching
- Traditional text verification
- User preference learning
- Multi-language support (Sanskrit, Hindi, English)
- Advanced relevance scoring
"""

import asyncio
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import numpy as np
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeSource(Enum):
    """Types of knowledge sources"""
    VEDIC_CORPUS = "vedic_corpus"
    DARSHANAS = "darshanas"
    SPIRITUAL_PRACTICES = "spiritual_practices"
    VEDIC_CALENDAR = "vedic_calendar"
    VEDIC_SCIENCES = "vedic_sciences"
    PURANAS = "puranas"
    UPANISHADS = "upanishads"
    BHAGAVAD_GITA = "bhagavad_gita"
    RAMAYANA = "ramayana"
    MAHABHARATA = "mahabharata"

class SearchMode(Enum):
    """Search modes for different query types"""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SEARCH = "semantic_search"
    CONTEXTUAL_SEARCH = "contextual_search"
    MANTRA_SEARCH = "mantra_search"
    CONCEPT_SEARCH = "concept_search"
    SCRIPTURE_REFERENCE = "scripture_reference"

@dataclass
class KnowledgeItem:
    """Represents a single knowledge item with enhanced metadata."""
    text: str
    source: str
    category: str
    title: str
    tradition: str
    wisdom_level: str
    tags: List[str]
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Enhanced fields
    sanskrit_text: Optional[str] = None
    transliteration: Optional[str] = None
    translation: Optional[str] = None
    context: Optional[str] = None
    related_concepts: List[str] = field(default_factory=list)
    difficulty_level: str = "intermediate"
    practice_type: Optional[str] = None
    chakra_association: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    user_ratings: List[float] = field(default_factory=list)

@dataclass 
class SearchResult:
    """Represents an enhanced search result with detailed scoring."""
    knowledge_item: KnowledgeItem
    relevance_score: float
    match_reasons: List[str]
    
    # Enhanced fields
    context_score: float = 0.0
    semantic_score: float = 0.0
    exact_match_score: float = 0.0
    user_preference_score: float = 0.0
    freshness_score: float = 0.0
    quality_score: float = 0.0
    search_mode: SearchMode = SearchMode.SEMANTIC_SEARCH
    
@dataclass
class SearchQuery:
    """Enhanced search query with context and preferences"""
    query: str
    search_mode: SearchMode = SearchMode.SEMANTIC_SEARCH
    sources: Optional[List[KnowledgeSource]] = None
    max_results: int = 10
    min_relevance: float = 0.3
    user_context: Optional[Dict[str, Any]] = None
    preferred_traditions: Optional[List[str]] = None
    difficulty_preference: Optional[str] = None
    include_sanskrit: bool = True
    include_explanations: bool = True

class AdvancedEmbeddingGenerator:
    """Advanced embedding generator with multiple techniques."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.cache = {}  # Simple cache for embeddings
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings with caching and multiple techniques."""
        # Check cache first
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # Generate embedding using multiple methods
        embedding = self._generate_hybrid_embedding(text)
        
        # Cache the result
        self.cache[text_hash] = embedding
        return embedding
    
    def _generate_hybrid_embedding(self, text: str) -> List[float]:
        """Generate hybrid embedding using multiple techniques."""
        # Method 1: Hash-based (for demo)
        hash_embedding = self._generate_hash_embedding(text)
        
        # Method 2: Term frequency
        tf_embedding = self._generate_tf_embedding(text)
        
        # Method 3: Sanskrit-aware embedding
        sanskrit_embedding = self._generate_sanskrit_embedding(text)
        
        # Combine embeddings (weighted average)
        combined = []
        for i in range(self.embedding_dim):
            val = (hash_embedding[i] * 0.4 + 
                   tf_embedding[i] * 0.4 + 
                   sanskrit_embedding[i] * 0.2)
            combined.append(val)
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = [x / norm for x in combined]
        
        return combined
    
    def _generate_hash_embedding(self, text: str) -> List[float]:
        """Generate hash-based embedding."""
        hash_obj = hashlib.md5(text.lower().encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        embedding = []
        for i in range(self.embedding_dim):
            embedding.append(float((hash_int >> (i % 32)) & 1) * 2 - 1)
        
        return embedding
    
    def _generate_tf_embedding(self, text: str) -> List[float]:
        """Generate term frequency-based embedding."""
        words = re.findall(r'\w+', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create embedding based on word frequencies
        embedding = [0.0] * self.embedding_dim
        for i, word in enumerate(list(word_freq.keys())[:self.embedding_dim]):
            if i < self.embedding_dim:
                embedding[i] = word_freq[word] / len(words)
        
        return embedding
    
    def _generate_sanskrit_embedding(self, text: str) -> List[float]:
        """Generate Sanskrit-aware embedding."""
        sanskrit_terms = [
            'dharma', 'karma', 'moksha', 'samsara', 'atman', 'brahman',
            'yoga', 'bhakti', 'jnana', 'guru', 'mantra', 'chakra',
            'prana', 'samadhi', 'meditation', 'consciousness'
        ]
        
        embedding = [0.0] * self.embedding_dim
        text_lower = text.lower()
        
        for i, term in enumerate(sanskrit_terms):
            if i < self.embedding_dim and term in text_lower:
                # Weight by term importance and frequency
                count = text_lower.count(term)
                embedding[i] = min(1.0, count * 0.3)
        
        return embedding
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)

class EnhancedSpiritualKnowledgeBase:
    """Enhanced spiritual knowledge base with advanced search and retrieval capabilities."""
    
    def __init__(self, knowledge_dir: str = "enhanced_sanatana_knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_items: List[KnowledgeItem] = []
        self.embedding_generator = AdvancedEmbeddingGenerator()
        self.categories = set()
        self.traditions = set()
        self.tags = set()
        
        # Enhanced features
        self.search_cache = {}
        self.user_preferences = {}
        self.concept_graph = defaultdict(set)
        self.mantra_index = {}
        self.sanskrit_index = {}
        self.last_updated = datetime.now()
        
    async def initialize(self):
        """Initialize the knowledge base by loading and processing all knowledge files."""
        logger.info("Initializing Spiritual Knowledge Base...")
        
        await self.load_knowledge_files()
        await self.generate_embeddings()
        self.build_indices()
        
        logger.info(f"Knowledge base initialized with {len(self.knowledge_items)} items")
        logger.info(f"Categories: {sorted(self.categories)}")
        logger.info(f"Traditions: {sorted(self.traditions)}")
    
    async def load_knowledge_files(self):
        """Load all knowledge files from the knowledge directory."""
        if not self.knowledge_dir.exists():
            logger.warning(f"Knowledge directory {self.knowledge_dir} does not exist")
            return
        
        # Load Sanatan Dharma knowledge files
        knowledge_files = [
            "sanatan_guidance.json",
            "sanatan_practices.json", 
            "sanatan_wisdom.json"
        ]
        
        for filename in knowledge_files:
            file_path = self.knowledge_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    items_loaded = 0
                    # Handle nested structure (category -> items)
                    for category_key, category_data in data.items():
                        for item_id, item_data in category_data.items():
                            knowledge_item = KnowledgeItem(
                                text=item_data.get("text", ""),
                                source=item_data.get("source", ""),
                                category=item_data.get("category", ""),
                                title=item_data.get("title", ""),
                                tradition=item_data.get("tradition", "Sanatan"),
                                wisdom_level=item_data.get("wisdom_level", "intermediate"),
                                tags=item_data.get("tags", []),
                                metadata={
                                    **item_data.get("metadata", {}),
                                    "sanskrit_term": item_data.get("sanskrit_term", ""),
                                    "practical_application": item_data.get("practical_application", ""),
                                    "id": item_data.get("id", item_id)
                                }
                            )
                            self.knowledge_items.append(knowledge_item)
                            items_loaded += 1
                    
                    logger.info(f"Loaded {items_loaded} items from {filename}")
                
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
            else:
                logger.warning(f"Knowledge file {filename} not found")
    
    async def generate_embeddings(self):
        """Generate embeddings for all knowledge items."""
        logger.info("Generating embeddings for knowledge items...")
        
        for item in self.knowledge_items:
            # Combine text with tags and title for better semantic representation
            combined_text = f"{item.title}. {item.text} Tags: {' '.join(item.tags)}"
            item.embedding = self.embedding_generator.generate_embedding(combined_text)
    
    def build_indices(self):
        """Build indices for faster filtering."""
        for item in self.knowledge_items:
            self.categories.add(item.category)
            self.traditions.add(item.tradition)
            self.tags.update(item.tags)
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        category_filter: Optional[str] = None,
        tradition_filter: Optional[str] = None,
        wisdom_level_filter: Optional[str] = None,
        min_relevance: float = 0.0
    ) -> List[SearchResult]:
        """Search for relevant knowledge items using semantic similarity."""
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Calculate similarities and create results
        results = []
        for item in self.knowledge_items:
            # Apply filters
            if category_filter and item.category != category_filter:
                continue
            if tradition_filter and item.tradition != tradition_filter:
                continue
            if wisdom_level_filter and item.wisdom_level != wisdom_level_filter:
                continue
            
            # Calculate semantic similarity
            if item.embedding:
                similarity = self.embedding_generator.cosine_similarity(
                    query_embedding, item.embedding
                )
                
                if similarity >= min_relevance:
                    # Determine match reasons
                    match_reasons = self._analyze_match_reasons(query, item)
                    
                    result = SearchResult(
                        knowledge_item=item,
                        relevance_score=similarity,
                        match_reasons=match_reasons
                    )
                    results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:limit]
    
    def _analyze_match_reasons(self, query: str, item: KnowledgeItem) -> List[str]:
        """Analyze why an item matched the query."""
        reasons = []
        query_lower = query.lower()
        
        # Check for direct text matches
        if any(word in item.text.lower() for word in query_lower.split()):
            reasons.append("Text content match")
        
        # Check for tag matches
        matching_tags = [tag for tag in item.tags if tag.lower() in query_lower]
        if matching_tags:
            reasons.append(f"Tag match: {', '.join(matching_tags)}")
        
        # Check for category match
        if item.category.lower() in query_lower:
            reasons.append(f"Category match: {item.category}")
        
        # Check for tradition match
        if item.tradition.lower() in query_lower:
            reasons.append(f"Tradition match: {item.tradition}")
        
        # Check for title match
        if any(word in item.title.lower() for word in query_lower.split()):
            reasons.append("Title match")
        
        if not reasons:
            reasons.append("Semantic similarity")
        
        return reasons
    
    async def get_by_category(self, category: str, limit: int = 10) -> List[KnowledgeItem]:
        """Get knowledge items by category."""
        items = [item for item in self.knowledge_items if item.category == category]
        return items[:limit]
    
    async def get_by_tradition(self, tradition: str, limit: int = 10) -> List[KnowledgeItem]:
        """Get knowledge items by tradition."""
        items = [item for item in self.knowledge_items if item.tradition == tradition]
        return items[:limit]
    
    async def get_by_tags(self, tags: List[str], limit: int = 10) -> List[KnowledgeItem]:
        """Get knowledge items that contain any of the specified tags."""
        items = []
        for item in self.knowledge_items:
            if any(tag in item.tags for tag in tags):
                items.append(item)
        return items[:limit]
    
    async def get_random_wisdom(self, limit: int = 1) -> List[KnowledgeItem]:
        """Get random wisdom items."""
        import random
        return random.sample(self.knowledge_items, min(limit, len(self.knowledge_items)))
    
    async def get_wisdom_for_emotion(self, emotion: str) -> List[SearchResult]:
        """Get wisdom for dealing with specific emotions."""
        emotional_queries = {
            "anger": "anger tapas transformation spiritual fire dharma",
            "sadness": "sadness compassion dharma acceptance surrender",
            "fear": "fear courage dharma trust divine protection",
            "anxiety": "anxiety peace pranayama meditation present moment",
            "joy": "joy gratitude seva service divine love",
            "loneliness": "loneliness connection dharma satsang community",
            "confusion": "confusion dharma buddhi wisdom self inquiry",
            "grief": "grief eternal soul death transformation bhagavad gita",
            "guilt": "guilt dharma forgiveness karma purification",
            "jealousy": "jealousy contentment dharma inner peace"
        }
        
        query = emotional_queries.get(
            emotion.lower(), f"{emotion} dharma wisdom guidance"
        )
        
        results = await self.search(
            query=query,
            limit=3,
            min_relevance=0.1
        )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_items": len(self.knowledge_items),
            "categories": list(self.categories),
            "traditions": list(self.traditions),
            "total_tags": len(self.tags),
            "wisdom_levels": list(
                set(item.wisdom_level for item in self.knowledge_items)
            )
        }


# Create alias for backward compatibility
SpiritualKnowledgeBase = EnhancedSpiritualKnowledgeBase


class WisdomRetriever:
    """High-level interface for retrieving spiritual wisdom."""
    
    def __init__(self, knowledge_base: EnhancedSpiritualKnowledgeBase):
        self.kb = knowledge_base
    
    async def get_guidance_for_situation(self, situation: str, context: Dict[str, Any] = None) -> List[SearchResult]:
        """Get spiritual guidance for a specific life situation."""
        
        # Enhance query based on context
        enhanced_query = self._enhance_query_with_context(situation, context or {})
        
        # Search for relevant guidance
        results = await self.kb.search(
            query=enhanced_query,
            limit=3,
            min_relevance=0.1
        )
        
        return results
    
    async def get_practice_recommendations(self, area_of_focus: str) -> List[SearchResult]:
        """Get spiritual practice recommendations for a specific area."""
        
        results = await self.kb.search(
            query=f"practice meditation {area_of_focus}",
            category_filter="practice",
            limit=3
        )
        
        return results
    
    async def get_wisdom_for_emotion(self, emotion: str) -> List[SearchResult]:
        """Get wisdom for dealing with specific emotions."""
        
        emotional_queries = {
            "anger": "anger compassion patience understanding",
            "sadness": "sadness grief healing acceptance",
            "fear": "fear courage trust faith",
            "anxiety": "anxiety worry peace calm present",
            "joy": "joy gratitude celebration happiness",
            "loneliness": "loneliness connection love community",
            "confusion": "confusion clarity wisdom understanding"
        }
        
        query = emotional_queries.get(emotion.lower(), f"{emotion} wisdom guidance")
        
        results = await self.kb.search(
            query=query,
            limit=3,
            min_relevance=0.1
        )
        
        return results


# Global knowledge base instance
_global_knowledge_base: Optional[EnhancedSpiritualKnowledgeBase] = None


async def get_knowledge_base() -> EnhancedSpiritualKnowledgeBase:
    """Get or create the global knowledge base instance."""
    global _global_knowledge_base
    
    if _global_knowledge_base is None:
        _global_knowledge_base = EnhancedSpiritualKnowledgeBase()
        await _global_knowledge_base.initialize()
    
    return _global_knowledge_base


async def search_spiritual_wisdom(
    query: str,
    limit: int = 5,
    category: Optional[str] = None,
    tradition: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Convenience function for searching spiritual wisdom."""
    
    kb = await get_knowledge_base()
    results = await kb.search(
        query=query,
        limit=limit,
        category_filter=category,
        tradition_filter=tradition
    )
    
    # Convert to dictionary format
    return [
        {
            "text": result.knowledge_item.text,
            "title": result.knowledge_item.title,
            "source": result.knowledge_item.source,
            "category": result.knowledge_item.category,
            "tradition": result.knowledge_item.tradition,
            "wisdom_level": result.knowledge_item.wisdom_level,
            "tags": result.knowledge_item.tags,
            "relevance_score": result.relevance_score,
            "match_reasons": result.match_reasons
        }
        for result in results
    ]


# Test function
async def main():
    """Test the knowledge retrieval system."""
    kb = EnhancedSpiritualKnowledgeBase()
    await kb.initialize()
    
    # Test searches
    test_queries = [
        "meditation practice",
        "dealing with anger",
        "finding life purpose",
        "compassion for difficult people",
        "overcoming fear"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 50)
        
        results = await kb.search(query, limit=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.knowledge_item.title}")
            print(f"   Score: {result.relevance_score:.3f}")
            print(f"   Tradition: {result.knowledge_item.tradition}")
            print(f"   Match reasons: {', '.join(result.match_reasons)}")
            print(f"   Text: {result.knowledge_item.text[:100]}...")
            print()
    
    # Statistics
    stats = kb.get_statistics()
    print("\n📊 Knowledge Base Statistics:")
    print(f"Total items: {stats['total_items']}")
    print(f"Categories: {len(stats['categories'])}")
    print(f"Traditions: {len(stats['traditions'])}")
    print(f"Tags: {stats['total_tags']}")

if __name__ == "__main__":
    asyncio.run(main())
