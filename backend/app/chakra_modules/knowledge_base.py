"""
Knowledge Base - Integrated into DharmaMind Backend
==================================================

Central repository for Hindu wisdom, concepts, and universal spiritual knowledge.
This module manages the storage, retrieval, and organization of philosophical concepts,
scriptural references, and cultural knowledge from diverse traditions.

Enhanced with Semantic Search and Vector-based Knowledge Retrieval.
"""

import asyncio
import logging
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import aiosqlite
import sys
import os

# Add knowledge_base directory to path for imports
knowledge_base_dir = Path(__file__).parent.parent.parent.parent / "knowledge_base"
sys.path.append(str(knowledge_base_dir))

try:
    from spiritual_knowledge_retrieval import (
        SpiritualKnowledgeBase, 
        WisdomRetriever, 
        SearchResult,
        search_spiritual_wisdom
    )
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Semantic search not available: {e}")
    SEMANTIC_SEARCH_AVAILABLE = False

try:
    from advanced_knowledge_enhancer import (
        AdvancedKnowledgeEnhancer,
        WisdomLevel,
        TraditionType,
        EnhancedWisdomEntry
    )
    ADVANCED_KNOWLEDGE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced knowledge system not available: {e}")
    ADVANCED_KNOWLEDGE_AVAILABLE = False

class KnowledgeBase:
    """
    Central knowledge repository for universal wisdom and spiritual knowledge
    
    This class manages the storage, retrieval, and organization of spiritual
    concepts, scriptural references, and universal wisdom from diverse traditions.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "./data/dharma_knowledge.db"
        self.logger = self._setup_logging()
        self.connection = None
        self.is_initialized = False
        
        # Initialize enhanced knowledge system
        if ADVANCED_KNOWLEDGE_AVAILABLE:
            self.advanced_enhancer = AdvancedKnowledgeEnhancer()
            self.logger.info("🚀 Advanced Knowledge Enhancement System loaded")
        else:
            self.advanced_enhancer = None
            self.logger.warning("⚠️ Advanced Knowledge Enhancement not available")
        
        # Ensure the data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for knowledge base"""
        logger = logging.getLogger("KnowledgeBase")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize the knowledge base"""
        try:
            self.logger.info("Initializing Knowledge Base...")
            
            # Create database connection (async)
            self.connection = await aiosqlite.connect(self.db_path)
            self.connection.row_factory = aiosqlite.Row
            
            # Create tables
            await self._create_tables()
            
            # Load initial knowledge
            await self._load_initial_knowledge()
            
            self.is_initialized = True
            self.logger.info("Knowledge Base initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Knowledge Base: {str(e)}")
            return False
    
    async def _create_tables(self):
        """Create database tables"""
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                sanskrit_term TEXT,
                definition TEXT,
                category TEXT,
                tradition TEXT,
                importance_level INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS scriptures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT,
                content TEXT,
                scripture_type TEXT,
                tradition TEXT,
                chapter TEXT,
                verse TEXT,
                translation TEXT,
                commentary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS wisdom (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theme TEXT NOT NULL,
                teaching TEXT,
                source TEXT,
                tradition TEXT,
                context TEXT,
                practical_application TEXT,
                relevance_score REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS practices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                tradition TEXT,
                category TEXT,
                difficulty_level INTEGER DEFAULT 1,
                duration_minutes INTEGER,
                instructions TEXT,
                benefits TEXT,
                precautions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table in tables:
            await self.connection.execute(table)
        
        await self.connection.commit()
    
    async def _load_initial_knowledge(self):
        """Load authentic Hindu scripture knowledge into the database"""
        
        # Check if authentic Hindu data already exists
        cursor = await self.connection.execute("SELECT COUNT(*) FROM concepts WHERE tradition = 'Hindu'")
        result = await cursor.fetchone()
        if result[0] > 0:
            self.logger.info("Authentic Hindu knowledge base already loaded")
            return
        
        self.logger.info("🕉️ Loading ONLY authentic Hindu scripture knowledge...")
        self.logger.info("📚 Sources: Vedas, Upanishads, Puranas, Gita, Mahabharata, Ramayana")
        
        try:
            # Import and run the authentic Hindu knowledge loader
            from pathlib import Path
            import sys
            
            # Add current directory to path
            current_dir = Path(__file__).parent.parent.parent.parent
            sys.path.append(str(current_dir))
            
            from authentic_hindu_knowledge_loader import load_authentic_hindu_knowledge
            
            # Load all authentic Hindu scriptures
            knowledge_base_dir = current_dir / "knowledge_base"
            db_path = self.db_path
            
            report = await load_authentic_hindu_knowledge(
                str(knowledge_base_dir), 
                db_path
            )
            
            self.logger.info("✅ Authentic Hindu scriptures loaded successfully!")
            self.logger.info(f"📚 Total entries: {report['loading_report']['total_entries_loaded']}")
            self.logger.info(f"🎯 Authenticity: {report['loading_report']['authenticity_level']}")
            self.logger.info(f"✅ Sanskrit coverage: {report['authenticity_verification']['sanskrit_coverage_percentage']:.1f}%")
            self.logger.info("🕉️ ALL KNOWLEDGE IS NOW FROM ORIGINAL HINDU TEXTS ONLY!")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading authentic Hindu knowledge: {e}")
            # Fallback to basic Hindu concepts only (no Buddhist, Taoist, etc.)
            await self._load_basic_hindu_concepts()
    
    async def _load_basic_hindu_concepts(self):
        """Fallback: Load basic authentic Hindu concepts only"""
        
        self.logger.info("Loading basic authentic Hindu concepts as fallback...")
        
        # ONLY authentic Hindu concepts from scriptures
        authentic_hindu_concepts = [
            ("Dharma", "धर्म", "Righteous duty from Mahabharata", "Philosophy", "Hindu", 10),
            ("Karma", "कर्म", "Action principle from Bhagavad Gita", "Philosophy", "Hindu", 10),
            ("Moksha", "मोक्ष", "Liberation goal from Upanishads", "Philosophy", "Hindu", 10),
            ("Ahimsa", "अहिंसा", "Non-violence from Vedas", "Ethics", "Hindu", 10),
            ("Yoga", "योग", "Union from Yoga Sutras", "Practice", "Hindu", 10),
            ("Dhyana", "ध्यान", "Meditation from Upanishads", "Practice", "Hindu", 10),
            ("Atman", "आत्मन्", "Soul from Upanishads", "Philosophy", "Hindu", 10),
            ("Brahman", "ब्रह्मन्", "Ultimate Reality from Upanishads", "Philosophy", "Hindu", 10),
            ("Om", "ॐ", "Sacred sound from Vedas", "Practice", "Hindu", 10),
            ("Tapas", "तपस्", "Austerity from Vedas", "Practice", "Hindu", 9)
        ]
        
        # Authentic Hindu wisdom from scriptures only
        authentic_wisdom = [
            ("Dharmic Action", "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन", "Bhagavad Gita 2.47", "Hindu", 
             "Right to action only", "Act without attachment to results", 1.0),
            ("Self-Knowledge", "तत्त्वमसि", "Chandogya Upanishad", "Hindu", 
             "Thou art That", "Realize your true nature as Brahman", 1.0),
            ("Unity Truth", "एकं सद्विप्रा बहुधा वदन्ति", "Rig Veda 1.164.46", "Hindu", 
             "Truth is one, sages call it by many names", "See unity in diversity", 1.0),
            ("Peace Mantra", "ॐ शान्तिः शान्तिः शान्तिः", "Upanishads", "Hindu", 
             "Peace in all realms", "Chant for inner and outer peace", 1.0),
            ("Divine Protection", "धर्मो रक्षति रक्षितः", "Mahabharata", "Hindu", 
             "Dharma protects those who protect it", "Uphold righteousness always", 1.0)
        ]
        
        # Authentic Hindu practices from scriptures
        authentic_practices = [
            ("Gayatri Mantra", "Sacred Vedic mantra for enlightenment", "Hindu", 
             "Mantra", 1, 10, "ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यम्", 
             "Purifies mind and awakens wisdom", "Chant with reverence"),
            ("Pranayama", "Vedic breathing from Yoga Sutras", "Hindu", 
             "Breathing", 2, 15, "Control of prana through breath regulation", 
             "Calms mind, energizes body", "Learn from qualified guru"),
            ("Meditation on Om", "Sacred sound meditation from Upanishads", "Hindu", 
             "Meditation", 1, 20, "Focus on the sound and vibration of Om", 
             "Connects to cosmic consciousness", "Practice regularly"),
            ("Sandhya Vandana", "Vedic twilight worship", "Hindu", 
             "Ritual", 3, 30, "Traditional prayers at sunrise and sunset", 
             "Spiritual purification and divine connection", "Learn proper procedures"),
            ("Japa Meditation", "Repetitive chanting from Puranas", "Hindu", 
             "Chanting", 1, 15, "Repeat divine names or mantras", 
             "Purifies heart and focuses mind", "Choose appropriate mantra")
        ]
        
        # Insert only authentic Hindu knowledge
        await self.connection.executemany(
            "INSERT INTO concepts (name, sanskrit_term, definition, category, tradition, importance_level) VALUES (?, ?, ?, ?, ?, ?)",
            authentic_hindu_concepts
        )
        
        await self.connection.executemany(
            "INSERT INTO wisdom (theme, teaching, source, tradition, context, practical_application, relevance_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            authentic_wisdom
        )
        
        await self.connection.executemany(
            "INSERT INTO practices (name, description, tradition, category, difficulty_level, duration_minutes, instructions, benefits, precautions) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            authentic_practices
        )
        
        await self.connection.commit()
        self.logger.info(f"✅ Loaded {len(authentic_hindu_concepts)} authentic Hindu concepts, {len(authentic_wisdom)} scriptures, {len(authentic_practices)} practices")
        self.logger.info("🕉️ All knowledge is from original Hindu texts: Vedas, Upanishads, Puranas, Gita, Mahabharata, Ramayana")
    
    async def search_concepts(self, query: str, tradition: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for concepts matching the query"""
        
        if not self.is_initialized:
            await self.initialize()
        
        base_query = """
            SELECT * FROM concepts 
            WHERE name LIKE ? OR sanskrit_term LIKE ? OR definition LIKE ?
        """
        
        params = [f"%{query}%", f"%{query}%", f"%{query}%"]
        
        if tradition:
            base_query += " AND tradition = ?"
            params.append(tradition)
        
        base_query += " ORDER BY importance_level DESC LIMIT ?"
        params.append(limit)
        
        cursor = await self.connection.execute(base_query, params)
        results = []
        async for row in cursor:
            results.append(dict(row))
        
        return results
    
    async def get_relevant_wisdom(self, context: str, tradition: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get wisdom relevant to the given context"""
        
        if not self.is_initialized:
            await self.initialize()
        
        base_query = """
            SELECT * FROM wisdom 
            WHERE theme LIKE ? OR context LIKE ? OR teaching LIKE ?
        """
        
        params = [f"%{context}%", f"%{context}%", f"%{context}%"]
        
        if tradition:
            base_query += " AND tradition = ?"
            params.append(tradition)
        
        base_query += " ORDER BY relevance_score DESC LIMIT ?"
        params.append(limit)
        
        cursor = await self.connection.execute(base_query, params)
        results = []
        async for row in cursor:
            results.append(dict(row))
        
        return results
    
    async def get_practices(self, category: Optional[str] = None, difficulty: Optional[int] = None, 
                          tradition: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get spiritual practices based on criteria"""
        
        if not self.is_initialized:
            await self.initialize()
        
        base_query = "SELECT * FROM practices WHERE 1=1"
        params = []
        
        if category:
            base_query += " AND category = ?"
            params.append(category)
        
        if difficulty:
            base_query += " AND difficulty_level <= ?"
            params.append(difficulty)
        
        if tradition:
            base_query += " AND tradition = ?"
            params.append(tradition)
        
        base_query += " ORDER BY difficulty_level ASC LIMIT ?"
        params.append(limit)
        
        cursor = await self.connection.execute(base_query, params)
        results = []
        async for row in cursor:
            results.append(dict(row))
        
        return results
    
    async def add_concept(self, name: str, sanskrit_term: str = "", definition: str = "", 
                         category: str = "", tradition: str = "", importance_level: int = 5) -> bool:
        """Add a new concept to the knowledge base"""
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            await self.connection.execute("""
                INSERT INTO concepts (name, sanskrit_term, definition, category, tradition, importance_level)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, sanskrit_term, definition, category, tradition, importance_level))
            
            await self.connection.commit()
            self.logger.info(f"Added concept: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding concept: {str(e)}")
            return False
    
    async def add_wisdom(self, theme: str, teaching: str, source: str = "", tradition: str = "",
                        context: str = "", practical_application: str = "", relevance_score: float = 0.5) -> bool:
        """Add wisdom teaching to the knowledge base"""
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            await self.connection.execute("""
                INSERT INTO wisdom (theme, teaching, source, tradition, context, practical_application, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (theme, teaching, source, tradition, context, practical_application, relevance_score))
            
            await self.connection.commit()
            self.logger.info(f"Added wisdom teaching: {theme}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding wisdom: {str(e)}")
            return False
    
    async def get_scripture_reference(self, title: str, tradition: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get scripture reference by title"""
        
        if not self.is_initialized:
            await self.initialize()
        
        base_query = "SELECT * FROM scriptures WHERE title LIKE ?"
        params = [f"%{title}%"]
        
        if tradition:
            base_query += " AND tradition = ?"
            params.append(tradition)
        
        base_query += " LIMIT 1"
        
        cursor = await self.connection.execute(base_query, params)
        row = await cursor.fetchone()
        return dict(row) if row else None
    
    async def get_concept_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific concept by name"""
        
        if not self.is_initialized:
            await self.initialize()
        
        cursor = await self.connection.execute(
            "SELECT * FROM concepts WHERE name = ? OR sanskrit_term = ? LIMIT 1", 
            (name, name)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    
    async def get_traditions(self) -> List[str]:
        """Get list of all traditions in the knowledge base"""
        
        if not self.is_initialized:
            await self.initialize()
        
        cursor = await self.connection.execute(
            "SELECT DISTINCT tradition FROM concepts WHERE tradition IS NOT NULL AND tradition != ''"
        )
        traditions = []
        async for row in cursor:
            traditions.append(row[0])
        
        return sorted(traditions)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get knowledge base status"""
        
        status = {
            "initialized": self.is_initialized,
            "database_path": self.db_path,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.is_initialized and self.connection:
            # Get counts
            cursor = await self.connection.execute("SELECT COUNT(*) FROM concepts")
            result = await cursor.fetchone()
            status["concept_count"] = result[0]
            
            cursor = await self.connection.execute("SELECT COUNT(*) FROM scriptures")
            result = await cursor.fetchone()
            status["scripture_count"] = result[0]
            
            cursor = await self.connection.execute("SELECT COUNT(*) FROM wisdom")
            result = await cursor.fetchone()
            status["wisdom_count"] = result[0]
            
            cursor = await self.connection.execute("SELECT COUNT(*) FROM practices")
            result = await cursor.fetchone()
            status["practice_count"] = result[0]
            
            # Get tradition distribution
            cursor = await self.connection.execute(
                "SELECT tradition, COUNT(*) FROM concepts WHERE tradition IS NOT NULL GROUP BY tradition"
            )
            traditions = {}
            async for row in cursor:
                traditions[row[0]] = row[1]
            status["traditions"] = traditions
        
        return status
    
    # ===== SEMANTIC SEARCH ENHANCEMENT =====
    
    async def search_wisdom_semantically(
        self, 
        query: str, 
        limit: int = 5,
        category: Optional[str] = None,
        tradition: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for wisdom using semantic similarity.
        Falls back to traditional keyword search if semantic search unavailable.
        """
        self.logger.info(f"Searching wisdom semantically: '{query}'")
        
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                # Use advanced semantic search
                results = await search_spiritual_wisdom(
                    query=query,
                    limit=limit,
                    category=category,
                    tradition=tradition
                )
                
                self.logger.info(f"Found {len(results)} semantic results for: '{query}'")
                return results
                
            except Exception as e:
                self.logger.warning(f"Semantic search failed, falling back to keyword search: {e}")
        
        # Fallback to traditional keyword search
        return await self._keyword_search_wisdom(query, limit, category, tradition)
    
    async def _keyword_search_wisdom(
        self, 
        query: str, 
        limit: int = 5,
        category: Optional[str] = None,
        tradition: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Traditional keyword-based wisdom search."""
        if not self.is_initialized:
            return []
        
        # Build search query with filters
        where_conditions = ["(teaching LIKE ? OR theme LIKE ? OR source LIKE ?)"]
        params = [f"%{query}%", f"%{query}%", f"%{query}%"]
        
        if category:
            where_conditions.append("theme = ?")
            params.append(category)
        
        if tradition:
            where_conditions.append("tradition = ?")
            params.append(tradition)
        
        sql = f"""
            SELECT theme, teaching, source, tradition, context, practical_application, relevance_score
            FROM wisdom 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY relevance_score DESC
            LIMIT ?
        """
        params.append(limit)
        
        try:
            cursor = await self.connection.execute(sql, params)
            results = []
            async for row in cursor:
                results.append({
                    "title": row[0],  # theme
                    "text": row[1],   # teaching
                    "source": row[2],
                    "tradition": row[3],
                    "category": row[0],  # theme as category
                    "wisdom_level": "traditional",
                    "tags": [],
                    "relevance_score": row[6] or 0.5,
                    "match_reasons": ["Keyword match"],
                    "context": row[4],
                    "practical_application": row[5]
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            return []
    
    async def get_guidance_for_situation(
        self, 
        situation: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get spiritual guidance for a specific life situation."""
        self.logger.info(f"Getting guidance for situation: '{situation}'")
        
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                # Import here to avoid circular imports
                kb = SpiritualKnowledgeBase()
                await kb.initialize()
                
                retriever = WisdomRetriever(kb)
                results = await retriever.get_guidance_for_situation(situation, context)
                
                # Convert to dictionary format
                return [
                    {
                        "title": result.knowledge_item.title,
                        "text": result.knowledge_item.text,
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
                
            except Exception as e:
                self.logger.warning(f"Semantic guidance search failed: {e}")
        
        # Fallback to keyword search
        return await self._keyword_search_wisdom(situation, limit=3)
    
    async def get_practice_recommendations(
        self, 
        area_of_focus: str,
        difficulty_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get spiritual practice recommendations."""
        self.logger.info(f"Getting practice recommendations for: '{area_of_focus}'")
        
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                kb = SpiritualKnowledgeBase()
                await kb.initialize()
                
                retriever = WisdomRetriever(kb)
                results = await retriever.get_practice_recommendations(area_of_focus)
                
                return [
                    {
                        "title": result.knowledge_item.title,
                        "text": result.knowledge_item.text,
                        "source": result.knowledge_item.source,
                        "category": result.knowledge_item.category,
                        "tradition": result.knowledge_item.tradition,
                        "wisdom_level": result.knowledge_item.wisdom_level,
                        "tags": result.knowledge_item.tags,
                        "relevance_score": result.relevance_score,
                        "practice_type": result.knowledge_item.metadata.get("practice_type", "unknown"),
                        "duration": result.knowledge_item.metadata.get("duration", "flexible")
                    }
                    for result in results
                ]
                
            except Exception as e:
                self.logger.warning(f"Semantic practice search failed: {e}")
        
        # Fallback to database search
        return await self._search_practices_database(area_of_focus, difficulty_level)
    
    async def _search_practices_database(
        self, 
        area_of_focus: str, 
        difficulty_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search practices in the database."""
        if not self.is_initialized:
            return []
        
        where_conditions = ["(name LIKE ? OR description LIKE ? OR category LIKE ?)"]
        params = [f"%{area_of_focus}%", f"%{area_of_focus}%", f"%{area_of_focus}%"]
        
        if difficulty_level:
            difficulty_map = {"beginner": 1, "intermediate": 2, "advanced": 3}
            if difficulty_level.lower() in difficulty_map:
                where_conditions.append("difficulty_level <= ?")
                params.append(difficulty_map[difficulty_level.lower()])
        
        sql = f"""
            SELECT name, description, tradition, category, difficulty_level, 
                   duration_minutes, instructions, benefits
            FROM practices 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY difficulty_level ASC
            LIMIT 3
        """
        
        try:
            cursor = await self.connection.execute(sql, params)
            results = []
            async for row in cursor:
                results.append({
                    "title": row[0],  # name
                    "text": row[1],   # description
                    "source": "database",
                    "category": row[3],  # category
                    "tradition": row[2],
                    "wisdom_level": "practical",
                    "tags": [row[3]] if row[3] else [],
                    "relevance_score": 0.7,
                    "match_reasons": ["Database match"],
                    "practice_type": row[3],
                    "duration": f"{row[5]}_minutes" if row[5] else "flexible",
                    "instructions": row[6],
                    "benefits": row[7]
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Database practice search failed: {e}")
            return []
    
    async def get_wisdom_for_emotion(self, emotion: str) -> List[Dict[str, Any]]:
        """Get wisdom for dealing with specific emotions."""
        self.logger.info(f"Getting wisdom for emotion: '{emotion}'")
        
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                kb = SpiritualKnowledgeBase()
                await kb.initialize()
                
                retriever = WisdomRetriever(kb)
                results = await retriever.get_wisdom_for_emotion(emotion)
                
                return [
                    {
                        "title": result.knowledge_item.title,
                        "text": result.knowledge_item.text,
                        "source": result.knowledge_item.source,
                        "category": result.knowledge_item.category,
                        "tradition": result.knowledge_item.tradition,
                        "wisdom_level": result.knowledge_item.wisdom_level,
                        "tags": result.knowledge_item.tags,
                        "relevance_score": result.relevance_score,
                        "match_reasons": result.match_reasons,
                        "emotion_context": emotion
                    }
                    for result in results
                ]
                
            except Exception as e:
                self.logger.warning(f"Semantic emotion search failed: {e}")
        
        # Fallback search
        emotion_keywords = {
            "anger": "anger patience compassion understanding",
            "sadness": "sadness grief healing acceptance sorrow",
            "fear": "fear courage trust faith confidence",
            "anxiety": "anxiety worry peace calm present moment",
            "joy": "joy gratitude celebration happiness bliss",
            "loneliness": "loneliness connection love community friendship",
            "confusion": "confusion clarity wisdom understanding insight"
        }
        
        search_term = emotion_keywords.get(emotion.lower(), emotion)
        return await self._keyword_search_wisdom(search_term, limit=3)
    
    # Enhanced Knowledge System Methods
    async def initialize_enhanced_knowledge(self) -> bool:
        """Initialize the enhanced knowledge system"""
        if not self.advanced_enhancer:
            self.logger.warning("Advanced knowledge enhancer not available")
            return False
        
        try:
            success = await self.advanced_enhancer.initialize_enhanced_system()
            if success:
                self.logger.info("✅ Enhanced knowledge system initialized successfully")
            return success
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced knowledge: {e}")
            return False
    
    async def search_advanced_wisdom(self, 
                                   query: str,
                                   wisdom_level: Optional[str] = None,
                                   tradition: Optional[str] = None,
                                   consciousness_level: Optional[str] = None,
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """Search enhanced wisdom with advanced filtering"""
        if not self.advanced_enhancer:
            self.logger.warning("Advanced knowledge enhancer not available")
            return await self._keyword_search_wisdom(query, limit)
        
        try:
            # Convert string parameters to enum values if needed
            wisdom_level_enum = None
            tradition_enum = None
            
            if wisdom_level and ADVANCED_KNOWLEDGE_AVAILABLE:
                try:
                    wisdom_level_enum = WisdomLevel(wisdom_level.lower())
                except ValueError:
                    pass
            
            if tradition and ADVANCED_KNOWLEDGE_AVAILABLE:
                try:
                    tradition_enum = TraditionType(tradition.lower())
                except ValueError:
                    pass
            
            results = await self.advanced_enhancer.search_enhanced_wisdom(
                query=query,
                wisdom_level=wisdom_level_enum,
                tradition=tradition_enum,
                limit=limit
            )
            
            self.logger.info(f"Found {len(results)} advanced wisdom entries for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Advanced wisdom search failed: {e}")
            return await self._keyword_search_wisdom(query, limit)
    
    async def get_practice_guidance(self, practice_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed practice guidance with progression information"""
        if not self.advanced_enhancer:
            return None
        
        try:
            progression = await self.advanced_enhancer.get_practice_progression(practice_id)
            if progression:
                self.logger.info(f"Retrieved practice guidance for: {practice_id}")
                return progression
        except Exception as e:
            self.logger.error(f"Failed to get practice guidance: {e}")
        
        return None
    
    async def get_wisdom_connections(self, wisdom_id: str) -> List[Dict[str, Any]]:
        """Get connected wisdom entries for deeper understanding"""
        if not self.advanced_enhancer:
            return []
        
        try:
            connections = await self.advanced_enhancer.get_wisdom_connections(wisdom_id)
            self.logger.info(f"Found {len(connections)} connected wisdom entries for: {wisdom_id}")
            return connections
        except Exception as e:
            self.logger.error(f"Failed to get wisdom connections: {e}")
            return []
    
    async def get_philosophical_framework(self, tradition: str, level: str = "intermediate") -> List[Dict[str, Any]]:
        """Get philosophical framework for specific tradition and level"""
        if not self.advanced_enhancer:
            return []
        
        try:
            results = await self.advanced_enhancer.search_enhanced_wisdom(
                query="philosophical framework",
                tradition=TraditionType(tradition.lower()) if ADVANCED_KNOWLEDGE_AVAILABLE else None,
                wisdom_level=WisdomLevel(level.lower()) if ADVANCED_KNOWLEDGE_AVAILABLE else None,
                limit=20
            )
            
            self.logger.info(f"Retrieved {len(results)} philosophical framework entries for {tradition}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get philosophical framework: {e}")
            return []
    
    async def get_consciousness_studies_integration(self, topic: str) -> List[Dict[str, Any]]:
        """Get integration between ancient wisdom and modern consciousness studies"""
        if not self.advanced_enhancer:
            return []
        
        try:
            results = await self.advanced_enhancer.search_enhanced_wisdom(
                query=f"consciousness {topic}",
                limit=15
            )
            
            # Filter for entries that have scientific correlates
            consciousness_studies = []
            for result in results:
                if result.get('scientific_correlates') and result['scientific_correlates']:
                    consciousness_studies.append(result)
            
            self.logger.info(f"Found {len(consciousness_studies)} consciousness studies integrations for: {topic}")
            return consciousness_studies
            
        except Exception as e:
            self.logger.error(f"Failed to get consciousness studies integration: {e}")
            return []
    
    async def get_dharmic_ai_guidance(self, context: str) -> List[Dict[str, Any]]:
        """Get dharmic principles for AI and technology contexts"""
        if not self.advanced_enhancer:
            return []
        
        try:
            results = await self.advanced_enhancer.search_enhanced_wisdom(
                query=f"dharmic AI technology {context}",
                limit=10
            )
            
            # Look for entries related to AI, technology, or modern applications
            ai_guidance = []
            for result in results:
                modern_apps = result.get('modern_applications', [])
                if any('AI' in app or 'technology' in app or 'digital' in app for app in modern_apps):
                    ai_guidance.append(result)
            
            if not ai_guidance:
                # Fallback to general dharmic principles
                ai_guidance = await self.advanced_enhancer.search_enhanced_wisdom(
                    query="dharma ethics principles",
                    limit=5
                )
            
            self.logger.info(f"Retrieved {len(ai_guidance)} dharmic AI guidance entries")
            return ai_guidance
            
        except Exception as e:
            self.logger.error(f"Failed to get dharmic AI guidance: {e}")
            return []

    async def close(self):
        """Close the knowledge base connection"""
        if self.connection:
            await self.connection.close()
            self.logger.info("Knowledge Base connection closed")

# Global knowledge base instance
_knowledge_base = None

def get_knowledge_base() -> KnowledgeBase:
    """Get global knowledge base instance"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base

# Export the main class
__all__ = ["KnowledgeBase", "get_knowledge_base"]
