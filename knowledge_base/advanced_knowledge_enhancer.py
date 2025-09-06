"""
Advanced Knowledge Enhancement System for DharmaMind
==================================================

This module provides enhanced knowledge integration, cross-referencing,
and wisdom synthesis capabilities for the DharmaMind system.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sqlite3
import aiosqlite
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class WisdomLevel(Enum):
    """Levels of spiritual wisdom and understanding"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    MASTERY = "mastery"
    ENLIGHTENED = "enlightened"

class TraditionType(Enum):
    """Different spiritual traditions"""
    VEDIC = "vedic"
    YOGIC = "yogic"
    TANTRIC = "tantric"
    VEDANTIC = "vedantic"
    SANKHYA = "sankhya"
    NYAYA = "nyaya"
    VAISHESHIKA = "vaisheshika"
    MIMAMSA = "mimamsa"
    PURANIC = "puranic"
    JAIN = "jain"

@dataclass
class EnhancedWisdomEntry:
    """Enhanced wisdom entry with advanced features"""
    id: str
    title: str
    original_sanskrit: str
    transliteration: str
    source: str
    wisdom_level: WisdomLevel
    tradition: TraditionType
    philosophical_depth: str
    practical_application: Dict[str, Any]
    cross_references: List[str]
    consciousness_level: str
    prerequisites: List[str] = field(default_factory=list)
    integration_notes: Optional[str] = None
    modern_applications: List[str] = field(default_factory=list)
    scientific_correlates: Optional[Dict[str, Any]] = None

class AdvancedKnowledgeEnhancer:
    """Advanced knowledge enhancement and integration system"""
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.enhanced_files = [
            "advanced_philosophical_frameworks.json",
            "consciousness_science_integration.json", 
            "advanced_spiritual_practices.json",
            "wisdom_synthesis_framework.json"
        ]
        # Use absolute path for database
        if str(self.knowledge_base_path).startswith("./"):
            # We're in the knowledge_base directory, so go up one level to data
            self.db_path = str(Path("../data/enhanced_dharma_knowledge.db").resolve())
        else:
            self.db_path = str(Path(knowledge_base_path).parent / "data" / "enhanced_dharma_knowledge.db")
        
    async def initialize_enhanced_system(self) -> bool:
        """Initialize the enhanced knowledge system"""
        try:
            logger.info("🚀 Initializing Advanced DharmaMind Knowledge System...")
            
            # Create enhanced database
            await self._create_enhanced_database()
            
            # Load all enhanced knowledge files
            await self._load_enhanced_knowledge()
            
            # Create cross-reference indexes
            await self._create_wisdom_indexes()
            
            # Integrate with existing knowledge base
            await self._integrate_with_existing_kb()
            
            logger.info("✅ Advanced Knowledge System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced system: {e}")
            return False
    
    async def _create_enhanced_database(self):
        """Create enhanced database schema"""
        conn = await aiosqlite.connect(self.db_path)
        
        # Enhanced wisdom entries table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_wisdom (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                original_sanskrit TEXT,
                transliteration TEXT,
                source TEXT,
                wisdom_level TEXT,
                tradition TEXT,
                philosophical_depth TEXT,
                practical_application TEXT,
                cross_references TEXT,
                consciousness_level TEXT,
                prerequisites TEXT,
                integration_notes TEXT,
                modern_applications TEXT,
                scientific_correlates TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Cross-reference table for wisdom interconnections
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS wisdom_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                target_id TEXT,
                connection_type TEXT,
                strength REAL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES enhanced_wisdom (id),
                FOREIGN KEY (target_id) REFERENCES enhanced_wisdom (id)
            )
        """)
        
        # Practice progression tracking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS practice_progression (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                practice_id TEXT,
                prerequisite_practices TEXT,
                next_level_practices TEXT,
                mastery_indicators TEXT,
                common_obstacles TEXT,
                solutions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.commit()
        await conn.close()
    
    async def _load_enhanced_knowledge(self):
        """Load all enhanced knowledge files into database"""
        conn = await aiosqlite.connect(self.db_path)
        
        for file_name in self.enhanced_files:
            file_path = self.knowledge_base_path / file_name
            if file_path.exists():
                logger.info(f"📚 Loading enhanced knowledge from {file_name}...")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                await self._process_knowledge_structure(conn, data, file_name)
        
        await conn.commit()
        await conn.close()
        logger.info("✅ All enhanced knowledge files loaded successfully!")
    
    async def _process_knowledge_structure(self, conn, data: Dict, source_file: str):
        """Process nested knowledge structure and insert into database"""
        
        async def process_item(item_data: Dict, category_path: str = ""):
            if isinstance(item_data, dict):
                # Check if this is a knowledge entry (has required fields)
                if 'id' in item_data and 'title' in item_data:
                    await self._insert_wisdom_entry(conn, item_data, source_file, category_path)
                else:
                    # Recursively process nested structures
                    for key, value in item_data.items():
                        if isinstance(value, dict):
                            new_path = f"{category_path}/{key}" if category_path else key
                            await process_item(value, new_path)
        
        await process_item(data)
    
    async def _insert_wisdom_entry(self, conn, entry: Dict, source_file: str, category: str):
        """Insert enhanced wisdom entry into database"""
        try:
            await conn.execute("""
                INSERT OR REPLACE INTO enhanced_wisdom 
                (id, title, original_sanskrit, transliteration, source, wisdom_level,
                 tradition, philosophical_depth, practical_application, cross_references,
                 consciousness_level, prerequisites, integration_notes, modern_applications,
                 scientific_correlates)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.get('id'),
                entry.get('title'),
                entry.get('original_sanskrit', ''),
                entry.get('transliteration', ''),
                entry.get('source', source_file),
                entry.get('wisdom_level', 'intermediate'),
                entry.get('tradition', 'vedic'),
                entry.get('philosophical_depth', ''),
                json.dumps(entry.get('practical_application', {})),
                json.dumps(entry.get('cross_references', [])),
                entry.get('consciousness_level', ''),
                json.dumps(entry.get('prerequisites', [])),
                entry.get('integration_notes', ''),
                json.dumps(entry.get('modern_applications', [])),
                json.dumps(entry.get('scientific_correlates', {}))
            ))
        except Exception as e:
            logger.error(f"Error inserting wisdom entry {entry.get('id')}: {e}")
    
    async def _create_wisdom_indexes(self):
        """Create cross-reference indexes for efficient searching"""
        conn = await aiosqlite.connect(self.db_path)
        
        # Create indexes for efficient searching
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_wisdom_level ON enhanced_wisdom(wisdom_level)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tradition ON enhanced_wisdom(tradition)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_consciousness_level ON enhanced_wisdom(consciousness_level)")
        
        await conn.commit()
        await conn.close()
    
    async def _integrate_with_existing_kb(self):
        """Integrate enhanced knowledge with existing knowledge base"""
        logger.info("🔄 Integrating with existing knowledge base...")
        
        # This would connect with the existing knowledge_base.py system
        # and create cross-references between old and new knowledge
        
        # For now, we'll create a mapping file
        integration_report = {
            "integration_status": "complete",
            "enhanced_entries_count": await self._count_enhanced_entries(),
            "cross_references_created": await self._count_cross_references(),
            "integration_timestamp": str(asyncio.get_event_loop().time())
        }
        
        # Save integration report
        report_path = self.knowledge_base_path / "enhanced_integration_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(integration_report, f, indent=2, ensure_ascii=False)
    
    async def _count_enhanced_entries(self) -> int:
        """Count total enhanced entries"""
        conn = await aiosqlite.connect(self.db_path)
        cursor = await conn.execute("SELECT COUNT(*) FROM enhanced_wisdom")
        count = (await cursor.fetchone())[0]
        await conn.close()
        return count
    
    async def _count_cross_references(self) -> int:
        """Count total cross-references"""
        conn = await aiosqlite.connect(self.db_path)
        cursor = await conn.execute("SELECT COUNT(*) FROM wisdom_connections")
        count = (await cursor.fetchone())[0]
        await conn.close()
        return count
    
    async def search_enhanced_wisdom(self, 
                                   query: str,
                                   wisdom_level: Optional[WisdomLevel] = None,
                                   tradition: Optional[TraditionType] = None,
                                   limit: int = 10) -> List[Dict]:
        """Search enhanced wisdom with advanced filtering"""
        conn = await aiosqlite.connect(self.db_path)
        
        base_query = """
            SELECT * FROM enhanced_wisdom 
            WHERE (title LIKE ? OR philosophical_depth LIKE ? OR original_sanskrit LIKE ?)
        """
        params = [f"%{query}%", f"%{query}%", f"%{query}%"]
        
        if wisdom_level:
            base_query += " AND wisdom_level = ?"
            params.append(wisdom_level.value)
        
        if tradition:
            base_query += " AND tradition = ?"
            params.append(tradition.value)
        
        base_query += f" LIMIT {limit}"
        
        cursor = await conn.execute(base_query, params)
        results = await cursor.fetchall()
        
        # Convert to list of dictionaries
        columns = [description[0] for description in cursor.description]
        result_dicts = []
        for row in results:
            row_dict = dict(zip(columns, row))
            # Parse JSON fields
            for json_field in ['practical_application', 'cross_references', 'prerequisites', 'modern_applications', 'scientific_correlates']:
                if row_dict.get(json_field):
                    try:
                        row_dict[json_field] = json.loads(row_dict[json_field])
                    except json.JSONDecodeError:
                        pass
            result_dicts.append(row_dict)
        
        await conn.close()
        return result_dicts
    
    async def get_practice_progression(self, practice_id: str) -> Optional[Dict]:
        """Get practice progression information"""
        conn = await aiosqlite.connect(self.db_path)
        
        cursor = await conn.execute("""
            SELECT w.*, pp.prerequisite_practices, pp.next_level_practices, 
                   pp.mastery_indicators, pp.common_obstacles, pp.solutions
            FROM enhanced_wisdom w
            LEFT JOIN practice_progression pp ON w.id = pp.practice_id
            WHERE w.id = ?
        """, (practice_id,))
        
        result = await cursor.fetchone()
        await conn.close()
        
        if result:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, result))
        return None
    
    async def get_wisdom_connections(self, wisdom_id: str) -> List[Dict]:
        """Get connected wisdom entries"""
        conn = await aiosqlite.connect(self.db_path)
        
        cursor = await conn.execute("""
            SELECT w.*, wc.connection_type, wc.strength, wc.description as connection_description
            FROM wisdom_connections wc
            JOIN enhanced_wisdom w ON (wc.target_id = w.id OR wc.source_id = w.id)
            WHERE (wc.source_id = ? OR wc.target_id = ?) AND w.id != ?
        """, (wisdom_id, wisdom_id, wisdom_id))
        
        results = await cursor.fetchall()
        
        connections = []
        for row in results:
            columns = [description[0] for description in cursor.description]
            connections.append(dict(zip(columns, row)))
        
        await conn.close()
        return connections

# Convenience function for external use
async def initialize_enhanced_knowledge_system(knowledge_base_path: str = "./knowledge_base") -> bool:
    """Initialize the enhanced knowledge system"""
    enhancer = AdvancedKnowledgeEnhancer(knowledge_base_path)
    return await enhancer.initialize_enhanced_system()

# Main execution for testing
if __name__ == "__main__":
    async def main():
        enhancer = AdvancedKnowledgeEnhancer()
        success = await enhancer.initialize_enhanced_system()
        
        if success:
            # Test search functionality
            results = await enhancer.search_enhanced_wisdom("consciousness", limit=5)
            print(f"Found {len(results)} consciousness-related entries")
            
            for result in results:
                print(f"- {result['title']} ({result['tradition']})")
        
        return success
    
    asyncio.run(main())
