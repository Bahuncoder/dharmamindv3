"""
🧘 Rishi Session Management System
=================================

Maintains conversation continuity and progressive guidance
across multiple sessions with each Rishi, enabling deep
spiritual mentorship and personalized development tracking.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import sqlite3
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class SpiritualProgress:
    """Track user's spiritual progress with a specific Rishi"""
    topics_explored: List[str] = field(default_factory=list)
    practices_given: List[str] = field(default_factory=list)
    practices_completed: List[str] = field(default_factory=list)
    depth_level: str = "beginner"  # beginner, intermediate, advanced
    insights_gained: List[str] = field(default_factory=list)
    challenges_addressed: List[str] = field(default_factory=list)
    mantras_learned: List[str] = field(default_factory=list)
    scriptures_studied: List[str] = field(default_factory=list)
    last_session_date: Optional[str] = None
    total_sessions: int = 0
    
@dataclass
class RishiSessionData:
    """Complete session data for a user-Rishi relationship"""
    user_id: str
    rishi_name: str
    first_session: str
    last_session: str
    progress: SpiritualProgress
    conversation_themes: List[str] = field(default_factory=list)
    personality_adaptation: Dict[str, Any] = field(default_factory=dict)
    next_recommended_topics: List[str] = field(default_factory=list)
    session_count: int = 0

class RishiSessionManager:
    """Manages ongoing relationships between users and Rishis"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "data/rishi_sessions"
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        self.db_path = Path(self.storage_path) / "rishi_sessions.db"
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for session storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rishi_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    rishi_name TEXT NOT NULL,
                    session_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, rishi_name)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    rishi_name TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response_summary TEXT,
                    topics_covered TEXT,
                    practices_given TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    async def get_session_data(self, user_id: str, rishi_name: str) -> Optional[RishiSessionData]:
        """Retrieve existing session data for user-Rishi pair"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT session_data FROM rishi_sessions WHERE user_id = ? AND rishi_name = ?",
                    (user_id, rishi_name)
                )
                row = cursor.fetchone()
                
                if row:
                    session_dict = json.loads(row[0])
                    # Convert dict back to dataclass
                    progress_data = session_dict.pop('progress')
                    progress = SpiritualProgress(**progress_data)
                    session_dict['progress'] = progress
                    return RishiSessionData(**session_dict)
                    
        except Exception as e:
            self.logger.error(f"Error retrieving session data: {e}")
            
        return None
    
    async def save_session_data(self, session_data: RishiSessionData):
        """Save or update session data"""
        try:
            # Convert to dict for JSON serialization
            session_dict = asdict(session_data)
            session_json = json.dumps(session_dict, default=str)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO rishi_sessions 
                    (user_id, rishi_name, session_data, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (session_data.user_id, session_data.rishi_name, session_json))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving session data: {e}")
    
    async def record_interaction(
        self, 
        user_id: str, 
        rishi_name: str, 
        query: str,
        response_summary: str,
        topics_covered: List[str],
        practices_given: List[str]
    ):
        """Record individual interaction for tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO session_interactions 
                    (user_id, rishi_name, query, response_summary, topics_covered, practices_given)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    user_id, rishi_name, query, response_summary,
                    json.dumps(topics_covered), json.dumps(practices_given)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error recording interaction: {e}")
    
    async def create_or_update_session(
        self, 
        user_id: str, 
        rishi_name: str,
        query: str,
        response_data: Dict[str, Any],
        user_context: Dict[str, Any] = None
    ) -> RishiSessionData:
        """Create new session or update existing one"""
        
        # Get existing session or create new
        session_data = await self.get_session_data(user_id, rishi_name)
        current_time = datetime.now().isoformat()
        
        if not session_data:
            # Create new session
            session_data = RishiSessionData(
                user_id=user_id,
                rishi_name=rishi_name,
                first_session=current_time,
                last_session=current_time,
                progress=SpiritualProgress(last_session_date=current_time),
                session_count=1
            )
        else:
            # Update existing session
            session_data.last_session = current_time
            session_data.session_count += 1
            session_data.progress.last_session_date = current_time
            session_data.progress.total_sessions += 1
        
        # Analyze and update progress
        await self._analyze_and_update_progress(session_data, query, response_data, user_context)
        
        # Save updated session
        await self.save_session_data(session_data)
        
        # Record this interaction
        await self.record_interaction(
            user_id, rishi_name, query,
            response_data.get('guidance', {}).get('primary_wisdom', '')[:200],
            session_data.conversation_themes[-3:],  # Last 3 themes
            response_data.get('practical_steps', [])
        )
        
        return session_data
    
    async def _analyze_and_update_progress(
        self, 
        session_data: RishiSessionData,
        query: str,
        response_data: Dict[str, Any],
        user_context: Dict[str, Any] = None
    ):
        """Analyze current interaction and update spiritual progress"""
        
        progress = session_data.progress
        
        # Extract topics from query
        topics = self._extract_topics_from_query(query)
        for topic in topics:
            if topic not in progress.topics_explored:
                progress.topics_explored.append(topic)
        
        # Add to conversation themes
        if topics:
            session_data.conversation_themes.extend(topics)
            # Keep only last 10 themes
            session_data.conversation_themes = session_data.conversation_themes[-10:]
        
        # Track practices given
        practical_steps = response_data.get('practical_steps', [])
        for step in practical_steps:
            if step not in progress.practices_given:
                progress.practices_given.append(step)
        
        # Track mantras
        guidance = response_data.get('guidance', {})
        mantras = guidance.get('mantras', [])
        for mantra in mantras:
            mantra_text = mantra.get('sanskrit', '') if isinstance(mantra, dict) else str(mantra)
            if mantra_text and mantra_text not in progress.mantras_learned:
                progress.mantras_learned.append(mantra_text)
        
        # Track scriptures
        scriptural_refs = guidance.get('scriptural_references', [])
        for ref in scriptural_refs:
            if isinstance(ref, dict):
                text_ref = ref.get('text', '')
                if text_ref and text_ref not in progress.scriptures_studied:
                    progress.scriptures_studied.append(text_ref)
        
        # Update depth level based on session count and topics
        session_data.progress = self._assess_spiritual_depth(progress, session_data.session_count)
        
        # Generate next recommended topics
        session_data.next_recommended_topics = self._generate_next_topics(
            session_data.rishi_name, progress, session_data.conversation_themes
        )
    
    def _extract_topics_from_query(self, query: str) -> List[str]:
        """Extract spiritual topics from user query"""
        query_lower = query.lower()
        topics = []
        
        topic_keywords = {
            'meditation': ['meditat', 'mindfulness', 'awareness', 'concentration'],
            'yoga': ['yoga', 'asana', 'pranayama', 'poses'],
            'dharma': ['dharma', 'duty', 'righteousness', 'ethics'],
            'karma': ['karma', 'action', 'consequences', 'work'],
            'devotion': ['devotion', 'bhakti', 'love', 'surrender'],
            'self_inquiry': ['self', 'identity', 'who am i', 'consciousness'],
            'suffering': ['suffering', 'pain', 'difficulty', 'stress'],
            'purpose': ['purpose', 'meaning', 'goal', 'direction'],
            'relationships': ['relationship', 'family', 'love', 'conflict'],
            'mind_control': ['mind', 'thoughts', 'control', 'discipline']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _assess_spiritual_depth(self, progress: SpiritualProgress, session_count: int) -> SpiritualProgress:
        """Assess and update user's spiritual depth level"""
        
        # Criteria for depth assessment
        topics_count = len(progress.topics_explored)
        practices_count = len(progress.practices_given)
        mantras_count = len(progress.mantras_learned)
        
        if session_count >= 10 and topics_count >= 8 and practices_count >= 5:
            progress.depth_level = "advanced"
        elif session_count >= 5 and topics_count >= 4 and practices_count >= 3:
            progress.depth_level = "intermediate"
        else:
            progress.depth_level = "beginner"
        
        return progress
    
    def _generate_next_topics(
        self, 
        rishi_name: str, 
        progress: SpiritualProgress, 
        recent_themes: List[str]
    ) -> List[str]:
        """Generate recommended topics for next session"""
        
        rishi_progressions = {
            'atri': {
                'beginner': ['basic meditation', 'breath awareness', 'simple austerity'],
                'intermediate': ['deeper tapasya', 'cosmic meditation', 'spiritual discipline'],
                'advanced': ['cosmic consciousness', 'divine union', 'supreme austerity']
            },
            'bhrigu': {
                'beginner': ['karma understanding', 'basic astrology', 'life patterns'],
                'intermediate': ['karmic insights', 'cosmic order', 'divine knowledge'],
                'advanced': ['karmic mastery', 'cosmic wisdom', 'divine sight']
            },
            'vashishta': {
                'beginner': ['dharmic living', 'basic wisdom', 'righteous conduct'],
                'intermediate': ['royal guidance', 'spiritual mastery', 'divine wisdom'],
                'advanced': ['supreme wisdom', 'guru consciousness', 'divine teaching']
            },
            'vishwamitra': {
                'beginner': ['Gayatri practice', 'spiritual discipline', 'transformation basics'],
                'intermediate': ['power development', 'spiritual achievement', 'mantra mastery'],
                'advanced': ['brahmarishi consciousness', 'divine power', 'spiritual transformation']
            },
            'gautama': {
                'beginner': ['meditation basics', 'dharmic conduct', 'righteousness'],
                'intermediate': ['deep meditation', 'ethical mastery', 'spiritual discipline'],
                'advanced': ['meditation mastery', 'dharmic perfection', 'spiritual authority']
            },
            'jamadagni': {
                'beginner': ['basic tapas', 'spiritual discipline', 'righteous action'],
                'intermediate': ['power through discipline', 'divine strength', 'tapas mastery'],
                'advanced': ['supreme discipline', 'divine power', 'spiritual warrior']
            },
            'kashyapa': {
                'beginner': ['universal love', 'life force awareness', 'cosmic connection'],
                'intermediate': ['pranic mastery', 'universal consciousness', 'cosmic creation'],
                'advanced': ['father consciousness', 'cosmic mastery', 'universal mind']
            }
        }
        
        level_topics = rishi_progressions.get(rishi_name, {}).get(progress.depth_level, [])
        
        # Filter out already explored topics
        unexplored_topics = [topic for topic in level_topics 
                           if topic not in progress.topics_explored]
        
        # Return 3 recommended topics
        return unexplored_topics[:3]
    
    async def get_session_summary(self, user_id: str, rishi_name: str) -> Dict[str, Any]:
        """Get comprehensive session summary for display"""
        session_data = await self.get_session_data(user_id, rishi_name)
        
        if not session_data:
            return {'exists': False}
        
        return {
            'exists': True,
            'session_count': session_data.session_count,
            'first_session': session_data.first_session,
            'last_session': session_data.last_session,
            'spiritual_level': session_data.progress.depth_level,
            'topics_explored': len(session_data.progress.topics_explored),
            'practices_given': len(session_data.progress.practices_given),
            'mantras_learned': len(session_data.progress.mantras_learned),
            'next_topics': session_data.next_recommended_topics,
            'recent_themes': session_data.conversation_themes[-5:]
        }
    
    async def get_personalized_greeting(
        self, 
        user_id: str, 
        rishi_name: str,
        base_greeting: str
    ) -> str:
        """Generate personalized greeting based on session history"""
        
        session_data = await self.get_session_data(user_id, rishi_name)
        
        if not session_data or session_data.session_count <= 1:
            return base_greeting
        
        # Personalize based on history
        days_since_last = self._days_since_last_session(session_data.last_session)
        
        if days_since_last > 7:
            return f"🙏 Welcome back after {days_since_last} days, devoted seeker. Your spiritual journey continues..."
        elif days_since_last > 1:
            return f"🕉️ Namaste again, dear student. Continuing our exploration from where we left off..."
        else:
            return f"🌟 Welcome back, sincere practitioner. Our ongoing dialogue deepens..."
    
    def _days_since_last_session(self, last_session: str) -> int:
        """Calculate days since last session"""
        try:
            last_date = datetime.fromisoformat(last_session.replace('Z', '+00:00'))
            return (datetime.now() - last_date).days
        except:
            return 0

# Global session manager instance
_session_manager = None

def get_session_manager() -> RishiSessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = RishiSessionManager()
    return _session_manager

# Export main classes
__all__ = [
    'RishiSessionManager',
    'RishiSessionData',
    'SpiritualProgress',
    'get_session_manager'
]
