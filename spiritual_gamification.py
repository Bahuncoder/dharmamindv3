#!/usr/bin/env python3
"""
🎮 DharmaMind Spiritual Gamification System
Engaging achievement and progress tracking with dharmic principles
"""

from fastapi import FastAPI, HTTPException
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AchievementCategory(str, Enum):
    MEDITATION = "meditation"
    WISDOM = "wisdom"
    COMPASSION = "compassion"
    DHARMA = "dharma"
    COMMUNITY = "community"
    MINDFULNESS = "mindfulness"
    SPIRITUAL_STUDY = "spiritual_study"
    SERVICE = "service"

class AchievementRarity(str, Enum):
    COMMON = "common"      # Easy to achieve
    UNCOMMON = "uncommon"  # Moderate effort
    RARE = "rare"          # Significant dedication
    EPIC = "epic"          # Exceptional achievement
    LEGENDARY = "legendary" # Extraordinary spiritual milestone

@dataclass
class SpiritualAchievement:
    """Represents a spiritual achievement or milestone"""
    id: str
    title: str
    description: str
    category: AchievementCategory
    rarity: AchievementRarity
    points: int
    requirements: Dict[str, Any]
    unlocked_by: Optional[str] = None
    unlocked_at: Optional[datetime] = None
    progress: float = 0.0  # 0-100%
    icon: str = "🏆"
    dharmic_wisdom: str = ""

@dataclass
class SpiritualLevel:
    """Spiritual development level with requirements"""
    level: int
    title: str
    description: str
    points_required: int
    abilities_unlocked: List[str]
    dharmic_teaching: str
    color: str
    icon: str

@dataclass
class UserProfile:
    """User's spiritual gamification profile"""
    user_id: str
    level: int
    total_points: int
    achievements: List[str]  # Achievement IDs
    current_streaks: Dict[str, int]
    longest_streaks: Dict[str, int]
    spiritual_stats: Dict[str, float]
    badges: List[str]
    created_at: datetime
    last_activity: datetime

class SpiritualGamificationEngine:
    """
    🎮 Spiritual Gamification Engine with Dharmic Principles
    Makes spiritual growth engaging while maintaining sacred respect
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.achievements = {}
        self.spiritual_levels = {}
        self.user_profiles = {}
        self._initialize_achievements()
        self._initialize_levels()
    
    def _initialize_achievements(self):
        """Initialize all spiritual achievements"""
        
        # Meditation Achievements
        meditation_achievements = [
            SpiritualAchievement(
                id="first_meditation",
                title="🧘 First Steps on the Path",
                description="Complete your first meditation session",
                category=AchievementCategory.MEDITATION,
                rarity=AchievementRarity.COMMON,
                points=10,
                requirements={"meditation_sessions": 1},
                icon="🌱",
                dharmic_wisdom="A journey of a thousand miles begins with a single step. - Lao Tzu"
            ),
            SpiritualAchievement(
                id="meditation_streak_7",
                title="🌟 Seven Sacred Days",
                description="Meditate for 7 consecutive days",
                category=AchievementCategory.MEDITATION,
                rarity=AchievementRarity.UNCOMMON,
                points=50,
                requirements={"meditation_streak": 7},
                icon="⭐",
                dharmic_wisdom="Consistency in practice leads to mastery. Daily meditation transforms the mind."
            ),
            SpiritualAchievement(
                id="meditation_streak_30",
                title="🌙 Lunar Cycle Mastery",
                description="Maintain meditation practice for 30 days",
                category=AchievementCategory.MEDITATION,
                rarity=AchievementRarity.RARE,
                points=200,
                requirements={"meditation_streak": 30},
                icon="🌙",
                dharmic_wisdom="यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः - Where there is devotion, there is victory."
            ),
            SpiritualAchievement(
                id="meditation_master",
                title="🕉️ Meditation Master",
                description="Complete 365 meditation sessions",
                category=AchievementCategory.MEDITATION,
                rarity=AchievementRarity.LEGENDARY,
                points=1000,
                requirements={"meditation_sessions": 365},
                icon="🕉️",
                dharmic_wisdom="The mind is everything. What you think you become. - Buddha"
            )
        ]
        
        # Wisdom Achievements
        wisdom_achievements = [
            SpiritualAchievement(
                id="first_insight",
                title="💡 First Glimpse of Wisdom",
                description="Receive your first spiritual insight",
                category=AchievementCategory.WISDOM,
                rarity=AchievementRarity.COMMON,
                points=15,
                requirements={"wisdom_insights": 1},
                icon="💡",
                dharmic_wisdom="The wise find pleasure in water; men of heroic strength in mountains. - Confucius"
            ),
            SpiritualAchievement(
                id="wisdom_seeker",
                title="📚 Dedicated Seeker",
                description="Collect 50 wisdom insights",
                category=AchievementCategory.WISDOM,
                rarity=AchievementRarity.RARE,
                points=300,
                requirements={"wisdom_insights": 50},
                icon="📚",
                dharmic_wisdom="ज्ञानं परमं बलम् - Knowledge is the ultimate strength"
            ),
            SpiritualAchievement(
                id="sanskrit_scholar",
                title="🕉️ Sanskrit Scholar",
                description="Learn 25 Sanskrit terms and meanings",
                category=AchievementCategory.SPIRITUAL_STUDY,
                rarity=AchievementRarity.EPIC,
                points=500,
                requirements={"sanskrit_terms": 25},
                icon="📜",
                dharmic_wisdom="मातृदेवो भव पितृदेवो भव - Honor your mother and father as divine"
            )
        ]
        
        # Compassion Achievements
        compassion_achievements = [
            SpiritualAchievement(
                id="loving_kindness",
                title="❤️ Heart Opening",
                description="Practice loving-kindness meditation",
                category=AchievementCategory.COMPASSION,
                rarity=AchievementRarity.UNCOMMON,
                points=40,
                requirements={"loving_kindness_sessions": 1},
                icon="❤️",
                dharmic_wisdom="May all beings be happy. May all beings be free from suffering."
            ),
            SpiritualAchievement(
                id="community_helper",
                title="🤝 Community Guide",
                description="Help 10 community members with spiritual guidance",
                category=AchievementCategory.COMMUNITY,
                rarity=AchievementRarity.RARE,
                points=250,
                requirements={"helped_members": 10},
                icon="🤝",
                dharmic_wisdom="सेवा धर्मो जगत्प्रीतिः - Service is the highest dharma and brings joy to the world"
            )
        ]
        
        # Dharma Achievements
        dharma_achievements = [
            SpiritualAchievement(
                id="dharmic_alignment",
                title="⚖️ Walking the Righteous Path",
                description="Maintain high dharmic alignment for 30 days",
                category=AchievementCategory.DHARMA,
                rarity=AchievementRarity.EPIC,
                points=400,
                requirements={"dharmic_alignment_days": 30, "min_alignment_score": 80},
                icon="⚖️",
                dharmic_wisdom="धर्मो रक्षति रक्षितः - Dharma protects those who protect dharma"
            ),
            SpiritualAchievement(
                id="truth_speaker",
                title="🗣️ Speaker of Truth",
                description="Demonstrate consistent truthfulness in interactions",
                category=AchievementCategory.DHARMA,
                rarity=AchievementRarity.RARE,
                points=300,
                requirements={"truthfulness_score": 90, "interactions": 50},
                icon="🗣️",
                dharmic_wisdom="सत्यमेव जयते - Truth alone triumphs"
            )
        ]
        
        # Combine all achievements
        all_achievements = (meditation_achievements + wisdom_achievements + 
                          compassion_achievements + dharma_achievements)
        
        # Store achievements by ID
        for achievement in all_achievements:
            self.achievements[achievement.id] = achievement
        
        logger.info(f"✅ Initialized {len(all_achievements)} spiritual achievements")
    
    def _initialize_levels(self):
        """Initialize spiritual development levels"""
        
        levels = [
            SpiritualLevel(
                level=1,
                title="🌱 Spiritual Seedling",
                description="Beginning your journey of spiritual awakening",
                points_required=0,
                abilities_unlocked=["Basic meditation guidance", "Simple breathing exercises"],
                dharmic_teaching="Every master was once a beginner. Every expert was once a novice.",
                color="#8FBC8F",
                icon="🌱"
            ),
            SpiritualLevel(
                level=2,
                title="🌿 Growing Seeker",
                description="Establishing regular spiritual practices",
                points_required=100,
                abilities_unlocked=["Loving-kindness meditation", "Basic dharma teachings"],
                dharmic_teaching="Consistency in small steps leads to great transformations.",
                color="#9ACD32",
                icon="🌿"
            ),
            SpiritualLevel(
                level=3,
                title="🌸 Blossoming Student",
                description="Developing deeper understanding and compassion",
                points_required=300,
                abilities_unlocked=["Advanced meditation techniques", "Sanskrit wisdom study"],
                dharmic_teaching="As the flower blooms naturally, so does wisdom unfold in a prepared mind.",
                color="#FFB6C1",
                icon="🌸"
            ),
            SpiritualLevel(
                level=4,
                title="🌳 Rooted Practitioner",
                description="Strong foundation in spiritual practices and wisdom",
                points_required=600,
                abilities_unlocked=["Dharma counseling", "Community guidance", "Advanced breathing"],
                dharmic_teaching="Like a tree with deep roots, the wise remain unshaken by life's storms.",
                color="#228B22",
                icon="🌳"
            ),
            SpiritualLevel(
                level=5,
                title="🦋 Transformed Being",
                description="Significant spiritual transformation and inner peace",
                points_required=1200,
                abilities_unlocked=["Emotional mastery", "Wisdom teaching", "Energy practices"],
                dharmic_teaching="The butterfly emerges transformed, no longer bound by its former limitations.",
                color="#9370DB",
                icon="🦋"
            ),
            SpiritualLevel(
                level=6,
                title="🌟 Radiant Guide",
                description="Embodying wisdom and serving as a spiritual guide",
                points_required=2000,
                abilities_unlocked=["Advanced teaching", "Healing practices", "Dharma transmission"],
                dharmic_teaching="Those who are enlightened light the way for others walking in darkness.",
                color="#FFD700",
                icon="🌟"
            ),
            SpiritualLevel(
                level=7,
                title="🕉️ Awakened Master",
                description="Profound spiritual realization and mastery",
                points_required=3500,
                abilities_unlocked=["Spiritual mastery", "Divine wisdom", "Universal compassion"],
                dharmic_teaching="बोधि स्वभावतः शुद्धं - Enlightenment is naturally pure and ever-present.",
                color="#FF69B4",
                icon="🕉️"
            )
        ]
        
        for level in levels:
            self.spiritual_levels[level.level] = level
        
        logger.info(f"✅ Initialized {len(levels)} spiritual levels")
    
    async def initialize(self):
        """Initialize the gamification engine"""
        try:
            # Connect to Redis
            self.redis_client = redis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
            logger.info("✅ Spiritual gamification engine initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize gamification: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user's spiritual profile"""
        try:
            if not self.redis_client:
                return self._create_default_profile(user_id)
            
            # Try to get existing profile
            profile_data = await self.redis_client.get(f"spiritual_profile:{user_id}")
            
            if profile_data:
                data = json.loads(profile_data)
                return UserProfile(
                    user_id=user_id,
                    level=data.get('level', 1),
                    total_points=data.get('total_points', 0),
                    achievements=data.get('achievements', []),
                    current_streaks=data.get('current_streaks', {}),
                    longest_streaks=data.get('longest_streaks', {}),
                    spiritual_stats=data.get('spiritual_stats', {}),
                    badges=data.get('badges', []),
                    created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
                    last_activity=datetime.fromisoformat(data.get('last_activity', datetime.now().isoformat()))
                )
            else:
                # Create new profile
                return await self._create_new_profile(user_id)
        
        except Exception as e:
            logger.error(f"❌ Error getting user profile: {e}")
            return self._create_default_profile(user_id)
    
    def _create_default_profile(self, user_id: str) -> UserProfile:
        """Create a default profile when Redis is unavailable"""
        return UserProfile(
            user_id=user_id,
            level=1,
            total_points=0,
            achievements=[],
            current_streaks={},
            longest_streaks={},
            spiritual_stats={},
            badges=[],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
    
    async def _create_new_profile(self, user_id: str) -> UserProfile:
        """Create and save a new user profile"""
        profile = UserProfile(
            user_id=user_id,
            level=1,
            total_points=0,
            achievements=[],
            current_streaks={},
            longest_streaks={},
            spiritual_stats={},
            badges=[],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        await self._save_profile(profile)
        logger.info(f"🆕 Created new spiritual profile for user {user_id}")
        return profile
    
    async def _save_profile(self, profile: UserProfile):
        """Save user profile to Redis"""
        if not self.redis_client:
            return
        
        profile_data = {
            'level': profile.level,
            'total_points': profile.total_points,
            'achievements': profile.achievements,
            'current_streaks': profile.current_streaks,
            'longest_streaks': profile.longest_streaks,
            'spiritual_stats': profile.spiritual_stats,
            'badges': profile.badges,
            'created_at': profile.created_at.isoformat(),
            'last_activity': profile.last_activity.isoformat()
        }
        
        await self.redis_client.set(
            f"spiritual_profile:{profile.user_id}",
            json.dumps(profile_data)
        )
    
    async def track_spiritual_activity(self, user_id: str, activity_type: str, 
                                     activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track spiritual activity and check for achievements"""
        try:
            profile = await self.get_user_profile(user_id)
            profile.last_activity = datetime.now()
            
            # Update streaks and stats based on activity
            result = {
                'user_id': user_id,
                'activity_type': activity_type,
                'points_earned': 0,
                'achievements_unlocked': [],
                'level_up': False,
                'new_level': profile.level,
                'streak_updates': {}
            }
            
            # Process different activity types
            if activity_type == 'meditation_session':
                result.update(await self._process_meditation_activity(profile, activity_data))
            elif activity_type == 'wisdom_insight':
                result.update(await self._process_wisdom_activity(profile, activity_data))
            elif activity_type == 'dharma_study':
                result.update(await self._process_dharma_activity(profile, activity_data))
            elif activity_type == 'community_help':
                result.update(await self._process_community_activity(profile, activity_data))
            
            # Check for achievements
            new_achievements = await self._check_achievements(profile, activity_type, activity_data)
            result['achievements_unlocked'].extend(new_achievements)
            
            # Add achievement points
            for achievement_id in new_achievements:
                if achievement_id in self.achievements:
                    result['points_earned'] += self.achievements[achievement_id].points
                    profile.total_points += self.achievements[achievement_id].points
                    profile.achievements.append(achievement_id)
            
            # Check for level up
            current_level = profile.level
            new_level = self._calculate_level(profile.total_points)
            if new_level > current_level:
                profile.level = new_level
                result['level_up'] = True
                result['new_level'] = new_level
                logger.info(f"🎉 User {user_id} leveled up to {new_level}!")
            
            # Save updated profile
            await self._save_profile(profile)
            
            # Log activity
            await self._log_activity(user_id, activity_type, activity_data, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error tracking spiritual activity: {e}")
            return {'error': str(e)}
    
    async def _process_meditation_activity(self, profile: UserProfile, 
                                         activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process meditation session activity"""
        duration = activity_data.get('duration_minutes', 0)
        quality = activity_data.get('quality_score', 50)  # 0-100
        
        # Update meditation stats
        if 'meditation_sessions' not in profile.spiritual_stats:
            profile.spiritual_stats['meditation_sessions'] = 0
        profile.spiritual_stats['meditation_sessions'] += 1
        
        if 'total_meditation_minutes' not in profile.spiritual_stats:
            profile.spiritual_stats['total_meditation_minutes'] = 0
        profile.spiritual_stats['total_meditation_minutes'] += duration
        
        # Update streak
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        if 'meditation_streak' not in profile.current_streaks:
            profile.current_streaks['meditation_streak'] = 1
        else:
            # Check if meditated yesterday
            last_meditation = profile.spiritual_stats.get('last_meditation_date', '')
            if last_meditation == yesterday:
                profile.current_streaks['meditation_streak'] += 1
            elif last_meditation != today:
                profile.current_streaks['meditation_streak'] = 1
        
        # Update longest streak
        current_streak = profile.current_streaks['meditation_streak']
        if current_streak > profile.longest_streaks.get('meditation_streak', 0):
            profile.longest_streaks['meditation_streak'] = current_streak
        
        profile.spiritual_stats['last_meditation_date'] = today
        
        # Calculate points based on duration and quality
        base_points = 5
        duration_bonus = min(duration, 60) // 5  # 1 point per 5 minutes, max 12
        quality_bonus = quality // 20  # 0-5 points based on quality
        
        points_earned = base_points + duration_bonus + quality_bonus
        profile.total_points += points_earned
        
        return {
            'points_earned': points_earned,
            'streak_updates': {
                'meditation_streak': profile.current_streaks['meditation_streak']
            }
        }
    
    async def _process_wisdom_activity(self, profile: UserProfile, 
                                     activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process wisdom insight activity"""
        insight_quality = activity_data.get('quality_score', 50)
        insight_depth = activity_data.get('depth_score', 50)
        
        # Update wisdom stats
        if 'wisdom_insights' not in profile.spiritual_stats:
            profile.spiritual_stats['wisdom_insights'] = 0
        profile.spiritual_stats['wisdom_insights'] += 1
        
        # Calculate points
        base_points = 10
        quality_bonus = insight_quality // 10  # 0-10 points
        depth_bonus = insight_depth // 10  # 0-10 points
        
        points_earned = base_points + quality_bonus + depth_bonus
        profile.total_points += points_earned
        
        return {'points_earned': points_earned}
    
    async def _process_dharma_activity(self, profile: UserProfile, 
                                     activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process dharma study activity"""
        study_type = activity_data.get('study_type', 'general')
        duration = activity_data.get('duration_minutes', 0)
        
        # Update dharma stats
        if 'dharma_study_sessions' not in profile.spiritual_stats:
            profile.spiritual_stats['dharma_study_sessions'] = 0
        profile.spiritual_stats['dharma_study_sessions'] += 1
        
        if study_type == 'sanskrit':
            if 'sanskrit_terms' not in profile.spiritual_stats:
                profile.spiritual_stats['sanskrit_terms'] = 0
            profile.spiritual_stats['sanskrit_terms'] += activity_data.get('terms_learned', 1)
        
        # Calculate points
        base_points = 8
        duration_bonus = min(duration, 30) // 5  # 1 point per 5 minutes, max 6
        
        points_earned = base_points + duration_bonus
        profile.total_points += points_earned
        
        return {'points_earned': points_earned}
    
    async def _process_community_activity(self, profile: UserProfile, 
                                        activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process community help activity"""
        help_type = activity_data.get('help_type', 'general')
        impact_score = activity_data.get('impact_score', 50)
        
        # Update community stats
        if 'helped_members' not in profile.spiritual_stats:
            profile.spiritual_stats['helped_members'] = 0
        profile.spiritual_stats['helped_members'] += 1
        
        # Calculate points based on impact
        base_points = 15
        impact_bonus = impact_score // 10  # 0-10 points
        
        points_earned = base_points + impact_bonus
        profile.total_points += points_earned
        
        return {'points_earned': points_earned}
    
    async def _check_achievements(self, profile: UserProfile, activity_type: str, 
                                activity_data: Dict[str, Any]) -> List[str]:
        """Check if user has unlocked any new achievements"""
        new_achievements = []
        
        for achievement_id, achievement in self.achievements.items():
            # Skip if already unlocked
            if achievement_id in profile.achievements:
                continue
            
            # Check requirements
            unlocked = True
            for req_key, req_value in achievement.requirements.items():
                
                if req_key == 'meditation_sessions':
                    if profile.spiritual_stats.get('meditation_sessions', 0) < req_value:
                        unlocked = False
                
                elif req_key == 'meditation_streak':
                    if profile.current_streaks.get('meditation_streak', 0) < req_value:
                        unlocked = False
                
                elif req_key == 'wisdom_insights':
                    if profile.spiritual_stats.get('wisdom_insights', 0) < req_value:
                        unlocked = False
                
                elif req_key == 'sanskrit_terms':
                    if profile.spiritual_stats.get('sanskrit_terms', 0) < req_value:
                        unlocked = False
                
                elif req_key == 'helped_members':
                    if profile.spiritual_stats.get('helped_members', 0) < req_value:
                        unlocked = False
                
                elif req_key == 'dharmic_alignment_days':
                    # This would require more complex tracking
                    # For now, using a placeholder
                    if profile.spiritual_stats.get('dharmic_alignment_days', 0) < req_value:
                        unlocked = False
                
                # Add more requirement checks as needed
            
            if unlocked:
                new_achievements.append(achievement_id)
                achievement.unlocked_by = profile.user_id
                achievement.unlocked_at = datetime.now()
                
                logger.info(f"🏆 Achievement unlocked: {achievement.title} for user {profile.user_id}")
        
        return new_achievements
    
    def _calculate_level(self, total_points: int) -> int:
        """Calculate spiritual level based on total points"""
        current_level = 1
        
        for level_num in sorted(self.spiritual_levels.keys()):
            level = self.spiritual_levels[level_num]
            if total_points >= level.points_required:
                current_level = level_num
            else:
                break
        
        return current_level
    
    async def _log_activity(self, user_id: str, activity_type: str, 
                          activity_data: Dict[str, Any], result: Dict[str, Any]):
        """Log spiritual activity for analytics"""
        if not self.redis_client:
            return
        
        log_entry = {
            'user_id': user_id,
            'activity_type': activity_type,
            'activity_data': activity_data,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.redis_client.lpush(
            f"spiritual_activity_log:{user_id}",
            json.dumps(log_entry)
        )
        # Keep only last 100 activities
        await self.redis_client.ltrim(f"spiritual_activity_log:{user_id}", 0, 99)
    
    async def get_achievements_list(self) -> List[Dict[str, Any]]:
        """Get list of all achievements"""
        achievements_list = []
        
        for achievement in self.achievements.values():
            achievements_list.append({
                'id': achievement.id,
                'title': achievement.title,
                'description': achievement.description,
                'category': achievement.category.value,
                'rarity': achievement.rarity.value,
                'points': achievement.points,
                'icon': achievement.icon,
                'dharmic_wisdom': achievement.dharmic_wisdom,
                'requirements': achievement.requirements
            })
        
        return achievements_list
    
    async def get_leaderboard(self, category: str = "overall", limit: int = 10) -> List[Dict[str, Any]]:
        """Get spiritual leaderboard"""
        try:
            if not self.redis_client:
                return []
            
            # Get all user profiles
            pattern = "spiritual_profile:*"
            keys = await self.redis_client.keys(pattern)
            
            users_data = []
            for key in keys:
                profile_data = await self.redis_client.get(key)
                if profile_data:
                    data = json.loads(profile_data)
                    user_id = key.decode().split(':')[1]
                    
                    users_data.append({
                        'user_id': user_id,
                        'level': data.get('level', 1),
                        'total_points': data.get('total_points', 0),
                        'achievements_count': len(data.get('achievements', [])),
                        'meditation_sessions': data.get('spiritual_stats', {}).get('meditation_sessions', 0),
                        'wisdom_insights': data.get('spiritual_stats', {}).get('wisdom_insights', 0),
                        'last_activity': data.get('last_activity', datetime.now().isoformat())
                    })
            
            # Sort based on category
            if category == "meditation":
                users_data.sort(key=lambda x: x['meditation_sessions'], reverse=True)
            elif category == "wisdom":
                users_data.sort(key=lambda x: x['wisdom_insights'], reverse=True)
            elif category == "achievements":
                users_data.sort(key=lambda x: x['achievements_count'], reverse=True)
            else:  # overall
                users_data.sort(key=lambda x: x['total_points'], reverse=True)
            
            return users_data[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error getting leaderboard: {e}")
            return []

# Create FastAPI app for gamification
app = FastAPI(title="DharmaMind Spiritual Gamification", version="1.0.0")
gamification = SpiritualGamificationEngine()

@app.on_event("startup")
async def startup_event():
    await gamification.initialize()

@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Get user's spiritual profile"""
    profile = await gamification.get_user_profile(user_id)
    
    # Convert to dict for JSON response
    profile_dict = {
        'user_id': profile.user_id,
        'level': profile.level,
        'total_points': profile.total_points,
        'achievements': profile.achievements,
        'current_streaks': profile.current_streaks,
        'longest_streaks': profile.longest_streaks,
        'spiritual_stats': profile.spiritual_stats,
        'badges': profile.badges,
        'created_at': profile.created_at.isoformat(),
        'last_activity': profile.last_activity.isoformat()
    }
    
    # Add level info
    if profile.level in gamification.spiritual_levels:
        level_info = gamification.spiritual_levels[profile.level]
        profile_dict['level_info'] = {
            'title': level_info.title,
            'description': level_info.description,
            'abilities_unlocked': level_info.abilities_unlocked,
            'dharmic_teaching': level_info.dharmic_teaching,
            'color': level_info.color,
            'icon': level_info.icon
        }
    
    return profile_dict

@app.post("/track/{user_id}")
async def track_activity(user_id: str, activity_type: str, activity_data: Dict[str, Any]):
    """Track spiritual activity and update achievements"""
    result = await gamification.track_spiritual_activity(user_id, activity_type, activity_data)
    return result

@app.get("/achievements")
async def get_achievements():
    """Get list of all achievements"""
    return await gamification.get_achievements_list()

@app.get("/leaderboard")
async def get_leaderboard(category: str = "overall", limit: int = 10):
    """Get spiritual leaderboard"""
    return await gamification.get_leaderboard(category, limit)

@app.get("/levels")
async def get_levels():
    """Get list of all spiritual levels"""
    levels_list = []
    for level in gamification.spiritual_levels.values():
        levels_list.append({
            'level': level.level,
            'title': level.title,
            'description': level.description,
            'points_required': level.points_required,
            'abilities_unlocked': level.abilities_unlocked,
            'dharmic_teaching': level.dharmic_teaching,
            'color': level.color,
            'icon': level.icon
        })
    return levels_list

if __name__ == "__main__":
    import uvicorn
    print("🎮 Starting DharmaMind Spiritual Gamification System...")
    print("🏆 Profile API: http://localhost:8081/profile/your_user_id")
    print("📊 Achievements: http://localhost:8081/achievements")
    print("🏅 Leaderboard: http://localhost:8081/leaderboard")
    uvicorn.run(app, host="0.0.0.0", port=8081)
