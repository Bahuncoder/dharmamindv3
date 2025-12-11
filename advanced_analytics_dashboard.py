#!/usr/bin/env python3
"""
🧠 DharmaMind Advanced Analytics & Insights Dashboard
Comprehensive spiritual progress tracking and visualization system
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import redis.asyncio as redis
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpiritualMetricType(str, Enum):
    MEDITATION_CONSISTENCY = "meditation_consistency"
    WISDOM_INTEGRATION = "wisdom_integration"
    CONSCIOUSNESS_LEVEL = "consciousness_level"
    EMOTIONAL_BALANCE = "emotional_balance"
    DHARMIC_ALIGNMENT = "dharmic_alignment"
    COMPASSION_DEVELOPMENT = "compassion_development"
    MINDFULNESS_PRACTICE = "mindfulness_practice"
    SPIRITUAL_INSIGHTS = "spiritual_insights"

@dataclass
class SpiritualProgress:
    """Individual spiritual progress tracking"""
    user_id: str
    metric_type: SpiritualMetricType
    score: float  # 0-100
    timestamp: datetime
    context: Dict[str, Any]
    insights: List[str]
    practices: List[str]

@dataclass
class DashboardMetrics:
    """Dashboard metrics for visualization"""
    total_users: int
    active_meditators: int
    avg_session_duration: float
    wisdom_insights_shared: int
    spiritual_paths_explored: Dict[str, int]
    consciousness_distribution: Dict[str, int]
    growth_trends: Dict[str, List[float]]

class AdvancedAnalyticsDashboard:
    """
    🧠 Advanced Analytics Dashboard for Spiritual Growth Tracking
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.active_sessions = {}
        self.spiritual_metrics = {}
        self.wisdom_insights = []
        
    async def initialize(self):
        """Initialize analytics dashboard"""
        try:
            # Connect to Redis for real-time data
            self.redis_client = redis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
            logger.info("✅ Analytics dashboard initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize analytics: {e}")
            return False
    
    async def track_spiritual_session(self, user_id: str, session_data: Dict[str, Any]):
        """Track a spiritual guidance session"""
        try:
            session_id = f"session_{user_id}_{datetime.now().timestamp()}"
            
            # Extract spiritual metrics
            meditation_quality = session_data.get('meditation_quality', 0)
            wisdom_gained = session_data.get('wisdom_insights', [])
            emotional_state = session_data.get('emotional_balance', 50)
            dharmic_alignment = session_data.get('dharmic_alignment', 0)
            
            # Create progress records
            progress_records = [
                SpiritualProgress(
                    user_id=user_id,
                    metric_type=SpiritualMetricType.MEDITATION_CONSISTENCY,
                    score=meditation_quality,
                    timestamp=datetime.now(),
                    context=session_data,
                    insights=wisdom_gained,
                    practices=session_data.get('practices', [])
                ),
                SpiritualProgress(
                    user_id=user_id,
                    metric_type=SpiritualMetricType.EMOTIONAL_BALANCE,
                    score=emotional_state,
                    timestamp=datetime.now(),
                    context=session_data,
                    insights=[],
                    practices=[]
                ),
                SpiritualProgress(
                    user_id=user_id,
                    metric_type=SpiritualMetricType.DHARMIC_ALIGNMENT,
                    score=dharmic_alignment,
                    timestamp=datetime.now(),
                    context=session_data,
                    insights=[],
                    practices=[]
                )
            ]
            
            # Store in Redis for real-time access
            if self.redis_client:
                for progress in progress_records:
                    key = f"spiritual_progress:{user_id}:{progress.metric_type.value}"
                    await self.redis_client.zadd(key, {
                        json.dumps({
                            'score': progress.score,
                            'timestamp': progress.timestamp.isoformat(),
                            'context': progress.context,
                            'insights': progress.insights,
                            'practices': progress.practices
                        }): progress.timestamp.timestamp()
                    })
                    # Keep only last 100 records per metric
                    await self.redis_client.zremrangebyrank(key, 0, -101)
            
            logger.info(f"📊 Tracked spiritual session for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Error tracking spiritual session: {e}")
            return None
    
    async def get_user_spiritual_journey(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive spiritual journey analytics for a user"""
        try:
            journey_data = {
                'user_id': user_id,
                'period_days': days,
                'metrics': {},
                'insights': [],
                'growth_patterns': {},
                'recommendations': []
            }
            
            if not self.redis_client:
                return journey_data
            
            # Get data for each spiritual metric
            for metric_type in SpiritualMetricType:
                key = f"spiritual_progress:{user_id}:{metric_type.value}"
                
                # Get last 30 days of data
                since_timestamp = (datetime.now() - timedelta(days=days)).timestamp()
                records = await self.redis_client.zrangebyscore(key, since_timestamp, '+inf', withscores=True)
                
                if records:
                    metric_data = []
                    for record_json, timestamp in records:
                        record = json.loads(record_json)
                        record['metric_timestamp'] = timestamp
                        metric_data.append(record)
                    
                    # Calculate trends and insights
                    scores = [r['score'] for r in metric_data]
                    if len(scores) > 1:
                        # Calculate growth trend
                        growth_trend = (scores[-1] - scores[0]) / max(scores[0], 1) * 100
                        
                        journey_data['metrics'][metric_type.value] = {
                            'current_score': scores[-1] if scores else 0,
                            'average_score': np.mean(scores) if scores else 0,
                            'growth_trend': growth_trend,
                            'data_points': len(scores),
                            'trend_direction': 'improving' if growth_trend > 5 else 'stable' if growth_trend > -5 else 'declining',
                            'historical_data': metric_data[-10:]  # Last 10 data points
                        }
                        
                        # Extract insights
                        all_insights = []
                        for record in metric_data:
                            all_insights.extend(record.get('insights', []))
                        
                        if all_insights:
                            journey_data['insights'].extend(all_insights[-5:])  # Last 5 insights
            
            # Generate personalized recommendations
            journey_data['recommendations'] = await self._generate_spiritual_recommendations(journey_data['metrics'])
            
            # Calculate overall spiritual progress score
            if journey_data['metrics']:
                overall_score = np.mean([m['current_score'] for m in journey_data['metrics'].values()])
                journey_data['overall_spiritual_progress'] = overall_score
                journey_data['spiritual_level'] = self._determine_spiritual_level(overall_score)
            
            logger.info(f"📈 Generated spiritual journey for user {user_id}")
            return journey_data
            
        except Exception as e:
            logger.error(f"❌ Error getting spiritual journey: {e}")
            return {'error': str(e)}
    
    async def get_community_analytics(self) -> Dict[str, Any]:
        """Get community-wide spiritual analytics"""
        try:
            community_data = {
                'total_active_users': 0,
                'community_metrics': {},
                'spiritual_paths_distribution': {},
                'wisdom_insights_shared': 0,
                'collective_growth_trends': {},
                'inspirational_stats': {}
            }
            
            if not self.redis_client:
                return community_data
            
            # Get all users with spiritual progress data
            user_pattern = "spiritual_progress:*"
            all_keys = await self.redis_client.keys(user_pattern)
            
            # Extract unique user IDs
            users = set()
            for key in all_keys:
                # Key format: spiritual_progress:user_id:metric_type
                parts = key.decode().split(':')
                if len(parts) >= 2:
                    users.add(parts[1])
            
            community_data['total_active_users'] = len(users)
            
            # Aggregate community metrics
            all_metrics = {metric.value: [] for metric in SpiritualMetricType}
            wisdom_count = 0
            
            for user_id in users:
                for metric_type in SpiritualMetricType:
                    key = f"spiritual_progress:{user_id}:{metric_type.value}"
                    recent_records = await self.redis_client.zrevrange(key, 0, 4)  # Last 5 records
                    
                    for record_json in recent_records:
                        record = json.loads(record_json)
                        all_metrics[metric_type.value].append(record['score'])
                        wisdom_count += len(record.get('insights', []))
            
            # Calculate community averages and trends
            for metric_type, scores in all_metrics.items():
                if scores:
                    community_data['community_metrics'][metric_type] = {
                        'average_score': np.mean(scores),
                        'median_score': np.median(scores),
                        'std_deviation': np.std(scores),
                        'participants': len(scores),
                        'distribution': {
                            'beginner (0-30)': len([s for s in scores if s <= 30]),
                            'intermediate (31-70)': len([s for s in scores if 30 < s <= 70]),
                            'advanced (71-100)': len([s for s in scores if s > 70])
                        }
                    }
            
            community_data['wisdom_insights_shared'] = wisdom_count
            
            # Generate inspirational community stats
            community_data['inspirational_stats'] = {
                'total_meditation_sessions': len(users) * 15,  # Estimated
                'collective_wisdom_score': np.mean([m['average_score'] for m in community_data['community_metrics'].values()]) if community_data['community_metrics'] else 0,
                'community_growth_momentum': 'Rising' if community_data['wisdom_insights_shared'] > 100 else 'Stable',
                'spiritual_diversity_index': len([m for m in community_data['community_metrics'].values() if m['participants'] > 5])
            }
            
            logger.info(f"🌍 Generated community analytics for {len(users)} users")
            return community_data
            
        except Exception as e:
            logger.error(f"❌ Error getting community analytics: {e}")
            return {'error': str(e)}
    
    async def _generate_spiritual_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate personalized spiritual recommendations based on metrics"""
        recommendations = []
        
        for metric_name, metric_data in metrics.items():
            score = metric_data.get('current_score', 0)
            trend = metric_data.get('trend_direction', 'stable')
            
            if metric_name == 'meditation_consistency':
                if score < 50:
                    recommendations.append("🧘 Consider establishing a daily meditation routine, even 5-10 minutes can create profound shifts")
                elif trend == 'declining':
                    recommendations.append("🌸 Your meditation practice may benefit from exploring new techniques like loving-kindness or walking meditation")
                else:
                    recommendations.append("✨ Your meditation consistency is excellent! Consider deepening your practice with longer sessions")
            
            elif metric_name == 'wisdom_integration':
                if score < 40:
                    recommendations.append("📖 Try journaling about spiritual insights to better integrate wisdom into daily life")
                elif trend == 'improving':
                    recommendations.append("🌟 Beautiful progress in wisdom integration! Consider sharing insights with others")
            
            elif metric_name == 'emotional_balance':
                if score < 50:
                    recommendations.append("❤️ Loving-kindness meditation could help cultivate emotional equilibrium and inner peace")
                elif score > 80:
                    recommendations.append("🕊️ Your emotional balance is inspiring! Consider mentoring others on this path")
            
            elif metric_name == 'dharmic_alignment':
                if score < 60:
                    recommendations.append("⚖️ Reflect on your daily choices through the lens of dharmic principles - right action, speech, and livelihood")
                else:
                    recommendations.append("🙏 Your dharmic alignment shines! Continue living as an example of conscious living")
        
        # Add general recommendations
        if len(recommendations) < 3:
            recommendations.extend([
                "🌱 Consider exploring different spiritual paths to find what resonates most deeply",
                "📚 Reading sacred texts like the Bhagavad Gita or Buddhist sutras can provide profound insights",
                "🤝 Connecting with like-minded spiritual seekers can accelerate your growth"
            ])
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _determine_spiritual_level(self, overall_score: float) -> str:
        """Determine spiritual development level based on overall score"""
        if overall_score >= 85:
            return "Advanced Practitioner"
        elif overall_score >= 70:
            return "Dedicated Seeker"
        elif overall_score >= 50:
            return "Developing Practitioner"
        elif overall_score >= 30:
            return "Beginning Explorer"
        else:
            return "New to the Path"
    
    async def generate_dashboard_html(self, user_id: Optional[str] = None) -> str:
        """Generate HTML dashboard for analytics visualization"""
        
        # Get data
        if user_id:
            journey_data = await self.get_user_spiritual_journey(user_id)
            community_data = None
        else:
            journey_data = None
            community_data = await self.get_community_analytics()
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🧠 DharmaMind Analytics Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ 
                    font-family: 'Segoe UI', sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0; padding: 20px; color: white;
                }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ 
                    background: rgba(255,255,255,0.1); 
                    border-radius: 15px; 
                    padding: 20px; 
                    margin: 20px 0;
                    backdrop-filter: blur(10px);
                }}
                .metric {{ 
                    display: inline-block; 
                    margin: 10px 15px; 
                    text-align: center;
                }}
                .metric-value {{ 
                    font-size: 2.5em; 
                    font-weight: bold; 
                    color: #FFD700;
                }}
                .metric-label {{ 
                    font-size: 0.9em; 
                    opacity: 0.8;
                }}
                .recommendation {{ 
                    background: rgba(255,255,255,0.05); 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 10px;
                    border-left: 4px solid #FFD700;
                }}
                .chart-container {{ 
                    position: relative; 
                    height: 300px; 
                    margin: 20px 0;
                }}
                h1, h2 {{ text-align: center; margin-bottom: 30px; }}
                .spiritual-level {{ 
                    font-size: 1.5em; 
                    color: #FFD700; 
                    text-align: center; 
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <h1>🕉️ DharmaMind Spiritual Analytics Dashboard</h1>
                
                {"" if not user_id else f'''
                <div class="card">
                    <h2>🧘 Your Spiritual Journey</h2>
                    <div class="spiritual-level">
                        Current Level: {journey_data.get("spiritual_level", "Beginning Explorer")}
                    </div>
                    
                    <div class="metric">
                        <div class="metric-value">{journey_data.get("overall_spiritual_progress", 0):.1f}</div>
                        <div class="metric-label">Overall Progress Score</div>
                    </div>
                    
                    <div class="metric">
                        <div class="metric-value">{journey_data.get("period_days", 30)}</div>
                        <div class="metric-label">Days Tracked</div>
                    </div>
                    
                    <div class="metric">
                        <div class="metric-value">{len(journey_data.get("insights", []))}</div>
                        <div class="metric-label">Wisdom Insights</div>
                    </div>
                    
                    <h3>📈 Recent Insights</h3>
                    {"".join([f'<div class="recommendation">💡 {insight}</div>' for insight in journey_data.get("insights", [])[:3]])}
                    
                    <h3>🌟 Personalized Recommendations</h3>
                    {"".join([f'<div class="recommendation">{rec}</div>' for rec in journey_data.get("recommendations", [])])}
                </div>
                '''}
                
                {"" if user_id else f'''
                <div class="card">
                    <h2>🌍 Community Spiritual Insights</h2>
                    
                    <div class="metric">
                        <div class="metric-value">{community_data.get("total_active_users", 0)}</div>
                        <div class="metric-label">Active Practitioners</div>
                    </div>
                    
                    <div class="metric">
                        <div class="metric-value">{community_data.get("wisdom_insights_shared", 0)}</div>
                        <div class="metric-label">Wisdom Insights Shared</div>
                    </div>
                    
                    <div class="metric">
                        <div class="metric-value">{community_data.get("inspirational_stats", {}).get("collective_wisdom_score", 0):.1f}</div>
                        <div class="metric-label">Collective Wisdom Score</div>
                    </div>
                    
                    <div class="metric">
                        <div class="metric-value">{community_data.get("inspirational_stats", {}).get("community_growth_momentum", "Stable")}</div>
                        <div class="metric-label">Growth Momentum</div>
                    </div>
                </div>
                '''}
                
                <div class="card">
                    <h2>🔮 Spiritual Insights Generator</h2>
                    <p>🧠 <strong>Advanced Analytics Capabilities:</strong></p>
                    <ul>
                        <li>📊 Real-time spiritual progress tracking</li>
                        <li>🧘 Meditation consistency analysis</li>
                        <li>💡 Wisdom integration assessment</li>
                        <li>❤️ Emotional balance monitoring</li>
                        <li>⚖️ Dharmic alignment evaluation</li>
                        <li>🌱 Personalized growth recommendations</li>
                        <li>🌍 Community spiritual insights</li>
                        <li>📈 Historical progress visualization</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h2>🚀 Next Steps</h2>
                    <div class="recommendation">
                        <strong>🎯 Implementation Plan:</strong><br>
                        1. Integrate with existing DharmaMind backend<br>
                        2. Add user authentication and privacy controls<br>
                        3. Create interactive charts and visualizations<br>
                        4. Implement real-time updates with WebSocket<br>
                        5. Add export functionality for personal records<br>
                        6. Create mobile-responsive design<br>
                        7. Add gamification elements for engagement
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template

# Create FastAPI app for dashboard
app = FastAPI(title="DharmaMind Analytics Dashboard", version="1.0.0")
dashboard = AdvancedAnalyticsDashboard()

@app.on_event("startup")
async def startup_event():
    await dashboard.initialize()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Get the main analytics dashboard"""
    return await dashboard.generate_dashboard_html()

@app.get("/user/{user_id}", response_class=HTMLResponse)
async def get_user_dashboard(user_id: str):
    """Get user-specific analytics dashboard"""
    return await dashboard.generate_dashboard_html(user_id)

@app.post("/track-session/{user_id}")
async def track_session(user_id: str, session_data: Dict[str, Any]):
    """Track a spiritual guidance session"""
    session_id = await dashboard.track_spiritual_session(user_id, session_data)
    return {"session_id": session_id, "status": "tracked"}

@app.get("/api/journey/{user_id}")
async def get_spiritual_journey(user_id: str, days: int = 30):
    """Get spiritual journey data as JSON"""
    return await dashboard.get_user_spiritual_journey(user_id, days)

@app.get("/api/community")
async def get_community_analytics():
    """Get community analytics as JSON"""
    return await dashboard.get_community_analytics()

if __name__ == "__main__":
    import uvicorn
    print("🧠 Starting DharmaMind Advanced Analytics Dashboard...")
    print("📊 Dashboard URL: http://localhost:8080")
    print("👤 User Dashboard: http://localhost:8080/user/your_user_id")
    print("🌍 Community Analytics: http://localhost:8080/api/community")
    uvicorn.run(app, host="0.0.0.0", port=8080)
