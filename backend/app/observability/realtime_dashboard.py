"""
📈 DharmaMind Real-time Dashboard System

Interactive dashboard for comprehensive system monitoring and analytics:

Core Features:
- Real-time system metrics visualization
- User behavior analytics dashboard
- Performance monitoring and alerting
- Business KPI tracking
- Custom dashboard creation
- Live data streaming with WebSockets
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import HTMLResponse
import redis.asyncio as redis

from ..observability.analytics_engine import AdvancedAnalyticsEngine, AnalyticsEventType
from ..observability.distributed_tracing import get_tracer

# Dashboard configuration
logger = logging.getLogger("dharmamind.dashboard")


class DashboardType(str, Enum):
    """Types of dashboards"""
    SYSTEM_OVERVIEW = "system_overview"
    USER_ANALYTICS = "user_analytics"
    PERFORMANCE_METRICS = "performance_metrics"
    BUSINESS_KPIS = "business_kpis"
    SECURITY_MONITORING = "security_monitoring"
    DHARMA_INSIGHTS = "dharma_insights"


class ChartType(str, Enum):
    """Types of charts"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    chart_type: ChartType
    data_source: str
    refresh_interval: int  # seconds
    size: str  # "small", "medium", "large", "full"
    position: Dict[str, int]  # {"x": 0, "y": 0, "w": 6, "h": 4}
    config: Dict[str, Any] = None


@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    widgets: List[DashboardWidget]
    auto_refresh: bool = True
    theme: str = "light"


class WebSocketManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.dashboard_subscribers: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, dashboard_id: Optional[str] = None):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        if dashboard_id:
            if dashboard_id not in self.dashboard_subscribers:
                self.dashboard_subscribers[dashboard_id] = set()
            self.dashboard_subscribers[dashboard_id].add(websocket)
        
        logger.info(f"📡 WebSocket connected for dashboard: {dashboard_id}")
    
    def disconnect(self, websocket: WebSocket, dashboard_id: Optional[str] = None):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        
        if dashboard_id and dashboard_id in self.dashboard_subscribers:
            self.dashboard_subscribers[dashboard_id].discard(websocket)
            
            # Clean up empty subscriber sets
            if not self.dashboard_subscribers[dashboard_id]:
                del self.dashboard_subscribers[dashboard_id]
        
        logger.info(f"📡 WebSocket disconnected from dashboard: {dashboard_id}")
    
    async def send_to_dashboard(self, dashboard_id: str, data: Dict[str, Any]):
        """Send data to all subscribers of a dashboard"""
        if dashboard_id not in self.dashboard_subscribers:
            return
        
        message = json.dumps(data)
        disconnected = set()
        
        for websocket in self.dashboard_subscribers[dashboard_id]:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected WebSockets
        for websocket in disconnected:
            self.disconnect(websocket, dashboard_id)
    
    async def broadcast_to_all(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        message = json.dumps(data)
        disconnected = set()
        
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected WebSockets
        for websocket in disconnected:
            self.disconnect(websocket)


class RealtimeDashboard:
    """
    Real-time dashboard system for comprehensive monitoring
    """
    
    def __init__(self, redis_client: redis.Redis, analytics_engine: AdvancedAnalyticsEngine):
        self.redis = redis_client
        self.analytics = analytics_engine
        self.websocket_manager = WebSocketManager()
        
        # Dashboard configurations
        self.dashboards: Dict[str, DashboardLayout] = {}
        self._initialize_default_dashboards()
        
        # Data update tasks
        self.update_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("📈 Real-time Dashboard System initialized")
    
    def _initialize_default_dashboards(self):
        """Initialize default dashboard layouts"""
        
        # System Overview Dashboard
        system_widgets = [
            DashboardWidget(
                widget_id="system_health",
                title="System Health",
                chart_type=ChartType.GAUGE,
                data_source="system_health",
                refresh_interval=30,
                size="medium",
                position={"x": 0, "y": 0, "w": 6, "h": 4}
            ),
            DashboardWidget(
                widget_id="request_rate",
                title="Request Rate",
                chart_type=ChartType.LINE,
                data_source="request_metrics",
                refresh_interval=10,
                size="large",
                position={"x": 6, "y": 0, "w": 12, "h": 4}
            ),
            DashboardWidget(
                widget_id="error_rate",
                title="Error Rate",
                chart_type=ChartType.LINE,
                data_source="error_metrics",
                refresh_interval=30,
                size="medium",
                position={"x": 0, "y": 4, "w": 6, "h": 4}
            ),
            DashboardWidget(
                widget_id="resource_usage",
                title="Resource Usage",
                chart_type=ChartType.BAR,
                data_source="resource_metrics",
                refresh_interval=60,
                size="medium",
                position={"x": 6, "y": 4, "w": 6, "h": 4}
            )
        ]
        
        self.dashboards["system_overview"] = DashboardLayout(
            dashboard_id="system_overview",
            name="System Overview",
            description="Comprehensive system health and performance monitoring",
            dashboard_type=DashboardType.SYSTEM_OVERVIEW,
            widgets=system_widgets
        )
        
        # User Analytics Dashboard
        user_widgets = [
            DashboardWidget(
                widget_id="active_users",
                title="Active Users",
                chart_type=ChartType.LINE,
                data_source="user_activity",
                refresh_interval=60,
                size="large",
                position={"x": 0, "y": 0, "w": 12, "h": 4}
            ),
            DashboardWidget(
                widget_id="user_engagement",
                title="User Engagement",
                chart_type=ChartType.PIE,
                data_source="engagement_metrics",
                refresh_interval=300,
                size="medium",
                position={"x": 12, "y": 0, "w": 6, "h": 4}
            ),
            DashboardWidget(
                widget_id="session_duration",
                title="Session Duration",
                chart_type=ChartType.HEATMAP,
                data_source="session_data",
                refresh_interval=300,
                size="large",
                position={"x": 0, "y": 4, "w": 12, "h": 4}
            )
        ]
        
        self.dashboards["user_analytics"] = DashboardLayout(
            dashboard_id="user_analytics",
            name="User Analytics",
            description="User behavior and engagement analysis",
            dashboard_type=DashboardType.USER_ANALYTICS,
            widgets=user_widgets
        )
        
        # Dharma Insights Dashboard
        dharma_widgets = [
            DashboardWidget(
                widget_id="wisdom_scores",
                title="Wisdom Scores Trend",
                chart_type=ChartType.LINE,
                data_source="wisdom_metrics",
                refresh_interval=300,
                size="large",
                position={"x": 0, "y": 0, "w": 12, "h": 4}
            ),
            DashboardWidget(
                widget_id="topic_popularity",
                title="Popular Topics",
                chart_type=ChartType.BAR,
                data_source="topic_metrics",
                refresh_interval=600,
                size="medium",
                position={"x": 12, "y": 0, "w": 6, "h": 4}
            ),
            DashboardWidget(
                widget_id="user_satisfaction",
                title="User Satisfaction",
                chart_type=ChartType.GAUGE,
                data_source="satisfaction_metrics",
                refresh_interval=300,
                size="medium",
                position={"x": 0, "y": 4, "w": 6, "h": 4}
            )
        ]
        
        self.dashboards["dharma_insights"] = DashboardLayout(
            dashboard_id="dharma_insights",
            name="Dharma Insights",
            description="Dharmic wisdom and spiritual guidance analytics",
            dashboard_type=DashboardType.DHARMA_INSIGHTS,
            widgets=dharma_widgets
        )
    
    async def start_dashboard_updates(self, dashboard_id: str):
        """Start real-time updates for a dashboard"""
        
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        if dashboard_id in self.update_tasks:
            return  # Already running
        
        async def update_loop():
            """Update loop for dashboard data"""
            while True:
                try:
                    dashboard = self.dashboards[dashboard_id]
                    
                    for widget in dashboard.widgets:
                        # Get updated data for widget
                        data = await self._get_widget_data(widget)
                        
                        # Send update to subscribers
                        await self.websocket_manager.send_to_dashboard(
                            dashboard_id,
                            {
                                "type": "widget_update",
                                "widget_id": widget.widget_id,
                                "data": data,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    
                    # Wait for next update cycle
                    await asyncio.sleep(30)  # Update every 30 seconds
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in dashboard update loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
        
        # Start update task
        self.update_tasks[dashboard_id] = asyncio.create_task(update_loop())
        logger.info(f"📈 Started dashboard updates for: {dashboard_id}")
    
    async def stop_dashboard_updates(self, dashboard_id: str):
        """Stop real-time updates for a dashboard"""
        
        if dashboard_id in self.update_tasks:
            self.update_tasks[dashboard_id].cancel()
            del self.update_tasks[dashboard_id]
            logger.info(f"📈 Stopped dashboard updates for: {dashboard_id}")
    
    async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget"""
        
        data_source = widget.data_source
        
        if data_source == "system_health":
            return await self._get_system_health_data()
        elif data_source == "request_metrics":
            return await self._get_request_metrics_data()
        elif data_source == "error_metrics":
            return await self._get_error_metrics_data()
        elif data_source == "resource_metrics":
            return await self._get_resource_metrics_data()
        elif data_source == "user_activity":
            return await self._get_user_activity_data()
        elif data_source == "engagement_metrics":
            return await self._get_engagement_metrics_data()
        elif data_source == "session_data":
            return await self._get_session_data()
        elif data_source == "wisdom_metrics":
            return await self._get_wisdom_metrics_data()
        elif data_source == "topic_metrics":
            return await self._get_topic_metrics_data()
        elif data_source == "satisfaction_metrics":
            return await self._get_satisfaction_metrics_data()
        else:
            return {"error": f"Unknown data source: {data_source}"}
    
    async def _get_system_health_data(self) -> Dict[str, Any]:
        """Get system health gauge data"""
        
        insights = await self.analytics.get_system_insights(hours=1)
        
        return {
            "chart_type": "gauge",
            "value": insights.health_score * 100,
            "min": 0,
            "max": 100,
            "title": "System Health Score",
            "units": "%",
            "color_ranges": [
                {"range": [0, 50], "color": "red"},
                {"range": [50, 80], "color": "yellow"},
                {"range": [80, 100], "color": "green"}
            ]
        }
    
    async def _get_request_metrics_data(self) -> Dict[str, Any]:
        """Get request rate time series data"""
        
        # Collect recent request data
        hours = 24
        timestamps = []
        request_counts = []
        
        for i in range(hours):
            hour = (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d-%H')
            system_key = f"analytics:system:{hour}"
            data = await self.redis.get(system_key)
            
            if data:
                system_data = json.loads(data)
                timestamps.append(hour)
                request_counts.append(system_data.get('total_events', 0))
        
        # Reverse to get chronological order
        timestamps.reverse()
        request_counts.reverse()
        
        return {
            "chart_type": "line",
            "x": timestamps,
            "y": request_counts,
            "title": "Request Rate Over Time",
            "x_title": "Time",
            "y_title": "Requests per Hour",
            "line_color": "blue"
        }
    
    async def _get_error_metrics_data(self) -> Dict[str, Any]:
        """Get error rate time series data"""
        
        # Collect recent error data
        hours = 24
        timestamps = []
        error_rates = []
        
        for i in range(hours):
            hour = (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d-%H')
            system_key = f"analytics:system:{hour}"
            data = await self.redis.get(system_key)
            
            if data:
                system_data = json.loads(data)
                total_events = system_data.get('total_events', 0)
                error_events = system_data.get('error_events', 0)
                error_rate = (error_events / total_events * 100) if total_events > 0 else 0
                
                timestamps.append(hour)
                error_rates.append(error_rate)
        
        # Reverse to get chronological order
        timestamps.reverse()
        error_rates.reverse()
        
        return {
            "chart_type": "line",
            "x": timestamps,
            "y": error_rates,
            "title": "Error Rate Over Time",
            "x_title": "Time",
            "y_title": "Error Rate (%)",
            "line_color": "red"
        }
    
    async def _get_resource_metrics_data(self) -> Dict[str, Any]:
        """Get resource utilization bar chart data"""
        
        insights = await self.analytics.get_system_insights(hours=1)
        
        resources = list(insights.resource_utilization.keys())
        utilization = [v * 100 for v in insights.resource_utilization.values()]
        
        return {
            "chart_type": "bar",
            "x": resources,
            "y": utilization,
            "title": "Resource Utilization",
            "x_title": "Resource Type",
            "y_title": "Utilization (%)",
            "bar_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        }
    
    async def _get_user_activity_data(self) -> Dict[str, Any]:
        """Get user activity time series data"""
        
        # Generate sample user activity data
        hours = 24
        timestamps = []
        active_users = []
        
        for i in range(hours):
            hour = datetime.now() - timedelta(hours=i)
            timestamps.append(hour.strftime('%H:00'))
            
            # Simulate user activity pattern
            base_users = 100
            time_factor = abs(12 - hour.hour) / 12  # Peak at noon
            active_users.append(int(base_users * (1 - time_factor * 0.5)))
        
        # Reverse to get chronological order
        timestamps.reverse()
        active_users.reverse()
        
        return {
            "chart_type": "line",
            "x": timestamps,
            "y": active_users,
            "title": "Active Users Over Time",
            "x_title": "Time",
            "y_title": "Active Users",
            "line_color": "green"
        }
    
    async def _get_engagement_metrics_data(self) -> Dict[str, Any]:
        """Get user engagement pie chart data"""
        
        return {
            "chart_type": "pie",
            "labels": ["Dharma Queries", "Meditation", "Wisdom Ratings", "Other"],
            "values": [45, 25, 20, 10],
            "title": "User Engagement by Feature",
            "colors": ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
        }
    
    async def _get_session_data(self) -> Dict[str, Any]:
        """Get session duration heatmap data"""
        
        # Generate sample session heatmap data
        hours = list(range(24))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Create sample data matrix
        import random
        data_matrix = []
        for day in days:
            day_data = []
            for hour in hours:
                # Simulate higher activity during certain hours
                if 9 <= hour <= 17:  # Business hours
                    value = random.randint(20, 60)
                elif 19 <= hour <= 22:  # Evening
                    value = random.randint(15, 40)
                else:
                    value = random.randint(5, 20)
                day_data.append(value)
            data_matrix.append(day_data)
        
        return {
            "chart_type": "heatmap",
            "z": data_matrix,
            "x": hours,
            "y": days,
            "title": "Session Duration Heatmap (minutes)",
            "colorscale": "Viridis"
        }
    
    async def _get_wisdom_metrics_data(self) -> Dict[str, Any]:
        """Get wisdom scores trend data"""
        
        # Generate sample wisdom trend data
        days = 30
        timestamps = []
        wisdom_scores = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            timestamps.append(date.strftime('%Y-%m-%d'))
            
            # Simulate improving wisdom scores over time
            base_score = 4.0
            trend = 0.01 * (days - i)  # Gradual improvement
            noise = (hash(date.strftime('%Y-%m-%d')) % 100) / 1000  # Consistent noise
            wisdom_scores.append(base_score + trend + noise)
        
        # Reverse to get chronological order
        timestamps.reverse()
        wisdom_scores.reverse()
        
        return {
            "chart_type": "line",
            "x": timestamps,
            "y": wisdom_scores,
            "title": "Wisdom Scores Trend",
            "x_title": "Date",
            "y_title": "Average Wisdom Score",
            "line_color": "purple"
        }
    
    async def _get_topic_metrics_data(self) -> Dict[str, Any]:
        """Get popular topics bar chart data"""
        
        return {
            "chart_type": "bar",
            "x": ["Meditation", "Compassion", "Mindfulness", "Dharma", "Wisdom", "Karma"],
            "y": [680, 520, 430, 380, 320, 280],
            "title": "Popular Topics",
            "x_title": "Topic",
            "y_title": "Query Count",
            "bar_colors": ["#8e44ad", "#3498db", "#e74c3c", "#f39c12", "#27ae60", "#34495e"]
        }
    
    async def _get_satisfaction_metrics_data(self) -> Dict[str, Any]:
        """Get user satisfaction gauge data"""
        
        return {
            "chart_type": "gauge",
            "value": 4.2,
            "min": 1,
            "max": 5,
            "title": "User Satisfaction Score",
            "units": "/5",
            "color_ranges": [
                {"range": [1, 2], "color": "red"},
                {"range": [2, 3], "color": "orange"},
                {"range": [3, 4], "color": "yellow"},
                {"range": [4, 5], "color": "green"}
            ]
        }
    
    async def get_dashboard_config(self, dashboard_id: str) -> Optional[DashboardLayout]:
        """Get dashboard configuration"""
        return self.dashboards.get(dashboard_id)
    
    async def create_custom_dashboard(self, dashboard_config: DashboardLayout) -> str:
        """Create a custom dashboard"""
        
        self.dashboards[dashboard_config.dashboard_id] = dashboard_config
        
        # Store in Redis for persistence
        config_key = f"dashboard:config:{dashboard_config.dashboard_id}"
        await self.redis.setex(
            config_key,
            3600 * 24 * 30,  # 30 days
            json.dumps(asdict(dashboard_config))
        )
        
        logger.info(f"📈 Created custom dashboard: {dashboard_config.dashboard_id}")
        return dashboard_config.dashboard_id
    
    async def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards"""
        
        dashboards = []
        for dashboard_id, dashboard in self.dashboards.items():
            dashboards.append({
                "id": dashboard_id,
                "name": dashboard.name,
                "description": dashboard.description,
                "type": dashboard.dashboard_type.value,
                "widget_count": len(dashboard.widgets)
            })
        
        return dashboards


# Create router for dashboard endpoints
dashboard_router = APIRouter(prefix="/api/dashboard", tags=["Real-time Dashboard"])

# Global dashboard instance (will be initialized in main.py)
_dashboard_instance: Optional[RealtimeDashboard] = None


def get_dashboard() -> RealtimeDashboard:
    """Get dashboard instance"""
    if _dashboard_instance is None:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    return _dashboard_instance


def initialize_dashboard(redis_client: redis.Redis, analytics_engine: AdvancedAnalyticsEngine) -> RealtimeDashboard:
    """Initialize dashboard system"""
    global _dashboard_instance
    _dashboard_instance = RealtimeDashboard(redis_client, analytics_engine)
    return _dashboard_instance


@dashboard_router.get("/dashboards")
async def get_dashboards(dashboard: RealtimeDashboard = Depends(get_dashboard)):
    """Get list of available dashboards"""
    return await dashboard.get_dashboard_list()


@dashboard_router.get("/dashboard/{dashboard_id}")
async def get_dashboard_config(
    dashboard_id: str,
    dashboard: RealtimeDashboard = Depends(get_dashboard)
):
    """Get dashboard configuration"""
    config = await dashboard.get_dashboard_config(dashboard_id)
    if not config:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return asdict(config)


@dashboard_router.websocket("/ws/{dashboard_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    dashboard_id: str,
    dashboard: RealtimeDashboard = Depends(get_dashboard)
):
    """WebSocket endpoint for real-time dashboard updates"""
    
    await dashboard.websocket_manager.connect(websocket, dashboard_id)
    
    try:
        # Start dashboard updates
        await dashboard.start_dashboard_updates(dashboard_id)
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle client messages (e.g., configuration changes)
                if message.get("type") == "refresh":
                    # Force refresh of dashboard data
                    dashboard_config = await dashboard.get_dashboard_config(dashboard_id)
                    if dashboard_config:
                        for widget in dashboard_config.widgets:
                            widget_data = await dashboard._get_widget_data(widget)
                            await websocket.send_text(json.dumps({
                                "type": "widget_update",
                                "widget_id": widget.widget_id,
                                "data": widget_data,
                                "timestamp": datetime.now().isoformat()
                            }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    
    finally:
        dashboard.websocket_manager.disconnect(websocket, dashboard_id)


@dashboard_router.get("/widget/{widget_id}/data")
async def get_widget_data(
    widget_id: str,
    dashboard_id: str,
    dashboard: RealtimeDashboard = Depends(get_dashboard)
):
    """Get data for a specific widget"""
    
    dashboard_config = await dashboard.get_dashboard_config(dashboard_id)
    if not dashboard_config:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    # Find widget
    widget = None
    for w in dashboard_config.widgets:
        if w.widget_id == widget_id:
            widget = w
            break
    
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")
    
    data = await dashboard._get_widget_data(widget)
    return data


# Export main components
__all__ = [
    "RealtimeDashboard",
    "DashboardLayout",
    "DashboardWidget",
    "DashboardType",
    "ChartType",
    "WebSocketManager",
    "dashboard_router",
    "initialize_dashboard",
    "get_dashboard"
]
