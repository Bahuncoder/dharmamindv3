/**
 * 📊 Advanced Analytics Dashboard Component
 * 
 * Comprehensive real-time analytics and performance monitoring dashboard
 * that connects to the DharmaMind backend analytics infrastructure.
 * 
 * Features:
 * - Real-time system metrics visualization
 * - User behavior analytics
 * - Performance monitoring with alerts
 * - Spiritual guidance insights
 * - Interactive charts and widgets
 * - WebSocket-powered live updates
 * - Mobile-responsive design
 * - Accessibility optimized
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChartBarIcon,
  CpuChipIcon,
  UserGroupIcon,
  HeartIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  EyeIcon,
  ChatBubbleLeftRightIcon,
  LightBulbIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';

// Types for dashboard data
interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  active_connections: number;
  response_time: number;
  error_rate: number;
}

interface UserAnalytics {
  active_users: number;
  total_sessions: number;
  average_session_duration: number;
  new_users_today: number;
  user_satisfaction: number;
  engagement_rate: number;
}

interface DharmaInsights {
  wisdom_requests: number;
  popular_topics: string[];
  satisfaction_score: number;
  guidance_effectiveness: number;
  spiritual_growth_metrics: number[];
}

interface PerformanceAlert {
  id: string;
  level: 'info' | 'warning' | 'error';
  message: string;
  timestamp: string;
  component?: string;
}

interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}

interface DashboardProps {
  dashboardId?: string;
  className?: string;
  refreshInterval?: number;
  isAdmin?: boolean;
  isMobile?: boolean;
  reduceMotion?: boolean;
  isHighContrast?: boolean;
}

const AdvancedAnalyticsDashboard: React.FC<DashboardProps> = ({
  dashboardId = 'system_overview',
  className = '',
  refreshInterval = 30000,
  isAdmin = false,
  isMobile = false,
  reduceMotion = false,
  isHighContrast = false
}) => {
  // State management
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpu_usage: 0,
    memory_usage: 0,
    disk_usage: 0,
    active_connections: 0,
    response_time: 0,
    error_rate: 0
  });

  const [userAnalytics, setUserAnalytics] = useState<UserAnalytics>({
    active_users: 0,
    total_sessions: 0,
    average_session_duration: 0,
    new_users_today: 0,
    user_satisfaction: 0,
    engagement_rate: 0
  });

  const [dharmaInsights, setDharmaInsights] = useState<DharmaInsights>({
    wisdom_requests: 0,
    popular_topics: [],
    satisfaction_score: 0,
    guidance_effectiveness: 0,
    spiritual_growth_metrics: []
  });

  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [chartData, setChartData] = useState<{
    [key: string]: ChartDataPoint[]
  }>({});

  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connecting');

  // Refs for cleanup
  const wsRef = useRef<WebSocket | null>(null);
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionStatus('connecting');

    try {
      // Connect to backend dashboard WebSocket
      const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/api/dashboard/ws/${dashboardId}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setIsConnected(true);
        setConnectionStatus('connected');
        setIsLoading(false);
        console.log('📊 Dashboard WebSocket connected');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
          setLastUpdate(new Date());
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onclose = () => {
        setIsConnected(false);
        setConnectionStatus('disconnected');
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('disconnected');
      };

    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
      setConnectionStatus('disconnected');
      // Fallback to REST API polling
      startPolling();
    }
  }, [dashboardId]);

  // Handle incoming WebSocket messages
  const handleWebSocketMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'system_metrics':
        setSystemMetrics(data.metrics);
        break;
      case 'user_analytics':
        setUserAnalytics(data.analytics);
        break;
      case 'dharma_insights':
        setDharmaInsights(data.insights);
        break;
      case 'performance_alert':
        setAlerts(prev => [data.alert, ...prev.slice(0, 9)]);
        break;
      case 'chart_data':
        setChartData(prev => ({
          ...prev,
          [data.chart_id]: data.data
        }));
        break;
      case 'widget_update':
        // Handle specific widget updates
        handleWidgetUpdate(data.widget_id, data.data);
        break;
    }
  }, []);

  // Handle widget-specific updates
  const handleWidgetUpdate = useCallback((widgetId: string, data: any) => {
    switch (widgetId) {
      case 'system_health':
        setSystemMetrics(prev => ({ ...prev, ...data }));
        break;
      case 'user_activity':
        setUserAnalytics(prev => ({ ...prev, ...data }));
        break;
      case 'wisdom_metrics':
        setDharmaInsights(prev => ({ ...prev, ...data }));
        break;
    }
  }, []);

  // Fallback REST API polling
  const startPolling = useCallback(() => {
    if (refreshTimerRef.current) {
      clearInterval(refreshTimerRef.current);
    }

    const poll = async () => {
      try {
        const response = await fetch(`/api/dashboard/${dashboardId}`);
        if (response.ok) {
          const data = await response.json();
          setSystemMetrics(data.system_metrics || systemMetrics);
          setUserAnalytics(data.user_analytics || userAnalytics);
          setDharmaInsights(data.dharma_insights || dharmaInsights);
          setLastUpdate(new Date());
        }
      } catch (error) {
        console.error('Error polling dashboard data:', error);
      }
    };

    // Initial poll
    poll();

    refreshTimerRef.current = setInterval(poll, refreshInterval);
  }, [dashboardId, refreshInterval, systemMetrics, userAnalytics, dharmaInsights]);

  // Lifecycle management
  useEffect(() => {
    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [connectWebSocket]);

  // Accessibility announcements
  const announceMetricChange = useCallback((metric: string, value: number) => {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      const utterance = new SpeechSynthesisUtterance(
        `${metric} updated to ${value}`
      );
      utterance.volume = 0.1;
      window.speechSynthesis.speak(utterance);
    }
  }, []);

  // Format numbers for display
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
  };

  // Format percentage
  const formatPercentage = (num: number): string => {
    return `${num.toFixed(1)}%`;
  };

  // Get status color based on value and thresholds
  const getStatusColor = (value: number, thresholds: { warning: number; danger: number }) => {
    if (isHighContrast) {
      return value >= thresholds.danger ? '#000000' : value >= thresholds.warning ? '#444444' : '#666666';
    }
    return value >= thresholds.danger ? '#ef4444' : value >= thresholds.warning ? '#f59e0b' : '#d4a854';
  };

  // Render connection status indicator
  const renderConnectionStatus = () => (
    <div className={`flex items-center gap-2 ${isMobile ? 'text-xs' : 'text-sm'}`}>
      <div
        className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-success-500' :
            connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
              'bg-red-500'
          }`}
        aria-label={`Connection status: ${connectionStatus}`}
      />
      <span className={`${isHighContrast ? 'text-black font-semibold' : 'text-gray-600'}`}>
        {connectionStatus === 'connected' ? 'Live' :
          connectionStatus === 'connecting' ? 'Connecting...' :
            'Disconnected'}
      </span>
      {lastUpdate && (
        <span className={`${isHighContrast ? 'text-black' : 'text-gray-500'} text-xs`}>
          Updated {lastUpdate.toLocaleTimeString()}
        </span>
      )}
    </div>
  );

  // Render metric card
  const renderMetricCard = (
    title: string,
    value: number | string,
    icon: React.ReactNode,
    trend?: number,
    format: 'number' | 'percentage' | 'custom' = 'number'
  ) => {
    const formattedValue = format === 'percentage' ? formatPercentage(Number(value)) :
      format === 'number' ? formatNumber(Number(value)) :
        value;

    return (
      <motion.div
        className={`${isHighContrast ? 'bg-white border-2 border-black' : 'bg-white/80 backdrop-blur-sm border border-gray-200/50'} rounded-xl p-4 shadow-lg ${isMobile ? 'min-h-[100px]' : 'min-h-[120px]'}`}
        initial={{ opacity: 0, y: reduceMotion ? 0 : 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: reduceMotion ? 0.1 : 0.3 }}
        whileHover={reduceMotion ? {} : { scale: 1.02 }}
        role="article"
        aria-label={`${title}: ${formattedValue}`}
      >
        <div className="flex items-center justify-between mb-2">
          <div className={`p-2 rounded-lg ${isHighContrast ? 'bg-gray-200' : 'bg-gold-100'}`}>
            {icon}
          </div>
          {trend !== undefined && (
            <div className={`flex items-center text-sm ${trend > 0 ? 'text-success-600' : trend < 0 ? 'text-red-600' : 'text-gray-600'
              }`}>
              <ArrowTrendingUpIcon
                className={`w-4 h-4 ${trend < 0 ? 'rotate-180' : ''}`}
              />
              <span>{Math.abs(trend)}%</span>
            </div>
          )}
        </div>
        <div>
          <h3 className={`${isHighContrast ? 'text-black font-bold' : 'text-gray-600 font-medium'} text-sm mb-1`}>
            {title}
          </h3>
          <p className={`${isHighContrast ? 'text-black font-bold text-2xl' : 'text-gray-900 font-bold text-xl'}`}>
            {formattedValue}
          </p>
        </div>
      </motion.div>
    );
  };

  // Render chart widget (simplified for demo)
  const renderChartWidget = (title: string, dataKey: string) => (
    <motion.div
      className={`${isHighContrast ? 'bg-white border-2 border-black' : 'bg-white/80 backdrop-blur-sm border border-gray-200/50'} rounded-xl p-6 shadow-lg`}
      initial={{ opacity: 0, scale: reduceMotion ? 1 : 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: reduceMotion ? 0.1 : 0.3 }}
    >
      <h3 className={`${isHighContrast ? 'text-black font-bold' : 'text-gray-900 font-semibold'} text-lg mb-4`}>
        {title}
      </h3>
      <div className={`h-48 ${isHighContrast ? 'bg-gray-100 border border-black' : 'bg-gray-50'} rounded-lg flex items-center justify-center`}>
        <div className="text-center">
          <ChartBarIcon className={`w-12 h-12 mx-auto mb-2 ${isHighContrast ? 'text-black' : 'text-gray-400'}`} />
          <p className={`${isHighContrast ? 'text-black font-semibold' : 'text-gray-600'} text-sm`}>
            Chart: {title}
          </p>
          <p className={`${isHighContrast ? 'text-black' : 'text-gray-500'} text-xs mt-1`}>
            {chartData[dataKey]?.length || 0} data points
          </p>
        </div>
      </div>
    </motion.div>
  );

  // Render alerts panel
  const renderAlertsPanel = () => (
    <motion.div
      className={`${isHighContrast ? 'bg-white border-2 border-black' : 'bg-white/80 backdrop-blur-sm border border-gray-200/50'} rounded-xl p-4 shadow-lg`}
      initial={{ opacity: 0, x: reduceMotion ? 0 : 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: reduceMotion ? 0.1 : 0.3 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <ExclamationTriangleIcon className={`w-5 h-5 ${isHighContrast ? 'text-black' : 'text-orange-500'}`} />
        <h3 className={`${isHighContrast ? 'text-black font-bold' : 'text-gray-900 font-semibold'} text-lg`}>
          Recent Alerts
        </h3>
      </div>

      <div className="space-y-2 max-h-60 overflow-y-auto">
        {alerts.length === 0 ? (
          <p className={`${isHighContrast ? 'text-black' : 'text-gray-500'} text-sm text-center py-4`}>
            No recent alerts
          </p>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-3 rounded-lg border-l-4 ${alert.level === 'error' ? 'border-red-500 bg-red-50' :
                  alert.level === 'warning' ? 'border-yellow-500 bg-yellow-50' :
                    'border-neutral-1000 bg-neutral-100'
                } ${isHighContrast ? 'border-2 border-black' : ''}`}
              role="alert"
              aria-label={`${alert.level} alert: ${alert.message}`}
            >
              <div className="flex justify-between items-start">
                <div>
                  <p className={`${isHighContrast ? 'text-black font-semibold' : 'text-gray-900'} text-sm font-medium`}>
                    {alert.message}
                  </p>
                  {alert.component && (
                    <p className={`${isHighContrast ? 'text-black' : 'text-gray-600'} text-xs mt-1`}>
                      Component: {alert.component}
                    </p>
                  )}
                </div>
                <span className={`${isHighContrast ? 'text-black' : 'text-gray-500'} text-xs`}>
                  {alert.timestamp}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </motion.div>
  );

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center p-8 ${className}`}>
        <div className="text-center">
          <motion.div
            animate={reduceMotion ? {} : { rotate: 360 }}
            transition={reduceMotion ? {} : { duration: 2, repeat: Infinity, ease: "linear" }}
            className={`w-8 h-8 border-2 border-t-transparent rounded-full mx-auto mb-4 ${isHighContrast ? 'border-black' : 'border-gold-500'
              }`}
          />
          <p className={`${isHighContrast ? 'text-black font-semibold' : 'text-gray-600'}`}>
            Loading dashboard...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`p-4 space-y-6 ${className} ${isMobile ? 'px-2' : ''}`}
      role="main"
      aria-label="Advanced Analytics Dashboard"
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className={`${isHighContrast ? 'text-black font-bold' : 'text-gray-900 font-semibold'} ${isMobile ? 'text-xl' : 'text-2xl'}`}>
            📊 Analytics Dashboard
          </h1>
          <p className={`${isHighContrast ? 'text-black' : 'text-gray-600'} text-sm mt-1`}>
            Real-time system monitoring and insights
          </p>
        </div>
        {renderConnectionStatus()}
      </div>

      {/* System Metrics Grid */}
      <div>
        <h2 className={`${isHighContrast ? 'text-black font-bold' : 'text-gray-800 font-semibold'} text-lg mb-4`}>
          System Performance
        </h2>
        <div className={`grid grid-cols-2 ${isMobile ? 'gap-3' : 'sm:grid-cols-3 lg:grid-cols-6 gap-4'}`}>
          {renderMetricCard(
            'CPU Usage',
            systemMetrics.cpu_usage,
            <CpuChipIcon className={`w-5 h-5 ${isHighContrast ? 'text-black' : 'text-gold-600'}`} />,
            undefined,
            'percentage'
          )}
          {renderMetricCard(
            'Memory',
            systemMetrics.memory_usage,
            <ChartBarIcon className={`w-5 h-5 ${isHighContrast ? 'text-black' : 'text-gold-600'}`} />,
            undefined,
            'percentage'
          )}
          {renderMetricCard(
            'Active Users',
            userAnalytics.active_users,
            <UserGroupIcon className={`w-5 h-5 ${isHighContrast ? 'text-black' : 'text-gold-600'}`} />,
            5
          )}
          {renderMetricCard(
            'Response Time',
            `${systemMetrics.response_time}ms`,
            <ClockIcon className={`w-5 h-5 ${isHighContrast ? 'text-black' : 'text-yellow-600'}`} />,
            -2,
            'custom'
          )}
          {renderMetricCard(
            'Satisfaction',
            userAnalytics.user_satisfaction,
            <HeartIcon className={`w-5 h-5 ${isHighContrast ? 'text-black' : 'text-red-600'}`} />,
            3,
            'percentage'
          )}
          {renderMetricCard(
            'Wisdom Requests',
            dharmaInsights.wisdom_requests,
            <SparklesIcon className={`w-5 h-5 ${isHighContrast ? 'text-black' : 'text-indigo-600'}`} />,
            8
          )}
        </div>
      </div>

      {/* Charts and Detailed Analytics */}
      <div className={`grid ${isMobile ? 'grid-cols-1 gap-4' : 'lg:grid-cols-2 gap-6'}`}>
        {renderChartWidget('User Activity Trends', 'user_activity')}
        {renderChartWidget('System Performance', 'system_performance')}
      </div>

      {/* Dharma Insights and Alerts */}
      <div className={`grid ${isMobile ? 'grid-cols-1 gap-4' : 'lg:grid-cols-3 gap-6'}`}>
        <div className="lg:col-span-2">
          {renderChartWidget('Spiritual Guidance Analytics', 'dharma_insights')}
        </div>
        <div>
          {renderAlertsPanel()}
        </div>
      </div>

      {/* Popular Topics */}
      {dharmaInsights.popular_topics.length > 0 && (
        <motion.div
          className={`${isHighContrast ? 'bg-white border-2 border-black' : 'bg-white/80 backdrop-blur-sm border border-gray-200/50'} rounded-xl p-6 shadow-lg`}
          initial={{ opacity: 0, y: reduceMotion ? 0 : 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: reduceMotion ? 0.1 : 0.3 }}
        >
          <div className="flex items-center gap-2 mb-4">
            <LightBulbIcon className={`w-5 h-5 ${isHighContrast ? 'text-black' : 'text-yellow-500'}`} />
            <h3 className={`${isHighContrast ? 'text-black font-bold' : 'text-gray-900 font-semibold'} text-lg`}>
              Popular Wisdom Topics
            </h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {dharmaInsights.popular_topics.map((topic, index) => (
              <span
                key={index}
                className={`px-3 py-1 rounded-full text-sm ${isHighContrast
                    ? 'bg-gray-200 text-black border border-black font-semibold'
                    : 'bg-gold-100 text-gold-800'
                  }`}
              >
                {topic}
              </span>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default AdvancedAnalyticsDashboard;