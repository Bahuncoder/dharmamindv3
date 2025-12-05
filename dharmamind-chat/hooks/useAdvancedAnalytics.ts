/**
 * 📊 Advanced Analytics Dashboard Hook
 * 
 * Custom React hook for managing dashboard data, WebSocket connections,
 * and real-time analytics integration with the DharmaMind backend.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

// Types
interface DashboardMetrics {
  system: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    active_connections: number;
    response_time: number;
    error_rate: number;
  };
  user: {
    active_users: number;
    total_sessions: number;
    average_session_duration: number;
    new_users_today: number;
    user_satisfaction: number;
    engagement_rate: number;
  };
  dharma: {
    wisdom_requests: number;
    popular_topics: string[];
    satisfaction_score: number;
    guidance_effectiveness: number;
    spiritual_growth_metrics: number[];
  };
}

interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}

interface PerformanceAlert {
  id: string;
  level: 'info' | 'warning' | 'error';
  message: string;
  timestamp: string;
  component?: string;
}

interface DashboardHookOptions {
  dashboardId: string;
  refreshInterval?: number;
  enableWebSocket?: boolean;
  enablePolling?: boolean;
  apiBaseUrl?: string;
  wsBaseUrl?: string;
}

interface DashboardState {
  metrics: DashboardMetrics;
  chartData: { [key: string]: ChartDataPoint[] };
  alerts: PerformanceAlert[];
  isConnected: boolean;
  isLoading: boolean;
  lastUpdate: Date | null;
  connectionStatus: 'connected' | 'connecting' | 'disconnected';
  error: string | null;
}

const useAdvancedAnalytics = (options: DashboardHookOptions) => {
  const {
    dashboardId,
    refreshInterval = 30000,
    enableWebSocket = true,
    enablePolling = true,
    apiBaseUrl = '/api',
    wsBaseUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
  } = options;

  // State
  const [state, setState] = useState<DashboardState>({
    metrics: {
      system: {
        cpu_usage: 0,
        memory_usage: 0,
        disk_usage: 0,
        active_connections: 0,
        response_time: 0,
        error_rate: 0
      },
      user: {
        active_users: 0,
        total_sessions: 0,
        average_session_duration: 0,
        new_users_today: 0,
        user_satisfaction: 0,
        engagement_rate: 0
      },
      dharma: {
        wisdom_requests: 0,
        popular_topics: [],
        satisfaction_score: 0,
        guidance_effectiveness: 0,
        spiritual_growth_metrics: []
      }
    },
    chartData: {},
    alerts: [],
    isConnected: false,
    isLoading: true,
    lastUpdate: null,
    connectionStatus: 'connecting',
    error: null
  });

  // Refs for cleanup
  const wsRef = useRef<WebSocket | null>(null);
  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Update state helper
  const updateState = useCallback((updates: Partial<DashboardState>) => {
    setState(prev => ({ ...prev, ...updates }));
  }, []);

  // WebSocket message handler
  const handleWebSocketMessage = useCallback((data: any) => {
    try {
      switch (data.type) {
        case 'metrics_update':
          updateState({ 
            metrics: data.metrics,
            lastUpdate: new Date()
          });
          break;

        case 'chart_data':
          updateState({
            chartData: {
              ...state.chartData,
              [data.chart_id]: data.data
            }
          });
          break;

        case 'performance_alert':
          updateState({
            alerts: [data.alert, ...state.alerts.slice(0, 9)]
          });
          break;

        case 'system_metrics':
          updateState({
            metrics: {
              ...state.metrics,
              system: data.metrics
            },
            lastUpdate: new Date()
          });
          break;

        case 'user_analytics':
          updateState({
            metrics: {
              ...state.metrics,
              user: data.analytics
            },
            lastUpdate: new Date()
          });
          break;

        case 'dharma_insights':
          updateState({
            metrics: {
              ...state.metrics,
              dharma: data.insights
            },
            lastUpdate: new Date()
          });
          break;

        case 'widget_update':
          handleWidgetUpdate(data.widget_id, data.data);
          break;

        default:
          console.log('Unknown WebSocket message type:', data.type);
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
      updateState({ error: 'Failed to process real-time update' });
    }
  }, [state, updateState]);

  // Handle widget-specific updates
  const handleWidgetUpdate = useCallback((widgetId: string, data: any) => {
    switch (widgetId) {
      case 'system_health':
        updateState({
          metrics: {
            ...state.metrics,
            system: { ...state.metrics.system, ...data }
          }
        });
        break;

      case 'user_activity':
        updateState({
          metrics: {
            ...state.metrics,
            user: { ...state.metrics.user, ...data }
          }
        });
        break;

      case 'wisdom_metrics':
        updateState({
          metrics: {
            ...state.metrics,
            dharma: { ...state.metrics.dharma, ...data }
          }
        });
        break;
    }
  }, [state, updateState]);

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (!enableWebSocket) return;

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    updateState({ connectionStatus: 'connecting' });

    try {
      const wsUrl = `${wsBaseUrl}/api/dashboard/ws/${dashboardId}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('📊 Dashboard WebSocket connected');
        updateState({
          isConnected: true,
          connectionStatus: 'connected',
          isLoading: false,
          error: null
        });
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          updateState({ error: 'Invalid data received from server' });
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        updateState({
          isConnected: false,
          connectionStatus: 'disconnected'
        });

        // Attempt to reconnect after 5 seconds if not intentionally closed
        if (event.code !== 1000) {
          retryTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, 5000);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateState({
          connectionStatus: 'disconnected',
          error: 'Connection error'
        });
      };

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      updateState({
        connectionStatus: 'disconnected',
        error: 'Failed to establish connection'
      });

      // Fallback to polling
      if (enablePolling) {
        startPolling();
      }
    }
  }, [enableWebSocket, wsBaseUrl, dashboardId, handleWebSocketMessage, enablePolling]);

  // REST API data fetching
  const fetchDashboardData = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/dashboard/${dashboardId}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      updateState({
        metrics: {
          system: data.system_metrics || state.metrics.system,
          user: data.user_analytics || state.metrics.user,
          dharma: data.dharma_insights || state.metrics.dharma
        },
        chartData: data.chart_data || state.chartData,
        alerts: data.alerts || state.alerts,
        lastUpdate: new Date(),
        isLoading: false,
        error: null
      });

    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      updateState({
        error: error instanceof Error ? error.message : 'Failed to fetch data',
        isLoading: false
      });
    }
  }, [apiBaseUrl, dashboardId, state, updateState]);

  // Polling mechanism
  const startPolling = useCallback(() => {
    if (!enablePolling) return;

    // Clear existing timer
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
    }

    // Initial fetch
    fetchDashboardData();

    // Set up polling
    pollTimerRef.current = setInterval(fetchDashboardData, refreshInterval);
  }, [enablePolling, fetchDashboardData, refreshInterval]);

  // Manual refresh
  const refresh = useCallback(async () => {
    updateState({ isLoading: true, error: null });
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Request refresh via WebSocket
      wsRef.current.send(JSON.stringify({ type: 'refresh' }));
    } else {
      // Fallback to REST API
      await fetchDashboardData();
    }
  }, [fetchDashboardData]);

  // Send custom message via WebSocket
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close(1000, 'Component unmounting');
      wsRef.current = null;
    }

    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
  }, []);

  // Initialize connection
  useEffect(() => {
    if (enableWebSocket) {
      connectWebSocket();
    } else if (enablePolling) {
      startPolling();
    }

    return cleanup;
  }, [connectWebSocket, startPolling, cleanup, enableWebSocket, enablePolling]);

  // Dashboard ID change handler
  useEffect(() => {
    cleanup();
    updateState({ isLoading: true, error: null });
    
    if (enableWebSocket) {
      connectWebSocket();
    } else if (enablePolling) {
      startPolling();
    }
  }, [dashboardId, connectWebSocket, startPolling, cleanup, enableWebSocket, enablePolling]);

  return {
    // State
    ...state,
    
    // Actions
    refresh,
    sendMessage,
    
    // Utils
    formatNumber: (num: number): string => {
      if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
      if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
      return num.toString();
    },
    
    formatPercentage: (num: number): string => `${num.toFixed(1)}%`,
    
    getStatusColor: (value: number, thresholds: { warning: number; danger: number }) => {
      return value >= thresholds.danger ? '#ef4444' : 
             value >= thresholds.warning ? '#f59e0b' : '#10b981';
    }
  };
};

export default useAdvancedAnalytics;
export type { DashboardMetrics, ChartDataPoint, PerformanceAlert, DashboardHookOptions, DashboardState };