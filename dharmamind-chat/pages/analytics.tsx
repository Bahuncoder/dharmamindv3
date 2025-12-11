/**
 * 📊 Analytics Dashboard Page
 * 
 * Full-screen analytics dashboard with advanced monitoring capabilities
 * and real-time data visualization for DharmaMind platform.
 */

import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import AdvancedAnalyticsDashboard from '../components/AdvancedAnalyticsDashboard';
import { 
  Cog6ToothIcon, 
  ArrowLeftIcon,
  Square2StackIcon,
  ComputerDesktopIcon,
  DevicePhoneMobileIcon
} from '@heroicons/react/24/outline';

// Dashboard types and configurations
const DASHBOARD_TYPES = {
  system_overview: {
    id: 'system_overview',
    name: 'System Overview',
    description: 'Comprehensive system health and performance monitoring',
    icon: ComputerDesktopIcon
  },
  user_analytics: {
    id: 'user_analytics', 
    name: 'User Analytics',
    description: 'User behavior and engagement analysis',
    icon: Square2StackIcon
  },
  dharma_insights: {
    id: 'dharma_insights',
    name: 'Dharma Insights', 
    description: 'Spiritual guidance and wisdom analytics',
    icon: Square2StackIcon
  }
};

const AnalyticsPage: React.FC = () => {
  const router = useRouter();
  const [selectedDashboard, setSelectedDashboard] = useState('system_overview');
  const [isMobile, setIsMobile] = useState(false);
  const [isHighContrast, setIsHighContrast] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // Detect mobile and accessibility preferences
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    const checkAccessibilityPrefs = () => {
      // Check for reduced motion preference
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      setReduceMotion(prefersReducedMotion);

      // Check for high contrast preference
      const prefersHighContrast = window.matchMedia('(prefers-contrast: high)').matches;
      setIsHighContrast(prefersHighContrast);
    };

    checkMobile();
    checkAccessibilityPrefs();

    window.addEventListener('resize', checkMobile);
    
    const motionMediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const contrastMediaQuery = window.matchMedia('(prefers-contrast: high)');
    
    motionMediaQuery.addEventListener('change', checkAccessibilityPrefs);
    contrastMediaQuery.addEventListener('change', checkAccessibilityPrefs);

    return () => {
      window.removeEventListener('resize', checkMobile);
      motionMediaQuery.removeEventListener('change', checkAccessibilityPrefs);
      contrastMediaQuery.removeEventListener('change', checkAccessibilityPrefs);
    };
  }, []);

  // Handle dashboard selection
  const handleDashboardChange = (dashboardId: string) => {
    setSelectedDashboard(dashboardId);
    // Update URL without full page reload
    router.push(`/analytics?dashboard=${dashboardId}`, undefined, { shallow: true });
  };

  // Handle back navigation
  const handleBack = () => {
    router.back();
  };

  // Render dashboard selector
  const renderDashboardSelector = () => (
    <div className={`flex flex-wrap gap-2 ${isMobile ? 'mb-4' : 'mb-6'}`}>
      {Object.values(DASHBOARD_TYPES).map((dashboard) => {
        const Icon = dashboard.icon;
        const isSelected = selectedDashboard === dashboard.id;
        
        return (
          <motion.button
            key={dashboard.id}
            onClick={() => handleDashboardChange(dashboard.id)}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-lg border transition-all
              ${isSelected 
                ? isHighContrast 
                  ? 'bg-black text-white border-black' 
                  : 'bg-gold-500 text-white border-gold-500'
                : isHighContrast
                  ? 'bg-white text-black border-black hover:bg-gray-100'
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
              }
              ${isMobile ? 'text-sm px-3 py-2' : ''}
            `}
            whileHover={reduceMotion ? {} : { scale: 1.02 }}
            whileTap={reduceMotion ? {} : { scale: 0.98 }}
            aria-pressed={isSelected}
            aria-label={`Switch to ${dashboard.name} dashboard`}
          >
            <Icon className={`w-4 h-4 ${isMobile ? 'w-3 h-3' : ''}`} />
            <span className={isMobile ? 'text-xs' : 'text-sm'}>
              {dashboard.name}
            </span>
          </motion.button>
        );
      })}
    </div>
  );

  // Render settings panel
  const renderSettingsPanel = () => (
    <motion.div
      initial={{ opacity: 0, x: 300 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 300 }}
      className={`
        fixed top-0 right-0 h-full w-80 z-50 p-6 shadow-2xl
        ${isHighContrast ? 'bg-white border-l-2 border-black' : 'bg-white border-l border-gray-200'}
        ${isMobile ? 'w-full' : ''}
      `}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className={`text-lg font-semibold ${isHighContrast ? 'text-black' : 'text-gray-900'}`}>
          Dashboard Settings
        </h3>
        <button
          onClick={() => setShowSettings(false)}
          className={`p-2 rounded-lg ${isHighContrast ? 'hover:bg-gray-100' : 'hover:bg-gray-100'}`}
          aria-label="Close settings"
        >
          <ArrowLeftIcon className={`w-5 h-5 ${isHighContrast ? 'text-black' : 'text-gray-600'}`} />
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label className={`block text-sm font-medium mb-2 ${isHighContrast ? 'text-black' : 'text-gray-700'}`}>
            Display Options
          </label>
          <div className="space-y-2">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={isHighContrast}
                onChange={(e) => setIsHighContrast(e.target.checked)}
                className="mr-2"
              />
              <span className={`text-sm ${isHighContrast ? 'text-black' : 'text-gray-600'}`}>
                High Contrast Mode
              </span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={reduceMotion}
                onChange={(e) => setReduceMotion(e.target.checked)}
                className="mr-2"
              />
              <span className={`text-sm ${isHighContrast ? 'text-black' : 'text-gray-600'}`}>
                Reduce Motion
              </span>
            </label>
          </div>
        </div>

        <div>
          <label className={`block text-sm font-medium mb-2 ${isHighContrast ? 'text-black' : 'text-gray-700'}`}>
            View Mode
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => setIsMobile(false)}
              className={`
                flex items-center gap-2 px-3 py-2 rounded-lg text-sm border
                ${!isMobile 
                  ? isHighContrast ? 'bg-black text-white' : 'bg-gold-500 text-white'
                  : isHighContrast ? 'bg-white text-black border-black' : 'bg-white text-gray-600 border-gray-300'
                }
              `}
            >
              <ComputerDesktopIcon className="w-4 h-4" />
              Desktop
            </button>
            <button
              onClick={() => setIsMobile(true)}
              className={`
                flex items-center gap-2 px-3 py-2 rounded-lg text-sm border
                ${isMobile 
                  ? isHighContrast ? 'bg-black text-white' : 'bg-gold-500 text-white'
                  : isHighContrast ? 'bg-white text-black border-black' : 'bg-white text-gray-600 border-gray-300'
                }
              `}
            >
              <DevicePhoneMobileIcon className="w-4 h-4" />
              Mobile
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );

  return (
    <>
      <Head>
        <title>Analytics Dashboard - DharmaMind</title>
        <meta name="description" content="Advanced analytics and monitoring dashboard for DharmaMind platform" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className={`min-h-screen ${isHighContrast ? 'bg-white' : 'bg-gradient-to-br from-gold-50 to-neutral-100'}`}>
        {/* Header */}
        <div className={`${isHighContrast ? 'bg-white border-b-2 border-black' : 'bg-white/80 backdrop-blur-sm border-b border-gray-200'} sticky top-0 z-40`}>
          <div className={`${isMobile ? 'px-4 py-3' : 'px-6 py-4'}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <button
                  onClick={handleBack}
                  className={`p-2 rounded-lg ${isHighContrast ? 'hover:bg-gray-100 text-black' : 'hover:bg-gray-100 text-gray-600'}`}
                  aria-label="Go back"
                >
                  <ArrowLeftIcon className="w-5 h-5" />
                </button>
                <div>
                  <h1 className={`${isHighContrast ? 'text-black font-bold' : 'text-gray-900 font-semibold'} ${isMobile ? 'text-lg' : 'text-xl'}`}>
                    Analytics Dashboard
                  </h1>
                  <p className={`${isHighContrast ? 'text-black' : 'text-gray-600'} text-sm`}>
                    {DASHBOARD_TYPES[selectedDashboard as keyof typeof DASHBOARD_TYPES]?.description}
                  </p>
                </div>
              </div>
              
              <button
                onClick={() => setShowSettings(true)}
                className={`p-2 rounded-lg ${isHighContrast ? 'hover:bg-gray-100 text-black' : 'hover:bg-gray-100 text-gray-600'}`}
                aria-label="Open dashboard settings"
              >
                <Cog6ToothIcon className="w-5 h-5" />
              </button>
            </div>
            
            <div className="mt-4">
              {renderDashboardSelector()}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <main className={`${isMobile ? 'p-4' : 'p-6'}`}>
          <AdvancedAnalyticsDashboard
            dashboardId={selectedDashboard}
            isMobile={isMobile}
            isHighContrast={isHighContrast}
            reduceMotion={reduceMotion}
          />
        </main>

        {/* Settings Panel */}
        {showSettings && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black bg-opacity-50 z-40"
              onClick={() => setShowSettings(false)}
            />
            {renderSettingsPanel()}
          </>
        )}
      </div>
    </>
  );
};

export default AnalyticsPage;