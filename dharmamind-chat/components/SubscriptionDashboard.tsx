/**
 * 🕉️ DharmaMind Subscription Dashboard
 * 
 * Enhanced subscription management with usage analytics,
 * real-time insights, and upgrade recommendations
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useSubscription } from '../hooks/useSubscription';
import { useColor } from '../contexts/ColorContext';
import { UpgradePrompt, UsageProgress } from './UpgradePrompt';
import CentralizedSubscriptionModal from './CentralizedSubscriptionModal';
import Logo from './Logo';

// ===============================
// TYPES
// ===============================

interface UsageInsight {
  title: string;
  value: string;
  trend: 'up' | 'down' | 'stable';
  description: string;
  color: string;
}

interface PredictiveInsight {
  type: 'usage_prediction' | 'upgrade_recommendation' | 'feature_suggestion';
  title: string;
  description: string;
  actionText?: string;
  action?: () => void;
  priority: 'high' | 'medium' | 'low';
}

// ===============================
// SUBSCRIPTION DASHBOARD
// ===============================

export const SubscriptionDashboard: React.FC = () => {
  const {
    currentSubscription,
    currentPlan,
    usage,
    checkFeatureUsage,
    isFreePlan,
    isPremiumPlan,
    isEnterprisePlan,
    formatPrice,
    isLoading
  } = useSubscription();

  const { currentTheme } = useColor();

  const [showSubscriptionModal, setShowSubscriptionModal] = useState(false);
  const [insights, setInsights] = useState<UsageInsight[]>([]);
  const [predictions, setPredictions] = useState<PredictiveInsight[]>([]);

  // ===============================
  // GENERATE INSIGHTS
  // ===============================

  useEffect(() => {
    generateUsageInsights();
    generatePredictiveInsights();
  }, [usage, currentPlan]);

  const generateUsageInsights = () => {
    if (!usage) return;

    const messageCheck = checkFeatureUsage('messages');
    const moduleCheck = checkFeatureUsage('wisdom_modules');

    const newInsights: UsageInsight[] = [
      {
        title: 'Messages This Month',
        value: `${messageCheck.usage}/${messageCheck.limit === -1 ? '∞' : messageCheck.limit}`,
        trend: messageCheck.percentage > 80 ? 'up' : messageCheck.percentage > 50 ? 'stable' : 'down',
        description: `${messageCheck.percentage.toFixed(1)}% of monthly limit used`,
        color: messageCheck.percentage > 90 ? 'text-red-600' : messageCheck.percentage > 80 ? 'text-gray-600' : 'text-green-600'
      },
      {
        title: 'Wisdom Modules Accessed',
        value: `${moduleCheck.usage}/${moduleCheck.limit === -1 ? '∞' : moduleCheck.limit}`,
        trend: moduleCheck.usage > 3 ? 'up' : 'stable',
        description: 'Different modules explored',
        color: 'text-blue-600'
      },
      {
        title: 'Days Until Reset',
        value: calculateDaysUntilReset().toString(),
        trend: 'stable',
        description: 'Until usage resets',
        color: 'text-purple-600'
      },
      {
        title: 'Plan Utilization',
        value: `${Math.round(messageCheck.percentage)}%`,
        trend: messageCheck.percentage > 50 ? 'up' : 'down',
        description: 'Overall plan usage',
        color: messageCheck.percentage > 80 ? 'text-red-600' : 'text-green-600'
      }
    ];

    setInsights(newInsights);
  };

  const generatePredictiveInsights = () => {
    if (!usage) return;

    const messageCheck = checkFeatureUsage('messages');
    const moduleCheck = checkFeatureUsage('wisdom_modules');
    const newPredictions: PredictiveInsight[] = [];

    // Usage prediction
    if (isFreePlan() && messageCheck.percentage > 80) {
      newPredictions.push({
        type: 'usage_prediction',
        title: 'Approaching Message Limit',
        description: `At your current pace, you'll reach your message limit in ${Math.ceil((messageCheck.limit - messageCheck.usage) / 2)} days.`,
        actionText: 'Upgrade Now',
        action: () => setShowSubscriptionModal(true),
        priority: 'high'
      });
    }

    // Upgrade recommendation
    if (isFreePlan() && messageCheck.usage > 30) {
      newPredictions.push({
        type: 'upgrade_recommendation',
        title: 'Consider Professional Plan',
        description: 'You\'re an active user! Professional plan offers unlimited messages and advanced features.',
        actionText: 'View Professional',
        action: () => setShowSubscriptionModal(true),
        priority: 'medium'
      });
    }

    // Feature suggestion
    if (isPremiumPlan() && moduleCheck.usage < 5) {
      newPredictions.push({
        type: 'feature_suggestion',
        title: 'Explore More Modules',
        description: 'You have access to all 32 wisdom modules. Try Advanced Meditation or Consciousness Studies!',
        priority: 'low'
      });
    }

    setPredictions(newPredictions);
  };

  const calculateDaysUntilReset = (): number => {
    if (!usage) return 0;
    const resetDate = new Date(usage.resetDate);
    const now = new Date();
    const diffTime = resetDate.getTime() - now.getTime();
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  };

  // ===============================
  // COMPONENTS
  // ===============================

  const InsightCard: React.FC<{ insight: UsageInsight; index: number }> = ({ insight, index }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="bg-white rounded-lg p-6 shadow-sm border border-gray-200"
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-600">{insight.title}</h3>
        <div className={`text-xs ${
          insight.trend === 'up' ? 'text-green-600' : 
          insight.trend === 'down' ? 'text-red-600' : 'text-gray-600'
        }`}>
          {insight.trend === 'up' ? '↗️' : insight.trend === 'down' ? '↘️' : '➡️'}
        </div>
      </div>
      <div className={`text-2xl font-bold ${insight.color} mb-1`}>
        {insight.value}
      </div>
      <p className="text-xs text-gray-500">{insight.description}</p>
    </motion.div>
  );

  const PredictionCard: React.FC<{ prediction: PredictiveInsight; index: number }> = ({ prediction, index }) => (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      className={`rounded-lg p-4 border-l-4 ${
        prediction.priority === 'high' ? 'border-red-500 bg-red-50' :
        prediction.priority === 'medium' ? 'border-gray-500 bg-gray-50' :
        'border-blue-500 bg-blue-50'
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h4 className={`font-medium text-sm ${
            prediction.priority === 'high' ? 'text-red-800' :
            prediction.priority === 'medium' ? 'text-gray-800' :
            'text-blue-800'
          }`}>
            {prediction.title}
          </h4>
          <p className={`text-xs mt-1 ${
            prediction.priority === 'high' ? 'text-red-700' :
            prediction.priority === 'medium' ? 'text-gray-700' :
            'text-blue-700'
          }`}>
            {prediction.description}
          </p>
        </div>
        {prediction.action && prediction.actionText && (
          <button
            onClick={prediction.action}
            className={`ml-3 px-3 py-1 text-xs font-medium rounded-full transition-colors ${
              prediction.priority === 'high' ? 'bg-red-600 hover:bg-red-700 text-white' :
              prediction.priority === 'medium' ? 'bg-gray-600 hover:bg-gray-700 text-white' :
              'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {prediction.actionText}
          </button>
        )}
      </div>
    </motion.div>
  );

  const FeatureComparisonCard: React.FC = () => (
    <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-6 border border-purple-200">
      <h3 className="text-lg font-semibold text-purple-900 mb-4">Plan Comparison</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Free Plan */}
        <div className={`bg-white rounded-lg p-4 border-2 ${
          isFreePlan() ? 'border-purple-500 ring-2 ring-purple-200' : 'border-gray-200'
        }`}>
          <div className="text-center mb-3">
            <h4 className="font-semibold text-gray-900">Free</h4>
            <p className="text-2xl font-bold text-gray-900">$0</p>
            <p className="text-xs text-gray-500">per month</p>
          </div>
          <ul className="space-y-1 text-xs text-gray-600">
            <li>✓ 50 messages/month</li>
            <li>✓ 5 basic wisdom modules</li>
            <li>✓ Community support</li>
            <li>✗ Advanced features</li>
          </ul>
        </div>

        {/* Professional Plan */}
        <div className={`bg-white rounded-lg p-4 border-2 ${
          isPremiumPlan() ? 'border-purple-500 ring-2 ring-purple-200' : 'border-gray-200'
        }`}>
          <div className="text-center mb-3">
            <h4 className="font-semibold text-purple-900">Professional</h4>
            <p className="text-2xl font-bold text-purple-900">{formatPrice(995)}</p>
            <p className="text-xs text-gray-500">per month</p>
          </div>
          <ul className="space-y-1 text-xs text-gray-600">
            <li>✓ Unlimited messages</li>
            <li>✓ All 32 wisdom modules</li>
            <li>✓ Advanced meditation guides</li>
            <li>✓ Priority support</li>
          </ul>
          {!isPremiumPlan() && (
            <button
              onClick={() => setShowSubscriptionModal(true)}
              className="w-full mt-3 px-3 py-2 bg-purple-600 text-white text-xs font-medium rounded-lg hover:bg-purple-700 transition-colors"
            >
              Upgrade Now
            </button>
          )}
        </div>

        {/* Enterprise Plan */}
        <div className={`bg-white rounded-lg p-4 border-2 ${
          isEnterprisePlan() ? 'border-purple-500 ring-2 ring-purple-200' : 'border-gray-200'
        }`}>
          <div className="text-center mb-3">
            <h4 className="font-semibold text-purple-900">Enterprise</h4>
            <p className="text-2xl font-bold text-purple-900">{formatPrice(2995)}</p>
            <p className="text-xs text-gray-500">per month</p>
          </div>
          <ul className="space-y-1 text-xs text-gray-600">
            <li>✓ Everything in Professional</li>
            <li>✓ Business dharma modules</li>
            <li>✓ Team collaboration</li>
            <li>✓ Custom integrations</li>
          </ul>
          {!isEnterprisePlan() && (
            <button
              onClick={() => setShowSubscriptionModal(true)}
              className="w-full mt-3 px-3 py-2 bg-purple-600 text-white text-xs font-medium rounded-lg hover:bg-purple-700 transition-colors"
            >
              Contact Sales
            </button>
          )}
        </div>
      </div>
    </div>
  );

  // ===============================
  // RENDER
  // ===============================

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600 mx-auto mb-2"></div>
          <p className="text-gray-600">Loading subscription data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Subscription Dashboard</h1>
          <p className="text-gray-600">
            Current Plan: <span className="font-medium text-purple-600">
              {currentPlan?.name || 'Basic'}
            </span>
          </p>
        </div>
        
        {isFreePlan() && (
          <button
            onClick={() => setShowSubscriptionModal(true)}
            className="px-4 py-2 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 transition-colors"
          >
            Upgrade Plan
          </button>
        )}
      </div>

      {/* Usage Insights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {insights.map((insight, index) => (
          <InsightCard key={insight.title} insight={insight} index={index} />
        ))}
      </div>

      {/* Usage Progress Bars */}
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Usage</h3>
        <div className="space-y-4">
          <UsageProgress feature="messages" />
          <UsageProgress feature="wisdom_modules" />
        </div>
      </div>

      {/* Predictive Insights */}
      {predictions.length > 0 && (
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Insights & Recommendations</h3>
          <div className="space-y-3">
            {predictions.map((prediction, index) => (
              <PredictionCard key={prediction.title} prediction={prediction} index={index} />
            ))}
          </div>
        </div>
      )}

      {/* Feature Comparison */}
      <FeatureComparisonCard />

      {/* Upgrade Prompt for Free Users */}
      {isFreePlan() && (
        <UpgradePrompt
          feature="messages"
          trigger="feature"
          onUpgrade={() => setShowSubscriptionModal(true)}
        />
      )}

      {/* Subscription Modal */}
      <CentralizedSubscriptionModal
        isOpen={showSubscriptionModal}
        onClose={() => setShowSubscriptionModal(false)}
      />
    </div>
  );
};

export default SubscriptionDashboard;
