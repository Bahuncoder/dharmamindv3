/**
 * 🕉️ DharmaMind Upgrade Prompt Component
 * 
 * Smart upgrade prompts that appear when users approach
 * or reach their subscription limits
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSubscription } from '../hooks/useSubscription';
import { useColors } from '../contexts/ColorContext';
import Logo from './Logo';

// ===============================
// TYPES
// ===============================

interface UpgradePromptProps {
  feature?: 'messages' | 'wisdom_modules' | 'api_requests';
  trigger?: 'warning' | 'limit' | 'feature';
  onUpgrade?: () => void;
  onDismiss?: () => void;
  className?: string;
}

// ===============================
// UPGRADE PROMPT COMPONENT
// ===============================

export const UpgradePrompt: React.FC<UpgradePromptProps> = ({
  feature = 'messages',
  trigger = 'warning',
  onUpgrade,
  onDismiss,
  className = ''
}) => {
  const {
    shouldShowUpgradePrompt,
    getUpgradeMessage,
    checkFeatureUsage,
    isFreePlan,
    formatPrice
  } = useSubscription();

  const { currentTheme } = useColors();

  // Don't show if not on free plan
  if (!isFreePlan()) return null;

  // Check if we should show the prompt
  const shouldShow = shouldShowUpgradePrompt(feature);
  if (!shouldShow && trigger !== 'feature') return null;

  const featureCheck = checkFeatureUsage(feature);
  const message = getUpgradeMessage(feature);

  const getPromptStyle = () => {
    const primaryColor = currentTheme.colors.primary;
    const primaryHover = currentTheme.colors.primaryHover;
    
    switch (trigger) {
      case 'limit':
        return {
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
          textColor: 'text-red-800',
          buttonStyle: { backgroundColor: '#dc2626', borderColor: '#dc2626' },
          buttonHoverStyle: { backgroundColor: '#b91c1c', borderColor: '#b91c1c' },
          icon: '🚫'
        };
      case 'warning':
        return {
          bgColor: 'bg-yellow-50',
          borderColor: 'border-yellow-200',
          textColor: 'text-yellow-800',
          buttonStyle: { backgroundColor: '#d97706', borderColor: '#d97706' },
          buttonHoverStyle: { backgroundColor: '#b45309', borderColor: '#b45309' },
          icon: '⚠️'
        };
      case 'feature':
        return {
          bgColor: 'bg-blue-50',
          borderColor: 'border-blue-200',
          textColor: 'text-blue-800',
          buttonStyle: { backgroundColor: primaryColor, borderColor: primaryColor },
          buttonHoverStyle: { backgroundColor: primaryHover, borderColor: primaryHover },
          icon: '✨'
        };
      default:
        return {
          bgColor: 'bg-gradient-to-br from-orange-50 to-amber-50',
          borderColor: 'border-orange-200',
          textColor: 'text-orange-800',
          buttonStyle: { backgroundColor: primaryColor, borderColor: primaryColor },
          buttonHoverStyle: { backgroundColor: primaryHover, borderColor: primaryHover },
          icon: '🕉️'
        };
    }
  };

  const style = getPromptStyle();

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className={`${style.bgColor} ${style.borderColor} border rounded-lg p-4 ${className}`}
    >
      <div className="flex items-start space-x-3">
        <div className="text-2xl">{style.icon}</div>
        
        <div className="flex-1">
          <h3 className={`font-semibold ${style.textColor} mb-2`}>
            {trigger === 'limit' && 'Limit Reached'}
            {trigger === 'warning' && 'Approaching Limit'}
            {trigger === 'feature' && 'Unlock Premium Features'}
          </h3>
          
          <p className={`text-sm ${style.textColor} mb-3`}>
            {message}
          </p>
          
          {/* Usage Progress (for warning/limit) */}
          {(trigger === 'warning' || trigger === 'limit') && (
            <div className="mb-3">
              <div className="flex justify-between text-xs mb-1">
                <span className={style.textColor}>
                  {feature === 'messages' && 'Messages'}
                  {feature === 'wisdom_modules' && 'Wisdom Modules'}
                  {feature === 'api_requests' && 'API Requests'}
                </span>
                <span className={style.textColor}>
                  {featureCheck.usage}/{featureCheck.limit === -1 ? '∞' : featureCheck.limit}
                </span>
              </div>
              <div className="w-full bg-white bg-opacity-50 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${
                    featureCheck.percentage >= 100 ? 'bg-red-500' :
                    featureCheck.percentage >= 90 ? 'bg-orange-500' :
                    featureCheck.percentage >= 80 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${Math.min(100, featureCheck.percentage)}%` }}
                />
              </div>
            </div>
          )}
          
          {/* Premium Features List */}
          {trigger === 'feature' && (
            <div className={`text-xs ${style.textColor} mb-3`}>
              <div className="grid grid-cols-1 gap-1">
                <div>✓ Unlimited spiritual conversations</div>
                <div>✓ Access to all 32 wisdom modules</div>
                <div>✓ Advanced meditation guidance</div>
                <div>✓ Priority AI responses</div>
                <div>✓ Personalized spiritual insights</div>
              </div>
            </div>
          )}
          
          {/* Action Buttons */}
          <div className="flex items-center space-x-2">
            <button
              onClick={onUpgrade}
              style={style.buttonStyle}
              className="px-4 py-2 text-white text-sm font-medium rounded-lg transition-colors hover:opacity-90"
              onMouseEnter={(e) => {
                Object.assign(e.currentTarget.style, style.buttonHoverStyle);
              }}
              onMouseLeave={(e) => {
                Object.assign(e.currentTarget.style, style.buttonStyle);
              }}
            >
              {trigger === 'limit' ? 'Upgrade Now' : 'View Plans'}
            </button>
            
            <div className="text-xs text-gray-600">
              Starting at {formatPrice(995)} per month
            </div>
            
            {onDismiss && trigger !== 'limit' && (
              <button
                onClick={onDismiss}
                className={`text-xs ${style.textColor} hover:underline ml-auto`}
              >
                Maybe later
              </button>
            )}
          </div>
        </div>
        
        {/* Close button (only for warnings) */}
        {onDismiss && trigger === 'warning' && (
          <button
            onClick={onDismiss}
            className={`${style.textColor} hover:opacity-75 text-lg leading-none`}
          >
            ×
          </button>
        )}
      </div>
    </motion.div>
  );
};

// ===============================
// MINI UPGRADE BANNER
// ===============================

interface MiniUpgradeBannerProps {
  onUpgrade?: () => void;
  className?: string;
}

export const MiniUpgradeBanner: React.FC<MiniUpgradeBannerProps> = ({
  onUpgrade,
  className = ''
}) => {
  const { usage, isFreePlan, checkFeatureUsage, formatPrice } = useSubscription();
  const { currentTheme } = useColors();

  if (!isFreePlan() || !usage) return null;

  const messageCheck = checkFeatureUsage('messages');
  const isNearLimit = messageCheck.percentage >= 75;

  if (!isNearLimit) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`text-white p-2 text-center ${className}`}
      style={{
        background: `linear-gradient(45deg, ${currentTheme.colors.primaryStart}, ${currentTheme.colors.primaryEnd})`
      }}
    >
      <div className="flex items-center justify-center space-x-4 text-sm">
        <span>
          🕉️ {messageCheck.remaining} messages left this month
        </span>
        <button
          onClick={onUpgrade}
          className="bg-white bg-opacity-20 hover:bg-opacity-30 px-3 py-1 rounded-full text-xs font-medium transition-all"
        >
          Upgrade for {formatPrice(995)}/month
        </button>
      </div>
    </motion.div>
  );
};

// ===============================
// FEATURE GATE COMPONENT
// ===============================

interface FeatureGateProps {
  feature: 'messages' | 'wisdom_modules' | 'api_requests';
  children: React.ReactNode;
  fallback?: React.ReactNode;
  onUpgrade?: () => void;
}

export const FeatureGate: React.FC<FeatureGateProps> = ({
  feature,
  children,
  fallback,
  onUpgrade
}) => {
  const { checkFeatureUsage } = useSubscription();
  
  const featureCheck = checkFeatureUsage(feature);
  
  if (featureCheck.canUse) {
    return <>{children}</>;
  }
  
  if (fallback) {
    return <>{fallback}</>;
  }
  
  return (
    <UpgradePrompt
      feature={feature}
      trigger="limit"
      onUpgrade={onUpgrade}
      className="mx-auto max-w-md"
    />
  );
};

// ===============================
// USAGE PROGRESS INDICATOR
// ===============================

interface UsageProgressProps {
  feature: 'messages' | 'wisdom_modules' | 'api_requests';
  showDetails?: boolean;
  className?: string;
}

export const UsageProgress: React.FC<UsageProgressProps> = ({
  feature,
  showDetails = true,
  className = ''
}) => {
  const { checkFeatureUsage, isFreePlan } = useSubscription();
  const { currentTheme } = useColors();
  
  if (!isFreePlan()) return null;
  
  const featureCheck = checkFeatureUsage(feature);
  
  const getFeatureName = () => {
    switch (feature) {
      case 'messages': return 'Messages';
      case 'wisdom_modules': return 'Wisdom Modules';
      case 'api_requests': return 'API Requests';
    }
  };
  
  const getProgressColor = () => {
    if (featureCheck.percentage >= 100) return '#dc2626'; // red-600
    if (featureCheck.percentage >= 90) return '#ea580c'; // orange-600
    if (featureCheck.percentage >= 80) return '#d97706'; // amber-600
    return currentTheme.colors.primary; // Use theme primary color for good progress
  };
  
  return (
    <div className={`${className}`}>
      {showDetails && (
        <div className="flex justify-between text-xs text-gray-600 mb-1">
          <span>{getFeatureName()}</span>
          <span>
            {featureCheck.usage}/{featureCheck.limit === -1 ? '∞' : featureCheck.limit}
          </span>
        </div>
      )}
      
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="h-2 rounded-full transition-all duration-300"
          style={{ 
            width: `${Math.min(100, featureCheck.percentage)}%`,
            backgroundColor: getProgressColor()
          }}
        />
      </div>
      
      {showDetails && featureCheck.percentage >= 80 && (
        <div className="text-xs text-orange-600 mt-1">
          {featureCheck.remaining > 0 
            ? `${featureCheck.remaining} remaining this month`
            : 'Limit reached - upgrade to continue'
          }
        </div>
      )}
    </div>
  );
};

export default UpgradePrompt;
