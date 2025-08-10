/**
 * 🕉️ DharmaMind Enhanced Chat Interface
 * 
 * Chat interface with integrated subscription features:
 * - Real-time usage tracking
 * - Feature gating with usage limits
 * - Upgrade prompts and premium features
 * - Seamless subscription management
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useSession } from 'next-auth/react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSubscription } from '../hooks/useSubscription';
import { useNavigation } from '../hooks/useNavigation';
import { useColors } from '../contexts/ColorContext';
import CentralizedSubscriptionModal from './CentralizedSubscriptionModal';
import SidebarQuotes from './SidebarQuotes';
import VoiceInput from './VoiceInput';
import Logo from './Logo';

// ===============================
// TYPES
// ===============================

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  module?: string;
  wisdom?: 'basic' | 'advanced' | 'premium';
}

interface UsageAlert {
  type: 'warning' | 'limit' | 'upgrade';
  feature: string;
  message: string;
  action?: () => void;
}

// ===============================
// ENHANCED CHAT INTERFACE
// ===============================

export const EnhancedChatInterface: React.FC = () => {
  const { data: session } = useSession();
  const { goToAuth } = useNavigation();
  const { currentTheme } = useColors();
  
  // Subscription hook with all enhanced features
  const {
    usage,
    isFreePlan,
    isPremiumPlan,
    trackUsage,
    checkFeatureUsage,
    shouldShowUpgradePrompt,
    getUpgradeMessage,
    currentPlan,
    isLoading: subscriptionLoading
  } = useSubscription();

  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedModule, setSelectedModule] = useState<string>('general');
  const [usageAlert, setUsageAlert] = useState<UsageAlert | null>(null);
  const [showSubscriptionModal, setShowSubscriptionModal] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // ===============================
  // WISDOM MODULES CONFIGURATION
  // ===============================

  const wisdomModules = [
    { id: 'general', name: 'General Dharma', tier: 'basic', description: 'Core spiritual teachings' },
    { id: 'meditation', name: 'Meditation Guide', tier: 'basic', description: 'Mindfulness practices' },
    { id: 'karma', name: 'Karma & Action', tier: 'basic', description: 'Understanding cause and effect' },
    { id: 'dharma_paths', name: 'Dharma Paths', tier: 'basic', description: 'Different spiritual paths' },
    { id: 'inner_peace', name: 'Inner Peace', tier: 'basic', description: 'Finding tranquility' },
    
    // Premium modules (Professional+)
    { id: 'advanced_meditation', name: 'Advanced Meditation', tier: 'advanced', description: 'Deep contemplative practices' },
    { id: 'consciousness', name: 'Consciousness Studies', tier: 'advanced', description: 'Exploring awareness itself' },
    { id: 'non_duality', name: 'Non-Duality', tier: 'advanced', description: 'Unity consciousness teachings' },
    { id: 'tantric_wisdom', name: 'Tantric Wisdom', tier: 'advanced', description: 'Sacred energy practices' },
    { id: 'vedantic_inquiry', name: 'Vedantic Inquiry', tier: 'advanced', description: 'Self-inquiry methods' },
    
    // Enterprise modules
    { id: 'business_dharma', name: 'Business Dharma', tier: 'premium', description: 'Ethical leadership' },
    { id: 'healing_arts', name: 'Healing Arts', tier: 'premium', description: 'Spiritual healing practices' },
    { id: 'prophecy_wisdom', name: 'Prophecy & Vision', tier: 'premium', description: 'Intuitive development' },
  ];

  // Filter modules based on subscription
  const availableModules = wisdomModules.filter(module => {
    if (module.tier === 'basic') return true;
    if (module.tier === 'advanced') return isPremiumPlan() || currentPlan?.tier === 'enterprise';
    if (module.tier === 'premium') return currentPlan?.tier === 'enterprise';
    return false;
  });

  // ===============================
  // USAGE TRACKING & ALERTS
  // ===============================

  const checkAndShowUsageAlert = useCallback((feature: string) => {
    const featureCheck = checkFeatureUsage(feature);
    
    if (featureCheck.hasReachedLimit) {
      setUsageAlert({
        type: 'limit',
        feature,
        message: getUpgradeMessage(feature),
        action: () => setShowSubscriptionModal(true)
      });
      return false;
    }
    
    if (featureCheck.percentage >= 80) {
      setUsageAlert({
        type: 'warning',
        feature,
        message: `You've used ${featureCheck.usage} of ${featureCheck.limit} ${feature} this month.`,
        action: () => setShowSubscriptionModal(true)
      });
    }
    
    return true;
  }, [checkFeatureUsage, getUpgradeMessage]);

  // ===============================
  // MESSAGE HANDLING
  // ===============================

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    // Check message usage before proceeding
    const canSendMessage = await trackUsage('messages');
    if (!canSendMessage) {
      checkAndShowUsageAlert('messages');
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage.trim(),
      isUser: true,
      timestamp: new Date(),
      module: selectedModule
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);

    try {
      // Simulate AI response with module-specific wisdom
      const selectedModuleData = wisdomModules.find(m => m.id === selectedModule);
      const response = await generateAIResponse(inputMessage, selectedModuleData);
      
      setTimeout(() => {
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: response,
          isUser: false,
          timestamp: new Date(),
          module: selectedModule,
          wisdom: selectedModuleData?.tier as any
        };
        
        setMessages(prev => [...prev, aiMessage]);
        setIsTyping(false);
      }, 1500 + Math.random() * 2000);

    } catch (error) {
      console.error('Failed to send message:', error);
      setIsTyping(false);
    }
  };

  const generateAIResponse = async (message: string, module: any): Promise<string> => {
    // Enhanced AI response logic with module-specific wisdom
    const responses = {
      general: [
        "The path of dharma teaches us that every moment is an opportunity for awakening. Your question touches the heart of spiritual inquiry.",
        "In the ancient wisdom traditions, we learn that suffering arises from attachment. Let me share some insights that might illuminate your path.",
        "The Buddha taught that understanding comes not from thinking about wisdom, but from embodying it in daily life."
      ],
      meditation: [
        "Meditation is the art of returning home to yourself. Your breath is always available as an anchor in the present moment.",
        "In stillness, we discover that peace was never lost - it was simply covered by the movements of mind.",
        "Begin with just watching your breath. No need to change anything, just observe with loving kindness."
      ],
      consciousness: [
        "Consciousness is not produced by the brain - rather, the brain is an activity within consciousness itself.",
        "The deepest inquiry leads us to question the very nature of the one who is asking the questions.",
        "Awareness is the unchanging background against which all experiences arise and dissolve."
      ]
    };

    const moduleResponses = responses[module?.id as keyof typeof responses] || responses.general;
    return moduleResponses[Math.floor(Math.random() * moduleResponses.length)];
  };

  // ===============================
  // USAGE MONITORING
  // ===============================

  const UsageMonitor: React.FC = () => {
    if (!usage || subscriptionLoading) return null;

    const messageCheck = checkFeatureUsage('messages');
    const moduleCheck = checkFeatureUsage('wisdom_modules');

    return (
      <div 
        className="border rounded-lg p-4 mb-4"
        style={{
          background: `linear-gradient(45deg, ${currentTheme.colors.primaryStart}15, ${currentTheme.colors.primaryEnd}15)`,
          borderColor: currentTheme.colors.primary + '40'
        }}
      >
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium" style={{ color: currentTheme.colors.primary }}>
            Monthly Usage
          </h3>
          {isFreePlan() && (
            <button
              onClick={() => setShowSubscriptionModal(true)}
              className="text-xs font-medium hover:underline"
              style={{ color: currentTheme.colors.primaryHover }}
            >
              Upgrade Plan
            </button>
          )}
        </div>
        
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span style={{ color: currentTheme.colors.primary }}>Messages</span>
            <span style={{ color: currentTheme.colors.primaryHover }}>
              {messageCheck.usage}/{messageCheck.limit === -1 ? '∞' : messageCheck.limit}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="h-2 rounded-full transition-all duration-300"
              style={{ 
                width: `${Math.min(100, messageCheck.percentage)}%`,
                background: `linear-gradient(45deg, ${currentTheme.colors.primaryStart}, ${currentTheme.colors.primaryEnd})`
              }}
            />
          </div>
          
          <div className="flex items-center justify-between text-xs mt-2">
            <span style={{ color: currentTheme.colors.primary }}>Available Modules</span>
            <span style={{ color: currentTheme.colors.primaryHover }}>{availableModules.length}/32</span>
          </div>
        </div>
      </div>
    );
  };

  // ===============================
  // USAGE ALERT COMPONENT
  // ===============================

  const UsageAlert: React.FC = () => {
    if (!usageAlert) return null;

    return (
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className={`fixed top-4 right-4 max-w-md rounded-lg p-4 shadow-lg z-50 ${
            usageAlert.type === 'limit' ? 'bg-red-50 border-red-200' :
            usageAlert.type === 'warning' ? 'bg-yellow-50 border-yellow-200' :
            'bg-blue-50 border-blue-200'
          }`}
        >
          <div className="flex items-start">
            <div className="flex-1">
              <p className={`text-sm font-medium ${
                usageAlert.type === 'limit' ? 'text-red-800' :
                usageAlert.type === 'warning' ? 'text-yellow-800' :
                'text-blue-800'
              }`}>
                {usageAlert.message}
              </p>
              {usageAlert.action && (
                <button
                  onClick={usageAlert.action}
                  className={`mt-2 text-xs font-medium underline ${
                    usageAlert.type === 'limit' ? 'text-red-600' :
                    usageAlert.type === 'warning' ? 'text-yellow-600' :
                    'text-blue-600'
                  }`}
                >
                  Upgrade Now
                </button>
              )}
            </div>
            <button
              onClick={() => setUsageAlert(null)}
              className="ml-4 text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>
        </motion.div>
      </AnimatePresence>
    );
  };

  // ===============================
  // MODULE SELECTOR
  // ===============================

  const ModuleSelector: React.FC = () => {
    return (
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Wisdom Module
        </label>
        <select
          value={selectedModule}
          onChange={(e) => setSelectedModule(e.target.value)}
          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
        >
          {availableModules.map(module => (
            <option key={module.id} value={module.id}>
              {module.name} - {module.description}
            </option>
          ))}
        </select>
        
        {isFreePlan() && (
          <p className="text-xs text-gray-500 mt-1">
            Upgrade to Professional to unlock {wisdomModules.length - availableModules.length} additional modules
          </p>
        )}
      </div>
    );
  };

  // ===============================
  // AUTO-SCROLL TO BOTTOM
  // ===============================

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ===============================
  // RENDER
  // ===============================

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Usage Monitor */}
      <UsageMonitor />
      
      {/* Module Selector */}
      <ModuleSelector />
      
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-8">
            <Logo size="xl" showText={false} className="justify-center mb-4" />
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              Welcome to DharmaMind
            </h2>
            <p className="text-gray-600 mb-4">
              Ask any question about spirituality, dharma, or life wisdom
            </p>
            {!session && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-md mx-auto">
                <p className="text-sm text-blue-800 mb-2">
                  You're in demo mode with limited features
                </p>
                <button
                  onClick={() => goToAuth('signup')}
                  className="text-sm font-medium underline"
                  style={{ color: currentTheme.colors.primary }}
                >
                  Sign up for unlimited access
                </button>
              </div>
            )}
          </div>
        )}

        {messages.map((message) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.isUser
                  ? 'text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
              style={{
                backgroundColor: message.isUser 
                  ? currentTheme.colors.primary 
                  : undefined
              }}
            >
              <p className="text-sm">{message.content}</p>
              {message.wisdom && (
                <div className={`text-xs mt-1 ${
                  message.isUser ? 'text-orange-100' : 'text-gray-500'
                }`}>
                  {message.wisdom === 'advanced' && '✨ Advanced Wisdom'}
                  {message.wisdom === 'premium' && '💎 Premium Insight'}
                </div>
              )}
            </div>
          </motion.div>
        ))}

        {isTyping && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="bg-gray-100 rounded-lg px-4 py-2">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
              </div>
            </div>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Ask about dharma, meditation, or spiritual wisdom..."
            className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
            disabled={isTyping}
          />
          <VoiceInput onTranscript={setInputMessage} />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isTyping}
            className="px-6 py-3 text-white rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            style={{
              backgroundColor: currentTheme.colors.primary,
              borderColor: currentTheme.colors.primary
            }}
          >
            Send
          </button>
        </div>
      </div>

      {/* Usage Alert */}
      <UsageAlert />

      {/* Subscription Modal */}
      <CentralizedSubscriptionModal
        isOpen={showSubscriptionModal}
        onClose={() => setShowSubscriptionModal(false)}
      />

      {/* Spiritual Quotes Sidebar (hidden on mobile) */}
      <div className="hidden lg:block">
        <SidebarQuotes />
      </div>
    </div>
  );
};

export default EnhancedChatInterface;
