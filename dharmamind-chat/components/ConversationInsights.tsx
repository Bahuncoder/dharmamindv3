import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TagIcon,
  ChartBarIcon,
  LightBulbIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  MinusIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { useConversationTags } from '../hooks/useConversationTags';

interface ConversationInsightsProps {
  isOpen: boolean;
  onClose: () => void;
  className?: string;
}

const ConversationInsights: React.FC<ConversationInsightsProps> = ({
  isOpen,
  onClose,
  className = ''
}) => {
  const {
    taggedConversations,
    availableTags,
    getMostUsedTags,
    getSpiritualProgress,
    getConversationsByCategory
  } = useConversationTags();

  const [activeTab, setActiveTab] = useState<'tags' | 'progress' | 'insights'>('tags');

  const mostUsedTags = getMostUsedTags(8);
  const spiritualProgress = getSpiritualProgress();

  const tabs = [
    {
      id: 'tags' as const,
      name: 'Spiritual Topics',
      icon: TagIcon,
      color: '#48bb78'
    },
    {
      id: 'progress' as const,
      name: 'Your Journey',
      icon: ChartBarIcon,
      color: '#4299e1'
    },
    {
      id: 'insights' as const,
      name: 'Wisdom Gained',
      icon: LightBulbIcon,
      color: '#ffd700'
    }
  ];

  const getProgressIcon = () => {
    switch (spiritualProgress.progressTrend) {
      case 'improving':
        return <ArrowTrendingUpIcon className="w-5 h-5 text-green-500" />;
      case 'declining':
        return <ArrowTrendingDownIcon className="w-5 h-5 text-red-500" />;
      default:
        return <MinusIcon className="w-5 h-5 text-gray-500" />;
    }
  };

  const getProgressColor = () => {
    if (spiritualProgress.averageAlignment >= 0.8) return 'text-green-500';
    if (spiritualProgress.averageAlignment >= 0.6) return 'text-emerald-500';
    return 'text-red-500';
  };

  const renderTagsContent = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {mostUsedTags.map(({ tag, count }) => (
          <motion.div
            key={tag.id}
            className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm"
            whileHover={{ scale: 1.02 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-lg">{tag.icon}</span>
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-white">
                    {tag.name}
                  </h4>
                  <p className="text-xs text-gray-500 capitalize">
                    {tag.category}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <div 
                  className="text-lg font-bold"
                  style={{ color: tag.color }}
                >
                  {count}
                </div>
                <div className="text-xs text-gray-500">
                  conversations
                </div>
              </div>
            </div>
            
            {/* Progress bar */}
            <div className="mt-2 bg-gray-200 dark:bg-gray-700 rounded-full h-1">
              <motion.div
                className="h-1 rounded-full"
                style={{ backgroundColor: tag.color }}
                initial={{ width: 0 }}
                animate={{ width: `${Math.min((count / Math.max(...mostUsedTags.map(t => t.count))) * 100, 100)}%` }}
                transition={{ duration: 1, delay: 0.2 }}
              />
            </div>
          </motion.div>
        ))}
      </div>
      
      {taggedConversations.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <TagIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Start chatting to see your spiritual topics!</p>
        </div>
      )}
    </div>
  );

  const renderProgressContent = () => (
    <div className="space-y-6">
      {/* Overall Progress */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-gray-800 dark:to-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-gray-900 dark:text-white">
            Spiritual Alignment
          </h3>
          {getProgressIcon()}
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <div className="flex justify-between text-sm text-gray-600 dark:text-gray-300 mb-1">
              <span>Progress</span>
              <span className={getProgressColor()}>
                {Math.round(spiritualProgress.averageAlignment * 100)}%
              </span>
            </div>
            <div className="bg-gray-200 dark:bg-gray-600 rounded-full h-2">
              <motion.div
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${spiritualProgress.averageAlignment * 100}%` }}
                transition={{ duration: 1.5, ease: 'easeOut' }}
              />
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4 mt-4 text-center">
          <div>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {spiritualProgress.totalConversations}
            </div>
            <div className="text-xs text-gray-500">Total Conversations</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400 capitalize">
              {spiritualProgress.progressTrend}
            </div>
            <div className="text-xs text-gray-500">Trend</div>
          </div>
        </div>
      </div>

      {/* Category Breakdown */}
      <div>
        <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
          Topics by Category
        </h3>
        <div className="space-y-2">
          {['spiritual', 'philosophical', 'emotional', 'practical'].map(category => {
            const conversations = getConversationsByCategory(category);
            const percentage = taggedConversations.length > 0 
              ? (conversations.length / taggedConversations.length) * 100 
              : 0;
            
            return (
              <div key={category} className="flex items-center justify-between">
                <span className="text-sm capitalize text-gray-700 dark:text-gray-300">
                  {category}
                </span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                    <motion.div
                      className="bg-blue-500 h-1 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${percentage}%` }}
                      transition={{ duration: 1, delay: 0.3 }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 w-8">
                    {Math.round(percentage)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );

  const renderInsightsContent = () => (
    <div className="space-y-4">
      {spiritualProgress.topInsights.length > 0 ? (
        <div className="space-y-3">
          {spiritualProgress.topInsights.map((insight, index) => (
            <motion.div
              key={index}
              className="p-3 bg-gradient-to-r from-gray-50 to-emerald-50 dark:from-gray-800 dark:to-gray-700 rounded-lg border-l-4 border-emerald-400"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="flex items-start space-x-2">
                <LightBulbIcon className="w-5 h-5 text-emerald-500 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-gray-700 dark:text-gray-300 italic">
                  "{insight}"
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <LightBulbIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Keep chatting to accumulate wisdom insights!</p>
        </div>
      )}
    </div>
  );

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div
          className={`
            bg-white dark:bg-gray-900 rounded-xl shadow-2xl 
            w-full max-w-4xl max-h-[90vh] overflow-hidden
            ${className}
          `}
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                Spiritual Journey Insights
              </h2>
              <button
                onClick={onClose}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                <XMarkIcon className="w-6 h-6" />
              </button>
            </div>
            
            {/* Tabs */}
            <div className="flex space-x-1 mt-4 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
              {tabs.map((tab) => {
                const IconComponent = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all
                      ${activeTab === tab.id
                        ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                      }
                    `}
                  >
                    <IconComponent className="w-4 h-4" />
                    <span>{tab.name}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.2 }}
              >
                {activeTab === 'tags' && renderTagsContent()}
                {activeTab === 'progress' && renderProgressContent()}
                {activeTab === 'insights' && renderInsightsContent()}
              </motion.div>
            </AnimatePresence>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default ConversationInsights;
