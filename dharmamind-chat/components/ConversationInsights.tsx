import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { XMarkIcon, ChartBarIcon, SparklesIcon, HeartIcon } from '@heroicons/react/24/outline';

interface ConversationInsightsProps {
  isOpen: boolean;
  onClose: () => void;
}

const ConversationInsights: React.FC<ConversationInsightsProps> = ({ isOpen, onClose }) => {
  const insights = {
    totalMessages: 42,
    wisdomTopics: ['Meditation', 'Dharma', 'Mindfulness', 'Compassion'],
    averageDharmicAlignment: 0.89,
    mostDiscussedTopic: 'Meditation Practice',
    spiritualGrowth: '+15% this month',
    favoriteTeachings: 3
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={(e) => {
          if (e.target === e.currentTarget) onClose();
        }}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="bg-white rounded-2xl shadow-2xl max-w-md w-full max-h-[80vh] overflow-hidden"
        >
          {/* Header */}
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <ChartBarIcon className="w-5 h-5 mr-2 text-emerald-600" />
                Conversation Insights
              </h2>
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100 transition-colors"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-emerald-50 p-4 rounded-xl">
                <div className="flex items-center justify-between">
                  <SparklesIcon className="w-8 h-8 text-emerald-600" />
                  <span className="text-2xl font-bold text-emerald-600">
                    {insights.totalMessages}
                  </span>
                </div>
                <p className="text-sm text-emerald-700 mt-2">Total Messages</p>
              </div>

              <div className="bg-purple-50 p-4 rounded-xl">
                <div className="flex items-center justify-between">
                  <HeartIcon className="w-8 h-8 text-purple-600" />
                  <span className="text-2xl font-bold text-purple-600">
                    {Math.round(insights.averageDharmicAlignment * 100)}%
                  </span>
                </div>
                <p className="text-sm text-purple-700 mt-2">Dharmic Alignment</p>
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Wisdom Topics</h3>
              <div className="flex flex-wrap gap-2">
                {insights.wisdomTopics.map((topic, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-emerald-100 text-emerald-800 rounded-full text-sm font-medium"
                  >
                    {topic}
                  </span>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-700">Most Discussed</span>
                <span className="font-medium text-gray-900">{insights.mostDiscussedTopic}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-700">Spiritual Growth</span>
                <span className="font-medium text-emerald-600">{insights.spiritualGrowth}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-700">Favorite Teachings</span>
                <span className="font-medium text-gray-900">{insights.favoriteTeachings}</span>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default ConversationInsights;
