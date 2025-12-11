/**
 * Dharmic Wisdom Card
 * Beautiful cards displaying dharmic insights, practices, and scripture references
 */

import React from 'react';
import { motion } from 'framer-motion';
import { 
  SparklesIcon, 
  HeartIcon, 
  BookOpenIcon,
  LightBulbIcon,
  FireIcon
} from '@heroicons/react/24/outline';

interface DharmicInsightCardProps {
  insights?: string[];
  practices?: string[];
  scriptures?: Array<{
    text: string;
    source: string;
    tradition: string;
  }>;
  context?: string;
}

export const DharmicInsightCard: React.FC<DharmicInsightCardProps> = ({
  insights = [],
  practices = [],
  scriptures = [],
  context
}) => {
  if (insights.length === 0 && practices.length === 0 && scriptures.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4 mt-4">
      {/* Spiritual Insights */}
      {insights.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-4 border border-purple-200 dark:border-purple-800"
        >
          <div className="flex items-center gap-2 mb-3">
            <SparklesIcon className="w-5 h-5 text-gold-600 dark:text-gold-400" />
            <h4 className="font-semibold text-purple-900 dark:text-purple-100">
              Spiritual Insights
            </h4>
          </div>
          <div className="space-y-2">
            {insights.map((insight, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="flex items-start gap-2"
              >
                <span className="text-gold-600 dark:text-gold-400 mt-1">✨</span>
                <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                  {insight}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Growth Practices */}
      {practices.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-br from-green-50 to-gold-50 dark:from-green-900/20 dark:to-gold-900/20 rounded-xl p-4 border border-green-200 dark:border-green-800"
        >
          <div className="flex items-center gap-2 mb-3">
            <LightBulbIcon className="w-5 h-5 text-success-600 dark:text-green-400" />
            <h4 className="font-semibold text-green-900 dark:text-green-100">
              Growth Practices
            </h4>
          </div>
          <div className="space-y-2">
            {practices.map((practice, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 + idx * 0.1 }}
                className="flex items-start gap-2"
              >
                <span className="text-success-600 dark:text-green-400 mt-1">🌱</span>
                <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                  {practice}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Scripture References */}
      {scriptures.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-4 border border-amber-200 dark:border-amber-800"
        >
          <div className="flex items-center gap-2 mb-3">
            <BookOpenIcon className="w-5 h-5 text-amber-600 dark:text-amber-400" />
            <h4 className="font-semibold text-amber-900 dark:text-amber-100">
              Sacred Teachings
            </h4>
          </div>
          <div className="space-y-3">
            {scriptures.map((scripture, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 + idx * 0.1 }}
                className="border-l-4 border-amber-400 dark:border-amber-600 pl-3"
              >
                <p className="text-sm italic text-gray-700 dark:text-gray-300 mb-1">
                  "{scripture.text}"
                </p>
                <p className="text-xs text-amber-700 dark:text-amber-400 font-medium">
                  — {scripture.source} ({scripture.tradition})
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Spiritual Context */}
      {context && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-center text-sm text-gray-500 dark:text-gray-400 italic"
        >
          Spiritual Context: {context}
        </motion.div>
      )}
    </div>
  );
};

/**
 * Dharmic Alignment Badge
 * Shows the dharmic alignment score with visual indicator
 */
interface DharmicAlignmentBadgeProps {
  score: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

export const DharmicAlignmentBadge: React.FC<DharmicAlignmentBadgeProps> = ({
  score,
  size = 'md',
  showLabel = true
}) => {
  const getColor = () => {
    if (score >= 0.9) return 'from-green-500 to-gold-500';
    if (score >= 0.7) return 'from-neutral-1000 to-cyan-500';
    if (score >= 0.5) return 'from-yellow-500 to-amber-500';
    return 'from-orange-500 to-red-500';
  };

  const getEmoji = () => {
    if (score >= 0.9) return '🧘';
    if (score >= 0.7) return '🕉️';
    if (score >= 0.5) return '🌸';
    return '🤔';
  };

  const sizeClasses = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1.5',
    lg: 'text-base px-4 py-2'
  };

  return (
    <motion.div
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      className={`inline-flex items-center gap-2 bg-gradient-to-r ${getColor()} text-white rounded-full ${sizeClasses[size]} font-medium shadow-lg`}
    >
      <span>{getEmoji()}</span>
      {showLabel && (
        <span>
          {Math.round(score * 100)}% Dharmic
        </span>
      )}
    </motion.div>
  );
};

/**
 * Wisdom Source Badge
 * Shows the source of wisdom (AI model, tradition, etc.)
 */
interface WisdomSourceBadgeProps {
  sources: string[];
  size?: 'sm' | 'md';
}

export const WisdomSourceBadge: React.FC<WisdomSourceBadgeProps> = ({
  sources,
  size = 'sm'
}) => {
  if (!sources || sources.length === 0) return null;

  const sizeClasses = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1.5'
  };

  return (
    <div className="flex flex-wrap gap-1">
      {sources.slice(0, 3).map((source, idx) => (
        <motion.span
          key={idx}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: idx * 0.05 }}
          className={`inline-flex items-center gap-1 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-full ${sizeClasses[size]}`}
        >
          <FireIcon className="w-3 h-3" />
          <span className="font-medium">{source}</span>
        </motion.span>
      ))}
      {sources.length > 3 && (
        <span className={`inline-flex items-center bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded-full ${sizeClasses[size]}`}>
          +{sources.length - 3} more
        </span>
      )}
    </div>
  );
};

/**
 * Quick Practice Card
 * Actionable practice card with instructions
 */
interface QuickPracticeCardProps {
  practice: {
    name: string;
    duration?: number;
    description: string;
    instructions?: string[];
  };
  onStart?: () => void;
}

export const QuickPracticeCard: React.FC<QuickPracticeCardProps> = ({
  practice,
  onStart
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      className="bg-gradient-to-br from-gold-50 to-cyan-50 dark:from-gold-900/20 dark:to-cyan-900/20 rounded-xl p-4 border border-gold-200 dark:border-gold-800 cursor-pointer"
      onClick={onStart}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <HeartIcon className="w-5 h-5 text-gold-600 dark:text-gold-400" />
          <h4 className="font-semibold text-gold-900 dark:text-gold-100">
            {practice.name}
          </h4>
        </div>
        {practice.duration && (
          <span className="text-xs bg-gold-100 dark:bg-gold-900 text-gold-700 dark:text-gold-300 px-2 py-1 rounded-full">
            {practice.duration} min
          </span>
        )}
      </div>
      
      <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
        {practice.description}
      </p>

      {practice.instructions && practice.instructions.length > 0 && (
        <div className="space-y-1">
          {practice.instructions.slice(0, 2).map((instruction, idx) => (
            <div key={idx} className="flex items-start gap-2">
              <span className="text-gold-600 dark:text-gold-400 text-xs mt-0.5">•</span>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                {instruction}
              </p>
            </div>
          ))}
          {practice.instructions.length > 2 && (
            <p className="text-xs text-gold-600 dark:text-gold-400 font-medium">
              +{practice.instructions.length - 2} more steps
            </p>
          )}
        </div>
      )}

      {onStart && (
        <button className="mt-3 w-full bg-gold-600 hover:bg-gold-700 text-white text-sm font-medium py-2 px-4 rounded-lg transition-colors">
          Begin Practice 🧘
        </button>
      )}
    </motion.div>
  );
};

export default DharmicInsightCard;
