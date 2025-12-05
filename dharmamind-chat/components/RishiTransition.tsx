/**
 * Rishi Transition Animation
 * Beautiful animation when switching between Rishi guides
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface RishiTransitionProps {
  show: boolean;
  fromRishi?: string;
  toRishi: string;
  toRishiName: string;
  onComplete: () => void;
}

const getRishiIcon = (rishiId: string) => {
  const icons: Record<string, string> = {
    marici: '☀️',
    atri: '🧘',
    angiras: '🔥',
    pulastya: '🗺️',
    pulaha: '🌬️',
    kratu: '⚡',
    daksha: '🎨',
    bhrigu: '⭐',
    vasishta: '📚',
    '': '🤖' // Standard AI
  };
  return icons[rishiId] || '🕉️';
};

const getRishiColor = (rishiId: string) => {
  const colors: Record<string, string> = {
    marici: 'from-yellow-400 to-orange-500',
    atri: 'from-purple-500 to-indigo-600',
    angiras: 'from-red-500 to-orange-600',
    pulastya: 'from-teal-400 to-pink-400',
    pulaha: 'from-blue-400 to-indigo-400',
    kratu: 'from-pink-500 to-yellow-500',
    daksha: 'from-teal-500 to-purple-700',
    bhrigu: 'from-pink-500 to-red-500',
    vasishta: 'from-blue-500 to-cyan-600',
    '': 'from-gray-500 to-gray-600'
  };
  return colors[rishiId] || 'from-purple-500 to-pink-600';
};

export const RishiTransition: React.FC<RishiTransitionProps> = ({
  show,
  fromRishi,
  toRishi,
  toRishiName,
  onComplete
}) => {
  React.useEffect(() => {
    if (show) {
      const timer = setTimeout(() => {
        onComplete();
      }, 2500); // 2.5 seconds transition
      
      return () => clearTimeout(timer);
    }
  }, [show, onComplete]);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{
            background: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(10px)'
          }}
        >
          <div className="relative">
            {/* Outgoing Rishi (if exists) */}
            {fromRishi && (
              <motion.div
                initial={{ scale: 1, opacity: 1 }}
                animate={{ scale: 0.5, opacity: 0, y: -50 }}
                transition={{ duration: 0.8, ease: "easeInOut" }}
                className="absolute inset-0 flex flex-col items-center justify-center"
              >
                <div className={`text-8xl mb-4`}>
                  {getRishiIcon(fromRishi)}
                </div>
                <motion.div
                  animate={{ opacity: [1, 0] }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                  className="text-white text-center"
                >
                  <p className="text-sm opacity-70">Transitioning from</p>
                  <p className="text-lg font-medium">{fromRishi || 'Standard AI'}</p>
                </motion.div>
              </motion.div>
            )}

            {/* Center Animation - Om Symbol */}
            <motion.div
              initial={{ scale: 0, rotate: 0 }}
              animate={{ 
                scale: [0, 1.5, 1], 
                rotate: [0, 360, 720],
                opacity: [0, 1, 0.8]
              }}
              transition={{ 
                duration: 1.5,
                times: [0, 0.5, 1],
                ease: "easeInOut",
                delay: 0.5
              }}
              className="flex items-center justify-center mb-8"
            >
              <div className="text-9xl">
                🕉️
              </div>
            </motion.div>

            {/* Incoming Rishi */}
            <motion.div
              initial={{ scale: 0.5, opacity: 0, y: 50 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 1.2, ease: "easeInOut" }}
              className="flex flex-col items-center justify-center"
            >
              <motion.div
                animate={{ 
                  scale: [1, 1.1, 1],
                  rotate: [0, 5, -5, 0]
                }}
                transition={{ 
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="text-8xl mb-6"
              >
                {getRishiIcon(toRishi)}
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.5 }}
                className="text-center"
              >
                <motion.p 
                  className="text-white text-sm opacity-70 mb-2"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  {toRishi === '' ? 'Returning to' : 'Entering the guidance of'}
                </motion.p>
                <motion.div
                  className={`text-3xl font-bold bg-gradient-to-r ${getRishiColor(toRishi)} bg-clip-text text-transparent mb-4`}
                  animate={{ scale: [1, 1.05, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  {toRishiName || 'Standard AI'}
                </motion.div>
                
                {toRishi !== '' && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0, 1, 0.8] }}
                    transition={{ delay: 1.8 }}
                    className="text-white text-sm italic max-w-md"
                  >
                    "A new chapter of wisdom begins..."
                  </motion.p>
                )}
              </motion.div>

              {/* Ripple effect */}
              <motion.div
                className="absolute inset-0 pointer-events-none"
                initial={{ scale: 0, opacity: 0.5 }}
                animate={{ scale: 3, opacity: 0 }}
                transition={{ duration: 2, delay: 1.2 }}
              >
                <div className={`w-full h-full rounded-full bg-gradient-to-r ${getRishiColor(toRishi)} opacity-20`} />
              </motion.div>
            </motion.div>
          </div>

          {/* Ambient particles */}
          {[...Array(12)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 rounded-full bg-white"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                y: [0, -50, 0],
                opacity: [0, 1, 0],
                scale: [0, 1, 0]
              }}
              transition={{
                duration: 3,
                delay: Math.random() * 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
          ))}
        </motion.div>
      )}
    </AnimatePresence>
  );
};

/**
 * Rishi Context Indicator
 * Small badge showing current Rishi context in chat
 */
interface RishiContextBadgeProps {
  rishiId: string;
  rishiName: string;
  messageCount: number;
  onSwitch?: () => void;
}

export const RishiContextBadge: React.FC<RishiContextBadgeProps> = ({
  rishiId,
  rishiName,
  messageCount,
  onSwitch
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r ${getRishiColor(rishiId)} text-white shadow-lg`}
    >
      <span className="text-lg">{getRishiIcon(rishiId)}</span>
      <div className="flex flex-col">
        <span className="text-sm font-semibold">{rishiName}</span>
        <span className="text-xs opacity-80">{messageCount} messages</span>
      </div>
      {onSwitch && (
        <button
          onClick={onSwitch}
          className="ml-2 hover:bg-white/20 rounded-full p-1 transition-colors"
          title="Switch Rishi"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
          </svg>
        </button>
      )}
    </motion.div>
  );
};

/**
 * Conversation History Sidebar
 * Shows all Rishi conversations
 */
interface RishiConversationHistory {
  rishiId: string;
  rishiName: string;
  messageCount: number;
  lastActive: Date;
}

interface ConversationHistoryProps {
  conversations: RishiConversationHistory[];
  currentRishi: string;
  onSelect: (rishiId: string) => void;
}

export const ConversationHistory: React.FC<ConversationHistoryProps> = ({
  conversations,
  currentRishi,
  onSelect
}) => {
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 px-2">
        Recent Conversations
      </h3>
      {conversations.map((conv, index) => (
        <motion.button
          key={conv.rishiId}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.05 }}
          onClick={() => onSelect(conv.rishiId)}
          className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-all ${
            currentRishi === conv.rishiId
              ? 'bg-white dark:bg-gray-800 shadow-md border-2 border-purple-400'
              : 'bg-gray-50 dark:bg-gray-900 hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
        >
          <span className="text-2xl">{getRishiIcon(conv.rishiId)}</span>
          <div className="flex-1 text-left">
            <div className="font-medium text-sm text-gray-800 dark:text-gray-200">
              {conv.rishiName}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              {conv.messageCount} messages • {formatRelativeTime(conv.lastActive)}
            </div>
          </div>
          {currentRishi === conv.rishiId && (
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          )}
        </motion.button>
      ))}
    </div>
  );
};

function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString();
}

export default RishiTransition;
