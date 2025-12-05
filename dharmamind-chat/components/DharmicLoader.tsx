/**
 * Dharmic Loading Indicator
 * Beautiful, spiritual loading animations for DharmaMind
 */

import React from 'react';
import { motion } from 'framer-motion';

interface DharmicLoaderProps {
  variant?: 'om' | 'lotus' | 'dharma-wheel' | 'pulse';
  size?: 'sm' | 'md' | 'lg';
  message?: string;
}

export const DharmicLoader: React.FC<DharmicLoaderProps> = ({ 
  variant = 'lotus', 
  size = 'md',
  message 
}) => {
  const sizeClasses = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16'
  };

  const textSizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base'
  };

  if (variant === 'om') {
    return (
      <div className="flex flex-col items-center justify-center gap-3">
        <motion.div
          className={`${sizeClasses[size]} flex items-center justify-center`}
          animate={{
            scale: [1, 1.2, 1],
            rotate: [0, 360],
            opacity: [0.6, 1, 0.6]
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          <span className="text-4xl">🕉️</span>
        </motion.div>
        {message && (
          <motion.p
            className={`text-gray-600 dark:text-gray-300 ${textSizeClasses[size]}`}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            {message}
          </motion.p>
        )}
      </div>
    );
  }

  if (variant === 'lotus') {
    return (
      <div className="flex flex-col items-center justify-center gap-3">
        <div className="relative" style={{ width: sizeClasses[size].split(' ')[0].replace('w-', '') + 'rem' }}>
          {[0, 1, 2, 3, 4, 5].map((petal) => (
            <motion.div
              key={petal}
              className="absolute inset-0 flex items-center justify-center"
              style={{
                transformOrigin: 'center',
                transform: `rotate(${petal * 60}deg)`
              }}
              animate={{
                scale: [1, 1.3, 1],
                opacity: [0.4, 1, 0.4]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: petal * 0.2,
                ease: "easeInOut"
              }}
            >
              <div className="w-2 h-6 bg-gradient-to-t from-pink-400 to-purple-500 rounded-full blur-sm" />
            </motion.div>
          ))}
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            animate={{
              scale: [1, 1.1, 1],
              rotate: [0, 360]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "linear"
            }}
          >
            <div className="w-4 h-4 bg-yellow-400 rounded-full shadow-lg" />
          </motion.div>
        </div>
        {message && (
          <motion.p
            className={`text-gray-600 dark:text-gray-300 ${textSizeClasses[size]} text-center`}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            {message}
          </motion.p>
        )}
      </div>
    );
  }

  if (variant === 'dharma-wheel') {
    return (
      <div className="flex flex-col items-center justify-center gap-3">
        <motion.div
          className={`${sizeClasses[size]} relative`}
          animate={{ rotate: 360 }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "linear"
          }}
        >
          {/* Wheel rim */}
          <div className="absolute inset-0 border-4 border-amber-500 rounded-full opacity-60" />
          
          {/* Spokes */}
          {[0, 1, 2, 3, 4, 5, 6, 7].map((spoke) => (
            <div
              key={spoke}
              className="absolute inset-0 flex items-center justify-center"
              style={{
                transform: `rotate(${spoke * 45}deg)`
              }}
            >
              <div className="w-0.5 h-full bg-amber-600 opacity-40" />
            </div>
          ))}
          
          {/* Center hub */}
          <div className="absolute inset-0 flex items-center justify-center">
            <motion.div
              className="w-3 h-3 bg-amber-500 rounded-full shadow-lg"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
          </div>
        </motion.div>
        {message && (
          <motion.p
            className={`text-gray-600 dark:text-gray-300 ${textSizeClasses[size]}`}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            {message}
          </motion.p>
        )}
      </div>
    );
  }

  // Pulse variant (default)
  return (
    <div className="flex flex-col items-center justify-center gap-3">
      <div className="flex gap-2">
        {[0, 1, 2].map((dot) => (
          <motion.div
            key={dot}
            className={`${size === 'sm' ? 'w-2 h-2' : size === 'md' ? 'w-3 h-3' : 'w-4 h-4'} bg-gradient-to-br from-purple-500 to-pink-500 rounded-full`}
            animate={{
              scale: [1, 1.5, 1],
              opacity: [0.4, 1, 0.4]
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              delay: dot * 0.3,
              ease: "easeInOut"
            }}
          />
        ))}
      </div>
      {message && (
        <motion.p
          className={`text-gray-600 dark:text-gray-300 ${textSizeClasses[size]}`}
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        >
          {message}
        </motion.p>
      )}
    </div>
  );
};

/**
 * Dharmic Typing Indicator
 * Shows when the AI is composing a response
 */
export const DharmicTypingIndicator: React.FC = () => {
  return (
    <div className="flex items-center gap-3 p-4 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-sm">
      <DharmicLoader variant="pulse" size="sm" />
      <div className="flex flex-col gap-1">
        <motion.p
          className="text-sm font-medium text-gray-700 dark:text-gray-200"
          animate={{ opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        >
          DharmaMind is reflecting...
        </motion.p>
        <motion.p
          className="text-xs text-gray-500 dark:text-gray-400"
          animate={{ opacity: [0.4, 0.8, 0.4] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          Channeling dharmic wisdom
        </motion.p>
      </div>
    </div>
  );
};

/**
 * Meditation Timer Display
 * Beautiful timer for contemplation sessions
 */
interface MeditationTimerProps {
  seconds: number;
  isActive: boolean;
  onPause?: () => void;
  onResume?: () => void;
  onStop?: () => void;
}

export const MeditationTimer: React.FC<MeditationTimerProps> = ({
  seconds,
  isActive,
  onPause,
  onResume,
  onStop
}) => {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl p-6 shadow-lg border border-purple-200 dark:border-purple-800"
    >
      <div className="flex flex-col items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🧘</span>
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
            Meditation in Progress
          </h3>
        </div>

        <motion.div
          className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-pink-600"
          animate={isActive ? {
            scale: [1, 1.05, 1],
          } : {}}
          transition={{
            duration: 1,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          {String(minutes).padStart(2, '0')}:{String(remainingSeconds).padStart(2, '0')}
        </motion.div>

        <div className="flex gap-3">
          {isActive ? (
            <button
              onClick={onPause}
              className="px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-lg transition-colors font-medium"
            >
              ⏸️ Pause
            </button>
          ) : (
            <button
              onClick={onResume}
              className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors font-medium"
            >
              ▶️ Resume
            </button>
          )}
          <button
            onClick={onStop}
            className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors font-medium"
          >
            ⏹️ Complete
          </button>
        </div>

        {isActive && (
          <motion.p
            className="text-sm text-gray-600 dark:text-gray-300 text-center"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            Breathe deeply. Stay present. 🌸
          </motion.p>
        )}
      </div>
    </motion.div>
  );
};

export default DharmicLoader;
