/**
 * Dharmic Loading Indicator
 * Beautiful, spiritual loading animations for DharmaMind
 */

import React from 'react';
import { motion } from 'framer-motion';

interface DharmicLoaderProps {
  variant?: 'om' | 'lotus' | 'dharma-wheel' | 'pulse' | 'mandala' | 'breath';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  message?: string;
  className?: string;
}

export const DharmicLoader: React.FC<DharmicLoaderProps> = ({ 
  variant = 'lotus', 
  size = 'md',
  message,
  className = ''
}) => {
  const sizeConfig = {
    sm: { container: 'w-8 h-8', text: 'text-xs', icon: 'text-2xl', petals: 'w-1.5 h-4' },
    md: { container: 'w-12 h-12', text: 'text-sm', icon: 'text-3xl', petals: 'w-2 h-6' },
    lg: { container: 'w-16 h-16', text: 'text-base', icon: 'text-4xl', petals: 'w-2.5 h-8' },
    xl: { container: 'w-24 h-24', text: 'text-lg', icon: 'text-5xl', petals: 'w-3 h-10' }
  };

  const config = sizeConfig[size];

  // Om Symbol Loader
  if (variant === 'om') {
    return (
      <div className={`flex flex-col items-center justify-center gap-4 ${className}`}>
        <div className="relative">
          {/* Glow background */}
          <motion.div
            className={`${config.container} absolute inset-0 bg-gold-500/20 rounded-full blur-xl`}
            animate={{ scale: [1, 1.5, 1], opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          />
          
          {/* Om symbol */}
          <motion.div
            className={`${config.container} flex items-center justify-center relative`}
            animate={{
              scale: [1, 1.15, 1],
              rotate: [0, 5, -5, 0]
            }}
            transition={{
              duration: 2.5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <span className={`${config.icon} filter drop-shadow-lg`}>🕉️</span>
          </motion.div>
          
          {/* Rotating ring */}
          <motion.div
            className={`${config.container} absolute inset-0 border-2 border-gold-500/30 rounded-full`}
            style={{ borderStyle: 'dashed' }}
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
          />
        </div>
        
        {message && (
          <motion.p
            className={`text-gray-600 dark:text-gray-300 ${config.text} text-center font-medium`}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            {message}
          </motion.p>
        )}
      </div>
    );
  }

  // Lotus Flower Loader
  if (variant === 'lotus') {
    return (
      <div className={`flex flex-col items-center justify-center gap-4 ${className}`}>
        <div className={`${config.container} relative`}>
          {/* Outer glow */}
          <motion.div
            className="absolute inset-0 bg-gold-500/10 rounded-full blur-xl"
            animate={{ scale: [1, 1.3, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          
          {/* Petals */}
          {[0, 1, 2, 3, 4, 5].map((petal) => (
            <motion.div
              key={petal}
              className="absolute inset-0 flex items-center justify-center"
              style={{
                transformOrigin: 'center',
                transform: `rotate(${petal * 60}deg)`
              }}
            >
              <motion.div
                className={`${config.petals} bg-gradient-to-t from-gold-500 to-gold-400 rounded-full`}
                animate={{
                  scaleY: [1, 1.3, 1],
                  opacity: [0.6, 1, 0.6]
                }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  delay: petal * 0.15,
                  ease: "easeInOut"
                }}
                style={{ filter: 'blur(1px)' }}
              />
            </motion.div>
          ))}
          
          {/* Center */}
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            <div className="w-3 h-3 bg-gradient-to-br from-yellow-400 to-amber-500 rounded-full shadow-lg" />
          </motion.div>
        </div>
        
        {message && (
          <motion.p
            className={`text-gray-600 dark:text-gray-300 ${config.text} text-center font-medium`}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            {message}
          </motion.p>
        )}
      </div>
    );
  }

  // Dharma Wheel Loader
  if (variant === 'dharma-wheel') {
    return (
      <div className={`flex flex-col items-center justify-center gap-4 ${className}`}>
        <motion.div
          className={`${config.container} relative`}
          animate={{ rotate: 360 }}
          transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
        >
          {/* Outer ring */}
          <div className="absolute inset-0 border-4 border-gold-500 rounded-full" />
          
          {/* Spokes */}
          {[0, 1, 2, 3, 4, 5, 6, 7].map((spoke) => (
            <div
              key={spoke}
              className="absolute inset-0 flex items-center justify-center"
              style={{ transform: `rotate(${spoke * 45}deg)` }}
            >
              <div className="w-0.5 h-full bg-gold-500 opacity-60" />
            </div>
          ))}
          
          {/* Inner ring */}
          <div className="absolute inset-2 border-2 border-gold-400 rounded-full" />
          
          {/* Center hub */}
          <div className="absolute inset-0 flex items-center justify-center">
            <motion.div
              className="w-3 h-3 bg-gold-600 rounded-full shadow-lg"
              animate={{ scale: [1, 1.3, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            />
          </div>
        </motion.div>
        
        {message && (
          <motion.p
            className={`text-gray-600 dark:text-gray-300 ${config.text} text-center font-medium`}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            {message}
          </motion.p>
        )}
      </div>
    );
  }

  // Mandala Loader
  if (variant === 'mandala') {
    return (
      <div className={`flex flex-col items-center justify-center gap-4 ${className}`}>
        <div className={`${config.container} relative`}>
          {/* Multiple rotating rings */}
          {[0, 1, 2].map((ring, i) => (
            <motion.div
              key={ring}
              className="absolute inset-0 flex items-center justify-center"
              animate={{ rotate: i % 2 === 0 ? 360 : -360 }}
              transition={{ duration: 3 + i, repeat: Infinity, ease: "linear" }}
            >
              <div 
                className={`rounded-full border-2 border-emerald-${500 - i * 100}`}
                style={{ 
                  width: `${100 - i * 20}%`, 
                  height: `${100 - i * 20}%`,
                  borderStyle: i === 1 ? 'dashed' : 'solid'
                }}
              />
            </motion.div>
          ))}
          
          {/* Center dots */}
          <div className="absolute inset-0 flex items-center justify-center">
            {[0, 1, 2, 3].map((dot) => (
              <motion.div
                key={dot}
                className="absolute w-1.5 h-1.5 bg-gold-500 rounded-full"
                style={{
                  transform: `rotate(${dot * 90}deg) translateX(${size === 'sm' ? 8 : size === 'md' ? 12 : size === 'lg' ? 16 : 24}px)`
                }}
                animate={{ scale: [1, 1.5, 1] }}
                transition={{ duration: 1, repeat: Infinity, delay: dot * 0.25 }}
              />
            ))}
          </div>
        </div>
        
        {message && (
          <motion.p
            className={`text-gray-600 dark:text-gray-300 ${config.text} text-center font-medium`}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            {message}
          </motion.p>
        )}
      </div>
    );
  }

  // Breath Loader (expanding circles)
  if (variant === 'breath') {
    return (
      <div className={`flex flex-col items-center justify-center gap-4 ${className}`}>
        <div className={`${config.container} relative flex items-center justify-center`}>
          {[0, 1, 2].map((circle) => (
            <motion.div
              key={circle}
              className="absolute rounded-full bg-gold-500"
              initial={{ scale: 0.5, opacity: 0.8 }}
              animate={{
                scale: [0.5, 1.5],
                opacity: [0.8, 0]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: circle * 0.6,
                ease: "easeOut"
              }}
              style={{ width: '100%', height: '100%' }}
            />
          ))}
          
          {/* Center stable dot */}
          <motion.div
            className="relative w-3 h-3 bg-gold-600 rounded-full shadow-lg"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        </div>
        
        {message && (
          <motion.p
            className={`text-gray-600 dark:text-gray-300 ${config.text} text-center font-medium`}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            {message}
          </motion.p>
        )}
      </div>
    );
  }

  // Pulse variant (default)
  return (
    <div className={`flex flex-col items-center justify-center gap-4 ${className}`}>
      <div className="flex gap-2">
        {[0, 1, 2].map((dot) => (
          <motion.div
            key={dot}
            className={`
              ${size === 'sm' ? 'w-2 h-2' : size === 'md' ? 'w-3 h-3' : size === 'lg' ? 'w-4 h-4' : 'w-5 h-5'} 
              bg-gradient-to-br from-gold-500 to-gold-500 rounded-full shadow-md
            `}
            animate={{
              scale: [1, 1.5, 1],
              opacity: [0.5, 1, 0.5]
            }}
            transition={{
              duration: 1.2,
              repeat: Infinity,
              delay: dot * 0.2,
              ease: "easeInOut"
            }}
          />
        ))}
      </div>
      
      {message && (
        <motion.p
          className={`text-gray-600 dark:text-gray-300 ${config.text} text-center font-medium`}
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
export const DharmicTypingIndicator: React.FC<{ wisdom?: string }> = ({ 
  wisdom = "DharmaMind is reflecting..."
}) => {
  const wisdoms = [
    "Channeling dharmic wisdom...",
    "Contemplating your question...",
    "Drawing from ancient texts...",
    "Finding the right words...",
    "Seeking clarity...",
  ];

  const [currentWisdom, setCurrentWisdom] = React.useState(wisdom);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setCurrentWisdom(wisdoms[Math.floor(Math.random() * wisdoms.length)]);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center gap-4 p-4 bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-2xl shadow-lg border border-gold-100 dark:border-gold-900"
    >
      <DharmicLoader variant="lotus" size="sm" />
      
      <div className="flex flex-col gap-0.5">
        <motion.p
          key={currentWisdom}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-sm font-semibold text-gray-800 dark:text-gray-200"
        >
          {currentWisdom}
        </motion.p>
        <div className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-pulse" />
          <span className="text-xs text-gold-600 dark:text-gold-400">
            Wisdom incoming
          </span>
        </div>
      </div>
    </motion.div>
  );
};

/**
 * Full Page Loader
 * Beautiful full-screen loading state
 */
export const DharmicPageLoader: React.FC<{ 
  message?: string;
  submessage?: string;
}> = ({
  message = "Preparing your spiritual journey...",
  submessage = "Please wait while we align the cosmic energies"
}) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Background orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div 
          className="absolute top-1/4 left-1/4 w-64 h-64 bg-gold-300/20 rounded-full blur-3xl"
          animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
          transition={{ duration: 4, repeat: Infinity }}
        />
        <motion.div 
          className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-gold-300/20 rounded-full blur-3xl"
          animate={{ scale: [1.2, 1, 1.2], opacity: [0.3, 0.5, 0.3] }}
          transition={{ duration: 5, repeat: Infinity }}
        />
      </div>
      
      <div className="relative z-10 flex flex-col items-center text-center px-6">
        {/* Logo animation */}
        <motion.div
          className="mb-8"
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ duration: 0.8, type: "spring" }}
        >
          <DharmicLoader variant="mandala" size="xl" />
        </motion.div>
        
        {/* Brand */}
        <motion.h1 
          className="text-3xl font-bold text-gray-800 dark:text-white mb-2"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          🕉️ DharmaMind
        </motion.h1>
        
        {/* Messages */}
        <motion.p
          className="text-lg text-gray-600 dark:text-gray-300 mb-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          {message}
        </motion.p>
        
        <motion.p
          className="text-sm text-gray-500 dark:text-gray-400"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
        >
          {submessage}
        </motion.p>
        
        {/* Progress indicator */}
        <motion.div
          className="mt-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          <DharmicLoader variant="pulse" size="sm" />
        </motion.div>
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
      className="bg-gradient-to-br from-gold-50 to-gold-50 dark:from-gold-900/20 dark:to-gold-900/20 rounded-2xl p-6 shadow-lg border border-gold-200 dark:border-gold-800"
    >
      <div className="flex flex-col items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🧘</span>
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
            Meditation in Progress
          </h3>
        </div>

        <motion.div
          className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-gold-600 to-gold-600"
          animate={isActive ? { scale: [1, 1.05, 1] } : {}}
          transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
        >
          {String(minutes).padStart(2, '0')}:{String(remainingSeconds).padStart(2, '0')}
        </motion.div>

        <div className="flex gap-3">
          {isActive ? (
            <button
              onClick={onPause}
              className="px-5 py-2.5 bg-amber-500 hover:bg-amber-600 text-white rounded-xl transition-all duration-300 font-semibold shadow-md hover:shadow-lg hover:-translate-y-0.5"
            >
              ⏸️ Pause
            </button>
          ) : (
            <button
              onClick={onResume}
              className="px-5 py-2.5 bg-gold-500 hover:bg-gold-600 text-white rounded-xl transition-all duration-300 font-semibold shadow-md hover:shadow-lg hover:-translate-y-0.5"
            >
              ▶️ Resume
            </button>
          )}
          <button
            onClick={onStop}
            className="px-5 py-2.5 bg-red-500 hover:bg-red-600 text-white rounded-xl transition-all duration-300 font-semibold shadow-md hover:shadow-lg hover:-translate-y-0.5"
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
