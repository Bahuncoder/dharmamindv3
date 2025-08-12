import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SparklesIcon } from '@heroicons/react/24/outline';

interface TypingIndicatorProps {
  isVisible: boolean;
  userName?: string;
  className?: string;
}

const TypingIndicator: React.FC<TypingIndicatorProps> = ({
  isVisible,
  userName = 'DharmaMind',
  className = ''
}) => {
  const [currentMessage, setCurrentMessage] = useState(0);
  
  const contemplationMessages = [
    'DharmaMind is contemplating...',
    'Seeking wisdom from ancient texts...',
    'Aligning with dharmic principles...',
    'Drawing from spiritual traditions...',
    'Channeling compassionate guidance...',
    'Consulting the path of enlightenment...'
  ];

  useEffect(() => {
    if (!isVisible) return;

    const interval = setInterval(() => {
      setCurrentMessage(prev => (prev + 1) % contemplationMessages.length);
    }, 2000);

    return () => clearInterval(interval);
  }, [isVisible, contemplationMessages.length]);

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -20, scale: 0.9 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
          className={`flex justify-start ${className}`}
        >
          <div className="max-w-[85%] md:max-w-[70%]">
            {/* Avatar */}
            <motion.div 
              className="flex items-center justify-center w-8 h-8 mb-2 rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 text-white shadow-lg"
              animate={{ 
                scale: [1, 1.1, 1],
                rotate: [0, 5, -5, 0]
              }}
              transition={{ 
                duration: 2, 
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <SparklesIcon className="w-4 h-4" />
            </motion.div>

            {/* Typing Bubble */}
            <motion.div
              className="relative bg-white/80 backdrop-blur-sm border border-gray-200/50 text-gray-800 rounded-2xl px-6 py-4 shadow-lg mr-4"
              animate={{ 
                boxShadow: [
                  "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
                  "0 20px 25px -5px rgba(16, 185, 129, 0.1)",
                  "0 10px 15px -3px rgba(0, 0, 0, 0.1)"
                ]
              }}
              transition={{ 
                duration: 2, 
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              {/* Floating Spiritual Elements */}
              <div className="absolute -top-1 -right-1 w-2 h-2 bg-emerald-400 rounded-full opacity-60">
                <motion.div
                  animate={{ 
                    scale: [1, 1.5, 1],
                    opacity: [0.6, 0.2, 0.6]
                  }}
                  transition={{ 
                    duration: 1.5, 
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                  className="w-full h-full bg-emerald-400 rounded-full"
                />
              </div>

              <div className="absolute -bottom-1 -left-1 w-1.5 h-1.5 bg-purple-400 rounded-full opacity-50">
                <motion.div
                  animate={{ 
                    scale: [1, 1.3, 1],
                    opacity: [0.5, 0.1, 0.5]
                  }}
                  transition={{ 
                    duration: 2, 
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: 0.5
                  }}
                  className="w-full h-full bg-purple-400 rounded-full"
                />
              </div>

              {/* Main Content */}
              <div className="flex items-center space-x-4">
                {/* Animated Dots */}
                <div className="flex space-x-1">
                  {[0, 1, 2].map((index) => (
                    <motion.div
                      key={index}
                      className="w-2 h-2 bg-emerald-500 rounded-full"
                      animate={{ 
                        scale: [1, 1.5, 1],
                        opacity: [0.5, 1, 0.5]
                      }}
                      transition={{ 
                        duration: 1,
                        repeat: Infinity,
                        delay: index * 0.2,
                        ease: "easeInOut"
                      }}
                    />
                  ))}
                </div>

                {/* Contemplation Message */}
                <motion.div
                  key={currentMessage}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  className="text-sm text-gray-600 font-medium"
                >
                  {contemplationMessages[currentMessage]}
                </motion.div>
              </div>

              {/* Mystical Glow Effect */}
              <motion.div
                className="absolute inset-0 rounded-2xl pointer-events-none"
                animate={{ 
                  background: [
                    'linear-gradient(135deg, rgba(16, 185, 129, 0) 0%, rgba(16, 185, 129, 0) 100%)',
                    'linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(139, 69, 19, 0.02) 100%)',
                    'linear-gradient(135deg, rgba(16, 185, 129, 0) 0%, rgba(16, 185, 129, 0) 100%)'
                  ]
                }}
                transition={{ 
                  duration: 3, 
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
            </motion.div>

            {/* Wisdom Particles */}
            <div className="relative h-0">
              {[...Array(3)].map((_, index) => (
                <motion.div
                  key={index}
                  className="absolute w-1 h-1 bg-yellow-400 rounded-full opacity-40"
                  style={{
                    left: `${20 + index * 15}px`,
                    top: `-${10 + index * 5}px`
                  }}
                  animate={{ 
                    y: [-20, -40, -20],
                    opacity: [0.4, 0.8, 0.4],
                    scale: [1, 1.2, 1]
                  }}
                  transition={{ 
                    duration: 2 + index * 0.5,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: index * 0.3
                  }}
                />
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default TypingIndicator;
