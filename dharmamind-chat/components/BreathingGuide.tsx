import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface BreathingGuideProps {
  isVisible: boolean;
  onClose: () => void;
  className?: string;
}

const BreathingGuide: React.FC<BreathingGuideProps> = ({
  isVisible,
  onClose,
  className = ''
}) => {
  const [phase, setPhase] = useState<'inhale' | 'hold' | 'exhale' | 'pause'>('inhale');
  const [cycleCount, setCycleCount] = useState(0);

  useEffect(() => {
    if (!isVisible) return;

    const cycle = async () => {
      // Inhale (4 seconds)
      setPhase('inhale');
      await new Promise(resolve => setTimeout(resolve, 4000));
      
      // Hold (4 seconds)
      setPhase('hold');
      await new Promise(resolve => setTimeout(resolve, 4000));
      
      // Exhale (6 seconds)
      setPhase('exhale');
      await new Promise(resolve => setTimeout(resolve, 6000));
      
      // Pause (2 seconds)
      setPhase('pause');
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setCycleCount(prev => prev + 1);
    };

    const interval = setInterval(cycle, 16000); // Total cycle time
    cycle(); // Start immediately

    return () => clearInterval(interval);
  }, [isVisible]);

  const getPhaseText = () => {
    switch (phase) {
      case 'inhale': return 'Breathe in slowly...';
      case 'hold': return 'Hold your breath...';
      case 'exhale': return 'Breathe out gently...';
      case 'pause': return 'Rest...';
    }
  };

  const getPhaseColor = () => {
    switch (phase) {
      case 'inhale': return 'from-blue-400 to-blue-600';
      case 'hold': return 'from-purple-400 to-purple-600';
      case 'exhale': return 'from-emerald-400 to-emerald-600';
      case 'pause': return 'from-gray-400 to-gray-600';
    }
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className={`fixed inset-0 bg-black/70 flex items-center justify-center z-50 ${className}`}
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            className="text-center"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Breathing Circle */}
            <motion.div
              className={`w-64 h-64 mx-auto mb-8 rounded-full bg-gradient-to-br ${getPhaseColor()} shadow-2xl flex items-center justify-center relative`}
              animate={{
                scale: phase === 'inhale' ? 1.2 : phase === 'exhale' ? 0.8 : 1,
              }}
              transition={{
                duration: phase === 'inhale' ? 4 : phase === 'exhale' ? 6 : phase === 'hold' ? 4 : 2,
                ease: "easeInOut"
              }}
            >
              <span className="text-6xl">🧘‍♂️</span>
              
              {/* Ripple Effect */}
              <motion.div
                className="absolute inset-0 rounded-full border-2 border-white/30"
                animate={{
                  scale: [1, 1.5, 1],
                  opacity: [0.5, 0, 0.5]
                }}
                transition={{
                  duration: 3,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
            </motion.div>

            {/* Instructions */}
            <motion.div
              key={phase}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-white text-center mb-6"
            >
              <h2 className="text-2xl font-medium mb-2">{getPhaseText()}</h2>
              <p className="text-white/80">Cycle {cycleCount + 1}</p>
            </motion.div>

            {/* Close Button */}
            <button
              onClick={onClose}
              className="text-white/60 hover:text-white transition-colors text-sm"
            >
              Click anywhere to close
            </button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default BreathingGuide;
