/**
 * 🕉️ Sacred Rishi Transition Animation
 * =====================================
 * 
 * Beautiful spiritual transition when switching between Rishi guides
 * Features sacred geometry, chakra animations, and meditative effects
 */

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RISHI_SACRED_COLORS } from './RishiAvatar';
import { RISHI_EXTENDED_DATA } from './RishiCard';

interface RishiTransitionProps {
  show: boolean;
  fromRishi?: string;
  toRishi: string;
  toRishiName: string;
  onComplete: () => void;
}

const getRishiIcon = (rishiId: string) => {
  return RISHI_SACRED_COLORS[rishiId]?.icon || '🕉️';
};

const getRishiGradient = (rishiId: string) => {
  const colors = RISHI_SACRED_COLORS[rishiId];
  if (!colors) return 'from-gold-500 to-gold-600';
  return `from-[${colors.primary}] to-[${colors.secondary}]`;
};

export const RishiTransition: React.FC<RishiTransitionProps> = ({
  show,
  fromRishi,
  toRishi,
  toRishiName,
  onComplete
}) => {
  const [phase, setPhase] = useState<'exit' | 'center' | 'enter'>('exit');
  
  const toColors = RISHI_SACRED_COLORS[toRishi] || RISHI_SACRED_COLORS[''];
  const fromColors = fromRishi ? RISHI_SACRED_COLORS[fromRishi] : null;
  const extendedData = RISHI_EXTENDED_DATA[toRishi];

  useEffect(() => {
    if (show) {
      setPhase('exit');
      
      const centerTimer = setTimeout(() => setPhase('center'), 600);
      const enterTimer = setTimeout(() => setPhase('enter'), 1400);
      const completeTimer = setTimeout(() => {
        onComplete();
      }, 2800);
      
      return () => {
        clearTimeout(centerTimer);
        clearTimeout(enterTimer);
        clearTimeout(completeTimer);
      };
    }
  }, [show, onComplete]);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center overflow-hidden"
        >
          {/* Animated Background */}
          <motion.div
            className="absolute inset-0"
            style={{
              background: `linear-gradient(135deg, 
                ${fromColors?.primary || '#1c1917'}20 0%, 
                #1c191790 30%,
                ${toColors.primary}20 70%,
                #1c191790 100%
              )`,
              backdropFilter: 'blur(20px)'
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />

          {/* Sacred Geometry Background */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            {/* Rotating Mandala */}
            <motion.div
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] opacity-10"
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 30, repeat: Infinity, ease: 'linear' }}
            >
              <svg viewBox="0 0 400 400" fill="none" stroke={toColors.primary} strokeWidth="0.5">
                {[...Array(12)].map((_, i) => (
                  <g key={i} transform={`rotate(${i * 30} 200 200)`}>
                    <path d="M200 50 L230 200 L200 350 L170 200 Z" opacity="0.5" />
                    <circle cx="200" cy="80" r="15" />
                  </g>
                ))}
                <circle cx="200" cy="200" r="150" />
                <circle cx="200" cy="200" r="100" />
                <circle cx="200" cy="200" r="50" />
              </svg>
            </motion.div>

            {/* Floating particles */}
            {[...Array(30)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 rounded-full"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  background: i % 2 === 0 ? toColors.primary : toColors.secondary,
                  boxShadow: `0 0 10px ${toColors.glow}`
                }}
                animate={{
                  y: [0, -100, 0],
                  x: [0, Math.random() * 50 - 25, 0],
                  opacity: [0, 0.8, 0],
                  scale: [0, 1.5, 0]
                }}
                transition={{
                  duration: 4 + Math.random() * 2,
                  delay: Math.random() * 2,
                  repeat: Infinity,
                  ease: 'easeInOut'
                }}
              />
            ))}
          </div>

          {/* Main Transition Content */}
          <div className="relative z-10 flex flex-col items-center justify-center">
            
            {/* Outgoing Rishi */}
            <AnimatePresence>
              {phase === 'exit' && fromRishi && fromColors && (
                <motion.div
                  className="absolute flex flex-col items-center"
                  initial={{ scale: 1, opacity: 1 }}
                  animate={{ scale: 0.3, opacity: 0, y: -100 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.6, ease: 'easeInOut' }}
                >
                  <motion.div
                    className="text-8xl mb-4"
                    animate={{ rotate: [0, 360] }}
                    transition={{ duration: 0.6 }}
                  >
                    {getRishiIcon(fromRishi)}
                  </motion.div>
                  <p className="text-white/60 text-sm">Departing...</p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Center Om Symbol */}
            <AnimatePresence>
              {phase === 'center' && (
                <motion.div
                  className="flex flex-col items-center"
                  initial={{ scale: 0, rotate: -180, opacity: 0 }}
                  animate={{ 
                    scale: [0, 2, 1.5],
                    rotate: [0, 360, 720],
                    opacity: [0, 1, 1]
                  }}
                  exit={{ scale: 3, opacity: 0 }}
                  transition={{ duration: 0.8, ease: 'easeInOut' }}
                >
                  {/* Om Symbol with Glow */}
                  <motion.div
                    className="relative"
                    animate={{
                      textShadow: [
                        `0 0 20px ${toColors.glow}`,
                        `0 0 60px ${toColors.glow}`,
                        `0 0 20px ${toColors.glow}`
                      ]
                    }}
                    transition={{ duration: 1, repeat: Infinity }}
                  >
                    <span className="text-9xl">🕉️</span>
                    
                    {/* Ripple Effects */}
                    {[...Array(3)].map((_, i) => (
                      <motion.div
                        key={i}
                        className="absolute inset-0 flex items-center justify-center pointer-events-none"
                        initial={{ scale: 0.5, opacity: 0.8 }}
                        animate={{ scale: 3, opacity: 0 }}
                        transition={{
                          duration: 1.5,
                          delay: i * 0.3,
                          repeat: Infinity,
                          ease: 'easeOut'
                        }}
                      >
                        <div 
                          className="w-32 h-32 rounded-full border-2"
                          style={{ borderColor: toColors.primary }}
                        />
                      </motion.div>
                    ))}
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Incoming Rishi */}
            <AnimatePresence>
              {phase === 'enter' && (
                <motion.div
                  className="flex flex-col items-center text-center"
                  initial={{ scale: 0.3, opacity: 0, y: 100 }}
                  animate={{ scale: 1, opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                >
                  {/* Rishi Avatar with Sacred Glow */}
                  <motion.div
                    className="relative mb-8"
                    animate={{ 
                      y: [0, -10, 0],
                      scale: [1, 1.05, 1]
                    }}
                    transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                  >
                    {/* Glow Ring */}
                    <motion.div
                      className="absolute inset-0 rounded-full"
                      style={{
                        background: `radial-gradient(circle, ${toColors.glow} 0%, transparent 70%)`,
                        filter: 'blur(30px)',
                        transform: 'scale(2)'
                      }}
                      animate={{
                        opacity: [0.3, 0.7, 0.3],
                        scale: [1.8, 2.2, 1.8]
                      }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                    
                    {/* Icon */}
                    <motion.span 
                      className="text-9xl relative z-10 block"
                      style={{
                        filter: `drop-shadow(0 0 30px ${toColors.glow})`
                      }}
                    >
                      {getRishiIcon(toRishi)}
                    </motion.span>
                  </motion.div>

                  {/* Rishi Name */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="space-y-2"
                  >
                    <motion.p
                      className="text-white/60 text-sm tracking-wider uppercase"
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      {toRishi === '' ? 'Returning to' : 'Entering the guidance of'}
                    </motion.p>
                    
                    <motion.h2
                      className="text-5xl font-bold text-white"
                      style={{
                        textShadow: `0 0 40px ${toColors.glow}`
                      }}
                      animate={{
                        textShadow: [
                          `0 0 20px ${toColors.glow}`,
                          `0 0 50px ${toColors.glow}`,
                          `0 0 20px ${toColors.glow}`
                        ]
                      }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      {toRishiName || 'Universal Guide'}
                    </motion.h2>

                    {/* Extended Data */}
                    {extendedData && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5 }}
                        className="mt-4 space-y-3"
                      >
                        <p 
                          className="text-lg font-medium"
                          style={{ color: toColors.secondary }}
                        >
                          {extendedData.title}
                        </p>
                        
                        <div className="flex items-center justify-center gap-4 text-sm text-white/70">
                          <span>🔥 {extendedData.element}</span>
                          <span>•</span>
                          <span>🌀 {extendedData.chakra}</span>
                        </div>
                        
                        <motion.p
                          className="text-xl italic text-white/80 max-w-md mt-4"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.8 }}
                        >
                          "{extendedData.keyTeaching}"
                        </motion.p>
                        
                        <motion.p
                          className="text-2xl mt-6"
                          style={{ color: toColors.primary }}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 1 }}
                        >
                          {extendedData.sacredMantra}
                        </motion.p>
                      </motion.div>
                    )}
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Skip Button */}
          <motion.button
            className="absolute bottom-8 right-8 px-4 py-2 rounded-xl bg-white/10 text-white/60 text-sm hover:bg-white/20 hover:text-white transition-all"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            onClick={onComplete}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Skip
          </motion.button>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

/**
 * 🕉️ Rishi Context Badge
 * Shows current Rishi in chat header
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
  const colors = RISHI_SACRED_COLORS[rishiId] || RISHI_SACRED_COLORS[''];
  const extendedData = RISHI_EXTENDED_DATA[rishiId];

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="inline-flex items-center gap-3 px-4 py-2.5 rounded-2xl shadow-lg"
      style={{
        background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`,
        boxShadow: `0 4px 20px ${colors.glow}`
      }}
    >
      <motion.span 
        className="text-2xl"
        animate={{ scale: [1, 1.1, 1] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        {colors.icon}
      </motion.span>
      
      <div className="flex flex-col">
        <span className="text-white font-semibold text-sm">{rishiName}</span>
        <span className="text-white/70 text-xs">
          {messageCount} message{messageCount !== 1 ? 's' : ''} 
          {extendedData && ` • ${extendedData.element}`}
        </span>
      </div>
      
      {onSwitch && (
        <motion.button
          onClick={onSwitch}
          className="ml-2 p-1.5 rounded-lg bg-white/20 hover:bg-white/30 transition-colors"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          title="Switch Guide"
        >
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
          </svg>
        </motion.button>
      )}
    </motion.div>
  );
};

/**
 * 🕉️ Mini Rishi Indicator
 * Compact version for message bubbles
 */
interface MiniRishiIndicatorProps {
  rishiId: string;
  rishiName: string;
}

export const MiniRishiIndicator: React.FC<MiniRishiIndicatorProps> = ({
  rishiId,
  rishiName
}) => {
  const colors = RISHI_SACRED_COLORS[rishiId] || RISHI_SACRED_COLORS[''];

  return (
    <div 
      className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium"
      style={{
        backgroundColor: `${colors.primary}15`,
        color: colors.primary
      }}
    >
      <span className="text-sm">{colors.icon}</span>
      <span>{rishiName}</span>
    </div>
  );
};

export default RishiTransition;
