/**
 * 🕉️ Sacred Rishi Avatar Component
 * ================================
 * 
 * Animated, sacred representation of each Rishi (Manas Putra)
 * Each avatar reflects the Rishi's domain, element, and chakra association
 */

import React from 'react';
import { motion } from 'framer-motion';

interface RishiAvatarProps {
  rishiId: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  animated?: boolean;
  showGlow?: boolean;
  showElement?: boolean;
  onClick?: () => void;
}

// Sacred colors for each Rishi based on their element and domain
const RISHI_SACRED_COLORS: Record<string, {
  primary: string;
  secondary: string;
  glow: string;
  element: string;
  icon: string;
  sanskrit: string;
  domain: string;
}> = {
  atri: {
    primary: '#6366f1', // Deep Indigo - Space/Akasha
    secondary: '#818cf8',
    glow: 'rgba(99, 102, 241, 0.4)',
    element: 'Ākāśa',
    icon: '🧘',
    sanskrit: 'अत्रि',
    domain: 'Meditation'
  },
  bhrigu: {
    primary: '#f59e0b', // Sacred Gold - Fire/Agni
    secondary: '#fbbf24',
    glow: 'rgba(245, 158, 11, 0.4)',
    element: 'Agni',
    icon: '⭐',
    sanskrit: 'भृगु',
    domain: 'Astrology'
  },
  vashishta: {
    primary: '#8b5cf6', // Royal Purple - Earth/Prithvi
    secondary: '#a78bfa',
    glow: 'rgba(139, 92, 246, 0.4)',
    element: 'Pṛthvī',
    icon: '📚',
    sanskrit: 'वशिष्ठ',
    domain: 'Dharma'
  },
  vishwamitra: {
    primary: '#f97316', // Blazing Orange - Fire/Agni
    secondary: '#fb923c',
    glow: 'rgba(249, 115, 22, 0.4)',
    element: 'Agni',
    icon: '⚡',
    sanskrit: 'विश्वामित्र',
    domain: 'Transformation'
  },
  gautama: {
    primary: '#10b981', // Healing Green - Water/Jala
    secondary: '#34d399',
    glow: 'rgba(16, 185, 129, 0.4)',
    element: 'Jala',
    icon: '💚',
    sanskrit: 'गौतम',
    domain: 'Compassion'
  },
  jamadagni: {
    primary: '#ef4444', // Crimson Red - Fire/Agni
    secondary: '#f87171',
    glow: 'rgba(239, 68, 68, 0.4)',
    element: 'Agni',
    icon: '🔥',
    sanskrit: 'जमदग्नि',
    domain: 'Discipline'
  },
  kashyapa: {
    primary: '#78716c', // Earth Brown - Earth/Prithvi
    secondary: '#a8a29e',
    glow: 'rgba(120, 113, 108, 0.4)',
    element: 'Pṛthvī',
    icon: '🌍',
    sanskrit: 'कश्यप',
    domain: 'Creation'
  },
  angiras: {
    primary: '#eab308', // Sacred Gold - Fire/Agni
    secondary: '#facc15',
    glow: 'rgba(234, 179, 8, 0.4)',
    element: 'Agni',
    icon: '🔱',
    sanskrit: 'अंगिरस',
    domain: 'Fire Wisdom'
  },
  pulastya: {
    primary: '#3b82f6', // Twilight Blue - Space/Akasha
    secondary: '#60a5fa',
    glow: 'rgba(59, 130, 246, 0.4)',
    element: 'Ākāśa',
    icon: '📖',
    sanskrit: 'पुलस्त्य',
    domain: 'Cosmic Knowledge'
  },
  // Fallback for Universal Guide
  '': {
    primary: '#10b981',
    secondary: '#34d399',
    glow: 'rgba(16, 185, 129, 0.4)',
    element: 'Pañcabhūta',
    icon: '🕉️',
    sanskrit: 'सर्वज्ञ',
    domain: 'Universal Wisdom'
  }
};

const SIZE_MAP = {
  sm: { container: 40, icon: 20, ring: 2 },
  md: { container: 56, icon: 28, ring: 3 },
  lg: { container: 72, icon: 36, ring: 4 },
  xl: { container: 96, icon: 48, ring: 5 }
};

export const RishiAvatar: React.FC<RishiAvatarProps> = ({
  rishiId,
  size = 'md',
  animated = true,
  showGlow = true,
  showElement = false,
  onClick
}) => {
  const colors = RISHI_SACRED_COLORS[rishiId] || RISHI_SACRED_COLORS[''];
  const dimensions = SIZE_MAP[size];

  return (
    <motion.div
      className="relative cursor-pointer"
      onClick={onClick}
      whileHover={animated ? { scale: 1.1 } : undefined}
      whileTap={animated ? { scale: 0.95 } : undefined}
    >
      {/* Glow Ring */}
      {showGlow && (
        <motion.div
          className="absolute inset-0 rounded-full"
          style={{
            background: `radial-gradient(circle, ${colors.glow} 0%, transparent 70%)`,
            filter: 'blur(8px)',
            transform: 'scale(1.5)'
          }}
          animate={animated ? {
            opacity: [0.5, 0.8, 0.5],
            scale: [1.3, 1.6, 1.3]
          } : undefined}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
        />
      )}
      
      {/* Main Avatar Circle */}
      <motion.div
        className="relative flex items-center justify-center rounded-full"
        style={{
          width: dimensions.container,
          height: dimensions.container,
          background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`,
          boxShadow: showGlow ? `0 0 20px ${colors.glow}` : undefined
        }}
        animate={animated ? {
          boxShadow: [
            `0 0 15px ${colors.glow}`,
            `0 0 30px ${colors.glow}`,
            `0 0 15px ${colors.glow}`
          ]
        } : undefined}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'easeInOut'
        }}
      >
        {/* Inner Ring */}
        <div
          className="absolute inset-1 rounded-full border-2 border-white/30"
          style={{ borderWidth: dimensions.ring }}
        />
        
        {/* Icon */}
        <span 
          className="relative z-10"
          style={{ fontSize: dimensions.icon }}
        >
          {colors.icon}
        </span>
      </motion.div>
      
      {/* Element Badge */}
      {showElement && (
        <motion.div
          className="absolute -bottom-1 -right-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold bg-white shadow-md"
          style={{ color: colors.primary }}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2 }}
        >
          {colors.element}
        </motion.div>
      )}
    </motion.div>
  );
};

// Export the colors for use in other components
export { RISHI_SACRED_COLORS };
export default RishiAvatar;

