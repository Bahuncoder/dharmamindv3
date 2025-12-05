/**
 * 🕉️ Sacred Rishi Card Component
 * ===============================
 * 
 * Beautiful, immersive card for each Manas Putra (Mind-Born Rishi)
 * Features domain-specific theming, animations, and sacred design
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RishiAvatar, RISHI_SACRED_COLORS } from './RishiAvatar';

interface Rishi {
  id: string;
  name: string;
  sanskrit?: string;
  specialization: string[];
  greeting?: string;
  available: boolean;
  requires_upgrade?: boolean;
  teaching_style?: string;
  archetype?: string;
}

interface RishiCardProps {
  rishi: Rishi;
  isSelected: boolean;
  isLocked: boolean;
  onSelect: () => void;
  size?: 'compact' | 'full';
  showDetails?: boolean;
}

// Extended Rishi data with sacred information
const RISHI_EXTENDED_DATA: Record<string, {
  title: string;
  element: string;
  chakra: string;
  sacredMantra: string;
  keyTeaching: string;
  color: string;
}> = {
  atri: {
    title: 'Master of Tapasya',
    element: 'Ākāśa (Space)',
    chakra: 'Sahasrara',
    sacredMantra: 'ॐ अत्रये नमः',
    keyTeaching: 'Through stillness, find the infinite',
    color: 'indigo'
  },
  bhrigu: {
    title: 'Father of Jyotisha',
    element: 'Agni (Fire)',
    chakra: 'Manipura',
    sacredMantra: 'ॐ भृगवे नमः',
    keyTeaching: 'Stars reveal karma, will shapes destiny',
    color: 'amber'
  },
  vashishta: {
    title: 'Raja Guru',
    element: 'Pṛthvī (Earth)',
    chakra: 'Muladhara',
    sacredMantra: 'ॐ वसिष्ठाय नमः',
    keyTeaching: 'Dharma protects those who protect dharma',
    color: 'purple'
  },
  vishwamitra: {
    title: 'Friend of the Universe',
    element: 'Agni (Fire)',
    chakra: 'Svadhisthana',
    sacredMantra: 'ॐ भूर्भुवः स्वः',
    keyTeaching: 'Transformation through determination',
    color: 'orange'
  },
  gautama: {
    title: 'Founder of Nyaya',
    element: 'Jala (Water)',
    chakra: 'Anahata',
    sacredMantra: 'ॐ गौतमाय नमः',
    keyTeaching: 'Compassion is the highest dharma',
    color: 'emerald'
  },
  jamadagni: {
    title: 'Lord of Sacred Fire',
    element: 'Agni (Fire)',
    chakra: 'Manipura',
    sacredMantra: 'ॐ जमदग्नये नमः',
    keyTeaching: 'Discipline is the fire that transforms',
    color: 'red'
  },
  kashyapa: {
    title: 'Cosmic Father',
    element: 'Pṛthvī (Earth)',
    chakra: 'Muladhara',
    sacredMantra: 'वसुधैव कुटुम्बकम्',
    keyTeaching: 'The world is one family',
    color: 'stone'
  },
  angiras: {
    title: 'Lord of Divine Fire',
    element: 'Agni (Fire)',
    chakra: 'Manipura',
    sacredMantra: 'ॐ अंगिरसे नमः',
    keyTeaching: 'Sacred fire carries prayers to divine',
    color: 'yellow'
  },
  pulastya: {
    title: 'Narrator of Puranas',
    element: 'Ākāśa (Space)',
    chakra: 'Vishuddha',
    sacredMantra: 'ॐ पुलस्त्याय नमः',
    keyTeaching: 'Every life is a sacred story',
    color: 'blue'
  }
};

export const RishiCard: React.FC<RishiCardProps> = ({
  rishi,
  isSelected,
  isLocked,
  onSelect,
  size = 'full',
  showDetails = true
}) => {
  const colors = RISHI_SACRED_COLORS[rishi.id] || RISHI_SACRED_COLORS[''];
  const extendedData = RISHI_EXTENDED_DATA[rishi.id];

  if (size === 'compact') {
    return (
      <motion.button
        onClick={onSelect}
        disabled={isLocked || !rishi.available}
        className={`relative flex items-center gap-3 p-3 rounded-xl transition-all duration-300 w-full text-left ${
          isSelected 
            ? 'bg-white shadow-lg ring-2' 
            : isLocked 
              ? 'bg-gray-50 opacity-60 cursor-not-allowed'
              : 'bg-white/80 hover:bg-white hover:shadow-md'
        }`}
        style={{
          ringColor: isSelected ? colors.primary : 'transparent'
        }}
        whileHover={!isLocked ? { scale: 1.02, y: -2 } : undefined}
        whileTap={!isLocked ? { scale: 0.98 } : undefined}
      >
        <RishiAvatar
          rishiId={rishi.id}
          size="sm"
          animated={isSelected}
          showGlow={isSelected}
        />
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm text-gray-900 truncate">
              {rishi.name}
            </span>
            {isSelected && (
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="px-1.5 py-0.5 rounded-full text-[10px] font-bold text-white"
                style={{ backgroundColor: colors.primary }}
              >
                Active
              </motion.span>
            )}
            {isLocked && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-amber-100 text-amber-700">
                PRO
              </span>
            )}
          </div>
          <p className="text-xs text-gray-500 truncate">
            {rishi.specialization.slice(0, 2).join(' • ')}
          </p>
        </div>
      </motion.button>
    );
  }

  return (
    <motion.button
      onClick={onSelect}
      disabled={isLocked || !rishi.available}
      className={`relative overflow-hidden rounded-2xl transition-all duration-300 ${
        isSelected 
          ? 'ring-2 shadow-xl' 
          : isLocked 
            ? 'opacity-60 cursor-not-allowed'
            : 'hover:shadow-lg'
      }`}
      style={{
        ringColor: isSelected ? colors.primary : 'transparent',
        background: isSelected 
          ? `linear-gradient(135deg, ${colors.primary}15 0%, ${colors.secondary}10 100%)`
          : 'white'
      }}
      whileHover={!isLocked ? { scale: 1.02, y: -4 } : undefined}
      whileTap={!isLocked ? { scale: 0.98 } : undefined}
    >
      {/* Sacred glow effect when selected */}
      {isSelected && (
        <motion.div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `radial-gradient(circle at center, ${colors.glow} 0%, transparent 70%)`,
            opacity: 0.3
          }}
          animate={{
            opacity: [0.2, 0.4, 0.2],
            scale: [1, 1.1, 1]
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
        />
      )}
      
      {/* Card Content */}
      <div className="relative p-5">
        {/* Header */}
        <div className="flex items-start gap-4">
          <RishiAvatar
            rishiId={rishi.id}
            size="lg"
            animated={isSelected}
            showGlow={isSelected}
            showElement={showDetails}
          />
          
          <div className="flex-1 min-w-0 text-left">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-bold text-lg text-gray-900">
                {rishi.name}
              </h3>
              {isLocked && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 font-semibold">
                  🔒 PRO
                </span>
              )}
            </div>
            
            {rishi.sanskrit && (
              <p className="text-sm font-medium mb-1" style={{ color: colors.primary }}>
                {rishi.sanskrit}
              </p>
            )}
            
            {extendedData && (
              <p className="text-xs text-gray-500 italic">
                {extendedData.title}
              </p>
            )}
          </div>
        </div>
        
        {/* Details Section */}
        {showDetails && extendedData && (
          <div className="mt-4 space-y-3">
            {/* Specializations */}
            <div className="flex flex-wrap gap-1.5">
              {rishi.specialization.slice(0, 3).map((spec, idx) => (
                <span
                  key={idx}
                  className="text-xs px-2 py-1 rounded-full"
                  style={{
                    backgroundColor: `${colors.primary}15`,
                    color: colors.primary
                  }}
                >
                  {spec}
                </span>
              ))}
            </div>
            
            {/* Sacred Info Grid */}
            <div className="grid grid-cols-2 gap-2 pt-2 border-t border-gray-100">
              <div className="text-left">
                <p className="text-[10px] uppercase tracking-wider text-gray-400">Element</p>
                <p className="text-xs font-medium text-gray-700">{extendedData.element}</p>
              </div>
              <div className="text-left">
                <p className="text-[10px] uppercase tracking-wider text-gray-400">Chakra</p>
                <p className="text-xs font-medium text-gray-700">{extendedData.chakra}</p>
              </div>
            </div>
            
            {/* Key Teaching */}
            <div 
              className="p-3 rounded-xl text-left"
              style={{
                background: `linear-gradient(135deg, ${colors.primary}08 0%, ${colors.secondary}05 100%)`,
                borderLeft: `3px solid ${colors.primary}`
              }}
            >
              <p className="text-xs italic text-gray-600">
                "{extendedData.keyTeaching}"
              </p>
            </div>
            
            {/* Sacred Mantra */}
            <div className="text-center pt-2">
              <motion.p 
                className="text-lg font-semibold"
                style={{ color: colors.primary }}
                animate={isSelected ? {
                  textShadow: [
                    `0 0 5px ${colors.glow}`,
                    `0 0 15px ${colors.glow}`,
                    `0 0 5px ${colors.glow}`
                  ]
                } : undefined}
                transition={{ duration: 2, repeat: Infinity }}
              >
                {extendedData.sacredMantra}
              </motion.p>
            </div>
          </div>
        )}
        
        {/* Selection Indicator */}
        {isSelected && (
          <motion.div
            className="absolute top-3 right-3"
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <div 
              className="w-6 h-6 rounded-full flex items-center justify-center text-white"
              style={{ backgroundColor: colors.primary }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            </div>
          </motion.div>
        )}
      </div>
    </motion.button>
  );
};

export { RISHI_EXTENDED_DATA };
export default RishiCard;

