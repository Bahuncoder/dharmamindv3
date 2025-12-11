/**
 * DharmaMind - Professional Guide Card
 * Clean, minimal card for displaying AI guide information
 */

import React from 'react';
import { motion } from 'framer-motion';
import { RishiAvatar, RISHI_COLORS } from './RishiAvatar';

// Extended data for guides
export const RISHI_EXTENDED_DATA: Record<string, {
  title: string;
  domain: string;
  description: string;
}> = {
  marici: {
    title: 'Light & Clarity',
    domain: 'Illumination',
    description: 'Guidance for finding clarity and understanding in complex situations.',
  },
  atri: {
    title: 'Meditation & Peace',
    domain: 'Inner Peace',
    description: 'Support for meditation practices and cultivating inner calm.',
  },
  angiras: {
    title: 'Sacred Traditions',
    domain: 'Rituals',
    description: 'Knowledge of spiritual practices and traditional wisdom.',
  },
  pulastya: {
    title: 'Knowledge & Learning',
    domain: 'Education',
    description: 'Guidance for learning, study, and intellectual growth.',
  },
  pulaha: {
    title: 'Wellness & Vitality',
    domain: 'Health',
    description: 'Support for physical and mental wellbeing practices.',
  },
  kratu: {
    title: 'Action & Discipline',
    domain: 'Practice',
    description: 'Guidance for building habits and taking purposeful action.',
  },
  bhrigu: {
    title: 'Wisdom & Insight',
    domain: 'Philosophy',
    description: 'Deep philosophical discussions and karmic understanding.',
  },
  vasishta: {
    title: 'Leadership & Ethics',
    domain: 'Guidance',
    description: 'Counsel on ethical decisions and leadership principles.',
  },
  daksha: {
    title: 'Creation & Skill',
    domain: 'Craft',
    description: 'Support for creative endeavors and skill development.',
  },
};

interface Rishi {
  id: string;
  name: string;
  sanskrit?: string;
  specialization: string[];
  greeting?: string;
  available: boolean;
  requires_upgrade?: boolean;
}

interface RishiCardProps {
  rishi: Rishi;
  isSelected?: boolean;
  isLocked?: boolean;
  onSelect?: () => void;
  variant?: 'compact' | 'full';
}

export const RishiCard: React.FC<RishiCardProps> = ({
  rishi,
  isSelected = false,
  isLocked = false,
  onSelect,
  variant = 'compact',
}) => {
  const extendedData = RISHI_EXTENDED_DATA[rishi.id];
  const colors = RISHI_COLORS[rishi.id] || RISHI_COLORS.default;

  if (variant === 'compact') {
    return (
      <motion.button
        onClick={!isLocked ? onSelect : undefined}
        disabled={isLocked}
        className={`
          w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all
          ${isSelected 
            ? 'bg-teal-50 border border-teal-200' 
            : isLocked
              ? 'opacity-50 cursor-not-allowed bg-gray-50 border border-gray-100'
              : 'bg-white hover:bg-gray-50 border border-gray-200 hover:border-gray-300'
          }
        `}
        whileHover={!isLocked ? { scale: 1.01 } : undefined}
        whileTap={!isLocked ? { scale: 0.99 } : undefined}
      >
        <RishiAvatar rishiId={rishi.id} rishiName={rishi.name} size="md" />
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-900 text-sm">{rishi.name}</span>
            {isLocked && (
              <span className="px-1.5 py-0.5 text-xs bg-gray-200 text-gray-600 rounded">Pro</span>
            )}
          </div>
          <div className="text-xs text-gray-500 truncate">
            {extendedData?.domain || rishi.specialization[0]}
          </div>
        </div>
        
        {isSelected && (
          <svg className="w-5 h-5 text-teal-600 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        )}
      </motion.button>
    );
  }

  // Full variant
  return (
    <motion.div
      className={`
        p-5 rounded-xl border transition-all
        ${isSelected 
          ? 'bg-teal-50 border-teal-200 shadow-sm' 
          : 'bg-white border-gray-200 hover:border-gray-300 hover:shadow-sm'
        }
      `}
      whileHover={{ y: -2 }}
    >
      <div className="flex items-start gap-4">
        <RishiAvatar rishiId={rishi.id} rishiName={rishi.name} size="lg" />
        
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold text-gray-900">{rishi.name}</h3>
            {isLocked && (
              <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded-full">Pro</span>
            )}
          </div>
          
          {extendedData && (
            <>
              <p className="text-sm text-gray-600 mb-2">{extendedData.title}</p>
              <p className="text-sm text-gray-500">{extendedData.description}</p>
            </>
          )}
          
          <div className="flex flex-wrap gap-1.5 mt-3">
            {rishi.specialization.slice(0, 3).map((spec, i) => (
              <span 
                key={i}
                className="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded-md"
              >
                {spec}
              </span>
            ))}
          </div>
        </div>
      </div>
      
      {onSelect && !isLocked && (
        <button
          onClick={onSelect}
          className={`
            w-full mt-4 py-2 px-4 rounded-lg text-sm font-medium transition-all
            ${isSelected 
              ? 'bg-teal-600 text-white hover:bg-teal-700' 
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }
          `}
        >
          {isSelected ? 'Selected' : 'Select Guide'}
        </button>
      )}
    </motion.div>
  );
};

export default RishiCard;
