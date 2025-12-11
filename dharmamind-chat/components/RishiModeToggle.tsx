/**
 * DharmaMind - Guide Mode Toggle
 * Clean, minimal toggle for switching AI guide modes
 */

import React from 'react';
import { motion } from 'framer-motion';
import { RishiAvatar } from './RishiAvatar';

interface Rishi {
  id: string;
  name: string;
  specialization: string[];
}

interface RishiModeToggleProps {
  selectedRishi: string;
  onOpenSelector: () => void;
  availableRishis?: Rishi[];
}

export const RishiModeToggle: React.FC<RishiModeToggleProps> = ({
  selectedRishi,
  onOpenSelector,
  availableRishis = [],
}) => {
  const selectedRishiData = availableRishis.find(r => r.id === selectedRishi);
  const isStandardMode = !selectedRishi || selectedRishi === '';

  return (
    <motion.button
      onClick={onOpenSelector}
      className="flex items-center gap-2.5 px-3 py-2 bg-white border border-gray-200 rounded-lg hover:border-gray-300 hover:bg-gray-50 transition-all duration-150 shadow-sm"
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
    >
      {/* Avatar */}
      {isStandardMode ? (
        <div className="w-7 h-7 rounded-full bg-gray-100 flex items-center justify-center">
          <svg className="w-3.5 h-3.5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </div>
      ) : (
        <RishiAvatar rishiId={selectedRishi} rishiName={selectedRishiData?.name} size="sm" />
      )}
      
      {/* Label */}
      <div className="text-left hidden sm:block">
        <div className="text-sm font-medium text-gray-900 leading-tight">
          {isStandardMode ? 'General' : selectedRishiData?.name}
        </div>
      </div>
      
      {/* Chevron */}
      <svg className="w-3.5 h-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </motion.button>
  );
};

export default RishiModeToggle;
