/**
 * 🕉️ Sacred Rishi Mode Toggle
 * ============================
 * 
 * Elegant toggle for switching between Universal Guide and Rishi modes
 * Features sacred design with smooth animations
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RishiAvatar, RISHI_SACRED_COLORS } from './RishiAvatar';
import { RISHI_EXTENDED_DATA } from './RishiCard';
import RishiCard from './RishiCard';

interface Rishi {
  id: string;
  name: string;
  sanskrit: string;
  specialization: string[];
  greeting: string;
  available: boolean;
  requires_upgrade?: boolean;
  teaching_style?: string;
  archetype?: string;
}

interface RishiModeToggleProps {
  onRishiSelect: (rishiId: string) => void;
  selectedRishi: string;
  userSubscription: string;
  isDemo?: boolean;
  availableRishis: Rishi[];
}

export const RishiModeToggle: React.FC<RishiModeToggleProps> = ({
  onRishiSelect,
  selectedRishi,
  userSubscription,
  isDemo = false,
  availableRishis
}) => {
  const [showPicker, setShowPicker] = useState(false);
  
  const selectedRishiData = availableRishis.find(r => r.id === selectedRishi);
  const isUniversalMode = !selectedRishi || selectedRishi === '';
  const colors = selectedRishi ? RISHI_SACRED_COLORS[selectedRishi] : RISHI_SACRED_COLORS[''];
  const extendedData = selectedRishi ? RISHI_EXTENDED_DATA[selectedRishi] : null;

  const canAccessRishi = (rishi: Rishi) => {
    if (isDemo) return true;
    if (!rishi.requires_upgrade) return true;
    return userSubscription !== 'basic';
  };

  return (
    <div className="rishi-mode-toggle relative">
      {/* Main Toggle Bar */}
      <motion.div
        className="flex items-center gap-1 p-1 rounded-2xl"
        style={{
          background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
          border: '2px solid rgba(16, 185, 129, 0.2)'
        }}
      >
        {/* Universal Mode Button */}
        <motion.button
          onClick={() => onRishiSelect('')}
          className={`relative flex items-center gap-2 px-4 py-2.5 rounded-xl transition-all duration-300 ${
            isUniversalMode ? 'text-white' : 'text-gray-600 hover:text-gray-800'
          }`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {isUniversalMode && (
            <motion.div
              className="absolute inset-0 rounded-xl"
              style={{
                background: 'linear-gradient(135deg, #10b981 0%, #0d9488 100%)',
                boxShadow: '0 4px 15px rgba(16, 185, 129, 0.3)'
              }}
              layoutId="activeMode"
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
            />
          )}
          <span className="relative z-10 text-lg">🕉️</span>
          <span className="relative z-10 text-sm font-semibold">Universal</span>
        </motion.button>

        {/* Rishi Mode Button */}
        <motion.button
          onClick={() => setShowPicker(!showPicker)}
          className={`relative flex items-center gap-2 px-4 py-2.5 rounded-xl transition-all duration-300 ${
            !isUniversalMode ? 'text-white' : 'text-gray-600 hover:text-gray-800'
          }`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {!isUniversalMode && (
            <motion.div
              className="absolute inset-0 rounded-xl"
              style={{
                background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`,
                boxShadow: `0 4px 15px ${colors.glow}`
              }}
              layoutId="activeMode"
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
            />
          )}
          <span className="relative z-10">
            {!isUniversalMode && selectedRishiData ? (
              <span className="text-lg">{RISHI_SACRED_COLORS[selectedRishi]?.icon}</span>
            ) : (
              <span className="text-lg">🧘</span>
            )}
          </span>
          <span className="relative z-10 text-sm font-semibold truncate max-w-24">
            {!isUniversalMode && selectedRishiData 
              ? selectedRishiData.name.split(' ').pop()
              : 'Rishi'
            }
          </span>
          <motion.svg 
            className="relative z-10 w-4 h-4"
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
            animate={{ rotate: showPicker ? 180 : 0 }}
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </motion.svg>
        </motion.button>
      </motion.div>

      {/* Rishi Quick Picker Dropdown */}
      <AnimatePresence>
        {showPicker && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="absolute top-full left-0 right-0 mt-2 p-3 bg-white rounded-2xl shadow-2xl border border-gray-100 z-50"
            style={{ minWidth: '320px' }}
          >
            <div className="text-xs uppercase tracking-wider text-gray-400 mb-3 px-1 font-medium">
              Select Guide
            </div>
            
            {/* Quick Grid */}
            <div className="grid grid-cols-3 gap-2">
              {availableRishis.slice(0, 9).map((rishi) => {
                const rishiColors = RISHI_SACRED_COLORS[rishi.id];
                const isSelected = selectedRishi === rishi.id;
                const isLocked = !canAccessRishi(rishi);
                
                return (
                  <motion.button
                    key={rishi.id}
                    onClick={() => {
                      if (!isLocked) {
                        onRishiSelect(rishi.id);
                        setShowPicker(false);
                      }
                    }}
                    disabled={isLocked}
                    className={`relative flex flex-col items-center p-3 rounded-xl transition-all duration-300 ${
                      isLocked 
                        ? 'opacity-50 cursor-not-allowed'
                        : 'hover:bg-gray-50'
                    }`}
                    style={{
                      background: isSelected ? `${rishiColors.primary}10` : undefined,
                      boxShadow: isSelected ? `0 0 0 2px ${rishiColors.primary}` : undefined
                    }}
                    whileHover={!isLocked ? { scale: 1.05 } : undefined}
                    whileTap={!isLocked ? { scale: 0.95 } : undefined}
                  >
                    <RishiAvatar 
                      rishiId={rishi.id} 
                      size="sm" 
                      animated={isSelected}
                      showGlow={isSelected}
                    />
                    <span 
                      className="text-xs font-medium mt-2 text-center truncate w-full"
                      style={{ color: isSelected ? rishiColors.primary : '#374151' }}
                    >
                      {rishi.name.split(' ').pop()}
                    </span>
                    {isLocked && (
                      <span className="absolute top-1 right-1 text-[10px] px-1 rounded bg-amber-100 text-amber-700">
                        PRO
                      </span>
                    )}
                    {isSelected && (
                      <motion.div
                        className="absolute -top-1 -right-1 w-4 h-4 rounded-full flex items-center justify-center"
                        style={{ backgroundColor: rishiColors.primary }}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                      >
                        <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      </motion.div>
                    )}
                  </motion.button>
                );
              })}
            </div>

            {/* Selected Rishi Info */}
            {!isUniversalMode && selectedRishiData && extendedData && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="mt-3 pt-3 border-t border-gray-100"
              >
                <div 
                  className="p-3 rounded-xl"
                  style={{
                    background: `linear-gradient(135deg, ${colors.primary}08 0%, ${colors.secondary}05 100%)`
                  }}
                >
                  <p 
                    className="text-center font-semibold"
                    style={{ color: colors.primary }}
                  >
                    {extendedData.sacredMantra}
                  </p>
                  <p className="text-xs text-gray-500 text-center mt-1 italic">
                    {extendedData.keyTeaching}
                  </p>
                </div>
              </motion.div>
            )}

            {/* Upgrade CTA */}
            {userSubscription === 'basic' && !isDemo && (
              <motion.div
                className="mt-3 p-2 rounded-xl bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 text-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <span className="text-xs text-amber-700">
                  ✨ <span className="font-semibold">Upgrade</span> for all 9 Rishis
                </span>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Click outside to close */}
      {showPicker && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setShowPicker(false)}
        />
      )}
    </div>
  );
};

export default RishiModeToggle;
