/**
 * 🕉️ Sacred Rishi Guidance Panel
 * ================================
 * 
 * Collapsible panel for Rishi selection integrated into chat
 * Features sacred design and smooth animations
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RishiAvatar, RISHI_SACRED_COLORS } from './RishiAvatar';
import { RISHI_EXTENDED_DATA } from './RishiCard';

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

interface RishiGuidancePanelProps {
  onRishiSelect: (rishiId: string) => void;
  selectedRishi: string;
  userSubscription: string;
  isDemo?: boolean;
  availableRishis: Rishi[];
}

export const RishiGuidancePanel: React.FC<RishiGuidancePanelProps> = ({
  onRishiSelect,
  selectedRishi,
  userSubscription,
  isDemo = false,
  availableRishis
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const selectedRishiData = availableRishis.find(r => r.id === selectedRishi);
  const colors = selectedRishi ? RISHI_SACRED_COLORS[selectedRishi] : RISHI_SACRED_COLORS[''];
  const extendedData = selectedRishi ? RISHI_EXTENDED_DATA[selectedRishi] : null;

  const canAccessRishi = (rishi: Rishi) => {
    if (isDemo) return true;
    if (!rishi.requires_upgrade) return true;
    return userSubscription !== 'basic';
  };

  return (
    <div className="rishi-guidance-panel mb-4">
      {/* Toggle Button */}
      <motion.button
        className="w-full flex items-center justify-between px-5 py-4 rounded-2xl text-sm font-medium transition-all duration-300"
        style={{
          background: selectedRishi 
            ? `linear-gradient(135deg, ${colors.primary}15 0%, ${colors.secondary}10 100%)`
            : 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(13, 148, 136, 0.05) 100%)',
          border: `2px solid ${selectedRishi ? colors.primary : 'rgb(16, 185, 129)'}40`,
          boxShadow: selectedRishi ? `0 4px 20px ${colors.glow}` : undefined
        }}
        onClick={() => setIsExpanded(!isExpanded)}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.99 }}
      >
        <div className="flex items-center gap-4">
          {selectedRishi && selectedRishiData ? (
            <RishiAvatar rishiId={selectedRishi} size="sm" animated={true} showGlow={false} />
          ) : (
            <span className="text-2xl">🕉️</span>
          )}
          <div className="text-left">
            <div 
              className="font-bold text-base"
              style={{ color: selectedRishi ? colors.primary : '#10b981' }}
            >
              {selectedRishi && selectedRishiData 
                ? selectedRishiData.name 
                : 'Spiritual Guidance'
              }
            </div>
            {selectedRishi && extendedData ? (
              <div className="text-xs mt-0.5" style={{ color: colors.secondary }}>
                {extendedData.title} • {extendedData.element}
              </div>
            ) : (
              <div className="text-xs text-gray-500 mt-0.5">
                {selectedRishi ? 'Active' : 'Select a Rishi guide'}
              </div>
            )}
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {selectedRishi && (
            <motion.span 
              className="px-3 py-1 rounded-full text-xs font-semibold text-white"
              style={{ backgroundColor: colors.primary }}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
            >
              Active
            </motion.span>
          )}
          <motion.span 
            className="text-gray-400"
            animate={{ rotate: isExpanded ? 180 : 0 }}
            transition={{ duration: 0.3 }}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </motion.span>
        </div>
      </motion.button>

      {/* Expandable Selection Panel */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div 
              className="mt-3 p-4 rounded-2xl shadow-lg border"
              style={{ 
                background: 'white',
                borderColor: 'rgba(0,0,0,0.05)'
              }}
            >
              <div className="text-xs text-gray-500 mb-4 font-medium uppercase tracking-wider">
                Choose your spiritual guide
              </div>
              
              {/* Universal Guide Option */}
              <motion.button
                className={`w-full flex items-center justify-between p-4 rounded-xl mb-3 transition-all duration-300 ${
                  !selectedRishi 
                    ? 'bg-gradient-to-r from-emerald-50 to-teal-50 border-2 border-emerald-300 shadow-md' 
                    : 'bg-gray-50 hover:bg-gray-100 border-2 border-transparent'
                }`}
                onClick={() => {
                  onRishiSelect('');
                  setIsExpanded(false);
                }}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
              >
                <div className="flex items-center gap-3">
                  <span className="text-2xl">🕉️</span>
                  <div className="text-left">
                    <span className={`font-semibold ${!selectedRishi ? 'text-emerald-700' : 'text-gray-700'}`}>
                      Universal Guide
                    </span>
                    <p className="text-xs text-gray-500">Balanced dharmic wisdom</p>
                  </div>
                </div>
                {!selectedRishi && (
                  <div className="w-6 h-6 rounded-full bg-emerald-500 flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                )}
              </motion.button>
              
              {/* Available Rishis */}
              <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                {availableRishis.map((rishi, index) => {
                  const rishiColors = RISHI_SACRED_COLORS[rishi.id] || RISHI_SACRED_COLORS[''];
                  const rishiExtended = RISHI_EXTENDED_DATA[rishi.id];
                  const isSelected = selectedRishi === rishi.id;
                  const isLocked = !canAccessRishi(rishi);
                  
                  return (
                    <motion.button
                      key={rishi.id}
                      className={`w-full flex items-center justify-between p-4 rounded-xl transition-all duration-300 ${
                        isSelected 
                          ? 'shadow-md' 
                          : isLocked 
                            ? 'opacity-60 cursor-not-allowed'
                            : 'hover:shadow-sm'
                      }`}
                      style={{
                        background: isSelected 
                          ? `linear-gradient(135deg, ${rishiColors.primary}15 0%, ${rishiColors.secondary}10 100%)`
                          : '#f9fafb',
                        border: `2px solid ${isSelected ? rishiColors.primary : 'transparent'}`
                      }}
                      onClick={() => {
                        if (!isLocked) {
                          onRishiSelect(rishi.id);
                          setIsExpanded(false);
                        }
                      }}
                      disabled={isLocked}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      whileHover={!isLocked ? { scale: 1.01 } : undefined}
                      whileTap={!isLocked ? { scale: 0.99 } : undefined}
                    >
                      <div className="flex items-center gap-3">
                        <RishiAvatar 
                          rishiId={rishi.id} 
                          size="sm" 
                          animated={isSelected}
                          showGlow={false}
                        />
                        <div className="text-left">
                          <div className="flex items-center gap-2">
                            <span 
                              className="font-semibold"
                              style={{ color: isSelected ? rishiColors.primary : '#374151' }}
                            >
                              {rishi.name}
                            </span>
                            {isLocked && (
                              <span className="text-xs px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 font-medium">
                                PRO
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-gray-500">
                            {rishiExtended?.title || rishi.specialization.slice(0, 2).join(' • ')}
                          </p>
                        </div>
                      </div>
                      
                      {isSelected && (
                        <motion.div 
                          className="w-6 h-6 rounded-full flex items-center justify-center"
                          style={{ backgroundColor: rishiColors.primary }}
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                        >
                          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                          </svg>
                        </motion.div>
                      )}
                    </motion.button>
                  );
                })}
              </div>
                
              {/* Subscription Info */}
              {userSubscription === 'basic' && !isDemo && (
                <motion.div 
                  className="mt-4 p-3 rounded-xl bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-lg">✨</span>
                    <div className="text-sm">
                      <span className="font-semibold text-amber-800">Unlock All Rishis</span>
                      <span className="text-amber-600 ml-1">• Upgrade for full access</span>
                    </div>
                  </div>
                </motion.div>
              )}
              
              {isDemo && (
                <motion.div 
                  className="mt-4 p-3 rounded-xl bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-lg">🚀</span>
                    <div className="text-sm">
                      <span className="font-semibold text-blue-800">Demo Mode</span>
                      <span className="text-blue-600 ml-1">• All 9 Rishis available</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick Info for Selected Rishi */}
      {selectedRishi && extendedData && !isExpanded && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-3 p-3 rounded-xl text-center"
          style={{
            background: `${colors.primary}08`,
            border: `1px solid ${colors.primary}20`
          }}
        >
          <motion.p 
            className="text-sm font-semibold"
            style={{ color: colors.primary }}
            animate={{
              textShadow: [
                `0 0 5px ${colors.glow}`,
                `0 0 10px ${colors.glow}`,
                `0 0 5px ${colors.glow}`
              ]
            }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            {extendedData.sacredMantra}
          </motion.p>
          <p className="text-xs text-gray-500 mt-1 italic">
            "{extendedData.keyTeaching}"
          </p>
        </motion.div>
      )}
    </div>
  );
};

export default RishiGuidancePanel;
