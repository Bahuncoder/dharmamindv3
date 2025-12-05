/**
 * 🕉️ Enhanced Sacred Rishi Selector
 * ==================================
 * 
 * Immersive selection experience for the Nine Manas Putra (Mind-Born Rishis)
 * Features sacred design, smooth animations, and spiritual theming
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { createPortal } from 'react-dom';
import { useColor } from '../contexts/ColorContext';
import { RishiAvatar, RISHI_SACRED_COLORS } from './RishiAvatar';
import RishiCard, { RISHI_EXTENDED_DATA } from './RishiCard';

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

interface RishiSelectorProps {
  onRishiSelect: (rishiId: string) => void;
  selectedRishi: string;
  userSubscription: string;
  isDemo?: boolean;
  availableRishis: Rishi[];
}

// Domain categories for filtering
const DOMAIN_CATEGORIES = [
  { id: 'all', label: 'All Guides', icon: '🕉️' },
  { id: 'spiritual', label: 'Spiritual Growth', icon: '🧘' },
  { id: 'wisdom', label: 'Wisdom & Knowledge', icon: '📚' },
  { id: 'action', label: 'Action & Discipline', icon: '⚡' },
  { id: 'healing', label: 'Healing & Compassion', icon: '💚' },
];

const RISHI_DOMAIN_MAP: Record<string, string[]> = {
  atri: ['spiritual'],
  bhrigu: ['wisdom', 'spiritual'],
  vashishta: ['wisdom', 'action'],
  vishwamitra: ['action', 'spiritual'],
  gautama: ['healing', 'wisdom'],
  jamadagni: ['action'],
  kashyapa: ['healing', 'wisdom'],
  angiras: ['spiritual', 'action'],
  pulastya: ['wisdom', 'spiritual'],
};

export const RishiSelector: React.FC<RishiSelectorProps> = ({
  onRishiSelect,
  selectedRishi,
  userSubscription,
  isDemo = false,
  availableRishis
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState('all');
  const [hoveredRishi, setHoveredRishi] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const { currentTheme } = useColor();
  const colors = currentTheme.colors;

  useEffect(() => {
    setMounted(true);
  }, []);

  const selectedRishiData = availableRishis.find(r => r.id === selectedRishi);
  const isStandardMode = !selectedRishi || selectedRishi === '';

  // Filter Rishis based on search and category
  const filteredRishis = availableRishis.filter(rishi => {
    const matchesSearch = 
      rishi.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      rishi.specialization.some(s => s.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesCategory = 
      activeCategory === 'all' ||
      RISHI_DOMAIN_MAP[rishi.id]?.includes(activeCategory);
    
    return matchesSearch && matchesCategory;
  });

  const canAccessRishi = (rishi: Rishi) => {
    if (isDemo) return true;
    if (!rishi.requires_upgrade) return true;
    return userSubscription !== 'basic';
  };

  const handleSelectRishi = (rishiId: string) => {
    onRishiSelect(rishiId);
    setIsExpanded(false);
    setSearchQuery('');
    setActiveCategory('all');
  };

  // Mini selector in sidebar
  return (
    <React.Fragment>
      <div className="rishi-selector">
        {/* Current Selection Display */}
        <div className="mb-3 px-1">
          <p className="text-[11px] uppercase tracking-wider text-gray-400 mb-2 font-medium">
            Spiritual Guide
          </p>
          
          {/* Mode Toggle */}
          <div className="flex items-center gap-2">
            {/* Universal Guide */}
            <motion.button
              onClick={() => !isStandardMode && onRishiSelect('')}
              className={`flex-1 py-2.5 px-3 rounded-xl transition-all duration-300 flex items-center justify-center gap-2 ${
                isStandardMode 
                  ? 'bg-gradient-to-r from-emerald-50 to-teal-50 shadow-sm border-2 border-emerald-200' 
                  : 'bg-gray-50 hover:bg-gray-100 border-2 border-transparent'
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <span className={`text-lg ${isStandardMode ? 'animate-pulse' : ''}`}>🕉️</span>
              <span className={`text-xs font-semibold ${isStandardMode ? 'text-emerald-700' : 'text-gray-500'}`}>
                Universal
              </span>
            </motion.button>

            {/* Rishi Mode Button */}
            <motion.button
              onClick={() => setIsExpanded(true)}
              className={`flex-1 py-2.5 px-3 rounded-xl transition-all duration-300 flex items-center justify-center gap-2 ${
                !isStandardMode 
                  ? 'shadow-sm border-2' 
                  : 'bg-gray-50 hover:bg-gray-100 border-2 border-transparent'
              }`}
              style={{
                background: !isStandardMode && selectedRishiData 
                  ? `linear-gradient(135deg, ${RISHI_SACRED_COLORS[selectedRishi]?.primary}15 0%, ${RISHI_SACRED_COLORS[selectedRishi]?.secondary}10 100%)`
                  : undefined,
                borderColor: !isStandardMode && selectedRishiData
                  ? RISHI_SACRED_COLORS[selectedRishi]?.primary
                  : undefined
              }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {!isStandardMode && selectedRishiData ? (
                <>
                  <RishiAvatar rishiId={selectedRishi} size="sm" animated={true} showGlow={false} />
                  <span 
                    className="text-xs font-semibold truncate"
                    style={{ color: RISHI_SACRED_COLORS[selectedRishi]?.primary }}
                  >
                    {selectedRishiData.name.split(' ')[1] || selectedRishiData.name}
                  </span>
                </>
              ) : (
                <>
                  <span className="text-lg">🧘</span>
                  <span className="text-xs font-semibold text-gray-500">
                    Rishis
                  </span>
                </>
              )}
              
              {/* Expand indicator */}
              <motion.svg 
                className="w-3 h-3 text-gray-400" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
                animate={{ rotate: isExpanded ? 180 : 0 }}
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </motion.svg>
            </motion.button>
          </div>
        </div>

        {/* Quick Access - Selected Rishi Info */}
        {!isStandardMode && selectedRishiData && RISHI_EXTENDED_DATA[selectedRishi] && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-3 p-3 rounded-xl text-center"
            style={{
              background: `linear-gradient(135deg, ${RISHI_SACRED_COLORS[selectedRishi]?.primary}10 0%, ${RISHI_SACRED_COLORS[selectedRishi]?.secondary}05 100%)`,
              border: `1px solid ${RISHI_SACRED_COLORS[selectedRishi]?.primary}30`
            }}
          >
            <p 
              className="text-sm font-semibold mb-1"
              style={{ color: RISHI_SACRED_COLORS[selectedRishi]?.primary }}
            >
              {RISHI_EXTENDED_DATA[selectedRishi].sacredMantra}
            </p>
            <p className="text-[10px] text-gray-500 italic">
              "{RISHI_EXTENDED_DATA[selectedRishi].keyTeaching}"
            </p>
          </motion.div>
        )}
      </div>

      {/* Full Selection Modal */}
      {isExpanded && mounted && createPortal(
        <AnimatePresence>
          <motion.div 
            className="fixed inset-0 flex items-center justify-center p-4"
            style={{ zIndex: 9999 }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Backdrop with sacred pattern */}
            <motion.div 
              className="absolute inset-0"
              style={{
                background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
                backdropFilter: 'blur(12px)'
              }}
              onClick={() => setIsExpanded(false)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            />
            
            {/* Modal Container */}
            <motion.div 
              className="relative bg-white rounded-3xl shadow-2xl max-w-4xl w-full max-h-[85vh] overflow-hidden"
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            >
              {/* Sacred Header */}
              <div className="relative px-8 py-6 bg-gradient-to-r from-emerald-50 via-white to-purple-50 border-b border-gray-100">
                {/* Decorative elements */}
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-400 via-amber-400 to-purple-400" />
                
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
                      <span className="text-3xl">🕉️</span>
                      <span>
                        <span className="bg-gradient-to-r from-emerald-600 to-purple-600 bg-clip-text text-transparent">
                          Nava Manas Putra
                        </span>
                        <span className="block text-sm font-normal text-gray-500 mt-0.5">
                          The Nine Mind-Born Sages
                        </span>
                      </span>
                    </h2>
                  </div>
                  
                  <motion.button
                    onClick={() => setIsExpanded(false)}
                    className="p-2 rounded-xl hover:bg-gray-100 transition-colors"
                    whileHover={{ scale: 1.1, rotate: 90 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </motion.button>
                </div>

                {/* Search and Filter */}
                <div className="mt-5 space-y-4">
                  {/* Search Input */}
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search by name, expertise, or domain..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full px-5 py-3 pl-12 rounded-2xl text-sm bg-white border-2 border-gray-100 focus:border-emerald-300 focus:ring-4 focus:ring-emerald-100 transition-all duration-300 outline-none"
                    />
                    <svg 
                      className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    {searchQuery && (
                      <motion.button
                        onClick={() => setSearchQuery('')}
                        className="absolute right-4 top-1/2 -translate-y-1/2 p-1 rounded-full hover:bg-gray-100"
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                      >
                        <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </motion.button>
                    )}
                  </div>

                  {/* Category Filters */}
                  <div className="flex flex-wrap gap-2">
                    {DOMAIN_CATEGORIES.map((category) => (
                      <motion.button
                        key={category.id}
                        onClick={() => setActiveCategory(category.id)}
                        className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
                          activeCategory === category.id
                            ? 'bg-emerald-100 text-emerald-700 shadow-sm'
                            : 'bg-gray-50 text-gray-600 hover:bg-gray-100'
                        }`}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <span className="mr-1.5">{category.icon}</span>
                        {category.label}
                      </motion.button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Rishi Grid */}
              <div className="p-6 overflow-y-auto max-h-[calc(85vh-250px)]">
                {/* Universal Guide Option */}
                <motion.div 
                  className="mb-6"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                >
                  <p className="text-xs uppercase tracking-wider text-gray-400 mb-3 px-1 font-medium">
                    Universal Guidance
                  </p>
                  <motion.button
                    onClick={() => handleSelectRishi('')}
                    className={`w-full p-5 rounded-2xl text-left transition-all duration-300 border-2 ${
                      isStandardMode
                        ? 'bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-300 shadow-lg'
                        : 'bg-gray-50 border-transparent hover:bg-gray-100 hover:border-gray-200'
                    }`}
                    whileHover={{ scale: 1.01, y: -2 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    <div className="flex items-center gap-4">
                      <div className={`w-16 h-16 rounded-2xl flex items-center justify-center text-3xl ${
                        isStandardMode 
                          ? 'bg-gradient-to-br from-emerald-400 to-teal-500 shadow-lg'
                          : 'bg-gray-200'
                      }`}>
                        🕉️
                      </div>
                      <div className="flex-1">
                        <h3 className={`text-lg font-bold ${isStandardMode ? 'text-emerald-700' : 'text-gray-700'}`}>
                          Universal Guide
                        </h3>
                        <p className="text-sm text-gray-500 mt-1">
                          Balanced wisdom from all dharmic traditions • General spiritual guidance
                        </p>
                      </div>
                      {isStandardMode && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="w-8 h-8 rounded-full bg-emerald-500 flex items-center justify-center"
                        >
                          <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                          </svg>
                        </motion.div>
                      )}
                    </div>
                  </motion.button>
                </motion.div>

                {/* Rishis */}
                <div>
                  <p className="text-xs uppercase tracking-wider text-gray-400 mb-3 px-1 font-medium">
                    Sacred Rishis ({filteredRishis.length})
                  </p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <AnimatePresence mode="popLayout">
                      {filteredRishis.map((rishi, index) => (
                        <motion.div
                          key={rishi.id}
                          initial={{ opacity: 0, scale: 0.9, y: 20 }}
                          animate={{ opacity: 1, scale: 1, y: 0 }}
                          exit={{ opacity: 0, scale: 0.9 }}
                          transition={{ delay: index * 0.05 }}
                          layout
                        >
                          <RishiCard
                            rishi={rishi}
                            isSelected={selectedRishi === rishi.id}
                            isLocked={!canAccessRishi(rishi)}
                            onSelect={() => handleSelectRishi(rishi.id)}
                            size="full"
                            showDetails={true}
                          />
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </div>

                  {/* Empty State */}
                  {filteredRishis.length === 0 && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-center py-12"
                    >
                      <span className="text-6xl mb-4 block">🔍</span>
                      <p className="text-gray-500 text-lg">No Rishis found matching your search</p>
                      <button
                        onClick={() => { setSearchQuery(''); setActiveCategory('all'); }}
                        className="mt-3 text-emerald-600 font-medium hover:text-emerald-700 transition-colors"
                      >
                        Clear filters
                      </button>
                    </motion.div>
                  )}
                </div>
              </div>

              {/* Footer Info */}
              {userSubscription === 'basic' && !isDemo && (
                <div className="px-6 py-4 bg-gradient-to-r from-amber-50 to-orange-50 border-t border-amber-100">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">✨</span>
                      <div>
                        <p className="text-sm font-semibold text-amber-800">Unlock All Rishis</p>
                        <p className="text-xs text-amber-600">Upgrade to access all 9 spiritual guides</p>
                      </div>
                    </div>
                    <motion.button
                      className="px-4 py-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold text-sm shadow-lg"
                      whileHover={{ scale: 1.05, boxShadow: '0 10px 30px rgba(245, 158, 11, 0.3)' }}
                      whileTap={{ scale: 0.95 }}
                    >
                      Upgrade Now
                    </motion.button>
                  </div>
                </div>
              )}
            </motion.div>
          </motion.div>
        </AnimatePresence>,
        document.body
      )}
    </React.Fragment>
  );
};

export default RishiSelector;
