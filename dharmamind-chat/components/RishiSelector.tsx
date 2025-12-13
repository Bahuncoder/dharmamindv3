/**
 * DharmaMind - AI Guide Selector
 * Professional, clean interface for selecting spiritual guides
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { createPortal } from 'react-dom';

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

// Clean domain categories without emojis
const DOMAIN_CATEGORIES = [
  { id: 'all', label: 'All' },
  { id: 'spiritual', label: 'Spiritual' },
  { id: 'wisdom', label: 'Wisdom' },
  { id: 'action', label: 'Practice' },
  { id: 'healing', label: 'Wellness' },
];

const RISHI_DOMAIN_MAP: Record<string, string[]> = {
  atri: ['spiritual'],
  bhrigu: ['wisdom', 'spiritual'],
  vasishta: ['wisdom', 'action'],
  angiras: ['spiritual', 'action'],
  pulastya: ['wisdom', 'spiritual'],
  pulaha: ['healing', 'spiritual'],
  kratu: ['action', 'spiritual'],
  marici: ['spiritual', 'wisdom'],
  daksha: ['action', 'wisdom'],
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
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const selectedRishiData = availableRishis.find(r => r.id === selectedRishi);
  const isStandardMode = !selectedRishi || selectedRishi === '';

  // Filter Rishis
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
  };

  // Compact trigger button
  const TriggerButton = () => (
    <button
      onClick={() => setIsExpanded(true)}
      className="flex items-center gap-3 px-4 py-2.5 bg-white border border-gray-200 rounded-lg hover:border-gray-300 hover:bg-gray-50 transition-all duration-150 shadow-sm"
    >
      {/* Avatar */}
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
        {isStandardMode ? (
          <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        ) : (
          <span className="text-sm font-medium text-gray-600">
            {selectedRishiData?.name.charAt(0) || 'G'}
          </span>
        )}
      </div>

      {/* Label */}
      <div className="text-left">
        <div className="text-sm font-medium text-gray-900">
          {isStandardMode ? 'DharmaMind' : selectedRishiData?.name}
        </div>
        <div className="text-xs text-gray-500">
          {isStandardMode ? 'Spiritual wisdom assistant' : selectedRishiData?.specialization[0]}
        </div>
      </div>

      {/* Chevron */}
      <svg className="w-4 h-4 text-gray-400 ml-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </button>
  );

  // Modal content
  const ModalContent = () => (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      onClick={() => setIsExpanded(false)}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" />

      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 10 }}
        transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
        onClick={e => e.stopPropagation()}
        className="relative w-full max-w-lg bg-white rounded-xl shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-100">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Select Guide</h2>
              <p className="text-sm text-gray-500 mt-0.5">Choose a specialized AI assistant</p>
            </div>
            <button
              onClick={() => setIsExpanded(false)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Search */}
          <div className="mt-4 relative">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Search guides..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 text-sm bg-gray-50 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gold-500/20 focus:border-gold-500 transition-all"
            />
          </div>

          {/* Category tabs */}
          <div className="flex gap-1 mt-4 overflow-x-auto pb-1">
            {DOMAIN_CATEGORIES.map(cat => (
              <button
                key={cat.id}
                onClick={() => setActiveCategory(cat.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md whitespace-nowrap transition-all ${activeCategory === cat.id
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
                  }`}
              >
                {cat.label}
              </button>
            ))}
          </div>
        </div>

        {/* Guide List */}
        <div className="max-h-80 overflow-y-auto p-2">
          {/* Standard AI Option */}
          <button
            onClick={() => handleSelectRishi('')}
            className={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all mb-1 ${isStandardMode
              ? 'bg-gold-50 border border-gold-200'
              : 'hover:bg-gray-50 border border-transparent'
              }`}
          >
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${isStandardMode ? 'bg-gold-100' : 'bg-gray-100'
              }`}>
              <svg className={`w-5 h-5 ${isStandardMode ? 'text-gold-600' : 'text-gray-500'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <div className="font-medium text-gray-900 text-sm">DharmaMind</div>
              <div className="text-xs text-gray-500 truncate">Spiritual wisdom assistant</div>
            </div>
            {isStandardMode && (
              <svg className="w-5 h-5 text-gold-600 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            )}
          </button>

          {/* Divider */}
          <div className="h-px bg-gray-100 my-2" />

          {/* Rishi Label */}
          <div className="px-2 py-1.5 mb-1">
            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Rishis (Sages)</span>
          </div>

          {/* Specialized Rishis */}
          {filteredRishis.map(rishi => {
            const isSelected = selectedRishi === rishi.id;
            const isLocked = !canAccessRishi(rishi);

            return (
              <button
                key={rishi.id}
                onClick={() => !isLocked && handleSelectRishi(rishi.id)}
                disabled={isLocked}
                className={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all mb-1 ${isSelected
                  ? 'bg-gold-50 border border-gold-200'
                  : isLocked
                    ? 'opacity-50 cursor-not-allowed border border-transparent'
                    : 'hover:bg-gray-50 border border-transparent'
                  }`}
              >
                {/* Avatar */}
                <div className={`w-10 h-10 rounded-full flex items-center justify-center font-medium text-sm ${isSelected
                  ? 'bg-gold-600 text-white'
                  : 'bg-gray-100 text-gray-600'
                  }`}>
                  {rishi.name.charAt(0)}
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-900 text-sm">{rishi.name}</span>
                    {isLocked && (
                      <span className="px-1.5 py-0.5 text-xs bg-gray-100 text-gray-500 rounded">Pro</span>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 truncate">
                    {rishi.specialization.slice(0, 2).join(' · ')}
                  </div>
                </div>

                {/* Check */}
                {isSelected && (
                  <svg className="w-5 h-5 text-gold-600 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                )}
              </button>
            );
          })}

          {filteredRishis.length === 0 && (
            <div className="py-8 text-center text-gray-500 text-sm">
              No guides found
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 bg-gray-50 border-t border-gray-100">
          <p className="text-xs text-gray-500 text-center">
            Each guide specializes in different aspects of wisdom and guidance
          </p>
        </div>
      </motion.div>
    </motion.div>
  );

  return (
    <>
      <TriggerButton />

      {mounted && createPortal(
        <AnimatePresence>
          {isExpanded && <ModalContent />}
        </AnimatePresence>,
        document.body
      )}
    </>
  );
};

export default RishiSelector;
