/**
 * DharmaMind - Guide Selection Panel
 * Professional, clean sidebar panel for guide selection
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RishiAvatar, RISHI_COLORS } from './RishiAvatar';
import { RISHI_EXTENDED_DATA } from './RishiCard';

interface Rishi {
  id: string;
  name: string;
  specialization: string[];
  available: boolean;
  requires_upgrade?: boolean;
}

interface RishiGuidancePanelProps {
  selectedRishi: string;
  onRishiSelect: (rishiId: string) => void;
  userSubscription: string;
  availableRishis: Rishi[];
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export const RishiGuidancePanel: React.FC<RishiGuidancePanelProps> = ({
  selectedRishi,
  onRishiSelect,
  userSubscription,
  availableRishis,
  isCollapsed = false,
  onToggleCollapse,
}) => {
  const [isExpanded, setIsExpanded] = useState(!isCollapsed);
  
  const canAccessRishi = (rishi: Rishi) => {
    if (!rishi.requires_upgrade) return true;
    return userSubscription !== 'basic';
  };

  const selectedRishiData = availableRishis.find(r => r.id === selectedRishi);
  const extendedData = selectedRishi ? RISHI_EXTENDED_DATA[selectedRishi] : null;

  return (
    <div className="bg-white border-l border-gray-200 h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-900">AI Guide</h3>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 hover:bg-gray-100 rounded transition-colors"
          >
            <svg 
              className={`w-4 h-4 text-gray-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
        
        {/* Current Selection */}
        {selectedRishiData && (
          <div className="mt-3 flex items-center gap-3">
            <RishiAvatar rishiId={selectedRishi} rishiName={selectedRishiData.name} size="sm" />
            <div>
              <div className="text-sm font-medium text-gray-900">{selectedRishiData.name}</div>
              <div className="text-xs text-gray-500">{extendedData?.domain || 'Specialized guide'}</div>
            </div>
          </div>
        )}
        
        {!selectedRishi && (
          <div className="mt-3 flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
              <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <div>
              <div className="text-sm font-medium text-gray-900">General Assistant</div>
              <div className="text-xs text-gray-500">Standard AI</div>
            </div>
          </div>
        )}
      </div>
      
      {/* Guide List */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex-1 overflow-y-auto"
          >
            <div className="p-3 space-y-1">
              {/* Standard AI Option */}
              <button
                onClick={() => onRishiSelect('')}
                className={`w-full flex items-center gap-3 p-2.5 rounded-lg text-left transition-all ${
                  !selectedRishi 
                    ? 'bg-gold-50 border border-gold-200' 
                    : 'hover:bg-gray-50 border border-transparent'
                }`}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  !selectedRishi ? 'bg-gold-100' : 'bg-gray-100'
                }`}>
                  <svg className={`w-4 h-4 ${!selectedRishi ? 'text-gold-600' : 'text-gray-500'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-900">General</div>
                  <div className="text-xs text-gray-500">Standard assistant</div>
                </div>
              </button>
              
              {/* Separator */}
              <div className="h-px bg-gray-100 my-2" />
              
              {/* Specialized Guides */}
              {availableRishis.map(rishi => {
                const isSelected = selectedRishi === rishi.id;
                const isLocked = !canAccessRishi(rishi);
                const data = RISHI_EXTENDED_DATA[rishi.id];
                
                return (
                  <button
                    key={rishi.id}
                    onClick={() => !isLocked && onRishiSelect(rishi.id)}
                    disabled={isLocked}
                    className={`w-full flex items-center gap-3 p-2.5 rounded-lg text-left transition-all ${
                      isSelected 
                        ? 'bg-gold-50 border border-gold-200' 
                        : isLocked
                          ? 'opacity-50 cursor-not-allowed border border-transparent'
                          : 'hover:bg-gray-50 border border-transparent'
                    }`}
                  >
                    <RishiAvatar rishiId={rishi.id} rishiName={rishi.name} size="sm" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm font-medium text-gray-900">{rishi.name}</span>
                        {isLocked && (
                          <svg className="w-3 h-3 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                          </svg>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 truncate">
                        {data?.domain || rishi.specialization[0]}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default RishiGuidancePanel;
