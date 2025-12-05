import React, { useState } from 'react';
import { useColor } from '../contexts/ColorContext';
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

export const RishiSelector: React.FC<RishiSelectorProps> = ({
  onRishiSelect,
  selectedRishi,
  userSubscription,
  isDemo = false,
  availableRishis
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [hoveredRishi, setHoveredRishi] = useState<string | null>(null);
  const { currentTheme } = useColor();
  const colors = currentTheme.colors;

  const selectedRishiData = availableRishis.find(r => r.id === selectedRishi);
  const isStandardMode = !selectedRishi || selectedRishi === '';

  // Filter Rishis based on search
  const filteredRishis = availableRishis.filter(rishi => 
    rishi.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    rishi.specialization.some(s => s.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const canAccessRishi = (rishi: Rishi) => {
    if (isDemo) return true;
    if (!rishi.requires_upgrade) return true;
    return userSubscription !== 'basic';
  };

  // Get icon SVG for each Rishi based on their personality
  const getRishiIcon = (rishiId: string) => {
    const iconMap: Record<string, JSX.Element> = {
      marici: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
      ),
      atri: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      angiras: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.879 16.121A3 3 0 1012.015 11L11 14H9c0 .768.293 1.536.879 2.121z" />
        </svg>
      ),
      pulastya: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      pulaha: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
        </svg>
      ),
      kratu: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
      daksha: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
      ),
      bhrigu: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
        </svg>
      ),
      vasishta: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      ),
    };
    return iconMap[rishiId] || null;
  };

  return (
    <React.Fragment>
      <div className="rishi-selector">
        {/* Horizontal Mode Switcher */}
        <div className="flex items-center gap-2 mb-3 px-1">
          {/* Universal Guide */}
          <button
            onClick={() => !isStandardMode && onRishiSelect('')}
            className={`flex-1 py-2 px-3 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2 ${
              isStandardMode ? 'bg-white shadow-sm' : 'hover:bg-gray-50'
            }`}
            style={{
              border: isStandardMode ? `1px solid ${colors.borderSecondary}` : '1px solid transparent'
            }}
          >
            <div className={`w-1.5 h-1.5 rounded-full ${isStandardMode ? 'bg-gray-400' : 'bg-gray-300'}`}></div>
            <span className={`text-xs font-medium ${isStandardMode ? 'text-gray-900' : 'text-gray-500'}`}>
              Universal
            </span>
          </button>

          {/* Rishi Mode */}
          <button
            onClick={() => setIsExpanded(true)}
            className={`flex-1 py-2 px-3 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2 ${
              !isStandardMode ? 'bg-white shadow-sm' : 'hover:bg-gray-50'
            }`}
            style={{
              border: !isStandardMode ? `1px solid ${colors.borderPrimary}` : '1px solid transparent'
            }}
          >
            <div className={`w-1.5 h-1.5 rounded-full ${!isStandardMode ? 'bg-emerald-500' : 'bg-gray-300'}`}></div>
            <span className={`text-xs font-medium ${!isStandardMode ? 'text-emerald-600' : 'text-gray-500'}`}>
              {!isStandardMode && selectedRishiData ? selectedRishiData.name : 'Rishi'}
            </span>
          </button>
        </div>
      </div>

      {/* Modal for Rishi Selection - Portaled to body for proper centering */}
      {isExpanded && typeof window !== 'undefined' && createPortal(
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4"
          style={{ 
            zIndex: 9999,
            animation: 'fadeIn 0.15s ease-out'
          }}
          onClick={() => setIsExpanded(false)}
        >
          <div 
            className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
            style={{
              animation: 'modalSlideIn 0.2s cubic-bezier(0.16, 1, 0.3, 1)'
            }}
          >
            {/* Modal Header */}
            <div className="px-6 py-4 border-b flex items-center justify-between" style={{ borderColor: colors.borderSecondary }}>
              <div>
                <h3 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
                  Select Your Guide
                </h3>
                <p className="text-xs mt-0.5" style={{ color: colors.textSecondary }}>
                  {availableRishis.length} spiritual guides available
                </p>
              </div>
              <button
                onClick={() => setIsExpanded(false)}
                className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto max-h-[calc(80vh-120px)]">
              {/* Search Bar */}
              <div className="relative mb-4">
                <input
                  type="text"
                  placeholder="Search by name or expertise..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full px-4 py-2.5 pl-10 rounded-lg text-sm transition-all duration-200 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                  style={{
                    backgroundColor: colors.backgroundSecondary,
                    border: `1px solid ${colors.borderSecondary}`,
                    color: colors.textPrimary
                  }}
                />
                <svg className="absolute left-3 top-3 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>

              {/* Results count */}
              {searchQuery && (
                <div className="text-xs px-2 mb-3" style={{ color: colors.textSecondary }}>
                  Found {filteredRishis.length} of {availableRishis.length} Rishis
                </div>
              )}

              {/* Rishi Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {filteredRishis.map((rishi) => (
                  <button
                    key={rishi.id}
                    className={`relative rounded-lg p-3 text-left transition-all duration-150 group ${
                      selectedRishi === rishi.id 
                        ? 'ring-2 ring-emerald-500 bg-emerald-50' 
                        : canAccessRishi(rishi) 
                          ? 'hover:bg-gray-50 border border-gray-200' 
                          : 'opacity-60 cursor-not-allowed border border-gray-200'
                    }`}
                    onClick={() => {
                      if (canAccessRishi(rishi)) {
                        onRishiSelect(rishi.id);
                        setIsExpanded(false);
                        setSearchQuery('');
                      }
                    }}
                    onMouseEnter={() => setHoveredRishi(rishi.id)}
                    onMouseLeave={() => setHoveredRishi(null)}
                    disabled={!canAccessRishi(rishi)}
                  >
                    <div className="flex items-start space-x-3">
                      {/* Rishi Icon */}
                      <div 
                        className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${
                          selectedRishi === rishi.id 
                            ? 'bg-emerald-100 text-emerald-600' 
                            : 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        {getRishiIcon(rishi.id)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-semibold text-sm truncate" style={{ color: colors.textPrimary }}>
                            {rishi.name}
                          </span>
                          {selectedRishi === rishi.id && (
                            <svg className="w-4 h-4 text-emerald-500 flex-shrink-0 ml-2" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                            </svg>
                          )}
                          {rishi.requires_upgrade && userSubscription === 'basic' && !isDemo && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 font-semibold flex-shrink-0 ml-2">
                              PRO
                            </span>
                          )}
                        </div>
                        <div className="text-xs mb-1.5" style={{ color: colors.textSecondary }}>
                          {rishi.sanskrit}
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {rishi.specialization.slice(0, 2).map((spec, idx) => (
                            <span key={idx} className="text-xs px-1.5 py-0.5 rounded" style={{
                              backgroundColor: colors.background,
                              color: colors.textSecondary
                            }}>
                              {spec}
                            </span>
                          ))}
                        </div>
                        {!canAccessRishi(rishi) && (
                          <div className="text-xs mt-2 flex items-center space-x-1 text-amber-600">
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd"/>
                            </svg>
                            <span>Upgrade required</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </button>
                ))}
              </div>

              {/* Empty State */}
              {filteredRishis.length === 0 && (
                <div className="text-center py-8">
                  <svg className="w-12 h-12 mx-auto mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-sm font-medium" style={{ color: colors.textSecondary }}>
                    No Rishis found matching "{searchQuery}"
                  </p>
                  <button
                    onClick={() => setSearchQuery('')}
                    className="mt-2 text-xs font-medium"
                    style={{ color: colors.borderPrimary }}
                  >
                    Clear search
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>,
        document.body
      )}
    </React.Fragment>
  );
};
