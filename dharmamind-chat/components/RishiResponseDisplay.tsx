/**
 * 🕉️ Sacred Rishi Response Display
 * ==================================
 * 
 * Beautiful, immersive display for Rishi wisdom responses
 * Features personality-driven theming, sacred animations, and rich content
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RISHI_SACRED_COLORS } from './RishiAvatar';
import { RISHI_EXTENDED_DATA } from './RishiCard';

interface ScripturalReference {
  text: string;
  sanskrit: string;
  translation: string;
  relevance: string;
}

interface Mantra {
  sanskrit: string;
  transliteration: string;
  meaning: string;
  usage: string;
}

interface RishiResponseData {
  mode: string;
  rishi_info: {
    name: string;
    sanskrit: string;
    specialization: string[];
    teaching_style?: string;
    archetype?: string;
    sacred_focus?: string;
  };
  greeting: string;
  guidance: {
    primary_wisdom: string;
    scriptural_references?: ScripturalReference[];
    mantras?: Mantra[];
    meditation_practice?: string;
    signature_mantra?: string;
    sanskrit_teaching?: string;
    teaching_style?: string;
  };
  practical_steps: string[];
  growth_opportunities: string[];
  personality_traits?: string[];
  wisdom_synthesis?: string;
  enhanced?: boolean;
  authentic?: boolean;
  session_continuity?: {
    conversation_count?: number;
  };
}

interface RishiResponseDisplayProps {
  response: RishiResponseData;
  onFollowUp?: (question: string) => void;
}

// Get Rishi ID from name
const getRishiId = (name: string): string => {
  const nameMap: Record<string, string> = {
    'Atri': 'atri',
    'Bhrigu': 'bhrigu',
    'Vashishta': 'vashishta',
    'Vishwamitra': 'vishwamitra',
    'Gautama': 'gautama',
    'Jamadagni': 'jamadagni',
    'Kashyapa': 'kashyapa',
    'Angiras': 'angiras',
    'Pulastya': 'pulastya',
    // Legacy names
    'Patanjali': 'atri',
    'Vyasa': 'vashishta',
    'Valmiki': 'gautama',
    'Shankara': 'vishwamitra',
    'Narada': 'pulastya',
  };
  
  for (const [key, value] of Object.entries(nameMap)) {
    if (name.includes(key)) return value;
  }
  return 'atri';
};

export const RishiResponseDisplay: React.FC<RishiResponseDisplayProps> = ({
  response,
  onFollowUp
}) => {
  const [activeTab, setActiveTab] = useState<'guidance' | 'personality' | 'practices' | 'mantras'>('guidance');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['primary']));

  const rishiId = getRishiId(response.rishi_info.name);
  const colors = RISHI_SACRED_COLORS[rishiId] || RISHI_SACRED_COLORS[''];
  const extendedData = RISHI_EXTENDED_DATA[rishiId];

  const tabs = [
    { id: 'guidance', label: 'Guidance', icon: '🌟' },
    { id: 'personality', label: 'Personality', icon: '🎭' },
    { id: 'practices', label: 'Practices', icon: '🧘' },
    { id: 'mantras', label: 'Mantras', icon: '🕉️' },
  ] as const;

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  return (
    <motion.div 
      className="overflow-hidden rounded-2xl shadow-xl border"
      style={{
        borderColor: `${colors.primary}30`,
        background: 'white'
      }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Sacred Header */}
      <div 
        className="relative p-6 overflow-hidden"
        style={{
          background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`
        }}
      >
        {/* Subtle Pattern Overlay */}
        <div 
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: `radial-gradient(circle at 20% 20%, white 1px, transparent 1px),
                             radial-gradient(circle at 80% 80%, white 1px, transparent 1px)`,
            backgroundSize: '30px 30px'
          }}
        />
        
        {/* Content */}
        <div className="relative flex items-start gap-4">
          {/* Animated Avatar */}
          <motion.div
            className="flex-shrink-0 w-16 h-16 rounded-2xl flex items-center justify-center text-4xl bg-white/20 backdrop-blur-sm"
            animate={{ 
              boxShadow: [
                `0 0 0 0 rgba(255,255,255,0.4)`,
                `0 0 0 15px rgba(255,255,255,0)`,
              ]
            }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            {colors.icon}
          </motion.div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 mb-2">
              <h3 className="text-xl font-bold text-white">
                {response.rishi_info.name}
              </h3>
              {response.enhanced && (
                <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-white/20 text-white">
                  ✨ Enhanced
                </span>
              )}
              {response.authentic && (
                <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-white/20 text-white">
                  🕉️ Authentic
                </span>
              )}
            </div>
            
            <p className="text-white/80 text-sm mb-1">
              {response.rishi_info.sanskrit}
              {extendedData && ` • ${extendedData.title}`}
            </p>
            
            {extendedData && (
              <div className="flex items-center gap-3 mt-2 text-xs text-white/70">
                <span>🔥 {extendedData.element}</span>
                <span>•</span>
                <span>🌀 {extendedData.chakra}</span>
              </div>
            )}
          </div>
        </div>

        {/* Specializations */}
        <div className="mt-4 flex flex-wrap gap-2">
          {response.rishi_info.specialization.slice(0, 4).map((spec, idx) => (
            <motion.span
              key={idx}
              className="px-3 py-1 rounded-full text-xs font-medium bg-white/15 text-white backdrop-blur-sm"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: idx * 0.1 }}
            >
              {spec}
            </motion.span>
          ))}
        </div>
      </div>

      {/* Greeting Quote */}
      <div 
        className="p-5 border-b"
        style={{
          background: `linear-gradient(90deg, ${colors.primary}08 0%, transparent 100%)`,
          borderColor: `${colors.primary}15`
        }}
      >
        <motion.p 
          className="text-gray-700 italic text-center leading-relaxed"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          "{response.greeting}"
        </motion.p>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b" style={{ borderColor: `${colors.primary}15` }}>
        {tabs.map((tab) => (
          <motion.button
            key={tab.id}
            className={`flex-1 py-3 px-4 text-sm font-medium transition-all duration-300 flex items-center justify-center gap-2 ${
              activeTab === tab.id
                ? 'border-b-2'
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}
            style={{
              borderColor: activeTab === tab.id ? colors.primary : 'transparent',
              color: activeTab === tab.id ? colors.primary : undefined,
              backgroundColor: activeTab === tab.id ? `${colors.primary}08` : undefined
            }}
            onClick={() => setActiveTab(tab.id)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <span>{tab.icon}</span>
            <span className="hidden sm:inline">{tab.label}</span>
          </motion.button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          {/* Guidance Tab */}
          {activeTab === 'guidance' && (
            <motion.div
              key="guidance"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="space-y-6"
            >
              {/* Primary Wisdom */}
              <div>
                <h4 
                  className="font-bold text-lg mb-4 flex items-center gap-2"
                  style={{ color: colors.primary }}
                >
                  <span>🌟</span>
                  Spiritual Guidance
                </h4>
                <div 
                  className="prose prose-sm max-w-none text-gray-700 p-4 rounded-xl"
                  style={{
                    background: `linear-gradient(135deg, ${colors.primary}05 0%, ${colors.secondary}05 100%)`,
                    borderLeft: `4px solid ${colors.primary}`
                  }}
                >
                  {response.guidance.primary_wisdom ? (
                    <div dangerouslySetInnerHTML={{ 
                      __html: response.guidance.primary_wisdom.replace(/\n/g, '<br>') 
                    }} />
                  ) : (
                    <p className="text-gray-500 italic">Wisdom is being channeled...</p>
                  )}
                </div>
              </div>

              {/* Practical Steps */}
              {response.practical_steps && response.practical_steps.length > 0 && (
                <div>
                  <h4 
                    className="font-bold text-lg mb-4 flex items-center gap-2"
                    style={{ color: colors.primary }}
                  >
                    <span>🎯</span>
                    Practical Steps
                  </h4>
                  <div className="space-y-3">
                    {response.practical_steps.map((step, index) => (
                      <motion.div
                        key={index}
                        className="flex items-start gap-4 p-4 rounded-xl"
                        style={{
                          background: `${colors.primary}08`
                        }}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <span 
                          className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm"
                          style={{ backgroundColor: colors.primary }}
                        >
                          {index + 1}
                        </span>
                        <p className="text-gray-700 text-sm leading-relaxed pt-1">{step}</p>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* Personality Tab */}
          {activeTab === 'personality' && (
            <motion.div
              key="personality"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="space-y-6"
            >
              {/* Sanskrit Teaching */}
              {response.guidance.sanskrit_teaching && (
                <div 
                  className="p-5 rounded-xl text-center"
                  style={{
                    background: `linear-gradient(135deg, ${colors.primary}10 0%, ${colors.secondary}10 100%)`,
                    border: `2px solid ${colors.primary}30`
                  }}
                >
                  <h4 
                    className="font-bold mb-3 flex items-center justify-center gap-2"
                    style={{ color: colors.primary }}
                  >
                    <span>📜</span>
                    Sanskrit Wisdom
                  </h4>
                  <motion.p 
                    className="text-2xl font-semibold"
                    style={{ color: colors.primary }}
                    animate={{
                      textShadow: [
                        `0 0 5px ${colors.glow}`,
                        `0 0 15px ${colors.glow}`,
                        `0 0 5px ${colors.glow}`
                      ]
                    }}
                    transition={{ duration: 3, repeat: Infinity }}
                  >
                    {response.guidance.sanskrit_teaching}
                  </motion.p>
                </div>
              )}

              {/* Personality Traits */}
              {response.personality_traits && response.personality_traits.length > 0 && (
                <div>
                  <h4 
                    className="font-bold text-lg mb-4 flex items-center gap-2"
                    style={{ color: colors.primary }}
                  >
                    <span>🎭</span>
                    Authentic Traits
                  </h4>
                  <div className="grid gap-3">
                    {response.personality_traits.map((trait, index) => (
                      <motion.div
                        key={index}
                        className="p-3 rounded-xl"
                        style={{
                          background: `${colors.primary}08`,
                          borderLeft: `3px solid ${colors.primary}`
                        }}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <p className="text-gray-700 text-sm">{trait}</p>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {/* Wisdom Synthesis */}
              {response.wisdom_synthesis && (
                <div 
                  className="p-5 rounded-xl"
                  style={{
                    background: `linear-gradient(135deg, ${colors.primary}05 0%, ${colors.secondary}05 100%)`
                  }}
                >
                  <h4 
                    className="font-bold mb-3 flex items-center gap-2"
                    style={{ color: colors.primary }}
                  >
                    <span>💫</span>
                    Wisdom Synthesis
                  </h4>
                  <p className="text-gray-700 italic leading-relaxed">
                    "{response.wisdom_synthesis}"
                  </p>
                </div>
              )}
            </motion.div>
          )}

          {/* Practices Tab */}
          {activeTab === 'practices' && (
            <motion.div
              key="practices"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="space-y-6"
            >
              {/* Meditation Practice */}
              {response.guidance.meditation_practice && (
                <div 
                  className="p-5 rounded-xl"
                  style={{
                    background: `linear-gradient(135deg, ${colors.primary}08 0%, ${colors.secondary}08 100%)`,
                    border: `1px solid ${colors.primary}20`
                  }}
                >
                  <h4 
                    className="font-bold mb-4 flex items-center gap-2"
                    style={{ color: colors.primary }}
                  >
                    <span>🧘</span>
                    Recommended Practice
                  </h4>
                  <div 
                    className="prose prose-sm max-w-none text-gray-700"
                    dangerouslySetInnerHTML={{ 
                      __html: response.guidance.meditation_practice.replace(/\n/g, '<br>') 
                    }}
                  />
                </div>
              )}

              {/* Growth Opportunities */}
              {response.growth_opportunities && response.growth_opportunities.length > 0 && (
                <div>
                  <h4 
                    className="font-bold text-lg mb-4 flex items-center gap-2"
                    style={{ color: colors.primary }}
                  >
                    <span>🌱</span>
                    Growth Opportunities
                  </h4>
                  <div className="grid gap-3">
                    {response.growth_opportunities.map((opportunity, index) => (
                      <motion.button
                        key={index}
                        className="p-4 rounded-xl text-left transition-all duration-300 hover:shadow-lg"
                        style={{
                          background: `${colors.primary}08`,
                          border: `1px solid ${colors.primary}20`
                        }}
                        onClick={() => onFollowUp && onFollowUp(opportunity)}
                        whileHover={{ scale: 1.02, x: 5 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <div className="flex items-center justify-between">
                          <p className="text-gray-700 text-sm">{opportunity}</p>
                          <svg 
                            className="w-5 h-5 flex-shrink-0 ml-3"
                            style={{ color: colors.primary }}
                            fill="none" 
                            stroke="currentColor" 
                            viewBox="0 0 24 24"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </div>
                      </motion.button>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* Mantras Tab */}
          {activeTab === 'mantras' && (
            <motion.div
              key="mantras"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="space-y-6"
            >
              {/* Signature Mantra */}
              {(response.guidance.signature_mantra || extendedData?.sacredMantra) && (
                <div 
                  className="p-6 rounded-2xl text-center"
                  style={{
                    background: `linear-gradient(135deg, ${colors.primary}10 0%, ${colors.secondary}10 100%)`,
                    border: `2px solid ${colors.primary}40`
                  }}
                >
                  <h4 
                    className="font-bold mb-4 flex items-center justify-center gap-2"
                    style={{ color: colors.primary }}
                  >
                    <span>🕉️</span>
                    Sacred Mantra
                  </h4>
                  <motion.p 
                    className="text-3xl font-bold"
                    style={{ color: colors.primary }}
                    animate={{
                      textShadow: [
                        `0 0 10px ${colors.glow}`,
                        `0 0 30px ${colors.glow}`,
                        `0 0 10px ${colors.glow}`
                      ]
                    }}
                    transition={{ duration: 3, repeat: Infinity }}
                  >
                    {response.guidance.signature_mantra || extendedData?.sacredMantra}
                  </motion.p>
                  <p className="mt-4 text-gray-500 text-sm italic">
                    The sacred vibration associated with {response.rishi_info.name}
                  </p>
                </div>
              )}

              {/* Additional Mantras */}
              {response.guidance.mantras && response.guidance.mantras.length > 0 ? (
                <div className="space-y-4">
                  {response.guidance.mantras.map((mantra, index) => (
                    <motion.div
                      key={index}
                      className="p-5 rounded-xl"
                      style={{
                        background: `${colors.primary}05`,
                        border: `1px solid ${colors.primary}20`
                      }}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <h5 
                        className="font-bold mb-3"
                        style={{ color: colors.primary }}
                      >
                        Sacred Mantra {index + 1}
                      </h5>
                      <div className="space-y-3">
                        <div 
                          className="p-4 rounded-lg text-center"
                          style={{ backgroundColor: `${colors.primary}10` }}
                        >
                          <p className="text-xl font-semibold" style={{ color: colors.primary }}>
                            {mantra.sanskrit}
                          </p>
                        </div>
                        <p className="text-center text-gray-600 italic">{mantra.transliteration}</p>
                        <div className="text-sm space-y-1">
                          <p><strong className="text-gray-700">Meaning:</strong> <span className="text-gray-600">{mantra.meaning}</span></p>
                          <p><strong className="text-gray-700">Usage:</strong> <span className="text-gray-600">{mantra.usage}</span></p>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : !response.guidance.signature_mantra && !extendedData?.sacredMantra ? (
                <div className="text-center py-12">
                  <motion.span 
                    className="text-6xl block mb-4"
                    animate={{ scale: [1, 1.1, 1], rotate: [0, 5, -5, 0] }}
                    transition={{ duration: 3, repeat: Infinity }}
                  >
                    🕉️
                  </motion.span>
                  <p className="text-gray-500">
                    Sacred mantras will appear here when provided by the Rishi
                  </p>
                </div>
              ) : null}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Session Info Footer */}
      {response.enhanced && response.session_continuity && (
        <div 
          className="px-6 py-4 border-t"
          style={{
            background: `${colors.primary}05`,
            borderColor: `${colors.primary}15`
          }}
        >
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">
              Session: {response.session_continuity.conversation_count || 1}
            </span>
            <span 
              className="font-medium flex items-center gap-1"
              style={{ color: colors.primary }}
            >
              <span>✨</span>
              Enhanced Guidance Active
            </span>
            <span className="text-gray-500">
              Personalized for your journey
            </span>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default RishiResponseDisplay;
