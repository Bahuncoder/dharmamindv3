import React, { useState } from 'react';

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
  session_continuity?: any;
}

interface RishiResponseDisplayProps {
  response: RishiResponseData;
  onFollowUp?: (question: string) => void;
}

export const RishiResponseDisplay: React.FC<RishiResponseDisplayProps> = ({
  response,
  onFollowUp
}) => {
  const [activeTab, setActiveTab] = useState<'guidance' | 'personality' | 'practices' | 'mantras'>('guidance');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  const getRishiIcon = (name: string) => {
    // Saptarishi icons
    if (name.includes('Atri')) return '🧘';
    if (name.includes('Bhrigu')) return '⭐';
    if (name.includes('Vashishta')) return '📚';
    if (name.includes('Vishwamitra')) return '🕉️';
    if (name.includes('Gautama')) return '🙏';
    if (name.includes('Jamadagni')) return '⚡';
    if (name.includes('Kashyapa')) return '🌍';
    
    // Legacy icons for backward compatibility
    if (name.includes('Patanjali')) return '🧘';
    if (name.includes('Vyasa')) return '📚';
    if (name.includes('Valmiki')) return '💖';
    if (name.includes('Shankara')) return '✨';
    if (name.includes('Narada')) return '🎵';
    return '🕉️';
  };

  const getRishiGradient = (name: string) => {
    // Saptarishi gradients
    if (name.includes('Atri')) return 'from-blue-600 to-indigo-700';
    if (name.includes('Bhrigu')) return 'from-yellow-500 to-orange-600';
    if (name.includes('Vashishta')) return 'from-green-600 to-emerald-700';
    if (name.includes('Vishwamitra')) return 'from-purple-600 to-violet-700';
    if (name.includes('Gautama')) return 'from-teal-600 to-cyan-700';
    if (name.includes('Jamadagni')) return 'from-red-600 to-pink-700';
    if (name.includes('Kashyapa')) return 'from-indigo-600 to-blue-700';
    
    // Legacy gradients for backward compatibility
    if (name.includes('Patanjali')) return 'from-blue-500 to-indigo-600';
    if (name.includes('Vyasa')) return 'from-green-500 to-emerald-600';
    if (name.includes('Valmiki')) return 'from-pink-500 to-rose-600';
    if (name.includes('Shankara')) return 'from-purple-500 to-violet-600';
    if (name.includes('Narada')) return 'from-yellow-500 to-orange-600';
    return 'from-orange-500 to-yellow-600';
  };

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
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      {/* Rishi Header */}
      <div className={`bg-gradient-to-r ${getRishiGradient(response.rishi_info.name)} text-white p-6`}>
        <div className="flex items-center space-x-4">
          <span className="text-4xl">{getRishiIcon(response.rishi_info.name)}</span>
          <div className="flex-1">
            <h3 className="text-xl font-bold">{response.rishi_info.name}</h3>
            <p className="text-white/90 text-sm">{response.rishi_info.sanskrit}</p>
            {response.rishi_info.archetype && (
              <p className="text-white/80 text-xs mt-1 italic">{response.rishi_info.archetype}</p>
            )}
            <div className="flex items-center space-x-3 mt-2">
              {response.enhanced && (
                <span className="inline-block bg-white/20 text-white text-xs px-2 py-1 rounded-full">
                  ✨ Enhanced Guidance
                </span>
              )}
              {response.authentic && (
                <span className="inline-block bg-white/20 text-white text-xs px-2 py-1 rounded-full">
                  🕉️ Authentic Personality
                </span>
              )}
            </div>
          </div>
        </div>
        
        {/* Sacred Focus */}
        {response.rishi_info.sacred_focus && (
          <div className="mt-3 text-white/80 text-sm">
            <span className="font-semibold">Sacred Focus:</span> {response.rishi_info.sacred_focus}
          </div>
        )}
      </div>

      {/* Greeting */}
      <div className="p-6 bg-gradient-to-r from-gray-50 to-white border-b border-gray-100">
        <p className="text-gray-700 italic text-center leading-relaxed">
          "{response.greeting}"
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex border-b border-gray-200">
        {['guidance', 'personality', 'practices', 'mantras'].map(tab => (
          <button
            key={tab}
            className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'border-b-2 border-orange-500 text-orange-600 bg-orange-50'
                : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
            }`}
            onClick={() => setActiveTab(tab as any)}
          >
            {tab === 'guidance' && '🌟 Guidance'}
            {tab === 'personality' && '🎭 Personality'}
            {tab === 'practices' && '🧘 Practices'}
            {tab === 'mantras' && '🕉️ Mantras'}
          </button>
        ))}
      </div>

      {/* Content Tabs */}
      <div className="p-6">
        {activeTab === 'guidance' && (
          <div className="space-y-6">
            {/* Primary Guidance */}
            <div>
              <h4 className="font-bold text-gray-800 mb-3 flex items-center">
                <span className="mr-2">🌟</span>
                Spiritual Guidance
              </h4>
              <div className="prose prose-sm max-w-none text-gray-700">
                {response.guidance.primary_wisdom ? (
                  <div dangerouslySetInnerHTML={{ 
                    __html: response.guidance.primary_wisdom.replace(/\n/g, '<br>') 
                  }} />
                ) : (
                  <p>Primary wisdom guidance would appear here.</p>
                )}
              </div>
            </div>

            {/* Practical Steps */}
            {response.practical_steps && response.practical_steps.length > 0 && (
              <div>
                <h4 className="font-bold text-gray-800 mb-3 flex items-center">
                  <span className="mr-2">🎯</span>
                  Practical Steps
                </h4>
                <div className="space-y-2">
                  {response.practical_steps.map((step, index) => (
                    <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                      <span className="bg-blue-500 text-white text-xs font-bold w-6 h-6 rounded-full flex items-center justify-center mt-0.5">
                        {index + 1}
                      </span>
                      <p className="text-gray-700 text-sm">{step}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'personality' && (
          <div className="space-y-6">
            {/* Sanskrit Teaching */}
            {response.guidance.sanskrit_teaching && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h4 className="font-bold text-yellow-800 mb-3 flex items-center">
                  <span className="mr-2">📜</span>
                  Sanskrit Wisdom
                </h4>
                <div className="bg-white p-3 rounded border">
                  <p className="font-mono text-gray-800 text-center text-lg mb-2">
                    {response.guidance.sanskrit_teaching}
                  </p>
                </div>
              </div>
            )}

            {/* Personality Traits */}
            {response.personality_traits && response.personality_traits.length > 0 && (
              <div>
                <h4 className="font-bold text-gray-800 mb-3 flex items-center">
                  <span className="mr-2">🎭</span>
                  Authentic Personality Traits
                </h4>
                <div className="grid gap-2">
                  {response.personality_traits.map((trait, index) => (
                    <div key={index} className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                      <p className="text-blue-800 text-sm">{trait}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Teaching Style */}
            {response.guidance.teaching_style && (
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h4 className="font-bold text-purple-800 mb-2 flex items-center">
                  <span className="mr-2">🎯</span>
                  Teaching Style
                </h4>
                <p className="text-purple-700 capitalize">{response.guidance.teaching_style.replace(/_/g, ' ')}</p>
              </div>
            )}

            {/* Wisdom Synthesis */}
            {response.wisdom_synthesis && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 className="font-bold text-green-800 mb-3 flex items-center">
                  <span className="mr-2">�</span>
                  Wisdom Synthesis
                </h4>
                <p className="text-green-700 italic">"{response.wisdom_synthesis}"</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'practices' && (
          <div className="space-y-6">
            {response.guidance.meditation_practice && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 className="font-bold text-green-800 mb-3 flex items-center">
                  <span className="mr-2">🧘</span>
                  Recommended Meditation Practice
                </h4>
                <div className="prose prose-sm max-w-none text-green-700">
                  <div dangerouslySetInnerHTML={{ 
                    __html: response.guidance.meditation_practice.replace(/\n/g, '<br>') 
                  }} />
                </div>
              </div>
            )}

            {/* Growth Opportunities */}
            {response.growth_opportunities && response.growth_opportunities.length > 0 && (
              <div>
                <h4 className="font-bold text-gray-800 mb-3 flex items-center">
                  <span className="mr-2">🌱</span>
                  Areas for Growth & Exploration
                </h4>
                <div className="grid gap-3">
                  {response.growth_opportunities.map((opportunity, index) => (
                    <div 
                      key={index} 
                      className="p-3 bg-purple-50 border border-purple-200 rounded-lg cursor-pointer hover:bg-purple-100 transition-colors"
                      onClick={() => onFollowUp && onFollowUp(opportunity)}
                    >
                      <p className="text-purple-800 text-sm">{opportunity}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'mantras' && (
          <div className="space-y-4">
            {/* Signature Mantra */}
            {response.guidance.signature_mantra && (
              <div className="border border-orange-200 rounded-lg p-4 bg-orange-50">
                <h5 className="font-bold text-orange-800 mb-3 flex items-center">
                  <span className="mr-2">🕉️</span>
                  Signature Mantra
                </h5>
                <div className="bg-white p-4 rounded border">
                  <p className="font-mono text-gray-800 text-center text-lg">
                    {response.guidance.signature_mantra}
                  </p>
                </div>
                <p className="text-orange-700 text-sm mt-3 text-center italic">
                  The sacred sound vibration associated with this Rishi
                </p>
              </div>
            )}

            {/* Additional Mantras */}
            {response.guidance.mantras && response.guidance.mantras.length > 0 ? (
              response.guidance.mantras.map((mantra, index) => (
                <div key={index} className="border border-orange-200 rounded-lg p-4 bg-orange-50">
                  <h5 className="font-bold text-orange-800 mb-3">Sacred Mantra {index + 1}</h5>
                  <div className="space-y-3">
                    <div className="bg-white p-3 rounded border">
                      <p className="font-mono text-gray-800 text-center text-lg">
                        {mantra.sanskrit}
                      </p>
                    </div>
                    <div className="bg-white p-3 rounded border">
                      <p className="text-gray-600 text-center italic">
                        {mantra.transliteration}
                      </p>
                    </div>
                    <div className="text-sm text-orange-700">
                      <p><strong>Meaning:</strong> {mantra.meaning}</p>
                      <p><strong>Usage:</strong> {mantra.usage}</p>
                    </div>
                  </div>
                </div>
              ))
            ) : !response.guidance.signature_mantra ? (
              <div className="text-center text-gray-500 py-8">
                <span className="text-4xl mb-4 block">🕉️</span>
                <p>Sacred mantras and chants will appear here when provided by the Rishi.</p>
              </div>
            ) : null}
          </div>
        )}
      </div>

      {/* Session Continuity Info */}
      {response.enhanced && response.session_continuity && (
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
          <div className="flex items-center justify-between text-xs text-gray-600">
            <span>Session: {response.session_continuity.conversation_count || 1}</span>
            <span>Enhanced Guidance Mode</span>
            <span>✨ Personalized for your journey</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default RishiResponseDisplay;
