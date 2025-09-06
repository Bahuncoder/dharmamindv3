import React, { useState, useRef, useEffect } from 'react';
import { 
  PaperAirplaneIcon, 
  SparklesIcon, 
  UserCircleIcon, 
  ChevronDownIcon,
  ArrowPathIcon,
  BookmarkIcon,
  ShareIcon,
  ChartBarIcon,
  ScaleIcon,
  HeartIcon,
  LightBulbIcon,
  UserGroupIcon,
  BriefcaseIcon,
  HomeIcon,
  AcademicCapIcon,
  GlobeAltIcon,
  MicrophoneIcon,
  StopIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SpiritualBackground from './SpiritualBackground';
import BreathingGuide from './BreathingGuide';
import VoiceWaveVisualization from './VoiceWaveVisualization';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  confidence?: number;
  dharmic_alignment?: number;
  modules_used?: string[];
  isFavorite?: boolean;
  isSaved?: boolean;
  category?: string;
  emotional_tone?: string;
}

interface LifeAspect {
  id: string;
  name: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  description: string;
  keywords: string[];
}

const LIFE_ASPECTS: LifeAspect[] = [
  {
    id: 'relationships',
    name: 'Relationships',
    icon: HeartIcon,
    color: 'text-rose-600',
    description: 'Navigate love, family, and connections',
    keywords: ['love', 'relationship', 'family', 'friendship', 'connection', 'partner']
  },
  {
    id: 'personal-growth',
    name: 'Personal Growth',
    icon: LightBulbIcon,
    color: 'text-amber-600',
    description: 'Evolve and transform yourself',
    keywords: ['growth', 'development', 'mindset', 'habits', 'transformation']
  },
  {
    id: 'community',
    name: 'Community & Service',
    icon: UserGroupIcon,
    color: 'text-green-600',
    description: 'Contribute to collective wellbeing',
    keywords: ['community', 'service', 'society', 'contribution', 'impact']
  },
  {
    id: 'career',
    name: 'Career & Purpose',
    icon: BriefcaseIcon,
    color: 'text-purple-600',
    description: 'Find meaningful work and purpose',
    keywords: ['career', 'work', 'purpose', 'calling', 'profession', 'job']
  },
  {
    id: 'life-balance',
    name: 'Life Balance',
    icon: HomeIcon,
    color: 'text-teal-600',
    description: 'Harmonize all aspects of life',
    keywords: ['balance', 'harmony', 'wellness', 'health', 'lifestyle']
  },
  {
    id: 'learning',
    name: 'Learning & Wisdom',
    icon: AcademicCapIcon,
    color: 'text-indigo-600',
    description: 'Expand knowledge and understanding',
    keywords: ['learning', 'wisdom', 'knowledge', 'education', 'study']
  },
  {
    id: 'global-impact',
    name: 'Global Impact',
    icon: GlobeAltIcon,
    color: 'text-emerald-600',
    description: 'Make a positive global difference',
    keywords: ['global', 'world', 'impact', 'change', 'humanity']
  }
];

const AdvancedSpiritualChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedAspect, setSelectedAspect] = useState<LifeAspect | null>(null);
  const [showAspectSelector, setShowAspectSelector] = useState(true);
  const [isListening, setIsListening] = useState(false);
  const [meditationMode, setMeditationMode] = useState(false);
  const [isBreathingActive, setIsBreathingActive] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-detect life aspect based on message content
  const detectLifeAspect = (message: string): LifeAspect | null => {
    const messageLower = message.toLowerCase();
    for (const aspect of LIFE_ASPECTS) {
      if (aspect.keywords.some(keyword => messageLower.includes(keyword))) {
        return aspect;
      }
    }
    return null;
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      role: 'user',
      timestamp: new Date(),
      category: selectedAspect?.id || 'general'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setShowAspectSelector(false);

    // Auto-detect aspect if not manually selected
    if (!selectedAspect) {
      const detectedAspect = detectLifeAspect(inputValue);
      if (detectedAspect) {
        setSelectedAspect(detectedAspect);
      }
    }

    try {
      // Simulate API response for demo
      setTimeout(() => {
        const aiResponse: Message = {
          id: (Date.now() + 1).toString(),
          content: `Thank you for sharing your thoughts about ${selectedAspect?.name.toLowerCase() || 'life'}. Here's some wisdom to consider:\n\n**Mindful Approach**: Every challenge is an opportunity for growth and deeper understanding.\n\n**Practical Steps**:\n- Practice daily mindfulness meditation\n- Reflect on your core values\n- Take aligned action with compassion\n\n*Remember: The journey of consciousness is unique to each soul. Trust your inner wisdom.*`,
          role: 'assistant',
          timestamp: new Date(),
          confidence: 0.92,
          dharmic_alignment: 0.87,
          category: selectedAspect?.id || 'general'
        };
        setMessages(prev => [...prev, aiResponse]);
        setIsLoading(false);
      }, 2000);
    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const selectLifeAspect = (aspect: LifeAspect) => {
    setSelectedAspect(aspect);
    setShowAspectSelector(false);
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  const startNewConversation = () => {
    setMessages([]);
    setSelectedAspect(null);
    setShowAspectSelector(true);
    setInputValue('');
  };

  return (
    <div className={`organic-chat-container min-h-screen relative overflow-hidden ${meditationMode ? 'meditation-mode' : ''}`}>
      {/* Spiritual Background */}
      <SpiritualBackground />
      
      {/* Breathing Guide */}
      <BreathingGuide 
        isVisible={isBreathingActive} 
        onClose={() => setIsBreathingActive(false)}
      />
      
      {/* Header with Aurora Effect */}
      <motion.header 
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="relative z-10 backdrop-blur-xl bg-white/5 border-b border-white/10 px-6 py-4"
      >
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <motion.div 
            className="flex items-center space-x-4"
            whileHover={{ scale: 1.02 }}
          >
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 1 }}
              className="h-12 w-12 bg-gradient-to-br from-emerald-400 to-blue-500 rounded-full flex items-center justify-center spiritual-glow"
            >
              <SparklesIcon className="h-6 w-6 text-white" />
            </motion.div>
            <div>
              <motion.h1 
                className="text-3xl font-bold bg-gradient-to-r from-emerald-400 via-blue-400 to-purple-400 bg-clip-text text-transparent"
                animate={{ 
                  backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'] 
                }}
                transition={{ 
                  duration: 3, 
                  repeat: Infinity, 
                  ease: "linear" 
                }}
              >
                Universal Dharma
              </motion.h1>
              <p className="text-sm text-white/70 font-medium">
                ✨ Conscious Living • Mindful Growth • Sacred Wisdom ✨
              </p>
            </div>
          </motion.div>
          
          <motion.div 
            className="flex items-center space-x-3"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            {selectedAspect && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="flex items-center space-x-2 bg-gradient-to-r from-emerald-100/20 to-blue-100/20 px-3 py-2 rounded-full backdrop-blur-md border border-white/20"
              >
                <selectedAspect.icon className={`h-5 w-5 ${selectedAspect.color}`} />
                <span className="text-sm font-medium text-white/90">
                  {selectedAspect.name}
                </span>
              </motion.div>
            )}
            
            {/* Meditation Mode Toggle */}
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setMeditationMode(!meditationMode)}
              className={`p-3 rounded-full backdrop-blur-md transition-all duration-300 ${
                meditationMode 
                  ? 'bg-purple-500/30 border border-purple-400/50 text-purple-200' 
                  : 'bg-white/10 border border-white/20 text-white/70 hover:bg-white/20'
              }`}
              title="Toggle Meditation Mode"
            >
              <SparklesIcon className="h-5 w-5" />
            </motion.button>

            {/* Voice Control */}
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setIsListening(!isListening)}
              className={`p-3 rounded-full backdrop-blur-md transition-all duration-300 ${
                isListening 
                  ? 'bg-red-500/30 border border-red-400/50 text-red-200' 
                  : 'bg-white/10 border border-white/20 text-white/70 hover:bg-white/20'
              }`}
              title={isListening ? "Stop Listening" : "Start Voice Input"}
            >
              {isListening ? <StopIcon className="h-5 w-5" /> : <MicrophoneIcon className="h-5 w-5" />}
            </motion.button>

            {/* Breathing Guide Toggle */}
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setIsBreathingActive(!isBreathingActive)}
              className={`p-3 rounded-full backdrop-blur-md transition-all duration-300 ${
                isBreathingActive 
                  ? 'bg-emerald-500/30 border border-emerald-400/50 text-emerald-200' 
                  : 'bg-white/10 border border-white/20 text-white/70 hover:bg-white/20'
              }`}
              title="Mindful Breathing Guide"
            >
              <HeartIcon className="h-5 w-5" />
            </motion.button>
          </motion.div>
        </div>
      </motion.header>

      {/* Life Aspect Selector */}
      <AnimatePresence>
        {showAspectSelector && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="relative z-10 backdrop-blur-md bg-white/5 border-b border-white/10 px-6 py-6"
          >
            <div className="max-w-7xl mx-auto">
              <h2 className="text-lg font-semibold text-white/90 mb-4 text-center">
                What aspect of life would you like guidance on?
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {LIFE_ASPECTS.map((aspect) => (
                  <motion.button
                    key={aspect.id}
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => selectLifeAspect(aspect)}
                    className="group p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20 hover:border-emerald-300/50 hover:bg-white/20 transition-all duration-200"
                  >
                    <aspect.icon className={`h-8 w-8 ${aspect.color} mx-auto mb-2 group-hover:scale-110 transition-transform`} />
                    <h3 className="text-sm font-medium text-white/90 mb-1">
                      {aspect.name}
                    </h3>
                    <p className="text-xs text-white/60">
                      {aspect.description}
                    </p>
                  </motion.button>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Messages Area */}
      <div className="flex-1 overflow-hidden relative z-10">
        <div className="h-full overflow-y-auto px-6 py-4 space-y-6">
          <div className="max-w-4xl mx-auto">
            {messages.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center py-12"
              >
                <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-emerald-400 to-blue-500 rounded-full flex items-center justify-center spiritual-glow">
                  <SparklesIcon className="h-10 w-10 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white/90 mb-2">
                  Welcome to Universal Dharma
                </h3>
                <p className="text-white/70 max-w-md mx-auto">
                  Your AI companion for ethical decision-making, personal growth, and conscious living. 
                  Ask me anything about life's challenges and opportunities.
                </p>
              </motion.div>
            ) : (
              <AnimatePresence>
                {messages.map((message, index) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 30, scale: 0.8 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ 
                      delay: index * 0.1,
                      type: "spring",
                      stiffness: 200,
                      damping: 20
                    }}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-6`}
                  >
                    <motion.div 
                      className={`organic-bubble ${message.role} fade-in-up max-w-3xl`}
                      whileHover={{ 
                        scale: 1.02,
                        rotateY: message.role === 'user' ? -2 : 2,
                      }}
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    >
                      <div className="prose prose-sm max-w-none text-white/90">
                        <ReactMarkdown 
                          remarkPlugins={[remarkGfm]} 
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                      
                      {/* Message Metadata */}
                      {message.role === 'assistant' && (
                        <div className="mt-4 pt-3 border-t border-white/20">
                          <div className="flex items-center justify-between text-xs text-white/60">
                            <div className="flex items-center space-x-4">
                              {message.confidence && (
                                <div className="flex items-center space-x-1">
                                  <ChartBarIcon className="h-3 w-3" />
                                  <span>Confidence: {Math.round(message.confidence * 100)}%</span>
                                </div>
                              )}
                              {message.dharmic_alignment && (
                                <div className="flex items-center space-x-1">
                                  <ScaleIcon className="h-3 w-3" />
                                  <span>Dharmic: {Math.round(message.dharmic_alignment * 100)}%</span>
                                </div>
                              )}
                            </div>
                            
                            <div className="flex items-center space-x-2">
                              <button className="hover:text-emerald-300 transition-colors">
                                <BookmarkIcon className="h-4 w-4" />
                              </button>
                              <button className="hover:text-emerald-300 transition-colors">
                                <ShareIcon className="h-4 w-4" />
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
            
            {/* Loading Indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start mb-6"
              >
                <div className="organic-bubble ai">
                  <div className="loading-dots-enhanced">
                    <div className="loading-dot loading-dot-1"></div>
                    <div className="loading-dot loading-dot-2"></div>
                    <div className="loading-dot loading-dot-3"></div>
                    <span className="loading-text">Contemplating wisdom...</span>
                  </div>
                </div>
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      {/* Input Area */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="relative z-10 p-6"
      >
        <div className="max-w-4xl mx-auto">
          <div className="mystical-input-container">
            <div className="flex items-end space-x-4 p-2">
              {/* New Conversation Button */}
              <motion.button
                whileHover={{ scale: 1.05, rotate: 180 }}
                whileTap={{ scale: 0.95 }}
                onClick={startNewConversation}
                className="p-3 rounded-full bg-white/10 backdrop-blur-md border border-white/20 text-white/70 hover:bg-white/20 hover:text-white transition-all duration-300"
                title="Start new conversation"
              >
                <ArrowPathIcon className="h-5 w-5" />
              </motion.button>

              {/* Voice Wave Visualization */}
              {isListening && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  className="flex-1"
                >
                  <VoiceWaveVisualization isActive={isListening} />
                </motion.div>
              )}

              {/* Input Field */}
              {!isListening && (
                <div className="flex-1 relative">
                  <textarea
                    ref={inputRef}
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={selectedAspect 
                      ? `Share your thoughts about ${selectedAspect.name.toLowerCase()}...` 
                      : "What wisdom do you seek today? ✨"
                    }
                    className="mystical-input"
                    rows={1}
                    style={{ minHeight: '48px', maxHeight: '120px' }}
                  />
                  
                  {/* Aspect Selector Button */}
                  <motion.button
                    whileHover={{ scale: 1.1, rotate: 180 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => setShowAspectSelector(!showAspectSelector)}
                    className="absolute right-12 top-1/2 transform -translate-y-1/2 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white/60 hover:text-white transition-all duration-300"
                    title="Select life aspect"
                  >
                    <ChevronDownIcon className={`h-4 w-4 transition-transform duration-300 ${showAspectSelector ? 'rotate-180' : ''}`} />
                  </motion.button>
                </div>
              )}

              {/* Send Button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="p-3 rounded-full bg-gradient-to-r from-emerald-500 via-blue-500 to-purple-500 hover:from-emerald-400 hover:via-blue-400 hover:to-purple-400 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed text-white shadow-lg hover:shadow-xl transition-all duration-300 spiritual-glow"
                style={{
                  background: 'linear-gradient(135deg, #10b981, #3b82f6, #8b5cf6)',
                  backgroundSize: '200% 200%',
                  animation: 'gradient-shift 3s ease infinite'
                }}
              >
                <PaperAirplaneIcon className="h-5 w-5" />
              </motion.button>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default AdvancedSpiritualChatInterface;
