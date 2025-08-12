import React, { useState, useRef, useEffect } from 'react';
import { 
  PaperAirplaneIcon, 
  SparklesIcon, 
  UserCircleIcon, 
  ChevronDownIcon,
  ClipboardDocumentIcon,
  ArrowPathIcon,
  BookmarkIcon,
  ShareIcon,
  StarIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiService, ChatMessage, ChatResponse } from '../utils/apiService';
import { useSubscription } from '../hooks/useSubscription';
import { useColors } from '../contexts/ColorContext';
import { useAuth } from '../contexts/AuthContext';
import { UpgradePrompt, MiniUpgradeBanner, UsageProgress } from './UpgradePrompt';
import CentralizedSubscriptionModal from './CentralizedSubscriptionModal';
import UserProfileMenu from './UserProfileMenu';
import Logo from './Logo';
import ToastContainer, { useToast } from './ToastContainer';
import { useConversationTags } from '../hooks/useConversationTags';
import { useTheme } from '../contexts/ThemeContext';
import ThemeToggle from './ThemeToggle';
import ConversationInsights from './ConversationInsights';
import EnhancedMessageInput from './EnhancedMessageInput';
import EnhancedMessageBubble from './EnhancedMessageBubble';
import ScrollToBottom from './ScrollToBottom';
import FloatingActionMenu from './FloatingActionMenu';
import TypingIndicator from './TypingIndicator';
import MessageSearch from './MessageSearch';

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
}

interface ChatInterfaceProps {
  onMessageSend?: (message: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onMessageSend }) => {
  // Authentication
  const { user, isAuthenticated } = useAuth();

  // Subscription features
  const {
    trackUsage,
    checkFeatureUsage,
    shouldShowUpgradePrompt,
    getUpgradeMessage,
    isFreePlan
  } = useSubscription();

  const { currentTheme } = useColors();

  // Toast notifications
  const {
    toasts,
    dismissToast,
    showCopySuccess,
    showFavoriteAdded,
    showFavoriteRemoved,
    showSavedToJournal,
    showRemovedFromJournal,
    showSpeaking,
    showShared,
    showError
  } = useToast();

  // Conversation tagging
  const { tagConversation, analyzeMessageForTags, availableTags } = useConversationTags();

  // Theme
  const { currentTheme: appTheme } = useTheme();

  // UI State
  const [showUserDropdown, setShowUserDropdown] = useState(false);

  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: '🕉️ Namaste! I am DharmaMind, your companion on the path of wisdom. How may I serve you today?',
      role: 'assistant',
      timestamp: new Date(),
      confidence: 1.0,
      dharmic_alignment: 1.0,
      modules_used: ['wisdom']
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [error, setError] = useState<string | null>(null);
  const [showSubscriptionModal, setShowSubscriptionModal] = useState(false);
  const [usageAlert, setUsageAlert] = useState<string | null>(null);
  const [hoveredMessageId, setHoveredMessageId] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<string | null>(null);
  const [savedMessages, setSavedMessages] = useState<string[]>([]);
  const [favoriteMessages, setFavoriteMessages] = useState<string[]>([]);
  const [showInsightsModal, setShowInsightsModal] = useState(false);
  const [currentTags, setCurrentTags] = useState<string[]>([]);
  const [messageReactions, setMessageReactions] = useState<Record<string, any>>({});
  const [showBreathingGuide, setShowBreathingGuide] = useState(false);
  const [meditationMode, setMeditationMode] = useState(false);
  const [showMessageSearch, setShowMessageSearch] = useState(false);
  const [showFloatingMenu, setShowFloatingMenu] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowUserDropdown(false);
      }
    };

    if (showUserDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [showUserDropdown]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    // Check message usage before proceeding (subscription feature)
    const canSendMessage = await trackUsage('messages');
    if (!canSendMessage) {
      const message = getUpgradeMessage('messages');
      setUsageAlert(message);
      setShowSubscriptionModal(true);
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      role: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);
    setUsageAlert(null);

    // Call parent handler if provided
    if (onMessageSend) {
      onMessageSend(inputMessage);
    }

    try {
      // Call the actual API
      const response = await apiService.sendChatMessage({
        message: inputMessage,
        conversation_id: conversationId,
        context: 'spiritual_guidance'
      });

      if (response.success && response.data) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: response.data.message,
          role: 'assistant',
          timestamp: new Date(),
          confidence: response.data.metadata?.confidence || 0.9,
          dharmic_alignment: response.data.metadata?.dharmic_alignment || 0.9,
          modules_used: response.data.metadata?.modules_used || []
        };

        setMessages(prev => [...prev, assistantMessage]);
        
        // Update conversation ID for continuity
        if (response.data.conversation_id) {
          setConversationId(response.data.conversation_id);
        }

        // Auto-tag conversation
        const allMessages = [...messages, userMessage, assistantMessage].map(m => m.content);
        const detectedTags = tagConversation(
          response.data.conversation_id || conversationId || Date.now().toString(),
          `Chat ${new Date().toLocaleDateString()}`,
          allMessages,
          assistantMessage.content,
          assistantMessage.dharmic_alignment
        );
        setCurrentTags(detectedTags);
      } else {
        // Handle API error
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: `I apologize, but I'm experiencing technical difficulties. ${response.error || 'Please try again in a moment.'}`,
          role: 'assistant',
          timestamp: new Date(),
          confidence: 0.1
        };
        setMessages(prev => [...prev, errorMessage]);
        setError(response.error || 'Failed to get response');
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'I apologize, but I\'m having trouble connecting right now. Please check your connection and try again.',
        role: 'assistant',
        timestamp: new Date(),
        confidence: 0.1
      };
      setMessages(prev => [...prev, errorMessage]);
      setError('Network error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getAlignmentColor = (alignment?: number) => {
    if (!alignment) return 'text-gray-400';
    if (alignment >= 0.8) return 'text-green-500';
    if (alignment >= 0.6) return 'text-emerald-400';
    return 'text-red-500';
  };

  const getConfidenceIcon = (confidence?: number) => {
    if (!confidence) return '🤔';
    if (confidence >= 0.9) return '🧘‍♂️';
    if (confidence >= 0.7) return '🕉️';
    if (confidence >= 0.5) return '🌸';
    return '🤔';
  };

  // Message action handlers
  const copyMessage = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      showCopySuccess();
    } catch (error) {
      console.error('Failed to copy message:', error);
      showError('Copy Failed', 'Unable to copy message to clipboard');
    }
  };

  const regenerateMessage = async (messageId: string) => {
    const messageIndex = messages.findIndex(msg => msg.id === messageId);
    if (messageIndex === -1 || messageIndex === 0) return;

    const userMessage = messages[messageIndex - 1];
    if (userMessage.role !== 'user') return;

    // Remove the current assistant message
    const updatedMessages = messages.filter(msg => msg.id !== messageId);
    setMessages(updatedMessages);
    
    // Regenerate response
    setIsLoading(true);
    try {
      const response = await apiService.sendChatMessage({
        message: userMessage.content,
        conversation_id: conversationId,
        context: 'spiritual_guidance'
      });

      if (response.success && response.data) {
        const newAssistantMessage: Message = {
          id: Date.now().toString(),
          content: response.data.message,
          role: 'assistant',
          timestamp: new Date(),
          confidence: response.data.metadata?.confidence || 0.9,
          dharmic_alignment: response.data.metadata?.dharmic_alignment || 0.9,
          modules_used: response.data.metadata?.modules_used || []
        };

        setMessages(prev => [...prev, newAssistantMessage]);
      }
    } catch (error) {
      console.error('Regeneration failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleFavorite = (messageId: string) => {
    const isCurrentlyFavorite = favoriteMessages.includes(messageId);
    
    setFavoriteMessages(prev => 
      isCurrentlyFavorite 
        ? prev.filter(id => id !== messageId)
        : [...prev, messageId]
    );

    if (isCurrentlyFavorite) {
      showFavoriteRemoved();
    } else {
      showFavoriteAdded();
    }
  };

  const toggleSaved = (messageId: string) => {
    const isCurrentlySaved = savedMessages.includes(messageId);
    
    setSavedMessages(prev => 
      isCurrentlySaved 
        ? prev.filter(id => id !== messageId)
        : [...prev, messageId]
    );

    if (isCurrentlySaved) {
      showRemovedFromJournal();
    } else {
      showSavedToJournal();
    }
  };

  const shareMessage = async (content: string) => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Wisdom from DharmaMind',
          text: content,
          url: window.location.href
        });
        showShared();
      } catch (error) {
        console.error('Sharing failed:', error);
        copyMessage(`"${content}" - Shared from DharmaMind`);
      }
    } else {
      copyMessage(`"${content}" - Shared from DharmaMind`);
      showShared();
    }
  };

  const speakMessage = (content: string, messageId: string) => {
    if (isPlaying === messageId) {
      speechSynthesis.cancel();
      setIsPlaying(null);
      return;
    }

    speechSynthesis.cancel();
    showSpeaking();
    
    const utterance = new SpeechSynthesisUtterance(content);
    utterance.rate = 0.7; // Slower for spiritual content
    utterance.pitch = 0.9; // Slightly deeper
    utterance.volume = 0.8;
    
    utterance.onstart = () => setIsPlaying(messageId);
    utterance.onend = () => setIsPlaying(null);
    utterance.onerror = () => {
      setIsPlaying(null);
      showError('Speech Error', 'Unable to speak this message');
    };
    
    speechSynthesis.speak(utterance);
  };

  const handleReaction = (messageId: string, reactionType: string) => {
    setMessageReactions(prev => {
      const messageReacts = prev[messageId] || [];
      const existingReaction = messageReacts.find((r: any) => r.type === reactionType);
      
      if (existingReaction) {
        // Toggle reaction
        existingReaction.userReacted = !existingReaction.userReacted;
        existingReaction.count += existingReaction.userReacted ? 1 : -1;
      } else {
        // Add new reaction
        messageReacts.push({
          type: reactionType,
          count: 1,
          userReacted: true
        });
      }
      
      return {
        ...prev,
        [messageId]: messageReacts.filter((r: any) => r.count > 0)
      };
    });
  };

  // Floating Action Menu Handlers
  const handleNewChat = () => {
    setMessages([{
      id: Date.now().toString(),
      content: '🕉️ Namaste! I am DharmaMind, your companion on the path of wisdom. How may I serve you today?',
      role: 'assistant',
      timestamp: new Date(),
      confidence: 1.0,
      dharmic_alignment: 1.0,
      modules_used: ['wisdom']
    }]);
    setConversationId(undefined);
  };

  const handleOpenNotes = () => {
    // Implement notes functionality
    console.log('Opening notes...');
  };

  const handleSearchHistory = () => {
    setShowMessageSearch(true);
  };

  const handleOpenSettings = () => {
    // Implement settings functionality
    console.log('Opening settings...');
  };

  const handleOpenJournal = () => {
    // Implement journal functionality
    console.log('Opening journal...');
  };

  const handleOpenInsights = () => {
    setShowInsightsModal(true);
  };

  const handleOpenCommunity = () => {
    // Implement community functionality
    console.log('Opening community...');
  };

  const handleSelectMessage = (result: any) => {
    // Navigate to the specific conversation/message
    console.log('Selected message:', result);
    setShowMessageSearch(false);
  };

  return (
    <div className={`flex flex-col h-full chat-container relative ${meditationMode ? 'meditation-mode' : ''}`}>
      {/* Advanced Spiritual Background */}
      <div className="floating-orbs-container">
        <div className="floating-orb large" style={{ top: '10%', left: '20%', animationDelay: '0s' }}></div>
        <div className="floating-orb medium" style={{ top: '60%', right: '15%', animationDelay: '2s' }}></div>
        <div className="floating-orb small" style={{ top: '30%', left: '70%', animationDelay: '4s' }}></div>
        <div className="floating-orb medium" style={{ bottom: '20%', left: '10%', animationDelay: '6s' }}></div>
        <div className="floating-orb small" style={{ top: '80%', right: '40%', animationDelay: '8s' }}></div>
      </div>
      
      {/* Sacred Geometry Background */}
      <div className="sacred-geometry-bg"></div>
      
      {/* Lotus Patterns */}
      <div className="lotus-pattern" style={{ top: '15%', right: '5%', animationDelay: '0s' }}></div>
      <div className="lotus-pattern" style={{ bottom: '25%', left: '3%', animationDelay: '10s' }}></div>
      
      {/* Breathing Guide */}
      {showBreathingGuide && (
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0, opacity: 0 }}
          className="breathing-guide"
          onClick={() => setShowBreathingGuide(false)}
          title="Click to hide breathing guide"
        >
          <div className="breathing-circle"></div>
        </motion.div>
      )}
      {/* Usage Alert Banner */}
      {isFreePlan() && shouldShowUpgradePrompt('messages') && (
        <MiniUpgradeBanner 
          onUpgrade={() => setShowSubscriptionModal(true)}
        />
      )}
      
      {/* Header */}
      <div className="flex-shrink-0 p-4 sm:p-6 chat-header-enhanced animate-slide-in-down relative z-10">
        <div className="organic-chat-container">
          <div className="flex items-center space-x-3 sm:space-x-4">
            <div className="interactive-element spiritual-glow">
              <Logo size="sm" showText={false} />
            </div>
            <div className="flex-1 min-w-0">
              <h1 className="text-xl sm:text-2xl typography-heading text-gray-900 truncate emerald-gradient-text">
                DharmaMind AI
              </h1>
              <p className="text-sm sm:text-base typography-caption text-gray-600 hidden sm:block mt-1">
                Your spiritual wisdom companion
              </p>
            </div>
            
            {/* Current Tags Display */}
            {currentTags.length > 0 && (
              <div className="hidden md:flex items-center space-x-1">
                {currentTags.slice(0, 2).map(tagId => {
                  const tag = availableTags.find(t => t.id === tagId);
                  return tag ? (
                    <span
                      key={tagId}
                      className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium"
                      style={{ backgroundColor: tag.color + '20', color: tag.color }}
                    >
                      <span className="mr-1">{tag.icon}</span>
                      {tag.name}
                    </span>
                  ) : null;
                })}
                {currentTags.length > 2 && (
                  <span className="text-xs text-gray-500">
                    +{currentTags.length - 2}
                  </span>
                )}
              </div>
            )}
            
            {/* Theme Toggle & Insights */}
            <div className="flex items-center space-x-3">
              <div className="interactive-element">
                <ThemeToggle size="sm" />
              </div>
              
              {/* Meditation Mode Toggle */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setMeditationMode(!meditationMode)}
                className={`p-2 rounded-full transition-all duration-300 ${
                  meditationMode 
                    ? 'bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400' 
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700'
                }`}
                title="Toggle Meditation Mode"
              >
                <SparklesIcon className="h-4 w-4" />
              </motion.button>

              {/* Breathing Guide Toggle */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowBreathingGuide(!showBreathingGuide)}
                className={`p-2 rounded-full transition-all duration-300 ${
                  showBreathingGuide 
                    ? 'bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400' 
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700'
                }`}
                title="Toggle Breathing Guide"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              </motion.button>
              
              <button
                onClick={() => setShowInsightsModal(true)}
                className="btn-premium btn-premium-ghost flex items-center space-x-2"
                title="View conversation insights"
              >
                <ChartBarIcon className="w-4 h-4" />
                <span className="hidden sm:inline typography-caption">
                  Insights
                </span>
              </button>
            </div>
            
            {/* Usage Progress in Header - Desktop Only */}
            {isFreePlan() && (
              <div className="hidden lg:block w-32">
                <UsageProgress 
                  feature="messages" 
                  showDetails={false}
                  className="mr-4"
                />
              </div>
            )}
            
            {/* User Profile Dropdown */}
            {isAuthenticated && user && (
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setShowUserDropdown(!showUserDropdown)}
                  className="btn-premium btn-premium-secondary flex items-center space-x-2 animate-pulse-emerald"
                  style={{ 
                    borderColor: currentTheme.colors.primary + '20',
                    backgroundColor: showUserDropdown ? currentTheme.colors.primary + '10' : 'transparent'
                  }}
                >
                  <UserCircleIcon className="h-5 w-5 sm:h-6 sm:w-6" />
                  <div className="hidden md:block text-left">
                    <p className="typography-caption text-sm">{user.first_name || 'User'}</p>
                    <p className="typography-caption text-xs opacity-75 capitalize">
                      {user.subscription_plan} Plan
                    </p>
                  </div>
                  <ChevronDownIcon 
                    className={`h-4 w-4 transition-transform duration-300 ${showUserDropdown ? 'rotate-180' : ''}`} 
                  />
                  {/* Mobile upgrade indicator */}
                  {isFreePlan() && (
                    <span className="md:hidden w-2.5 h-2.5 bg-emerald-400 rounded-full animate-pulse-emerald"></span>
                  )}
                </button>

                {/* Dropdown Menu */}
                <AnimatePresence>
                  {showUserDropdown && (
                    <UserProfileMenu
                      onUpgrade={() => setShowSubscriptionModal(true)}
                      onClose={() => setShowUserDropdown(false)}
                    />
                  )}
                </AnimatePresence>
              </div>
            )}
            
            {error && (
              <div className="bg-red-100 text-red-800 px-2 sm:px-3 py-1 rounded-full text-xs">
                <span className="hidden sm:inline">Connection issue</span>
                <span className="sm:hidden">!</span>
              </div>
            )}
          </div>
          
          {/* Mobile Usage Progress */}
          {isFreePlan() && (
            <div className="lg:hidden mt-3">
              <UsageProgress feature="messages" />
            </div>
          )}
        </div>
      </div>

      {/* Messages Container */}
      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-6 sm:space-y-8 scroll-smooth chat-messages-enhanced chat-messages-container organic-chat-container"
        style={{
          scrollbarWidth: 'thin',
          scrollbarColor: '#10b981 transparent'
        }}
      >
        <AnimatePresence mode="popLayout">
          {messages.map((message, index) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.95 }}
              transition={{ 
                duration: 0.4, 
                delay: index * 0.1,
                type: "spring",
                stiffness: 500,
                damping: 30
              }}
              className="animate-fade-in-scale"
            >
              <div className={`organic-bubble ${message.role === 'user' ? 'user' : 'ai'} fade-in-up`}>
                <EnhancedMessageBubble
                  key={message.id}
                  message={message}
                  isHovered={hoveredMessageId === message.id}
                  onHover={(hovered: boolean) => setHoveredMessageId(hovered ? message.id : null)}
                  onCopy={copyMessage}
                  onRegenerate={message.role === 'assistant' ? regenerateMessage : undefined}
                  onToggleFavorite={toggleFavorite}
                  onToggleSaved={toggleSaved}
                  onSpeak={speakMessage}
                  onShare={shareMessage}
                  onReact={(messageId: string, reaction: string) => handleReaction(messageId, reaction)}
                  isPlaying={isPlaying === message.id}
                />
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Enhanced Typing indicator */}
        <TypingIndicator isVisible={isLoading} />

        <div ref={messagesEndRef} />
      </div>

      <ScrollToBottom 
        messagesContainerRef={messagesContainerRef}
      />

      {/* Usage Alert */}
      {usageAlert && (
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="p-4 sm:p-6"
        >
          <div className="glass-morphism rounded-xl p-4">
            <UpgradePrompt
              feature="messages"
              trigger="limit"
              onUpgrade={() => setShowSubscriptionModal(true)}
              onDismiss={() => setUsageAlert(null)}
            />
          </div>
        </motion.div>
      )}

      {/* Input Area */}
      <div className="flex-shrink-0 p-6 sm:p-8 chat-input-enhanced animate-slide-in-up relative z-10">
        <div className="max-w-4xl mx-auto">
          <div className="mystical-input-container">
            <EnhancedMessageInput
              value={inputMessage}
              onChange={setInputMessage}
              onSend={handleSendMessage}
              isLoading={isLoading}
              showVoiceInput={true}
              placeholder="Ask me anything about spiritual wisdom, dharma, or life guidance..."
              disabled={isLoading}
            />
          </div>
        </div>
      </div>

      {/* Subscription Modal */}
      <CentralizedSubscriptionModal
        isOpen={showSubscriptionModal}
        onClose={() => setShowSubscriptionModal(false)}
      />

      {/* Toast Notifications */}
      <ToastContainer
        toasts={toasts}
        onDismiss={dismissToast}
      />

      {/* Conversation Insights Modal */}
      <ConversationInsights
        isOpen={showInsightsModal}
        onClose={() => setShowInsightsModal(false)}
      />

      {/* Floating Action Menu */}
      {showFloatingMenu && (
        <FloatingActionMenu
          onNewChat={handleNewChat}
          onOpenNotes={handleOpenNotes}
          onSearchHistory={handleSearchHistory}
          onOpenSettings={handleOpenSettings}
          onOpenJournal={handleOpenJournal}
          onOpenInsights={handleOpenInsights}
          onOpenCommunity={handleOpenCommunity}
        />
      )}

      {/* Message Search Modal */}
      <MessageSearch
        isOpen={showMessageSearch}
        onClose={() => setShowMessageSearch(false)}
        onSelectMessage={handleSelectMessage}
      />
    </div>
  );
};

export default ChatInterface;
