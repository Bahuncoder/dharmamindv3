import React, { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/router';
import { UserCircleIcon } from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';
<<<<<<< HEAD
import { useColor } from '../contexts/ColorContext';
=======
import { useColors } from '../contexts/ColorContext';
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
import { useSubscription } from '../hooks/useSubscription';
import VoiceInput from './VoiceInput';
import UserProfileMenu from './UserProfileMenu';
import CentralizedSubscriptionModal from './CentralizedSubscriptionModal';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  isTyping?: boolean;
  wisdom_score?: number;
  dharmic_alignment?: number;
}

interface MobileChatInterfaceProps {
  onSendMessage: (message: string) => Promise<any>;
  messages: Message[];
  isLoading?: boolean;
  className?: string;
}

const MobileChatInterface: React.FC<MobileChatInterfaceProps> = ({
  onSendMessage,
  messages,
  isLoading = false,
  className = ''
}) => {
  const router = useRouter();
  const [inputText, setInputText] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [showVoiceInput, setShowVoiceInput] = useState(false);
  const [isVoiceSupported, setIsVoiceSupported] = useState(false);
  const [showUserDropdown, setShowUserDropdown] = useState(false);
  const [showSubscriptionModal, setShowSubscriptionModal] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Authentication and subscription
  const { user, isAuthenticated } = useAuth();
<<<<<<< HEAD
  const { currentTheme } = useColor();
=======
  const { currentTheme } = useColors();
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
  const { isFreePlan } = useSubscription();

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

  // Check voice support on mount
  useEffect(() => {
    const checkVoiceSupport = () => {
      const isSupported = typeof window !== 'undefined' && 
        (window.SpeechRecognition || window.webkitSpeechRecognition);
      setIsVoiceSupported(!!isSupported);
    };
    
    checkVoiceSupport();
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 120)}px`;
    }
  }, [inputText]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || isSending) return;

    const messageText = inputText.trim();
    setInputText('');
    setIsSending(true);

    try {
      await onSendMessage(messageText);
    } catch (error) {
      console.error('Failed to send message:', error);
      // Show error feedback
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleVoiceTranscript = (transcript: string) => {
    setInputText(transcript);
    setShowVoiceInput(false);
    // Auto-send voice messages
    setTimeout(() => {
      if (transcript.trim()) {
        handleSendMessage();
      }
    }, 500);
  };

  const handleVoiceError = (error: string) => {
    console.error('Voice input error:', error);
    // Show user-friendly error
    setShowVoiceInput(false);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getSuggestions = () => {
    const suggestions = [
      "How can I find inner peace?",
      "Guide me in meditation",
      "What is my life purpose?",
      "Help with relationship advice",
      "Spiritual wisdom for stress"
    ];
    return suggestions.slice(0, 3);
  };

  return (
    <div className={`mobile-chat-interface ${className}`}>
      {/* Mobile Header */}
      <div className="mobile-header">
        <div className="mobile-header-content">
          <div className="flex items-center">
            <button
              onClick={() => router.back()}
              className="p-2 hover:bg-stone-100 rounded-lg mr-2"
              aria-label="Go back"
            >
              <span className="text-xl">←</span>
            </button>
            <div className="flex items-center">
              <div className="w-8 h-8 bg-gradient-to-br from-gray-100 to-emerald-100 rounded-full flex items-center justify-center mr-3">
                <span className="text-lg">🕉️</span>
              </div>
              <div>
                <h1 className="font-medium text-stone-800">DharmaMind</h1>
                <p className="text-xs text-stone-500">AI Spiritual Companion</p>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {isVoiceSupported && (
              <button
                onClick={() => setShowVoiceInput(!showVoiceInput)}
                className={`p-2 rounded-lg transition-colors ${
                  showVoiceInput ? 'bg-emerald-100 text-emerald-600' : 'hover:bg-stone-100'
                }`}
                aria-label="Toggle voice input"
              >
                <span className="text-lg">🎙️</span>
              </button>
            )}
            
            {/* User Profile Button - Mobile */}
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setShowUserDropdown(!showUserDropdown)}
                className="p-2 hover:bg-stone-100 rounded-lg transition-colors flex items-center space-x-1"
                aria-label="User profile"
              >
                <UserCircleIcon className="h-5 w-5 text-gray-600" />
                {!isAuthenticated && (
                  <span className="w-2 h-2 bg-yellow-400 rounded-full"></span>
                )}
                {isAuthenticated && isFreePlan() && (
                  <span className="w-2 h-2 bg-yellow-400 rounded-full"></span>
                )}
              </button>

              {/* Mobile User Dropdown */}
              {showUserDropdown && (
                <div className="absolute right-0 top-full mt-2 z-50">
                  <UserProfileMenu
                    onUpgrade={() => setShowSubscriptionModal(true)}
                    onClose={() => setShowUserDropdown(false)}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div 
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 pb-20"
        style={{ maxHeight: 'calc(100vh - 140px)' }}
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-8">
            <div className="w-16 h-16 bg-gradient-to-br from-amber-100 to-emerald-100 rounded-full flex items-center justify-center mb-4">
              <span className="text-2xl">🕉️</span>
            </div>
            <h2 className="text-xl font-bold text-stone-800 mb-2">
              Welcome to DharmaMind
            </h2>
            <p className="text-stone-600 mb-6 max-w-sm">
              Your AI companion for spiritual guidance, wisdom, and inner clarity.
            </p>
            
            {/* Quick Start Suggestions */}
            <div className="w-full max-w-sm space-y-2">
              <p className="text-sm font-medium text-stone-700 mb-3">Try asking:</p>
              {getSuggestions().map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => setInputText(suggestion)}
                  className="w-full text-left p-3 bg-white border border-stone-200 rounded-lg hover:border-stone-300 transition-colors"
                >
                  <span className="text-sm text-stone-700">{suggestion}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                    message.sender === 'user'
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white ml-12'
                      : 'bg-white border border-stone-200 text-stone-800 mr-12'
                  }`}
                >
                  {message.sender === 'ai' && (
                    <div className="flex items-center mb-2">
                      <span className="text-lg mr-2">🕉️</span>
                      <span className="text-xs font-medium text-stone-500">DharmaMind</span>
                    </div>
                  )}
                  
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {message.isTyping ? (
                      <span className="inline-flex">
                        <span className="animate-pulse">●</span>
                        <span className="animate-pulse delay-200">●</span>
                        <span className="animate-pulse delay-400">●</span>
                      </span>
                    ) : (
                      message.content
                    )}
                  </p>
                  
                  <div className="flex items-center justify-between mt-2">
                    <span className={`text-xs ${
                      message.sender === 'user' ? 'text-blue-100' : 'text-stone-400'
                    }`}>
                      {formatTime(message.timestamp)}
                    </span>
                    
                    {message.sender === 'ai' && message.wisdom_score && (
                      <div className="flex items-center space-x-2 text-xs text-stone-500">
                        <span>💡 {message.wisdom_score.toFixed(0)}</span>
                        {message.dharmic_alignment && (
                          <span>🕉️ {message.dharmic_alignment.toFixed(0)}%</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {isSending && (
              <div className="flex justify-start">
                <div className="bg-white border border-stone-200 text-stone-800 rounded-2xl px-4 py-3 mr-12">
                  <div className="flex items-center mb-2">
                    <span className="text-lg mr-2">🕉️</span>
                    <span className="text-xs font-medium text-stone-500">DharmaMind</span>
                  </div>
                  <div className="flex items-center text-sm">
                    <div className="flex space-x-1 mr-3">
                      <div className="w-2 h-2 bg-stone-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-stone-400 rounded-full animate-bounce delay-100"></div>
                      <div className="w-2 h-2 bg-stone-400 rounded-full animate-bounce delay-200"></div>
                    </div>
                    <span className="text-stone-500">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Voice Input Overlay */}
      {showVoiceInput && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-1000">
          <div className="bg-white rounded-2xl p-6 m-4 max-w-sm w-full">
            <div className="text-center mb-4">
              <h3 className="text-lg font-bold text-stone-800 mb-2">Voice Input</h3>
              <p className="text-sm text-stone-600">Tap the microphone and speak your question</p>
            </div>
            
            <div className="flex justify-center mb-4">
              <VoiceInput
                onTranscript={handleVoiceTranscript}
                onError={handleVoiceError}
                className="voice-input-large"
              />
            </div>
            
            <div className="flex space-x-3">
              <button
                onClick={() => setShowVoiceInput(false)}
                className="flex-1 py-2 px-4 bg-stone-100 text-stone-700 rounded-lg font-medium"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="chat-input-area">
        <div className="chat-input-container">
          <textarea
            ref={inputRef}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask for spiritual guidance..."
            className="chat-input"
            rows={1}
            disabled={isSending}
          />
          
          {isVoiceSupported && (
            <button
              onClick={() => setShowVoiceInput(true)}
              className="voice-input-btn"
              disabled={isSending}
              aria-label="Voice input"
            >
              🎙️
            </button>
          )}
          
          <button
            onClick={handleSendMessage}
            disabled={!inputText.trim() || isSending}
            className="chat-send-btn"
            aria-label="Send message"
          >
            {isSending ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <span className="text-lg">→</span>
            )}
          </button>
        </div>
      </div>

      <style jsx>{`
        .mobile-chat-interface {
          display: flex;
          flex-direction: column;
          height: 100vh;
          background: linear-gradient(to bottom, #f7fafc, #edf2f7);
        }
        
        .voice-input-large {
          width: 80px;
          height: 80px;
          font-size: 2rem;
        }
        
        @media (max-width: 768px) {
          .mobile-chat-interface {
            height: 100vh;
            height: 100dvh; /* Dynamic viewport height for mobile */
          }
        }
        
        /* Custom scrollbar */
        .mobile-chat-interface ::-webkit-scrollbar {
          width: 4px;
        }
        
        .mobile-chat-interface ::-webkit-scrollbar-track {
          background: transparent;
        }
        
        .mobile-chat-interface ::-webkit-scrollbar-thumb {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 2px;
        }
        
        .mobile-chat-interface ::-webkit-scrollbar-thumb:hover {
          background: rgba(0, 0, 0, 0.3);
        }
      `}</style>
      
      {/* Subscription Modal */}
      <CentralizedSubscriptionModal
        isOpen={showSubscriptionModal}
        onClose={() => setShowSubscriptionModal(false)}
      />
    </div>
  );
};

export default MobileChatInterface;
