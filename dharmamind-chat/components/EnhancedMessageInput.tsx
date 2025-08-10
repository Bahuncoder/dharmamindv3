import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  PaperAirplaneIcon,
  FaceSmileIcon,
  MicrophoneIcon,
  PhotoIcon,
  ChevronUpIcon
} from '@heroicons/react/24/outline';

interface EnhancedMessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  isLoading?: boolean;
  placeholder?: string;
  disabled?: boolean;
  showVoiceInput?: boolean;
  onVoiceInput?: () => void;
  className?: string;
}

const EnhancedMessageInput: React.FC<EnhancedMessageInputProps> = ({
  value,
  onChange,
  onSend,
  isLoading = false,
  placeholder = "Ask me anything about life, purpose, ethics, or personal growth...",
  disabled = false,
  showVoiceInput = true,
  onVoiceInput,
  className = ''
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxHeight = 120; // Max 5 lines approximately
      textareaRef.current.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
      
      // Expand container if content is more than 2 lines
      setIsExpanded(scrollHeight > 60);
    }
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !isLoading && !disabled) {
        onSend();
      }
    }
  };

  const handleSend = () => {
    if (value.trim() && !isLoading && !disabled) {
      onSend();
    }
  };

  const canSend = value.trim().length > 0 && !isLoading && !disabled;

  return (
    <motion.div
      ref={containerRef}
      className={`
        chat-input-container relative
        ${isFocused ? 'ring-2 ring-blue-500 ring-opacity-20' : ''}
        ${isExpanded ? 'rounded-2xl' : 'rounded-full'}
        ${className}
      `}
      animate={{
        borderRadius: isExpanded ? 16 : 24,
      }}
      transition={{ duration: 0.2 }}
    >
      {/* Input area */}
      <div className="flex items-end p-2">
        {/* Quick actions (left side) */}
        <div className="flex items-center space-x-2 px-2">
          {/* Voice input button */}
          {showVoiceInput && (
            <motion.button
              type="button"
              onClick={onVoiceInput}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title="Voice input"
            >
              <MicrophoneIcon className="w-5 h-5" />
            </motion.button>
          )}
        </div>

        {/* Text input */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder={placeholder}
            disabled={disabled}
            className={`
              w-full border-none outline-none resize-none
              text-gray-900 placeholder-gray-500
              bg-transparent px-3 py-2
              text-base leading-6
              ${isExpanded ? 'min-h-[48px]' : 'h-12'}
            `}
            style={{
              maxHeight: '120px'
            }}
          />
          
          {/* Character count for long messages */}
          <AnimatePresence>
            {value.length > 200 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="absolute bottom-2 right-2 text-xs text-gray-400 bg-white px-2 py-1 rounded-full shadow-sm"
              >
                {value.length}/500
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Send button */}
        <div className="flex items-center space-x-2 px-2">
          <motion.button
            type="button"
            onClick={handleSend}
            disabled={!canSend}
            className={`
              p-2 rounded-full flex items-center justify-center
              transition-all duration-200 ease-in-out
              ${canSend
                ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              }
            `}
            whileHover={canSend ? { scale: 1.05 } : {}}
            whileTap={canSend ? { scale: 0.95 } : {}}
            title={canSend ? "Send message" : "Type a message first"}
          >
            {isLoading ? (
              <motion.div
                className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              />
            ) : (
              <PaperAirplaneIcon className="w-5 h-5" />
            )}
          </motion.button>
        </div>
      </div>

      {/* Quick suggestions when focused and empty */}
      <AnimatePresence>
        {isFocused && value.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute bottom-full left-0 right-0 mb-2 bg-white rounded-lg shadow-lg border border-gray-200 p-3"
          >
            <div className="text-xs font-medium text-gray-700 mb-2">Quick suggestions:</div>
            <div className="flex flex-wrap gap-2">
              {[
                "What's my life purpose?",
                "Help me make an ethical decision",
                "How can I grow personally?",
                "I need guidance on..."
              ].map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => onChange(suggestion)}
                  className="text-xs px-3 py-1 bg-gray-50 hover:bg-gray-100 rounded-full text-gray-600 hover:text-gray-800 transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Typing indicator */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute top-full left-4 mt-2 text-xs text-gray-500 flex items-center space-x-2"
          >
            <div className="flex space-x-1">
              <motion.div
                className="w-2 h-2 bg-gray-400 rounded-full"
                animate={{ y: [0, -4, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
              />
              <motion.div
                className="w-2 h-2 bg-gray-400 rounded-full"
                animate={{ y: [0, -4, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0.1 }}
              />
              <motion.div
                className="w-2 h-2 bg-gray-400 rounded-full"
                animate={{ y: [0, -4, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
              />
            </div>
            <span>AI is thinking...</span>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default EnhancedMessageInput;
