import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PaperAirplaneIcon,
  MicrophoneIcon,
  StopIcon,
  PhotoIcon,
  FaceSmileIcon,
  PlusIcon
} from '@heroicons/react/24/outline';

interface EnhancedChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onVoiceRecord?: () => void;
  isLoading: boolean;
  isRecording?: boolean;
  placeholder?: string;
  maxLength?: number;
  showVoiceInput?: boolean;
  showAttachments?: boolean;
  showEmoji?: boolean;
}

const EnhancedChatInput: React.FC<EnhancedChatInputProps> = ({
  value,
  onChange,
  onSend,
  onVoiceRecord,
  isLoading,
  isRecording = false,
  placeholder = "Type your message...",
  maxLength = 2000,
  showVoiceInput = true,
  showAttachments = false,
  showEmoji = false
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const [showActions, setShowActions] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [value]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !isLoading) {
        onSend();
      }
    }
  };

  const handleSend = () => {
    if (value.trim() && !isLoading) {
      onSend();
    }
  };

  const characterCount = value.length;
  const isNearLimit = characterCount > maxLength * 0.8;
  const isOverLimit = characterCount > maxLength;

  return (
    <div className="enhanced-input-container">
      <div className="relative">
        {/* Input Field Container */}
        <motion.div
          className={`enhanced-input-wrapper ${isFocused ? 'focused' : ''}`}
          animate={{
            scale: isFocused ? 1.02 : 1,
            boxShadow: isFocused 
              ? '0 0 0 3px rgba(102, 126, 234, 0.1), 0 8px 25px rgba(0, 0, 0, 0.1)'
              : '0 2px 10px rgba(0, 0, 0, 0.05)'
          }}
          transition={{ duration: 0.2 }}
        >
          {/* Action Buttons - Left Side */}
          <div className="input-actions-left">
            <AnimatePresence>
              {showActions && (
                <motion.div
                  className="flex items-center gap-2"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.2 }}
                >
                  {showAttachments && (
                    <button
                      type="button"
                      className="action-button"
                      title="Attach file"
                    >
                      <PhotoIcon className="w-5 h-5" />
                    </button>
                  )}
                  
                  {showEmoji && (
                    <button
                      type="button"
                      className="action-button"
                      title="Add emoji"
                    >
                      <FaceSmileIcon className="w-5 h-5" />
                    </button>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            <button
              type="button"
              className="action-button plus-button"
              onClick={() => setShowActions(!showActions)}
              title="More options"
            >
              <motion.div
                animate={{ rotate: showActions ? 45 : 0 }}
                transition={{ duration: 0.2 }}
              >
                <PlusIcon className="w-5 h-5" />
              </motion.div>
            </button>
          </div>

          {/* Text Input */}
          <div className="input-field-container">
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => onChange(e.target.value)}
              onKeyPress={handleKeyPress}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder={placeholder}
              className="enhanced-textarea"
              rows={1}
              disabled={isLoading}
              maxLength={maxLength}
            />
            
            {/* Character Counter */}
            <AnimatePresence>
              {(isNearLimit || isFocused) && (
                <motion.div
                  className={`character-counter ${isOverLimit ? 'over-limit' : isNearLimit ? 'near-limit' : ''}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  transition={{ duration: 0.2 }}
                >
                  {characterCount}/{maxLength}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Action Buttons - Right Side */}
          <div className="input-actions-right">
            {/* Voice Recording Button */}
            {showVoiceInput && (
              <motion.button
                type="button"
                className={`voice-button ${isRecording ? 'recording' : ''}`}
                onClick={onVoiceRecord}
                disabled={isLoading}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title={isRecording ? "Stop recording" : "Voice message"}
              >
                <AnimatePresence mode="wait">
                  {isRecording ? (
                    <motion.div
                      key="stop"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      transition={{ duration: 0.2 }}
                    >
                      <StopIcon className="w-5 h-5" />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="mic"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      transition={{ duration: 0.2 }}
                    >
                      <MicrophoneIcon className="w-5 h-5" />
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.button>
            )}

            {/* Send Button */}
            <motion.button
              type="button"
              className={`send-button ${value.trim() && !isLoading ? 'active' : 'inactive'}`}
              onClick={handleSend}
              disabled={!value.trim() || isLoading || isOverLimit}
              whileHover={{ scale: value.trim() && !isLoading ? 1.05 : 1 }}
              whileTap={{ scale: value.trim() && !isLoading ? 0.95 : 1 }}
              title="Send message"
            >
              <AnimatePresence mode="wait">
                {isLoading ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="loading-spinner"
                  >
                    <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 0 1 4 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                  </motion.div>
                ) : (
                  <motion.div
                    key="send"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    transition={{ duration: 0.2 }}
                  >
                    <PaperAirplaneIcon className="w-5 h-5" />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.button>
          </div>
        </motion.div>

        {/* Recording Indicator */}
        <AnimatePresence>
          {isRecording && (
            <motion.div
              className="recording-indicator"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              transition={{ duration: 0.3 }}
            >
              <div className="recording-dot" />
              <span>Recording...</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default EnhancedChatInput;
