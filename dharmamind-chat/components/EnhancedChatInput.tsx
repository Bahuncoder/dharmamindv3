import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PaperAirplaneIcon,
  MicrophoneIcon,
  StopIcon,
  PhotoIcon,
  FaceSmileIcon,
  PlusIcon,
  XMarkIcon
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
  isMobile?: boolean;
  reduceMotion?: boolean;
  isHighContrast?: boolean;
  onFileUpload?: (file: File) => void;
  supportedFileTypes?: string[];
  voiceRecordingTime?: number;
}

const EnhancedChatInput: React.FC<EnhancedChatInputProps> = ({
  value,
  onChange,
  onSend,
  onVoiceRecord,
  isLoading,
  isRecording = false,
  placeholder = "Share your thoughts or ask for guidance...",
  maxLength = 2000,
  showVoiceInput = true,
  showAttachments = false,
  showEmoji = false,
  isMobile = false,
  reduceMotion = false,
  isHighContrast = false,
  onFileUpload,
  supportedFileTypes = ['image/*', 'text/*', '.pdf'],
  voiceRecordingTime = 0
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const [showActions, setShowActions] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Enhanced auto-resize with better mobile support
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const maxHeight = isMobile ? 100 : 120;
      const newHeight = Math.min(textareaRef.current.scrollHeight, maxHeight);
      textareaRef.current.style.height = `${newHeight}px`;
    }
  }, [value, isMobile]);

  // Enhanced focus management
  useEffect(() => {
    if (isFocused && textareaRef.current && !isMobile) {
      textareaRef.current.focus();
    }
  }, [isFocused, isMobile]);

  // Mobile keyboard handling
  useEffect(() => {
    if (isMobile) {
      const handleViewportChange = () => {
        // Adjust input position when virtual keyboard appears
        if (textareaRef.current) {
          const element = textareaRef.current;
          if (document.activeElement === element) {
            setTimeout(() => {
              element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 100);
          }
        }
      };

      window.addEventListener('resize', handleViewportChange);
      return () => window.removeEventListener('resize', handleViewportChange);
    }
  }, [isMobile]);

  // Enhanced keyboard handling
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !isLoading) {
        onSend();
        // Haptic feedback on mobile
        if (isMobile && 'vibrate' in navigator) {
          navigator.vibrate(50);
        }
      }
    }
    
    // Additional keyboard shortcuts
    if (e.key === 'Escape') {
      setShowActions(false);
      textareaRef.current?.blur();
    }
  };

  const handleSend = () => {
    if (value.trim() && !isLoading) {
      onSend();
      // Haptic feedback on mobile
      if (isMobile && 'vibrate' in navigator) {
        navigator.vibrate(50);
      }
    }
  };

  // File drag and drop handlers
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (showAttachments) {
      setIsDragOver(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    
    if (showAttachments && onFileUpload) {
      const files = Array.from(e.dataTransfer.files);
      files.forEach(file => {
        if (supportedFileTypes.some(type => 
          type === file.type || type.includes('*') && file.type.startsWith(type.split('/')[0])
        )) {
          onFileUpload(file);
          setUploadedFiles(prev => [...prev, file]);
        }
      });
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    files.forEach(file => {
      if (onFileUpload) {
        onFileUpload(file);
        setUploadedFiles(prev => [...prev, file]);
      }
    });
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const formatRecordingTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const characterCount = value.length;
  const isNearLimit = characterCount > maxLength * 0.8;
  const isOverLimit = characterCount > maxLength;

  return (
    <div 
      className={`enhanced-input-container ${isMobile ? 'mobile-optimized' : ''} ${isHighContrast ? 'high-contrast' : ''}`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* File Upload Overlay */}
      <AnimatePresence>
        {isDragOver && (
          <motion.div
            className="absolute inset-0 z-10 flex items-center justify-center bg-emerald-50 dark:bg-emerald-900/20 border-2 border-dashed border-emerald-500 rounded-lg"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            role="region"
            aria-label="File drop zone"
          >
            <div className="text-center">
              <PhotoIcon className="w-12 h-12 mx-auto text-emerald-500 mb-2" />
              <p className="text-emerald-700 dark:text-emerald-300 font-medium">Drop files here to upload</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Uploaded Files Display */}
      {uploadedFiles.length > 0 && (
        <div className="mb-3" role="region" aria-label="Uploaded files">
          <div className="flex flex-wrap gap-2">
            {uploadedFiles.map((file, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full text-sm"
              >
                <span className="truncate max-w-32">{file.name}</span>
                <button
                  onClick={() => removeFile(index)}
                  className="text-gray-500 hover:text-red-500 transition-colors"
                  aria-label={`Remove ${file.name}`}
                >
                  <XMarkIcon className="w-4 h-4" />
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      <div className="relative">
        {/* Input Field Container */}
        <motion.div
          className={`enhanced-input-wrapper ${isFocused ? 'focused' : ''} ${isDragOver ? 'drag-over' : ''}`}
          animate={{
            scale: isFocused && !reduceMotion ? 1.02 : 1
          }}
          transition={{ duration: reduceMotion ? 0 : 0.2 }}
          style={{
            border: '2px solid var(--color-border-primary, #10b981)',
            boxShadow: isFocused 
              ? '0 0 0 3px var(--color-shadow-light, rgba(0, 0, 0, 0.1)), 0 8px 25px var(--color-shadow-medium, rgba(0, 0, 0, 0.1))'
              : '0 2px 10px var(--color-shadow-light, rgba(0, 0, 0, 0.05))'
          }}
          role="group"
          aria-label="Message input area"
        >
          {/* Action Buttons - Left Side */}
          <div className="input-actions-left" role="toolbar" aria-label="Input actions">
            <AnimatePresence>
              {(showActions || isMobile) && (
                <motion.div
                  className="flex items-center gap-2"
                  initial={{ opacity: 0, x: reduceMotion ? 0 : -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: reduceMotion ? 0 : -20 }}
                  transition={{ duration: reduceMotion ? 0 : 0.2 }}
                >
                  {showAttachments && (
                    <>
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className={`action-button ${isMobile ? 'mobile-touch' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
                        title="Attach file (images, documents)"
                        aria-label="Attach file"
                      >
                        <PhotoIcon className={`${isMobile ? 'w-6 h-6' : 'w-5 h-5'}`} />
                      </button>
                      <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept={supportedFileTypes.join(',')}
                        onChange={handleFileSelect}
                        className="hidden"
                        aria-label="File upload input"
                      />
                    </>
                  )}
                  
                  {showEmoji && (
                    <button
                      type="button"
                      className={`action-button ${isMobile ? 'mobile-touch' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
                      title="Add emoji"
                      aria-label="Add emoji"
                    >
                      <FaceSmileIcon className={`${isMobile ? 'w-6 h-6' : 'w-5 h-5'}`} />
                    </button>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            <button
              type="button"
              className={`action-button plus-button ${isMobile ? 'mobile-touch' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
              onClick={() => setShowActions(!showActions)}
              title="More options"
              aria-label="Toggle additional options"
              aria-expanded={showActions}
            >
              <motion.div
                animate={{ rotate: showActions && !reduceMotion ? 45 : 0 }}
                transition={{ duration: reduceMotion ? 0 : 0.2 }}
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
              onFocus={() => {
                setIsFocused(true);
                if (isMobile) {
                  setShowActions(true);
                }
              }}
              onBlur={() => setIsFocused(false)}
              placeholder={placeholder}
              className={`enhanced-textarea ${isMobile ? 'mobile-optimized' : ''} ${isHighContrast ? 'high-contrast' : ''}`}
              rows={1}
              disabled={isLoading}
              maxLength={maxLength}
              aria-label="Message input"
              aria-describedby="character-counter voice-status"
              aria-invalid={isOverLimit}
              autoComplete="off"
              autoCorrect="on"
              autoCapitalize="sentences"
              spellCheck="true"
            />
            
            {/* Character Counter */}
            <AnimatePresence>
              {(isNearLimit || isFocused) && (
                <motion.div
                  id="character-counter"
                  className={`character-counter ${isOverLimit ? 'over-limit' : isNearLimit ? 'near-limit' : ''}`}
                  initial={{ opacity: 0, y: reduceMotion ? 0 : 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: reduceMotion ? 0 : 10 }}
                  transition={{ duration: reduceMotion ? 0 : 0.2 }}
                  role="status"
                  aria-label={`${characterCount} of ${maxLength} characters used`}
                >
                  {characterCount}/{maxLength}
                  {isOverLimit && (
                    <span className="sr-only">Character limit exceeded</span>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Action Buttons - Right Side */}
          <div className="input-actions-right" role="toolbar" aria-label="Send and voice actions">
            {/* Voice Recording Button */}
            {showVoiceInput && (
              <motion.button
                type="button"
                className={`voice-button ${isRecording ? 'recording' : ''} ${isMobile ? 'mobile-touch' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
                onClick={() => {
                  onVoiceRecord?.();
                  // Haptic feedback on mobile
                  if (isMobile && 'vibrate' in navigator) {
                    navigator.vibrate(isRecording ? [100] : [50]);
                  }
                }}
                disabled={isLoading}
                whileHover={{ scale: !reduceMotion ? 1.05 : 1 }}
                whileTap={{ scale: !reduceMotion ? 0.95 : 1 }}
                title={isRecording ? `Stop recording (${formatRecordingTime(voiceRecordingTime)})` : "Record voice message"}
                aria-label={isRecording ? "Stop voice recording" : "Start voice recording"}
                aria-pressed={isRecording}
                aria-describedby="voice-status"
              >
                <AnimatePresence mode="wait">
                  {isRecording ? (
                    <motion.div
                      key="stop"
                      initial={{ opacity: 0, scale: reduceMotion ? 1 : 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: reduceMotion ? 1 : 0.8 }}
                      transition={{ duration: reduceMotion ? 0 : 0.2 }}
                    >
                      <StopIcon className={`${isMobile ? 'w-6 h-6' : 'w-5 h-5'}`} />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="mic"
                      initial={{ opacity: 0, scale: reduceMotion ? 1 : 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: reduceMotion ? 1 : 0.8 }}
                      transition={{ duration: reduceMotion ? 0 : 0.2 }}
                    >
                      <MicrophoneIcon className={`${isMobile ? 'w-6 h-6' : 'w-5 h-5'}`} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.button>
            )}

            {/* Send Button */}
            <motion.button
              type="button"
              className={`send-button ${value.trim() && !isLoading ? 'active' : 'inactive'} ${isMobile ? 'mobile-touch' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
              onClick={handleSend}
              disabled={!value.trim() || isLoading || isOverLimit}
              whileHover={{ scale: value.trim() && !isLoading && !reduceMotion ? 1.05 : 1 }}
              whileTap={{ scale: value.trim() && !isLoading && !reduceMotion ? 0.95 : 1 }}
              title={isOverLimit ? "Message too long" : isLoading ? "Sending..." : "Send message (Enter)"}
              aria-label={isLoading ? "Sending message" : "Send message"}
              aria-disabled={!value.trim() || isLoading || isOverLimit}
              aria-keyshortcuts="Enter"
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

        {/* Enhanced Recording Indicator */}
        <AnimatePresence>
          {isRecording && (
            <motion.div
              id="voice-status"
              className={`recording-indicator ${isMobile ? 'mobile-recording' : ''} ${isHighContrast ? 'high-contrast' : ''}`}
              initial={{ opacity: 0, y: reduceMotion ? 0 : 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: reduceMotion ? 0 : 10 }}
              transition={{ duration: reduceMotion ? 0 : 0.3 }}
              role="status"
              aria-live="polite"
              aria-label={`Recording voice message for ${formatRecordingTime(voiceRecordingTime)}`}
            >
              <div className="flex items-center gap-3">
                <div className="recording-animation">
                  <div className="recording-dot" />
                  <div className="recording-pulse" />
                </div>
                <div className="recording-info">
                  <span className="recording-text">Recording voice message...</span>
                  <span className="recording-time">{formatRecordingTime(voiceRecordingTime)}</span>
                </div>
                {isMobile && (
                  <motion.button
                    onClick={() => onVoiceRecord?.()}
                    className={`recording-stop-mobile ${isHighContrast ? 'high-contrast-button' : ''}`}
                    whileTap={{ scale: reduceMotion ? 1 : 0.95 }}
                    aria-label="Stop recording"
                  >
                    <StopIcon className="w-5 h-5" />
                  </motion.button>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default EnhancedChatInput;
