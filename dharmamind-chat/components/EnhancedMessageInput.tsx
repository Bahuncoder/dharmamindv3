import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PaperAirplaneIcon,
  MicrophoneIcon,
  PhotoIcon,
  FaceSmileIcon,
  PaperClipIcon,
  SparklesIcon,
  XMarkIcon,
  DocumentIcon,
  MapIcon
} from '@heroicons/react/24/outline';
import VoiceInput from './VoiceInput';

interface EnhancedMessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  isLoading?: boolean;
  showVoiceInput?: boolean;
  showAttachments?: boolean;
  placeholder?: string;
  disabled?: boolean;
  maxLength?: number;
}

const EnhancedMessageInput: React.FC<EnhancedMessageInputProps> = ({
  value,
  onChange,
  onSend,
  isLoading = false,
  showVoiceInput = true,
  showAttachments = true,
  placeholder = "",
  disabled = false,
  maxLength = 2000
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const [showVoice, setShowVoice] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [textareaHeight, setTextareaHeight] = useState('auto');
  
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Common spiritual emojis
  const spiritualEmojis = [
    '🙏', '🕉️', '☮️', '🧘‍♂️', '🧘‍♀️', '🌸', '🪷', '✨', 
    '💫', '🌟', '🔮', '📿', '🕯️', '🌺', '🌼', '🦋', 
    '🌅', '🌄', '🌙', '⭐', '💖', '💙', '💜', '🤍'
  ];

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      const newHeight = Math.min(scrollHeight, 120);
      textareaRef.current.style.height = `${newHeight}px`;
      setTextareaHeight(`${newHeight}px`);
    }
  }, [value]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !isLoading && !disabled) {
        onSend();
      }
    }
  };

  const handleEmojiSelect = (emoji: string) => {
    onChange(value + emoji);
    setShowEmojiPicker(false);
    textareaRef.current?.focus();
  };

  const handleVoiceInput = (transcript: string) => {
    onChange(value + transcript);
    setShowVoice(false);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setAttachments(prev => [...prev, ...files]);
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    setAttachments(prev => [...prev, ...files]);
  };

  const getCharacterCount = () => {
    const count = value.length;
    const remaining = maxLength - count;
    const isNearLimit = remaining < 100;
    const isOverLimit = remaining < 0;
    
    return {
      count,
      remaining,
      isNearLimit,
      isOverLimit,
      percentage: (count / maxLength) * 100
    };
  };

  const characterInfo = getCharacterCount();

  // Function to get file type icon
  const getFileIcon = (filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'pdf':
        return <DocumentIcon className="w-4 h-4 text-red-500" />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'webp':
      case 'bmp':
      case 'svg':
        return <PhotoIcon className="w-4 h-4 text-success-500" />;
      case 'doc':
      case 'docx':
      case 'rtf':
      case 'odt':
        return <DocumentIcon className="w-4 h-4 text-neutral-1000" />;
      case 'xls':
      case 'xlsx':
      case 'csv':
        return <DocumentIcon className="w-4 h-4 text-gold-500" />;
      case 'ppt':
      case 'pptx':
        return <DocumentIcon className="w-4 h-4 text-orange-500" />;
      case 'mp4':
      case 'mov':
      case 'avi':
        return <DocumentIcon className="w-4 h-4 text-gold-500" />;
      case 'mp3':
      case 'wav':
        return <DocumentIcon className="w-4 h-4 text-indigo-500" />;
      case 'zip':
      case 'rar':
        return <DocumentIcon className="w-4 h-4 text-yellow-500" />;
      default:
        return <DocumentIcon className="w-4 h-4 text-gray-500" />;
    }
  };

  // Function to format file size
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="relative">
      {/* Attachments Preview */}
      <AnimatePresence>
        {attachments.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-3 p-3 bg-gray-50 rounded-xl border border-gray-200"
          >
            <div className="flex flex-wrap gap-2">
              {attachments.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center space-x-2 bg-white px-3 py-2 rounded-lg border border-gray-200 max-w-xs"
                >
                  {getFileIcon(file.name)}
                  <div className="flex flex-col min-w-0 flex-1">
                    <span className="text-sm text-gray-700 truncate">
                      {file.name}
                    </span>
                    <span className="text-xs text-gray-500">
                      {formatFileSize(file.size)}
                    </span>
                  </div>
                  <button
                    onClick={() => removeAttachment(index)}
                    className="text-gray-400 hover:text-red-500 transition-colors flex-shrink-0"
                    title="Remove file"
                  >
                    <XMarkIcon className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Input Container */}
      <div
        className={`relative rounded-2xl border-2 transition-all duration-300 ${
          isFocused
            ? 'shadow-lg'
            : ''
        } ${
          disabled ? 'opacity-50 cursor-not-allowed' : ''
        }`}
        style={{
          borderColor: 'var(--color-border-primary)',
          backgroundColor: 'var(--color-background)',
          ...(isDragOver && {
            backgroundColor: 'var(--color-background-secondary)'
          })
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Drag Overlay */}
        <AnimatePresence>
          {isDragOver && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 rounded-2xl border-2 border-dashed flex items-center justify-center z-10"
              style={{
                backgroundColor: 'var(--color-background-secondary)',
                borderColor: 'var(--color-border-primary)'
              }}
            >
              <div className="text-center" style={{ color: 'var(--color-primary)' }}>
                <DocumentIcon className="w-8 h-8 mx-auto mb-2" />
                <p className="text-sm font-medium">Drop files here</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="flex items-end p-4 space-x-3">
          {/* Attachment Button */}
          {showAttachments && (
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
              className="flex-shrink-0 p-2 rounded-xl transition-all duration-200 disabled:opacity-50"
              style={{
                backgroundColor: 'var(--color-background-secondary)',
                color: 'var(--color-text-secondary)',
                border: `1px solid var(--color-border-primary)`
              }}
              title="Attach files (PDF, Images, Documents)"
            >
              <PaperClipIcon className="w-5 h-5" />
            </motion.button>
          )}

          {/* Text Input */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => onChange(e.target.value)}
              onKeyDown={handleKeyPress}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder={placeholder}
              disabled={disabled || isLoading}
              maxLength={maxLength}
              className="enhanced-textarea"
              style={{ height: textareaHeight }}
            />
            
            {/* Character Counter */}
            {(characterInfo.isNearLimit || isFocused) && (
              <div className={`text-xs mt-1 text-right ${
                characterInfo.isOverLimit ? 'text-red-500' : 
                characterInfo.isNearLimit ? 'text-orange-500' : 'text-gray-400'
              }`}>
                {characterInfo.remaining < 0 ? `${Math.abs(characterInfo.remaining)} over limit` : `${characterInfo.remaining} left`}
              </div>
            )}
          </div>

          {/* Right Actions */}
          <div className="flex items-center space-x-2">
            {/* Emoji Button */}
            <div className="relative">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowEmojiPicker(!showEmojiPicker)}
                disabled={disabled}
                className="p-2 rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-600 hover:text-yellow-600 transition-all duration-200 disabled:opacity-50"
                title="Add emoji"
              >
                <FaceSmileIcon className="w-5 h-5" />
              </motion.button>

              {/* Emoji Picker */}
              <AnimatePresence>
                {showEmojiPicker && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9, y: 10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.9, y: 10 }}
                    className="absolute bottom-full right-0 mb-2 p-3 bg-white rounded-xl shadow-lg border border-gray-200 z-20"
                  >
                    <div className="grid grid-cols-6 gap-2 max-w-64">
                      {spiritualEmojis.map((emoji, index) => (
                        <button
                          key={index}
                          onClick={() => handleEmojiSelect(emoji)}
                          className="text-lg hover:bg-gray-100 rounded-lg p-2 transition-colors"
                        >
                          {emoji}
                        </button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Voice Input Button */}
            {showVoiceInput && (
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowVoice(!showVoice)}
                disabled={disabled}
                className="p-2 rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-600 hover:text-gold-600 transition-all duration-200 disabled:opacity-50"
                title="Voice input"
              >
                <MicrophoneIcon className="w-5 h-5" />
              </motion.button>
            )}

            {/* Send Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onSend}
              disabled={!value.trim() || isLoading || disabled || characterInfo.isOverLimit}
              className="p-3 rounded-xl text-white shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-lg"
              style={{
                backgroundColor: 'var(--color-border-primary)',
                border: `2px solid var(--color-border-primary)`
              }}
              title={isLoading ? 'Sending...' : 'Send message'}
            >
              {isLoading ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                />
              ) : (
                <PaperAirplaneIcon className="w-5 h-5" />
              )}
            </motion.button>
          </div>
        </div>

        {/* Floating Accent */}
        <motion.div
          className="absolute inset-0 rounded-2xl pointer-events-none"
          initial={{ opacity: 0 }}
          animate={{ 
            opacity: isFocused ? 0.3 : 0
          }}
          style={{
            background: 'var(--color-border-primary)'
          }}
          transition={{ duration: 0.3 }}
        />
      </div>

      {/* Voice Input Modal */}
      <AnimatePresence>
        {showVoice && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="bg-white rounded-2xl p-6 max-w-md w-full mx-4"
            >
              <div className="text-center">
                <button
                  onClick={() => setShowVoice(false)}
                  className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
                >
                  <XMarkIcon className="w-6 h-6" />
                </button>
                <VoiceInput
                  onTranscript={handleVoiceInput}
                  spiritualMode={true}
                  showVisualFeedback={true}
                />
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".txt,.pdf,.doc,.docx,.rtf,.odt,.jpg,.jpeg,.png,.gif,.webp,.bmp,.svg,.mp4,.mov,.avi,.mp3,.wav,.zip,.rar,.xls,.xlsx,.ppt,.pptx,.csv"
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  );
};

export default EnhancedMessageInput;
