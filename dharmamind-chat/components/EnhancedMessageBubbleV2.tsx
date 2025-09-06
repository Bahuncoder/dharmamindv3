import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ClipboardDocumentIcon,
  ArrowPathIcon,
  StarIcon,
  BookmarkIcon,
  ShareIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  HandThumbUpIcon,
  HandThumbDownIcon,
  HeartIcon,
  SparklesIcon,
  EyeIcon,
  CheckIcon
} from '@heroicons/react/24/outline';
import {
  StarIcon as StarIconSolid,
  BookmarkIcon as BookmarkIconSolid,
  HeartIcon as HeartIconSolid,
  HandThumbUpIcon as HandThumbUpIconSolid
} from '@heroicons/react/24/solid';

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

interface EnhancedMessageBubbleV2Props {
  message: Message;
  isHovered?: boolean;
  onHover?: (hovered: boolean) => void;
  onCopy: (content: string) => void;
  onRegenerate?: (messageId: string) => void;
  onToggleFavorite: (messageId: string) => void;
  onToggleSaved: (messageId: string) => void;
  onSpeak?: (content: string, messageId: string) => void;
  onShare?: (content: string) => void;
  onReact: (messageId: string, reaction: string) => void;
  isPlaying?: boolean;
}

const EnhancedMessageBubbleV2: React.FC<EnhancedMessageBubbleV2Props> = ({
  message,
  isHovered = false,
  onHover,
  onCopy,
  onRegenerate,
  onToggleFavorite,
  onToggleSaved,
  onSpeak,
  onShare,
  onReact,
  isPlaying = false
}) => {
  const [showActions, setShowActions] = useState(false);
  const [showReactions, setShowReactions] = useState(false);
  const [copied, setCopied] = useState(false);
  const [localHovered, setLocalHovered] = useState(false);
  const bubbleRef = useRef<HTMLDivElement>(null);

  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  const reactionEmojis = ['👍', '❤️', '🤔', '💯', '🙏', '✨'];

  useEffect(() => {
    if (copied) {
      const timer = setTimeout(() => setCopied(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [copied]);

  const handleCopy = async () => {
    await onCopy(message.content);
    setCopied(true);
  };

  const handleMouseEnter = () => {
    setLocalHovered(true);
    setShowActions(true);
    onHover?.(true);
  };

  const handleMouseLeave = () => {
    setLocalHovered(false);
    setShowActions(false);
    setShowReactions(false);
    onHover?.(false);
  };

  const getAlignmentColor = (alignment?: number) => {
    if (!alignment) return 'text-gray-400';
    if (alignment >= 0.9) return 'text-primary';
    if (alignment >= 0.7) return 'text-primary opacity-80';
    if (alignment >= 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceIndicator = (confidence?: number) => {
    if (!confidence) return '🤔';
    if (confidence >= 0.9) return '🧘‍♂️';
    if (confidence >= 0.7) return '🕉️';
    if (confidence >= 0.5) return '🌸';
    return '🤔';
  };

  const formatTimestamp = (timestamp: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true
    }).format(timestamp);
  };

  return (
    <motion.div
      ref={bubbleRef}
      className={`enhanced-message-container ${isUser ? 'user-message' : 'assistant-message'}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      <div className="message-layout">
        {/* Avatar */}
        <motion.div
          className={`enhanced-avatar ${isUser ? 'avatar-user' : 'avatar-ai'}`}
          whileHover={{ scale: 1.05 }}
          transition={{ duration: 0.2 }}
        >
          {isUser ? (
            <span className="avatar-text">U</span>
          ) : (
            <span className="avatar-text">AI</span>
          )}
          
          {/* Status Indicator */}
          <div className={`status-indicator ${isAssistant && message.confidence ? 'confident' : ''}`} />
        </motion.div>

        {/* Message Content */}
        <div className="message-content-wrapper">
          <motion.div
            className={`enhanced-message-bubble ${isUser ? 'message-user-enhanced' : 'message-ai-enhanced'}`}
            whileHover={{ y: -2 }}
            transition={{ duration: 0.2 }}
          >
            {/* Message Text */}
            <div className="message-text">
              {isUser ? (
                <p className="user-message-text">{message.content}</p>
              ) : (
                <div className="ai-message-markdown">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                      ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                      ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                      code: ({ children }) => (
                        <code className="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono">
                          {children}
                        </code>
                      ),
                      pre: ({ children }) => (
                        <pre className="bg-gray-50 p-3 rounded-lg overflow-x-auto mb-2">
                          {children}
                        </pre>
                      ),
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-4 border-blue-300 pl-4 italic text-gray-700 mb-2">
                          {children}
                        </blockquote>
                      ),
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
              )}
            </div>

            {/* AI Message Metadata */}
            {isAssistant && (message.confidence || message.dharmic_alignment) && (
              <div className="enhanced-message-metadata">
                <div className="metadata-left">
                  {message.confidence && (
                    <div className="confidence-badge">
                      <span className="indicator-emoji">{getConfidenceIndicator(message.confidence)}</span>
                      <span className="indicator-text">
                        {Math.round(message.confidence * 100)}% confident
                      </span>
                    </div>
                  )}
                </div>
                
                <div className="metadata-right">
                  {message.dharmic_alignment && (
                    <div className={`alignment-indicator ${getAlignmentColor(message.dharmic_alignment)}`}>
                      <SparklesIcon className="w-3 h-3" />
                      <span>Dharmic: {Math.round(message.dharmic_alignment * 100)}%</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Modules Used */}
            {isAssistant && message.modules_used && message.modules_used.length > 0 && (
              <div className="modules-container">
                <div className="modules-title">
                  <EyeIcon className="w-3 h-3" />
                  <span>Wisdom Sources</span>
                </div>
                <div className="modules-list">
                  {message.modules_used.map((module, index) => (
                    <span key={index} className="module-badge">
                      {module}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </motion.div>

          {/* Message Actions */}
          <AnimatePresence>
            {showActions && (
              <motion.div
                className="enhanced-message-actions"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                transition={{ duration: 0.2 }}
              >
                <div className="actions-row">
                  {/* Copy Button */}
                  <motion.button
                    className="enhanced-action-button"
                    onClick={handleCopy}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    title="Copy message"
                  >
                    {copied ? (
                      <CheckIcon className="w-4 h-4 text-green-500" />
                    ) : (
                      <ClipboardDocumentIcon className="w-4 h-4" />
                    )}
                  </motion.button>

                  {/* Regenerate Button (AI messages only) */}
                  {isAssistant && onRegenerate && (
                    <motion.button
                      className="enhanced-action-button"
                      onClick={() => onRegenerate(message.id)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      title="Regenerate response"
                    >
                      <ArrowPathIcon className="w-4 h-4" />
                    </motion.button>
                  )}

                  {/* Favorite Button */}
                  <motion.button
                    className={`enhanced-action-button ${message.isFavorite ? 'active' : ''}`}
                    onClick={() => onToggleFavorite(message.id)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    title={message.isFavorite ? "Remove from favorites" : "Add to favorites"}
                  >
                    {message.isFavorite ? (
                      <StarIconSolid className="w-4 h-4 text-yellow-400" />
                    ) : (
                      <StarIcon className="w-4 h-4" />
                    )}
                  </motion.button>

                  {/* Save Button */}
                  <motion.button
                    className={`enhanced-action-button ${message.isSaved ? 'active' : ''}`}
                    onClick={() => onToggleSaved(message.id)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    title={message.isSaved ? "Remove from saved" : "Save message"}
                  >
                    {message.isSaved ? (
                      <BookmarkIconSolid className="w-4 h-4 text-blue-500" />
                    ) : (
                      <BookmarkIcon className="w-4 h-4" />
                    )}
                  </motion.button>

                  {/* Speak Button */}
                  {onSpeak && (
                    <motion.button
                      className="enhanced-action-button"
                      onClick={() => onSpeak(message.content, message.id)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      title={isPlaying ? "Stop speaking" : "Read aloud"}
                    >
                      {isPlaying ? (
                        <SpeakerXMarkIcon className="w-4 h-4" />
                      ) : (
                        <SpeakerWaveIcon className="w-4 h-4" />
                      )}
                    </motion.button>
                  )}

                  {/* Reactions Button */}
                  <motion.button
                    className="enhanced-action-button"
                    onClick={() => setShowReactions(!showReactions)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    title="React to message"
                  >
                    <HeartIcon className="w-4 h-4" />
                  </motion.button>

                  {/* Share Button */}
                  {onShare && (
                    <motion.button
                      className="enhanced-action-button"
                      onClick={() => onShare(message.content)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      title="Share message"
                    >
                      <ShareIcon className="w-4 h-4" />
                    </motion.button>
                  )}
                </div>

                {/* Timestamp */}
                <div className="enhanced-timestamp">
                  {formatTimestamp(message.timestamp)}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Reaction Picker */}
          <AnimatePresence>
            {showReactions && (
              <motion.div
                className="reaction-picker"
                initial={{ opacity: 0, scale: 0.8, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.8, y: 10 }}
                transition={{ duration: 0.2 }}
              >
                {reactionEmojis.map((emoji, index) => (
                  <motion.button
                    key={emoji}
                    className="reaction-button"
                    onClick={() => {
                      onReact(message.id, emoji);
                      setShowReactions(false);
                    }}
                    whileHover={{ scale: 1.2 }}
                    whileTap={{ scale: 0.9 }}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    {emoji}
                  </motion.button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
};

export default EnhancedMessageBubbleV2;
