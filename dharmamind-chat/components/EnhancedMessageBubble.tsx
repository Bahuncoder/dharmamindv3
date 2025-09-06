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
  DocumentDuplicateIcon
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

interface EnhancedMessageBubbleProps {
  message: Message;
  isHovered: boolean;
  onHover: (hovered: boolean) => void;
  onCopy: (content: string) => void;
  onRegenerate?: (messageId: string) => void;
  onToggleFavorite: (messageId: string) => void;
  onToggleSaved: (messageId: string) => void;
  onSpeak: (content: string, messageId: string) => void;
  onShare: (content: string) => void;
  onReact: (messageId: string, reaction: string) => void;
  isPlaying?: boolean;
}

const EnhancedMessageBubble: React.FC<EnhancedMessageBubbleProps> = ({
  message,
  isHovered,
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
  const [expandedView, setExpandedView] = useState(false);
  const bubbleRef = useRef<HTMLDivElement>(null);

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

  const reactionEmojis = [
    { emoji: '🙏', name: 'gratitude' },
    { emoji: '💡', name: 'insight' },
    { emoji: '❤️', name: 'love' },
    { emoji: '✨', name: 'wisdom' },
    { emoji: '🕉️', name: 'peace' },
    { emoji: '🌸', name: 'beauty' }
  ];

  return (
    <motion.div
      ref={bubbleRef}
      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} group`}
      onMouseEnter={() => {
        onHover(true);
        setShowActions(true);
      }}
      onMouseLeave={() => {
        onHover(false);
        setShowActions(false);
        setShowReactions(false);
      }}
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, type: "spring" }}
    >
      <div className={`max-w-[85%] md:max-w-[70%] ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
        {/* Avatar for AI messages */}
        {message.role === 'assistant' && (
          <motion.div 
            className="flex items-center justify-center w-8 h-8 mb-2 rounded-full text-white shadow-lg"
            style={{
              backgroundColor: 'var(--color-border-primary)',
              border: `2px solid var(--color-border-primary)`
            }}
            whileHover={{ scale: 1.1 }}
            transition={{ type: "spring", stiffness: 400 }}
          >
            <SparklesIcon className="w-4 h-4" />
          </motion.div>
        )}

        {/* Message Bubble */}
        <motion.div
          className={`relative rounded-2xl px-4 py-3 shadow-lg ${
            message.role === 'user'
              ? 'text-gray-800 ml-4'
              : 'text-gray-800 mr-4'
          } ${expandedView ? 'transform scale-105' : ''}`}
          style={{
            backgroundColor: message.role === 'user' ? 'var(--color-background)' : 'var(--color-background)',
            border: `1px solid var(--color-border-primary)`,
            backdropFilter: 'blur(8px)'
          }}
          whileHover={{ 
            scale: message.role === 'assistant' ? 1.02 : 1.01,
            boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"
          }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          {/* Content */}
          <div className="relative z-10">
            {message.role === 'assistant' ? (
              <div className="prose prose-sm max-w-none prose-gray dark:prose-invert">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
                    ul: ({ children }) => <ul className="ml-4 mb-2 space-y-1">{children}</ul>,
                    ol: ({ children }) => <ol className="ml-4 mb-2 space-y-1">{children}</ol>,
                    li: ({ children }) => <li className="text-sm">{children}</li>,
                    strong: ({ children }) => (
                      <strong 
                        className="font-semibold"
                        style={{ color: 'var(--color-primary)' }}
                      >
                        {children}
                      </strong>
                    ),
                    em: ({ children }) => (
                      <em 
                        className="italic"
                        style={{ color: 'var(--color-primary)' }}
                      >
                        {children}
                      </em>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 pl-4 italic py-2 rounded-r-lg" style={{ 
                        borderLeftColor: 'var(--color-border-primary)', 
                        backgroundColor: 'var(--color-background-secondary)' 
                      }}>
                        {children}
                      </blockquote>
                    ),
                    code: ({ children }) => (
                      <code 
                        className="px-2 py-1 rounded text-xs font-mono"
                        style={{ 
                          backgroundColor: 'var(--color-background-secondary)',
                          color: 'var(--color-primary)'
                        }}
                      >
                        {children}
                      </code>
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="leading-relaxed">{message.content}</p>
            )}
          </div>

          {/* Message metadata for AI responses */}
          {message.role === 'assistant' && (
            <div className="flex items-center justify-between mt-3 pt-2 border-t border-gray-200/50">
              <div className="flex items-center space-x-3 text-xs">
                {message.confidence !== undefined && (
                  <div className="flex items-center space-x-1">
                    <span className="text-gray-500">Confidence:</span>
                    <span className={getAlignmentColor(message.confidence)}>
                      {getConfidenceIndicator(message.confidence)}
                    </span>
                    <span className="text-gray-600">
                      {Math.round((message.confidence || 0) * 100)}%
                    </span>
                  </div>
                )}
                {message.dharmic_alignment !== undefined && (
                  <div className="flex items-center space-x-1">
                    <span className="text-gray-500">Dharmic:</span>
                    <span className={getAlignmentColor(message.dharmic_alignment)}>
                      🕉️
                    </span>
                    <span className="text-gray-600">
                      {Math.round((message.dharmic_alignment || 0) * 100)}%
                    </span>
                  </div>
                )}
              </div>
              
              <div className="text-xs text-gray-500">
                {formatTimestamp(message.timestamp)}
              </div>
            </div>
          )}

          {/* User message timestamp */}
          {message.role === 'user' && (
            <div className="text-xs mt-2 text-right opacity-75" style={{ color: 'var(--color-text-secondary)' }}>
              {formatTimestamp(message.timestamp)}
            </div>
          )}

          {/* Modules used tags */}
          {message.modules_used && message.modules_used.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {message.modules_used.map((module, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium capitalize"
                  style={{
                    backgroundColor: 'var(--color-background-secondary)',
                    color: 'var(--color-text-secondary)',
                    border: `1px solid var(--color-border-primary)`
                  }}
                >
                  {module}
                </span>
              ))}
            </div>
          )}
        </motion.div>

        {/* Action Buttons */}
        <AnimatePresence>
          {showActions && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.9 }}
              transition={{ duration: 0.2 }}
              className={`flex items-center space-x-1 mt-2 ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {/* Copy Button */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleCopy}
                className="p-2 rounded-full backdrop-blur-sm border transition-all duration-200 shadow-sm"
                style={{
                  backgroundColor: 'var(--color-background)',
                  borderColor: 'var(--color-border-primary)',
                  color: 'var(--color-text-secondary)'
                }}
                title={copied ? 'Copied!' : 'Copy message'}
              >
                {copied ? (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="w-4 h-4"
                    style={{ color: 'var(--color-border-primary)' }}
                  >
                    ✓
                  </motion.div>
                ) : (
                  <ClipboardDocumentIcon className="w-4 h-4" />
                )}
              </motion.button>

              {/* Favorite Button */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => onToggleFavorite(message.id)}
                className="p-2 rounded-full backdrop-blur-sm border transition-all duration-200 shadow-sm"
                style={{
                  backgroundColor: 'var(--color-background)',
                  borderColor: 'var(--color-border-primary)',
                  color: 'var(--color-text-secondary)'
                }}
                title={message.isFavorite ? 'Remove from favorites' : 'Add to favorites'}
              >
                {message.isFavorite ? (
                  <StarIconSolid className="w-4 h-4 text-yellow-500" />
                ) : (
                  <StarIcon className="w-4 h-4" />
                )}
              </motion.button>

              {/* Save Button */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => onToggleSaved(message.id)}
                className="p-2 rounded-full backdrop-blur-sm border transition-all duration-200 shadow-sm"
                style={{
                  backgroundColor: 'var(--color-background)',
                  borderColor: 'var(--color-border-primary)',
                  color: 'var(--color-text-secondary)'
                }}
                title={message.isSaved ? 'Remove from saved' : 'Save message'}
              >
                {message.isSaved ? (
                  <BookmarkIconSolid className="w-4 h-4 text-blue-600" />
                ) : (
                  <BookmarkIcon className="w-4 h-4" />
                )}
              </motion.button>

              {/* Speak Button */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => onSpeak(message.content, message.id)}
                className="p-2 rounded-full backdrop-blur-sm border transition-all duration-200 shadow-sm"
                style={{
                  backgroundColor: 'var(--color-background)',
                  borderColor: 'var(--color-border-primary)',
                  color: 'var(--color-text-secondary)'
                }}
                title={isPlaying ? 'Stop speaking' : 'Read aloud'}
              >
                {isPlaying ? (
                  <SpeakerXMarkIcon className="w-4 h-4 text-purple-600" />
                ) : (
                  <SpeakerWaveIcon className="w-4 h-4" />
                )}
              </motion.button>

              {/* Regenerate Button (AI messages only) */}
              {message.role === 'assistant' && onRegenerate && (
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => onRegenerate(message.id)}
                  className="p-2 rounded-full backdrop-blur-sm border transition-all duration-200 shadow-sm"
                  style={{
                    backgroundColor: 'var(--color-background)',
                    borderColor: 'var(--color-border-primary)',
                    color: 'var(--color-text-secondary)'
                  }}
                  title="Regenerate response"
                >
                  <ArrowPathIcon className="w-4 h-4" />
                </motion.button>
              )}

              {/* Share Button */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => onShare(message.content)}
                className="p-2 rounded-full backdrop-blur-sm border transition-all duration-200 shadow-sm"
                style={{
                  backgroundColor: 'var(--color-background)',
                  borderColor: 'var(--color-border-primary)',
                  color: 'var(--color-text-secondary)'
                }}
                title="Share message"
              >
                <ShareIcon className="w-4 h-4" />
              </motion.button>

              {/* Reactions Button */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowReactions(!showReactions)}
                className="p-2 rounded-full bg-white/80 backdrop-blur-sm border border-gray-200/50 text-gray-600 hover:text-pink-600 hover:border-pink-300 transition-all duration-200 shadow-sm"
                title="React to message"
              >
                <HeartIcon className="w-4 h-4" />
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Reaction Picker */}
        <AnimatePresence>
          {showReactions && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.9 }}
              transition={{ duration: 0.2 }}
              className={`flex items-center space-x-2 mt-2 p-2 bg-white/90 backdrop-blur-sm border border-gray-200/50 rounded-xl shadow-lg ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {reactionEmojis.map((reaction) => (
                <motion.button
                  key={reaction.name}
                  whileHover={{ scale: 1.2 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => {
                    onReact(message.id, reaction.name);
                    setShowReactions(false);
                  }}
                  className="text-lg hover:bg-gray-100 rounded-full p-1 transition-all duration-200"
                  title={`React with ${reaction.name}`}
                >
                  {reaction.emoji}
                </motion.button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default EnhancedMessageBubble;
