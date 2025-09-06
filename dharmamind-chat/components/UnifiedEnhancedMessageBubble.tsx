import React, { useState, useRef } from 'react';
import { 
  ClipboardDocumentIcon,
  ArrowPathIcon,
  BookmarkIcon,
  ShareIcon,
  StarIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import { 
  BookmarkIcon as BookmarkSolidIcon,
  StarIcon as StarSolidIcon,
} from '@heroicons/react/24/solid';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useColor } from '../contexts/ColorContext';

interface Message {
  id: string;
  content: string;
  role?: 'user' | 'assistant';
  sender?: 'user' | 'ai';
  timestamp: Date;
  confidence?: number;
  dharmic_alignment?: number;
  modules_used?: string[];
  isFavorite?: boolean;
  isSaved?: boolean;
}

interface UnifiedEnhancedMessageBubbleProps {
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

const UnifiedEnhancedMessageBubble: React.FC<UnifiedEnhancedMessageBubbleProps> = ({
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
  isPlaying = false,
}) => {
  const [showActions, setShowActions] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const bubbleRef = useRef<HTMLDivElement>(null);
  const colorContext = useColor();
  const colors = colorContext.currentTheme.colors;

  // Normalize the role field to handle both 'role' and 'sender' properties
  const getUserRole = (): 'user' | 'assistant' => {
    if (message.role) {
      return message.role;
    }
    if (message.sender) {
      return message.sender === 'user' ? 'user' : 'assistant';
    }
    return 'user'; // fallback
  };

  const userRole = getUserRole();

  const handleCopy = async () => {
    await onCopy(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSpeak = () => {
    if (onSpeak) {
      onSpeak(message.content, message.id);
    }
  };

  const handleShare = () => {
    if (onShare) {
      onShare(message.content);
    }
  };

  const reactions = ['👍', '❤️', '😊', '🤔', '🙏'];

  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const renderMessageContent = () => {
    return (
      <div className="prose prose-emerald dark:prose-invert max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ className, children, ...props }: any) {
              return (
                <code className={`${className} bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-sm`} {...props}>
                  {children}
                </code>
              );
            },
            p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
            ul: ({ children }) => <ul className="list-disc pl-6 mb-3">{children}</ul>,
            ol: ({ children }) => <ol className="list-decimal pl-6 mb-3">{children}</ol>,
            li: ({ children }) => <li className="mb-1">{children}</li>,
            blockquote: ({ children }) => (
              <blockquote className="border-l-4 border-emerald-500 pl-4 italic my-4 text-gray-600 dark:text-gray-300">
                {children}
              </blockquote>
            ),
            h1: ({ children }) => <h1 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">{children}</h1>,
            h2: ({ children }) => <h2 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">{children}</h2>,
            h3: ({ children }) => <h3 className="text-lg font-medium mb-2 text-gray-900 dark:text-white">{children}</h3>,
          }}
        >
          {message.content}
        </ReactMarkdown>
      </div>
    );
  };

  return (
    <motion.div
      ref={bubbleRef}
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ 
        duration: 0.3,
        ease: [0.4, 0, 0.2, 1]
      }}
      className={`group relative mb-6 ${
        userRole === 'user' 
          ? 'ml-auto max-w-[85%] sm:max-w-[75%]' 
          : 'mr-auto max-w-[95%] sm:max-w-[85%]'
      }`}
      onMouseEnter={() => {
        setShowActions(true);
        if (onHover) onHover(true);
      }}
      onMouseLeave={() => {
        setShowActions(false);
        if (onHover) onHover(false);
      }}
    >
      {/* Message Bubble */}
      <div
        className={`relative rounded-2xl px-4 py-3 shadow-lg transition-all duration-200 ${
          userRole === 'user'
            ? 'bg-gradient-to-r from-emerald-500 to-emerald-600 text-white ml-auto'
            : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-700'
        } ${showActions || isHovered ? 'shadow-xl scale-[1.02]' : ''}`}
        style={{
          background: userRole === 'user' 
            ? `linear-gradient(135deg, ${colors.primaryStart}, ${colors.primaryEnd})` 
            : undefined,
        }}
      >
        {/* AI Badge and Metadata */}
        {userRole === 'assistant' && (
          <div className="flex items-center gap-2 mb-2 text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
              <span>DharmaMind AI</span>
            </div>
            
            {message.confidence && (
              <div className="flex items-center gap-1">
                <ChartBarIcon className="w-3 h-3" />
                <span>{Math.round(message.confidence * 100)}%</span>
              </div>
            )}
            
            {message.dharmic_alignment && (
              <div className="flex items-center gap-1">
                <span className="text-emerald-600">🕉</span>
                <span>{Math.round(message.dharmic_alignment * 100)}%</span>
              </div>
            )}
          </div>
        )}

        {/* Message Content */}
        <div className="message-content">
          {renderMessageContent()}
        </div>

        {/* Modules Used */}
        {userRole === 'assistant' && message.modules_used && (
          <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
            <div className="text-xs text-gray-500 dark:text-gray-400">
              <span className="font-medium">Modules: </span>
              {message.modules_used.join(' • ')}
            </div>
          </div>
        )}

        {/* Timestamp */}
        <div className={`text-xs mt-2 ${
          userRole === 'user' 
            ? 'text-emerald-100' 
            : 'text-gray-500 dark:text-gray-400'
        }`}>
          {formatTimestamp(message.timestamp)}
        </div>

        {/* Status Indicators */}
        <div className="absolute -right-2 -bottom-1 flex gap-1">
          {message.isFavorite && (
            <div className="w-4 h-4 bg-yellow-500 rounded-full flex items-center justify-center">
              <StarSolidIcon className="w-2 h-2 text-white" />
            </div>
          )}
          {message.isSaved && (
            <div className="w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center">
              <BookmarkSolidIcon className="w-2 h-2 text-white" />
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <AnimatePresence>
        {(showActions || isHovered) && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.2 }}
            className={`absolute top-0 ${
              userRole === 'user' ? '-left-12' : '-right-12'
            } flex flex-col gap-1`}
          >
            {/* Copy Button */}
            <button
              onClick={handleCopy}
              className="p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group"
              title="Copy message"
            >
              <ClipboardDocumentIcon className="w-4 h-4 text-gray-600 dark:text-gray-400 group-hover:text-emerald-600" />
            </button>

            {/* Favorite Button */}
            <button
              onClick={() => onToggleFavorite(message.id)}
              className="p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group"
              title={message.isFavorite ? "Remove from favorites" : "Add to favorites"}
            >
              {message.isFavorite ? (
                <StarSolidIcon className="w-4 h-4 text-yellow-500" />
              ) : (
                <StarIcon className="w-4 h-4 text-gray-600 dark:text-gray-400 group-hover:text-yellow-500" />
              )}
            </button>

            {/* Save Button */}
            <button
              onClick={() => onToggleSaved(message.id)}
              className="p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group"
              title={message.isSaved ? "Remove from saved" : "Save message"}
            >
              {message.isSaved ? (
                <BookmarkSolidIcon className="w-4 h-4 text-blue-500" />
              ) : (
                <BookmarkIcon className="w-4 h-4 text-gray-600 dark:text-gray-400 group-hover:text-blue-500" />
              )}
            </button>

            {/* Speak Button */}
            {onSpeak && (
              <button
                onClick={handleSpeak}
                className="p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group"
                title={isPlaying ? "Stop speaking" : "Speak message"}
              >
                {isPlaying ? (
                  <SpeakerXMarkIcon className="w-4 h-4 text-red-500" />
                ) : (
                  <SpeakerWaveIcon className="w-4 h-4 text-gray-600 dark:text-gray-400 group-hover:text-emerald-600" />
                )}
              </button>
            )}

            {/* Share Button */}
            {onShare && (
              <button
                onClick={handleShare}
                className="p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group"
                title="Share message"
              >
                <ShareIcon className="w-4 h-4 text-gray-600 dark:text-gray-400 group-hover:text-emerald-600" />
              </button>
            )}

            {/* Regenerate Button (AI messages only) */}
            {userRole === 'assistant' && onRegenerate && (
              <button
                onClick={() => onRegenerate(message.id)}
                className="p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group"
                title="Regenerate response"
              >
                <ArrowPathIcon className="w-4 h-4 text-gray-600 dark:text-gray-400 group-hover:text-emerald-600" />
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Reaction Bar */}
      <AnimatePresence>
        {(showActions || isHovered) && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.2, delay: 0.1 }}
            className={`mt-2 flex gap-1 ${
              userRole === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            {reactions.map((reaction) => (
              <button
                key={reaction}
                onClick={() => onReact(message.id, reaction)}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors text-sm"
                title={`React with ${reaction}`}
              >
                {reaction}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Copy Success Notification */}
      <AnimatePresence>
        {copied && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full mb-2 px-3 py-1 bg-emerald-600 text-white text-sm rounded-lg shadow-lg"
          >
            Copied!
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default UnifiedEnhancedMessageBubble;
