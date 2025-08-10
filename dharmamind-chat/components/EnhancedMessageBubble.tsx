import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ClipboardDocumentIcon,
  ArrowPathIcon,
  BookmarkIcon,
  ShareIcon,
  StarIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  HeartIcon,
  HandThumbUpIcon,
  FaceSmileIcon
} from '@heroicons/react/24/outline';

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
  reactions?: { type: string; count: number; userReacted: boolean }[];
}

interface EnhancedMessageBubbleProps {
  message: Message;
  isHovered: boolean;
  onHover: (hovered: boolean) => void;
  onCopy: (content: string) => void;
  onRegenerate?: (messageId: string) => void;
  onToggleFavorite: (messageId: string) => void;
  onToggleSaved: (messageId: string) => void;
  onShare: (content: string) => void;
  onSpeak: (content: string, messageId: string) => void;
  onReact: (messageId: string, reaction: string) => void;
  isPlaying?: boolean;
  className?: string;
}

const EnhancedMessageBubble: React.FC<EnhancedMessageBubbleProps> = ({
  message,
  isHovered,
  onHover,
  onCopy,
  onRegenerate,
  onToggleFavorite,
  onToggleSaved,
  onShare,
  onSpeak,
  onReact,
  isPlaying = false,
  className = ''
}) => {
  const [showReactions, setShowReactions] = useState(false);
  const bubbleRef = useRef<HTMLDivElement>(null);

  const getConfidenceColor = (confidence?: number) => {
    if (!confidence) return 'bg-gray-100 text-gray-600';
    if (confidence >= 0.8) return 'bg-green-100 text-green-700';
    if (confidence >= 0.6) return 'bg-emerald-100 text-emerald-700';
    return 'bg-red-100 text-red-700';
  };

  const getAlignmentColor = (alignment?: number) => {
    if (!alignment) return 'text-gray-400';
    if (alignment >= 0.8) return 'text-green-500';
    if (alignment >= 0.6) return 'text-emerald-400';
    return 'text-red-500';
  };

  const reactions = [
    { type: 'like', icon: HandThumbUpIcon, label: 'Helpful' },
    { type: 'love', icon: HeartIcon, label: 'Inspiring' },
    { type: 'smile', icon: FaceSmileIcon, label: 'Uplifting' }
  ];

  const isUser = message.role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} ${className}`}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
    >
      <div className="relative group max-w-[85%] sm:max-w-md lg:max-w-lg">
        {/* Message bubble */}
        <motion.div
          ref={bubbleRef}
          className={`
            relative px-4 py-3 rounded-2xl transition-all duration-200
            ${isUser
              ? 'bg-gradient-to-br from-blue-600 to-purple-600 text-white ml-auto rounded-br-md'
              : 'bg-white text-gray-800 border border-gray-200 shadow-sm rounded-bl-md'
            }
            ${isHovered ? 'shadow-lg transform -translate-y-0.5' : ''}
          `}
          whileHover={{ scale: 1.01 }}
        >
          {/* Assistant message header */}
          {!isUser && (
            <div className="flex items-center justify-between mb-2 text-xs">
              <div className="flex items-center space-x-2">
                {message.confidence && (
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(message.confidence)}`}>
                    {Math.round(message.confidence * 100)}% confident
                  </span>
                )}
                {message.dharmic_alignment && (
                  <span className={`${getAlignmentColor(message.dharmic_alignment)}`}>
                    ⚖️ {Math.round(message.dharmic_alignment * 100)}% aligned
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Message content */}
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>

          {/* Modules used */}
          {message.modules_used && message.modules_used.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {message.modules_used.map((module, index) => (
                <span
                  key={index}
                  className="inline-block bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded-full"
                >
                  {module}
                </span>
              ))}
            </div>
          )}

          {/* Timestamp */}
          <div className={`mt-2 text-xs opacity-70 ${isUser ? 'text-white' : 'text-gray-500'}`}>
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>

          {/* Message reactions */}
          {message.reactions && message.reactions.length > 0 && (
            <div className="mt-2 flex items-center space-x-1">
              {message.reactions.map((reaction) => (
                <button
                  key={reaction.type}
                  onClick={() => onReact(message.id, reaction.type)}
                  className={`
                    text-xs px-2 py-1 rounded-full border transition-colors
                    ${reaction.userReacted
                      ? 'bg-blue-100 border-blue-300 text-blue-700'
                      : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100'
                    }
                  `}
                >
                  {reaction.type} {reaction.count}
                </button>
              ))}
            </div>
          )}
        </motion.div>

        {/* Action buttons */}
        <AnimatePresence>
          {(isHovered || window.innerWidth <= 768) && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8, y: 5 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.8, y: 5 }}
              transition={{ duration: 0.2 }}
              className={`
                absolute flex items-center space-x-1 mt-2
                ${isUser ? 'right-0' : 'left-0'}
              `}
            >
              <div className="flex items-center space-x-1 bg-white rounded-full shadow-lg border border-gray-200 px-2 py-1">
                {/* Copy */}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => onCopy(message.content)}
                  className="p-1.5 hover:bg-gray-100 rounded-full transition-colors"
                  title="Copy message"
                >
                  <ClipboardDocumentIcon className="w-3.5 h-3.5 text-gray-600" />
                </motion.button>

                {/* Speak (Assistant messages only) */}
                {!isUser && (
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => onSpeak(message.content, message.id)}
                    className={`p-1.5 hover:bg-gray-100 rounded-full transition-colors ${isPlaying ? 'text-blue-600' : 'text-gray-600'}`}
                    title={isPlaying ? "Stop speaking" : "Speak message"}
                  >
                    {isPlaying ? (
                      <SpeakerXMarkIcon className="w-3.5 h-3.5" />
                    ) : (
                      <SpeakerWaveIcon className="w-3.5 h-3.5" />
                    )}
                  </motion.button>
                )}

                {/* Regenerate (Assistant messages only) */}
                {!isUser && onRegenerate && (
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => onRegenerate(message.id)}
                    className="p-1.5 hover:bg-gray-100 rounded-full transition-colors"
                    title="Regenerate response"
                  >
                    <ArrowPathIcon className="w-3.5 h-3.5 text-gray-600" />
                  </motion.button>
                )}

                {/* Favorite */}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => onToggleFavorite(message.id)}
                  className="p-1.5 hover:bg-gray-100 rounded-full transition-colors"
                  title={message.isFavorite ? "Remove from favorites" : "Add to favorites"}
                >
                  <StarIcon 
                    className={`w-3.5 h-3.5 ${message.isFavorite ? 'text-emerald-500 fill-current' : 'text-gray-600'}`} 
                  />
                </motion.button>

                {/* Save */}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => onToggleSaved(message.id)}
                  className="p-1.5 hover:bg-gray-100 rounded-full transition-colors"
                  title={message.isSaved ? "Remove from saved" : "Save message"}
                >
                  <BookmarkIcon 
                    className={`w-3.5 h-3.5 ${message.isSaved ? 'text-green-600 fill-current' : 'text-gray-600'}`} 
                  />
                </motion.button>

                {/* Share */}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => onShare(message.content)}
                  className="p-1.5 hover:bg-gray-100 rounded-full transition-colors"
                  title="Share message"
                >
                  <ShareIcon className="w-3.5 h-3.5 text-gray-600" />
                </motion.button>

                {/* Reactions toggle */}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setShowReactions(!showReactions)}
                  className="p-1.5 hover:bg-gray-100 rounded-full transition-colors"
                  title="Add reaction"
                >
                  <FaceSmileIcon className="w-3.5 h-3.5 text-gray-600" />
                </motion.button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Reaction picker */}
        <AnimatePresence>
          {showReactions && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8, y: -10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.8, y: -10 }}
              className={`
                absolute ${isUser ? 'right-0' : 'left-0'} mt-1 z-10
                bg-white rounded-lg shadow-lg border border-gray-200 p-2
                flex items-center space-x-2
              `}
            >
              {reactions.map((reaction) => {
                const IconComponent = reaction.icon;
                return (
                  <motion.button
                    key={reaction.type}
                    whileHover={{ scale: 1.2 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => {
                      onReact(message.id, reaction.type);
                      setShowReactions(false);
                    }}
                    className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                    title={reaction.label}
                  >
                    <IconComponent className="w-4 h-4 text-gray-600" />
                  </motion.button>
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default EnhancedMessageBubble;
