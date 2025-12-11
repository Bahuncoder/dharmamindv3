import React, { useState, useRef, useEffect } from 'react';
import {
    ClipboardDocumentIcon,
    ArrowPathIcon,
    BookmarkIcon,
    ShareIcon,
    StarIcon,
    SpeakerWaveIcon,
    SpeakerXMarkIcon,
    ChartBarIcon,
    EyeIcon,
    CheckIcon
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
    isMobile?: boolean;
    reduceMotion?: boolean;
    isHighContrast?: boolean;
    messageIndex?: number;
    totalMessages?: number;
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
    isMobile = false,
    reduceMotion = false,
    isHighContrast = false,
    messageIndex = 0,
    totalMessages = 0
}) => {
    const [showActions, setShowActions] = useState(false);
    const [copied, setCopied] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const [isVisible, setIsVisible] = useState(false);
    const [hasBeenRead, setHasBeenRead] = useState(false);
    const bubbleRef = useRef<HTMLDivElement>(null);
    const colorContext = useColor();
    const colors = colorContext.currentTheme.colors;

    // Enhanced mobile detection and accessibility
    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                setIsVisible(entry.isIntersecting);
                if (entry.isIntersecting && !hasBeenRead) {
                    setHasBeenRead(true);
                    // Haptic feedback for mobile when message becomes visible
                    if (isMobile && 'vibrate' in navigator) {
                        navigator.vibrate(50);
                    }
                }
            },
            { threshold: 0.5 }
        );

        if (bubbleRef.current) {
            observer.observe(bubbleRef.current);
        }

        return () => observer.disconnect();
    }, [isMobile, hasBeenRead]);

    // Enhanced keyboard navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!showActions) return;
            
            switch (e.key) {
                case 'c':
                case 'C':
                    if (e.metaKey || e.ctrlKey) {
                        e.preventDefault();
                        handleCopy();
                    }
                    break;
                case 'f':
                case 'F':
                    e.preventDefault();
                    onToggleFavorite(message.id);
                    break;
                case 's':
                case 'S':
                    e.preventDefault();
                    onToggleSaved(message.id);
                    break;
                case 'r':
                case 'R':
                    if (onRegenerate && userRole === 'assistant') {
                        e.preventDefault();
                        onRegenerate(message.id);
                    }
                    break;
            }
        };

        if (showActions) {
            document.addEventListener('keydown', handleKeyDown);
            return () => document.removeEventListener('keydown', handleKeyDown);
        }
    }, [showActions, message.id, onToggleFavorite, onToggleSaved, onRegenerate]);

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

    // Enhanced reaction emojis from V1 and V2
    const reactions = ['👍', '❤️', '😊', '🤔', '🙏', '✨', '💡', '🌸'];

    // Helper functions from V1 and V2
    const getAlignmentColor = (alignment?: number) => {
        if (!alignment) return 'text-gray-400';
        if (alignment >= 0.9) return 'text-emerald-500';
        if (alignment >= 0.7) return 'text-emerald-400';
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
            initial={{ opacity: 0, y: reduceMotion ? 0 : 20, scale: reduceMotion ? 1 : 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: reduceMotion ? 0 : -20, scale: reduceMotion ? 1 : 0.95 }}
            transition={{
                duration: reduceMotion ? 0.1 : 0.3,
                ease: [0.4, 0, 0.2, 1]
            }}
            className={`group relative mb-6 ${userRole === 'user'
                    ? 'ml-auto max-w-[85%] sm:max-w-[75%]'
                    : 'mr-auto max-w-[95%] sm:max-w-[85%]'
                } ${isMobile ? 'touch-manipulation' : ''} ${isHighContrast ? 'high-contrast-message' : ''}`}
            onMouseEnter={() => {
                if (!isMobile) {
                    setShowActions(true);
                    if (onHover) onHover(true);
                }
            }}
            onMouseLeave={() => {
                if (!isMobile) {
                    setShowActions(false);
                    if (onHover) onHover(false);
                }
            }}
            onTouchStart={() => {
                if (isMobile) {
                    setShowActions(true);
                    if (onHover) onHover(true);
                }
            }}
            role="article"
            aria-label={`${userRole === 'user' ? 'Your' : 'DharmaMind AI'} message ${messageIndex + 1} of ${totalMessages}`}
            aria-live={userRole === 'assistant' ? 'polite' : undefined}
            tabIndex={0}
        >
            {/* Message Bubble */}
            <div
                className={`relative rounded-2xl px-4 py-3 shadow-lg transition-all duration-200 ${userRole === 'user'
                        ? 'bg-gradient-to-r from-emerald-500 to-emerald-600 text-white ml-auto'
                        : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-700'
                    } ${showActions || isHovered ? 'shadow-xl scale-[1.02]' : ''} ${isMobile ? 'touch-friendly-spacing' : ''} ${isHighContrast ? 'high-contrast-bubble' : ''}`}
                style={{
                    background: userRole === 'user'
                        ? `linear-gradient(135deg, ${colors.primaryStart}, ${colors.primaryEnd})`
                        : undefined,
                }}
                role="group"
                aria-labelledby={`message-${message.id}-header`}
            >
                {/* AI Badge and Metadata */}
                {userRole === 'assistant' && (
                    <header 
                        id={`message-${message.id}-header`}
                        className="flex items-center gap-2 mb-2 text-xs text-gray-500 dark:text-gray-400"
                        role="banner"
                    >
                        <div className="flex items-center gap-1">
                            <div 
                                className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"
                                role="status"
                                aria-label="AI responding"
                            ></div>
                            <span className="font-medium">DharmaMind AI</span>
                        </div>

                        {message.confidence && (
                            <div className="flex items-center gap-1" role="status" aria-label={`AI confidence: ${Math.round(message.confidence * 100)}%`}>
                                <span className={getAlignmentColor(message.confidence)}>
                                    {getConfidenceIndicator(message.confidence)}
                                </span>
                                <span>{Math.round(message.confidence * 100)}% confident</span>
                            </div>
                        )}

                        {message.dharmic_alignment && (
                            <div className="flex items-center gap-1" role="status" aria-label={`Dharmic alignment: ${Math.round(message.dharmic_alignment * 100)}%`}>
                                <span className={getAlignmentColor(message.dharmic_alignment)}>🕉️</span>
                                <span>Dharmic: {Math.round(message.dharmic_alignment * 100)}%</span>
                            </div>
                        )}
                    </header>
                )}

                {/* Message Content */}
                <main 
                    className="message-content"
                    role="main"
                    aria-describedby={`message-${message.id}-metadata`}
                >
                    {renderMessageContent()}
                </main>

                {/* Modules Used - Enhanced Display */}
                {userRole === 'assistant' && message.modules_used && message.modules_used.length > 0 && (
                    <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
                        <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-1">
                            <span className="font-medium">🧠 Wisdom Sources:</span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                            {message.modules_used.map((module, index) => (
                                <span
                                    key={index}
                                    className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-emerald-100 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200 capitalize"
                                >
                                    {module}
                                </span>
                            ))}
                        </div>
                    </div>
                )}

                {/* Timestamp and Metadata */}
                <footer 
                    id={`message-${message.id}-metadata`}
                    className={`text-xs mt-2 ${userRole === 'user'
                        ? 'text-emerald-100'
                        : 'text-gray-500 dark:text-gray-400'
                    }`}
                    role="contentinfo"
                >
                    <time dateTime={new Date(message.timestamp).toISOString()}>
                        {formatTimestamp(new Date(message.timestamp))}
                    </time>
                    {isVisible && (
                        <span className="sr-only">
                            Message is currently visible
                        </span>
                    )}
                </footer>

                {/* Status Indicators */}
                <div className="absolute -right-2 -bottom-1 flex gap-1" role="group" aria-label="Message status indicators">
                    {message.isFavorite && (
                        <div 
                            className="w-4 h-4 bg-yellow-500 rounded-full flex items-center justify-center"
                            role="img"
                            aria-label="Favorited message"
                        >
                            <StarSolidIcon className="w-2 h-2 text-white" />
                        </div>
                    )}
                    {message.isSaved && (
                        <div 
                            className="w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center"
                            role="img"
                            aria-label="Saved message"
                        >
                            <BookmarkSolidIcon className="w-2 h-2 text-white" />
                        </div>
                    )}
                    {hasBeenRead && (
                        <div 
                            className="w-4 h-4 bg-green-500 rounded-full flex items-center justify-center"
                            role="img"
                            aria-label="Message has been read"
                        >
                            <CheckIcon className="w-2 h-2 text-white" />
                        </div>
                    )}
                </div>
            </div>

            {/* Action Buttons */}
            <AnimatePresence>
                {(showActions || isHovered) && (
                    <motion.div
                        initial={{ opacity: 0, y: reduceMotion ? 0 : 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: reduceMotion ? 0 : 10 }}
                        transition={{ duration: reduceMotion ? 0.1 : 0.2 }}
                        className={`absolute top-0 ${userRole === 'user' ? '-left-12' : '-right-12'
                            } ${isMobile ? '-left-16 -right-16' : ''} flex flex-col gap-1`}
                        role="toolbar"
                        aria-label="Message actions"
                    >
                        {/* Copy Button */}
                        <button
                            onClick={handleCopy}
                            className={`p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group ${isMobile ? 'p-3 text-base' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
                            title="Copy message (Ctrl+C)"
                            aria-label={`Copy ${userRole === 'user' ? 'your' : 'AI'} message to clipboard`}
                            aria-keyshortcuts="Control+C"
                        >
                            <ClipboardDocumentIcon className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} text-gray-600 dark:text-gray-400 group-hover:text-emerald-600`} />
                        </button>

                        {/* Favorite Button */}
                        <button
                            onClick={() => onToggleFavorite(message.id)}
                            className={`p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group ${isMobile ? 'p-3 text-base' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
                            title={message.isFavorite ? "Remove from favorites (F)" : "Add to favorites (F)"}
                            aria-label={message.isFavorite ? "Remove from favorites" : "Add to favorites"}
                            aria-keyshortcuts="F"
                            aria-pressed={message.isFavorite}
                        >
                            {message.isFavorite ? (
                                <StarSolidIcon className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} text-yellow-500`} />
                            ) : (
                                <StarIcon className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} text-gray-600 dark:text-gray-400 group-hover:text-yellow-500`} />
                            )}
                        </button>

                        {/* Save Button */}
                        <button
                            onClick={() => onToggleSaved(message.id)}
                            className={`p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group ${isMobile ? 'p-3 text-base' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
                            title={message.isSaved ? "Remove from saved (S)" : "Save message (S)"}
                            aria-label={message.isSaved ? "Remove from saved" : "Save message"}
                            aria-keyshortcuts="S"
                            aria-pressed={message.isSaved}
                        >
                            {message.isSaved ? (
                                <BookmarkSolidIcon className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} text-blue-500`} />
                            ) : (
                                <BookmarkIcon className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} text-gray-600 dark:text-gray-400 group-hover:text-blue-500`} />
                            )}
                        </button>

                        {/* Speak Button */}
                        {onSpeak && (
                            <button
                                onClick={handleSpeak}
                                className={`p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group ${isMobile ? 'p-3 text-base' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
                                title={isPlaying ? "Stop speaking message" : "Speak message aloud"}
                                aria-label={isPlaying ? "Stop speaking" : "Speak message"}
                                aria-pressed={isPlaying}
                            >
                                {isPlaying ? (
                                    <SpeakerXMarkIcon className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} text-red-500`} />
                                ) : (
                                    <SpeakerWaveIcon className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} text-gray-600 dark:text-gray-400 group-hover:text-emerald-600`} />
                                )}
                            </button>
                        )}

                        {/* Share Button */}
                        {onShare && (
                            <button
                                onClick={handleShare}
                                className={`p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group ${isMobile ? 'p-3 text-base' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
                                title="Share message"
                                aria-label="Share this message"
                            >
                                <ShareIcon className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} text-gray-600 dark:text-gray-400 group-hover:text-emerald-600`} />
                            </button>
                        )}

                        {/* Regenerate Button (AI messages only) */}
                        {userRole === 'assistant' && onRegenerate && (
                            <button
                                onClick={() => onRegenerate(message.id)}
                                className={`p-2 bg-white dark:bg-gray-800 rounded-full shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group ${isMobile ? 'p-3 text-base' : ''} ${isHighContrast ? 'high-contrast-button' : ''}`}
                                title="Regenerate response (R)"
                                aria-label="Regenerate AI response"
                                aria-keyshortcuts="R"
                            >
                                <ArrowPathIcon className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} text-gray-600 dark:text-gray-400 group-hover:text-emerald-600`} />
                            </button>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Reaction Bar */}
            <AnimatePresence>
                {(showActions || isHovered || isMobile) && (
                    <motion.div
                        initial={{ opacity: 0, y: reduceMotion ? 0 : 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: reduceMotion ? 0 : 10 }}
                        transition={{ duration: reduceMotion ? 0.1 : 0.2, delay: 0.1 }}
                        className={`mt-2 flex gap-1 ${userRole === 'user' ? 'justify-end' : 'justify-start'
                            } ${isMobile ? 'flex-wrap gap-2' : ''}`}
                        role="toolbar"
                        aria-label="Message reactions"
                    >
                        {reactions.map((reaction) => (
                            <button
                                key={reaction}
                                onClick={() => {
                                    onReact(message.id, reaction);
                                    // Haptic feedback on mobile
                                    if (isMobile && 'vibrate' in navigator) {
                                        navigator.vibrate(50);
                                    }
                                }}
                                className={`${isMobile ? 'p-2 text-lg' : 'p-1 text-sm'} hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500 ${isHighContrast ? 'high-contrast-reaction' : ''}`}
                                title={`React with ${reaction}`}
                                aria-label={`React with ${reaction} emoji`}
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
                        initial={{ opacity: 0, scale: reduceMotion ? 1 : 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: reduceMotion ? 1 : 0.8 }}
                        className={`absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full mb-2 px-3 py-1 bg-emerald-600 text-white text-sm rounded-lg shadow-lg ${isMobile ? 'text-base px-4 py-2' : ''}`}
                        role="status"
                        aria-live="polite"
                        aria-label="Message copied to clipboard"
                    >
                        <span className="flex items-center gap-2">
                            <CheckIcon className={`${isMobile ? 'w-4 h-4' : 'w-3 h-3'}`} />
                            Copied!
                        </span>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

export default UnifiedEnhancedMessageBubble;
