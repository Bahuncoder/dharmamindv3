import React, { useState } from 'react';
import {
    ClipboardDocumentIcon,
    ArrowPathIcon,
    CheckIcon
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Image from 'next/image';

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
    onCopy,
    onRegenerate,
    reduceMotion = false,
}) => {
    const [showActions, setShowActions] = useState(false);
    const [copied, setCopied] = useState(false);

    // Normalize the role
    const getUserRole = (): 'user' | 'assistant' => {
        if (message.role) return message.role;
        if (message.sender) return message.sender === 'user' ? 'user' : 'assistant';
        return 'user';
    };

    const userRole = getUserRole();
    const isUser = userRole === 'user';

    const handleCopy = async () => {
        await onCopy(message.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
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
            initial={{ opacity: 0, y: reduceMotion ? 0 : 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className={`group py-5 ${isUser ? 'bg-neutral-50 dark:bg-neutral-800/50' : ''}`}
            onMouseEnter={() => setShowActions(true)}
            onMouseLeave={() => setShowActions(false)}
        >
            <div className="max-w-3xl mx-auto px-4">
                {/* Header */}
                <div className="flex items-center gap-3 mb-2">
                    {isUser ? (
                        <div className="w-7 h-7 rounded-full bg-gold-500 flex items-center justify-center flex-shrink-0">
                            <span className="text-white text-xs font-semibold">Y</span>
                        </div>
                    ) : (
                        <div className="w-7 h-7 rounded-full overflow-hidden flex-shrink-0 border border-gold-500/30">
                            <Image
                                src="/logo.jpeg"
                                alt="DharmaMind"
                                width={28}
                                height={28}
                                className="object-cover"
                            />
                        </div>
                    )}
                    <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                        {isUser ? 'You' : 'DharmaMind'}
                    </span>
                </div>

                {/* Message Content */}
                <div className="pl-10">
                    <div className="text-neutral-800 dark:text-neutral-200 leading-7">
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                                p: ({ children }) => (
                                    <p className="mb-3 last:mb-0">{children}</p>
                                ),
                                code: ({ className, children, ...props }: any) => {
                                    const isInline = !className;
                                    if (isInline) {
                                        return (
                                            <code className="bg-neutral-100 dark:bg-neutral-700 px-1.5 py-0.5 rounded text-sm font-mono text-neutral-800 dark:text-neutral-200" {...props}>
                                                {children}
                                            </code>
                                        );
                                    }
                                    return (
                                        <code className={`${className} text-sm`} {...props}>
                                            {children}
                                        </code>
                                    );
                                },
                                pre: ({ children }) => (
                                    <pre className="bg-neutral-900 dark:bg-neutral-950 text-neutral-100 p-4 rounded-lg overflow-x-auto my-4 text-sm">
                                        {children}
                                    </pre>
                                ),
                                ul: ({ children }) => (
                                    <ul className="list-disc pl-6 mb-3 space-y-1">{children}</ul>
                                ),
                                ol: ({ children }) => (
                                    <ol className="list-decimal pl-6 mb-3 space-y-1">{children}</ol>
                                ),
                                li: ({ children }) => (
                                    <li className="leading-7">{children}</li>
                                ),
                                blockquote: ({ children }) => (
                                    <blockquote className="border-l-3 border-gold-500 pl-4 my-4 text-neutral-600 dark:text-neutral-400 italic">
                                        {children}
                                    </blockquote>
                                ),
                                h1: ({ children }) => (
                                    <h1 className="text-xl font-semibold mb-3 mt-6 first:mt-0 text-neutral-900 dark:text-white">{children}</h1>
                                ),
                                h2: ({ children }) => (
                                    <h2 className="text-lg font-semibold mb-2 mt-5 first:mt-0 text-neutral-900 dark:text-white">{children}</h2>
                                ),
                                h3: ({ children }) => (
                                    <h3 className="text-base font-semibold mb-2 mt-4 first:mt-0 text-neutral-900 dark:text-white">{children}</h3>
                                ),
                                a: ({ href, children }) => (
                                    <a href={href} className="text-gold-600 hover:text-gold-700 hover:underline" target="_blank" rel="noopener noreferrer">
                                        {children}
                                    </a>
                                ),
                                strong: ({ children }) => (
                                    <strong className="font-semibold text-neutral-900 dark:text-white">{children}</strong>
                                ),
                                em: ({ children }) => (
                                    <em className="italic">{children}</em>
                                ),
                            }}
                        >
                            {message.content}
                        </ReactMarkdown>
                    </div>

                    {/* Action Buttons - Show on hover */}
                    <div className={`flex items-center gap-1 mt-3 transition-opacity duration-150 ${showActions ? 'opacity-100' : 'opacity-0'}`}>
                        <button
                            onClick={handleCopy}
                            className="p-1.5 text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded transition-colors"
                            title="Copy message"
                        >
                            {copied ? (
                                <CheckIcon className="w-4 h-4 text-green-500" />
                            ) : (
                                <ClipboardDocumentIcon className="w-4 h-4" />
                            )}
                        </button>

                        {!isUser && onRegenerate && (
                            <button
                                onClick={() => onRegenerate(message.id)}
                                className="p-1.5 text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded transition-colors"
                                title="Regenerate response"
                            >
                                <ArrowPathIcon className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default UnifiedEnhancedMessageBubble;
