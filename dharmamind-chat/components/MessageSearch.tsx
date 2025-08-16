import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MagnifyingGlassIcon,
  XMarkIcon,
  ClockIcon,
  TagIcon,
  SparklesIcon,
  DocumentTextIcon,
  FunnelIcon,
  CalendarIcon
} from '@heroicons/react/24/outline';

interface SearchResult {
  id: string;
  content: string;
  timestamp: Date;
  type: 'user' | 'assistant';
  conversationId: string;
  tags?: string[];
  confidence?: number;
  dharmic_alignment?: number;
  context?: string;
}

interface MessageSearchProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectMessage: (result: SearchResult) => void;
  className?: string;
}

const MessageSearch: React.FC<MessageSearchProps> = ({
  isOpen,
  onClose,
  onSelectMessage,
  className = ''
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState<'all' | 'wisdom' | 'guidance' | 'meditation'>('all');
  const [dateFilter, setDateFilter] = useState<'all' | 'today' | 'week' | 'month'>('all');
  const [showFilters, setShowFilters] = useState(false);
  
  const searchInputRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  // Focus search input when modal opens
  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isOpen]);

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Mock search function (replace with actual search implementation)
  const performSearch = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Mock results - replace with actual search API
    const mockResults: SearchResult[] = [
      {
        id: '1',
        content: 'What is the meaning of dharma in Buddhism?',
        timestamp: new Date(Date.now() - 86400000), // 1 day ago
        type: 'user',
        conversationId: 'conv1',
        tags: ['dharma', 'buddhism'],
        context: 'Asked about fundamental Buddhist concepts'
      },
      {
        id: '2',
        content: 'Dharma refers to the cosmic law and order, but in Buddhism it specifically means the teachings of Buddha and the path to enlightenment...',
        timestamp: new Date(Date.now() - 86400000),
        type: 'assistant',
        conversationId: 'conv1',
        confidence: 0.95,
        dharmic_alignment: 0.98,
        tags: ['dharma', 'buddhism', 'enlightenment'],
        context: 'Explanation of dharma in Buddhist context'
      },
      {
        id: '3',
        content: 'How can I practice mindfulness meditation?',
        timestamp: new Date(Date.now() - 172800000), // 2 days ago
        type: 'user',
        conversationId: 'conv2',
        tags: ['meditation', 'mindfulness'],
        context: 'Seeking guidance on meditation practice'
      },
      {
        id: '4',
        content: 'Guide me through a loving-kindness meditation',
        timestamp: new Date(Date.now() - 604800000), // 1 week ago
        type: 'user',
        conversationId: 'conv3',
        tags: ['meditation', 'loving-kindness'],
        context: 'Requested specific meditation guidance'
      }
    ];

    const filteredResults = mockResults.filter(result =>
      result.content.toLowerCase().includes(query.toLowerCase()) ||
      (result.tags && result.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase())))
    );

    setSearchResults(filteredResults);
    setIsSearching(false);
  };

  // Debounced search
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      performSearch(searchQuery);
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  const formatRelativeTime = (timestamp: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - timestamp.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMinutes = Math.floor(diffMs / (1000 * 60));

    if (diffDays > 0) {
      return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    } else if (diffHours > 0) {
      return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else if (diffMinutes > 0) {
      return `${diffMinutes} minute${diffMinutes > 1 ? 's' : ''} ago`;
    } else {
      return 'Just now';
    }
  };

  const highlightMatch = (text: string, query: string) => {
    if (!query) return text;
    
    const regex = new RegExp(`(${query})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, index) => (
      regex.test(part) ? (
        <mark key={index} className="bg-yellow-200 text-yellow-900 px-1 rounded">
          {part}
        </mark>
      ) : part
    ));
  };

  const getTypeIcon = (type: 'user' | 'assistant') => {
    return type === 'user' ? (
      <div className="w-6 h-6 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center text-xs font-medium">
        U
      </div>
    ) : (
      <div className="w-6 h-6 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center">
        <SparklesIcon className="w-3 h-3" />
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-start justify-center pt-16"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <motion.div
        ref={modalRef}
        initial={{ opacity: 0, scale: 0.95, y: -20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: -20 }}
        className={`bg-white rounded-2xl shadow-2xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden ${className}`}
      >
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center">
              <MagnifyingGlassIcon className="w-5 h-5 mr-2 text-emerald-600" />
              Search Messages
            </h2>
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100 transition-colors"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Search Input */}
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search for wisdom, guidance, or specific topics..."
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition-all"
            />
            {isSearching && (
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full"
                />
              </div>
            )}
          </div>

          {/* Filters */}
          <div className="flex items-center justify-between mt-4">
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center px-3 py-2 text-sm text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                <FunnelIcon className="w-4 h-4 mr-1" />
                Filters
              </button>
              
              {searchResults.length > 0 && (
                <span className="text-sm text-gray-500">
                  {searchResults.length} result{searchResults.length !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          </div>

          {/* Filter Options */}
          <AnimatePresence>
            {showFilters && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-4 p-4 bg-gray-50 rounded-xl border border-gray-200"
              >
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Content Type
                    </label>
                    <select
                      value={selectedFilter}
                      onChange={(e) => setSelectedFilter(e.target.value as any)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                    >
                      <option value="all">All Messages</option>
                      <option value="wisdom">Wisdom & Teachings</option>
                      <option value="guidance">Personal Guidance</option>
                      <option value="meditation">Meditation & Practice</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Time Period
                    </label>
                    <select
                      value={dateFilter}
                      onChange={(e) => setDateFilter(e.target.value as any)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                    >
                      <option value="all">All Time</option>
                      <option value="today">Today</option>
                      <option value="week">This Week</option>
                      <option value="month">This Month</option>
                    </select>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Results */}
        <div className="p-6 max-h-96 overflow-y-auto">
          {searchQuery && !isSearching && searchResults.length === 0 && (
            <div className="text-center py-8">
              <DocumentTextIcon className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500">No messages found matching your search.</p>
              <p className="text-sm text-gray-400 mt-1">Try different keywords or check your filters.</p>
            </div>
          )}

          {!searchQuery && (
            <div className="text-center py-8">
              <MagnifyingGlassIcon className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500">Start typing to search your conversations.</p>
              <p className="text-sm text-gray-400 mt-1">Find wisdom, guidance, and insights from your past exchanges.</p>
            </div>
          )}

          <div className="space-y-3">
            <AnimatePresence>
              {searchResults.map((result) => (
                <motion.div
                  key={result.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="p-4 border border-gray-200 rounded-xl hover:border-emerald-300 hover:bg-emerald-50/50 cursor-pointer transition-all group"
                  onClick={() => onSelectMessage(result)}
                >
                  <div className="flex items-start space-x-3">
                    {getTypeIcon(result.type)}
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-600">
                          {result.type === 'user' ? 'You asked' : 'DharmaMind replied'}
                        </span>
                        <div className="flex items-center text-xs text-gray-400">
                          <ClockIcon className="w-3 h-3 mr-1" />
                          {formatRelativeTime(result.timestamp)}
                        </div>
                      </div>
                      
                      <p className="text-gray-800 line-clamp-3 group-hover:text-gray-900">
                        {highlightMatch(result.content, searchQuery)}
                      </p>
                      
                      {result.tags && (
                        <div className="flex flex-wrap gap-1 mt-2">
                          {result.tags.slice(0, 3).map((tag) => (
                            <span
                              key={tag}
                              className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800"
                            >
                              <TagIcon className="w-3 h-3 mr-1" />
                              {tag}
                            </span>
                          ))}
                          {result.tags.length > 3 && (
                            <span className="text-xs text-gray-500">
                              +{result.tags.length - 3} more
                            </span>
                          )}
                        </div>
                      )}
                      
                      {result.dharmic_alignment && (
                        <div className="flex items-center mt-2 text-xs text-gray-500">
                          <SparklesIcon className="w-3 h-3 mr-1 text-emerald-500" />
                          Dharmic alignment: {Math.round(result.dharmic_alignment * 100)}%
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default MessageSearch;
