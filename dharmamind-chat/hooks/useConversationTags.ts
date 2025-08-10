import { useState, useEffect } from 'react';

export interface ConversationTag {
  id: string;
  name: string;
  color: string;
  icon: string;
  category: 'spiritual' | 'practical' | 'emotional' | 'philosophical';
  keywords: string[];
}

export interface TaggedConversation {
  id: string;
  title: string;
  tags: string[];
  timestamp: Date;
  messageCount: number;
  lastMessage: string;
  spiritualAlignment?: number;
  insights: string[];
}

const SPIRITUAL_TAGS: ConversationTag[] = [
  {
    id: 'meditation',
    name: 'Meditation',
    color: '#9f7aea',
    icon: '🧘‍♂️',
    category: 'spiritual',
    keywords: ['meditation', 'mindfulness', 'breathing', 'awareness', 'present moment', 'zazen', 'vipassana']
  },
  {
    id: 'karma',
    name: 'Karma & Action',
    color: '#4299e1',
    icon: '⚖️',
    category: 'philosophical',
    keywords: ['karma', 'action', 'consequence', 'deed', 'intention', 'cause and effect']
  },
  {
    id: 'dharma',
    name: 'Dharma Path',
    color: '#48bb78',
    icon: '🛤️',
    category: 'spiritual',
    keywords: ['dharma', 'path', 'purpose', 'duty', 'righteousness', 'way', 'teaching']
  },
  {
    id: 'suffering',
    name: 'Suffering & Liberation',
    color: '#ed8936',
    icon: '🌅',
    category: 'emotional',
    keywords: ['suffering', 'pain', 'liberation', 'freedom', 'attachment', 'desire', 'dukkha', 'moksha']
  },
  {
    id: 'relationships',
    name: 'Relationships',
    color: '#f56565',
    icon: '💞',
    category: 'practical',
    keywords: ['relationship', 'love', 'family', 'friendship', 'compassion', 'kindness', 'forgiveness']
  },
  {
    id: 'wisdom',
    name: 'Ancient Wisdom',
    color: '#ffd700',
    icon: '📜',
    category: 'philosophical',
    keywords: ['wisdom', 'knowledge', 'understanding', 'insight', 'enlightenment', 'awakening']
  },
  {
    id: 'yoga',
    name: 'Yoga & Practice',
    color: '#38b2ac',
    icon: '🕉️',
    category: 'spiritual',
    keywords: ['yoga', 'asana', 'pranayama', 'practice', 'union', 'balance', 'harmony']
  },
  {
    id: 'emotions',
    name: 'Emotional Healing',
    color: '#9f7aea',
    icon: '💚',
    category: 'emotional',
    keywords: ['emotion', 'healing', 'trauma', 'grief', 'anger', 'fear', 'joy', 'peace']
  },
  {
    id: 'purpose',
    name: 'Life Purpose',
    color: '#ed64a6',
    icon: '🌟',
    category: 'philosophical',
    keywords: ['purpose', 'meaning', 'calling', 'mission', 'destiny', 'fulfillment', 'service']
  },
  {
    id: 'mantras',
    name: 'Mantras & Chants',
    color: '#d69e2e',
    icon: '🎵',
    category: 'spiritual',
    keywords: ['mantra', 'chant', 'prayer', 'sacred sound', 'om', 'repetition', 'devotion']
  }
];

export const useConversationTags = () => {
  const [taggedConversations, setTaggedConversations] = useState<TaggedConversation[]>([]);
  const [availableTags] = useState<ConversationTag[]>(SPIRITUAL_TAGS);

  // Load tagged conversations from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('dharmamind-tagged-conversations');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setTaggedConversations(parsed.map((conv: any) => ({
          ...conv,
          timestamp: new Date(conv.timestamp)
        })));
      } catch (error) {
        console.error('Failed to load tagged conversations:', error);
      }
    }
  }, []);

  // Save to localStorage whenever tagged conversations change
  useEffect(() => {
    localStorage.setItem('dharmamind-tagged-conversations', JSON.stringify(taggedConversations));
  }, [taggedConversations]);

  const analyzeMessageForTags = (message: string): string[] => {
    const messageLower = message.toLowerCase();
    const detectedTags: string[] = [];

    availableTags.forEach(tag => {
      const hasKeyword = tag.keywords.some(keyword => 
        messageLower.includes(keyword.toLowerCase())
      );
      
      if (hasKeyword) {
        detectedTags.push(tag.id);
      }
    });

    return detectedTags;
  };

  const extractInsights = (messages: string[]): string[] => {
    const insights: string[] = [];
    
    messages.forEach(message => {
      // Look for insight patterns
      const insightPatterns = [
        /remember that (.*?)[\.\!\?]/gi,
        /the key is (.*?)[\.\!\?]/gi,
        /understand that (.*?)[\.\!\?]/gi,
        /wisdom tells us (.*?)[\.\!\?]/gi,
        /the path teaches (.*?)[\.\!\?]/gi
      ];

      insightPatterns.forEach(pattern => {
        const matches = message.match(pattern);
        if (matches) {
          matches.forEach(match => {
            const insight = match.replace(pattern, '$1').trim();
            if (insight.length > 10 && insight.length < 150) {
              insights.push(insight);
            }
          });
        }
      });
    });

    return insights.slice(0, 3); // Return top 3 insights
  };

  const tagConversation = (
    conversationId: string,
    title: string,
    messages: string[],
    lastMessage: string,
    spiritualAlignment?: number
  ) => {
    // Analyze all messages for tags
    const allTags = new Set<string>();
    messages.forEach(message => {
      const tags = analyzeMessageForTags(message);
      tags.forEach(tag => allTags.add(tag));
    });

    // Extract insights
    const insights = extractInsights(messages);

    const taggedConversation: TaggedConversation = {
      id: conversationId,
      title,
      tags: Array.from(allTags),
      timestamp: new Date(),
      messageCount: messages.length,
      lastMessage: lastMessage.substring(0, 100) + (lastMessage.length > 100 ? '...' : ''),
      spiritualAlignment,
      insights
    };

    setTaggedConversations(prev => {
      const filtered = prev.filter(conv => conv.id !== conversationId);
      return [taggedConversation, ...filtered].slice(0, 50); // Keep last 50 conversations
    });

    return Array.from(allTags);
  };

  const getConversationsByTag = (tagId: string): TaggedConversation[] => {
    return taggedConversations.filter(conv => conv.tags.includes(tagId));
  };

  const getConversationsByCategory = (category: string): TaggedConversation[] => {
    const categoryTags = availableTags
      .filter(tag => tag.category === category)
      .map(tag => tag.id);
    
    return taggedConversations.filter(conv => 
      conv.tags.some(tag => categoryTags.includes(tag))
    );
  };

  const getMostUsedTags = (limit: number = 5): { tag: ConversationTag; count: number }[] => {
    const tagCounts = new Map<string, number>();
    
    taggedConversations.forEach(conv => {
      conv.tags.forEach(tagId => {
        tagCounts.set(tagId, (tagCounts.get(tagId) || 0) + 1);
      });
    });

    return Array.from(tagCounts.entries())
      .map(([tagId, count]) => ({
        tag: availableTags.find(t => t.id === tagId)!,
        count
      }))
      .filter(item => item.tag)
      .sort((a, b) => b.count - a.count)
      .slice(0, limit);
  };

  const getSpiritualProgress = (): {
    averageAlignment: number;
    totalConversations: number;
    topInsights: string[];
    progressTrend: 'improving' | 'stable' | 'declining';
  } => {
    const alignments = taggedConversations
      .map(conv => conv.spiritualAlignment)
      .filter(alignment => alignment !== undefined) as number[];

    const averageAlignment = alignments.length > 0 
      ? alignments.reduce((sum, val) => sum + val, 0) / alignments.length 
      : 0;

    // Get all insights and sort by frequency/relevance
    const allInsights = taggedConversations
      .flatMap(conv => conv.insights)
      .slice(0, 10);

    // Calculate progress trend
    const recentAlignments = alignments.slice(0, 5);
    const olderAlignments = alignments.slice(5, 10);
    
    let progressTrend: 'improving' | 'stable' | 'declining' = 'stable';
    if (recentAlignments.length > 0 && olderAlignments.length > 0) {
      const recentAvg = recentAlignments.reduce((sum, val) => sum + val, 0) / recentAlignments.length;
      const olderAvg = olderAlignments.reduce((sum, val) => sum + val, 0) / olderAlignments.length;
      
      if (recentAvg > olderAvg + 0.1) progressTrend = 'improving';
      else if (recentAvg < olderAvg - 0.1) progressTrend = 'declining';
    }

    return {
      averageAlignment,
      totalConversations: taggedConversations.length,
      topInsights: allInsights,
      progressTrend
    };
  };

  return {
    taggedConversations,
    availableTags,
    tagConversation,
    analyzeMessageForTags,
    getConversationsByTag,
    getConversationsByCategory,
    getMostUsedTags,
    getSpiritualProgress
  };
};
