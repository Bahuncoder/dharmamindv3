/**
 * 🕉️ DharmaMind Smart Personalized Suggestions
 * 
 * Intelligent, adaptive suggestions that learn from user interactions
 * and evolve based on user preferences and behavior patterns
 */

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';

interface PersonalizedSuggestionsProps {
  onSuggestionClick: (suggestion: string) => void;
  messages: any[];
  className?: string;
}

interface UserInteraction {
  suggestion: string;
  timestamp: Date;
  timeOfDay: string;
  dayOfWeek: string;
  category: string;
  frequency: number;
}

interface SuggestionCategory {
  name: string;
  keywords: string[];
  suggestions: string[];
  weight: number;
}

// Dynamic suggestion categories that learn and adapt
const suggestionCategories: SuggestionCategory[] = [
  {
    name: 'meditation',
    keywords: ['meditat', 'breath', 'mindful', 'calm', 'peace', 'relax'],
    suggestions: [
      "Guide me through meditation",
      "Lead me through breathing exercises", 
      "Help me with mindfulness practice",
      "Guide me through loving-kindness meditation",
      "Teach me body scan meditation",
      "Help me with walking meditation"
    ],
    weight: 1.0
  },
  {
    name: 'wisdom',
    keywords: ['wisdom', 'dharma', 'teach', 'understand', 'explain', 'meaning'],
    suggestions: [
      "Share wisdom about life's challenges",
      "Help me understand dharmic principles",
      "Explain the nature of suffering",
      "Guide me toward right action",
      "What does this situation teach me?",
      "Help me see the bigger picture"
    ],
    weight: 1.0
  },
  {
    name: 'emotional',
    keywords: ['feel', 'emotion', 'stress', 'anxious', 'sad', 'happy', 'angry'],
    suggestions: [
      "Help me process these emotions mindfully",
      "Guide me through emotional healing",
      "How can I find peace with difficult feelings?",
      "Help me transform negative emotions",
      "Guide me toward emotional balance",
      "Support me through this emotional challenge"
    ],
    weight: 1.0
  },
  {
    name: 'growth',
    keywords: ['grow', 'develop', 'improve', 'change', 'transform', 'evolve'],
    suggestions: [
      "How can I grow from this experience?",
      "Guide me toward personal transformation",
      "Help me develop spiritual maturity",
      "What practices support my growth?",
      "How can I evolve consciously?",
      "Support my journey of self-discovery"
    ],
    weight: 1.0
  },
  {
    name: 'practical',
    keywords: ['daily', 'routine', 'practice', 'habit', 'schedule', 'work'],
    suggestions: [
      "Help me create mindful daily routines",
      "Guide me in applying dharma at work",
      "How can I stay centered in daily life?",
      "Help me build sustainable spiritual practices",
      "Guide me in mindful decision-making",
      "Support my work-life spiritual balance"
    ],
    weight: 1.0
  }
];

const PersonalizedSuggestions: React.FC<PersonalizedSuggestionsProps> = ({ 
  onSuggestionClick, 
  messages,
  className = ''
}) => {
  const { user, isAuthenticated } = useAuth();
  const [currentSuggestions, setCurrentSuggestions] = useState<string[]>([]);
  const [userInteractions, setUserInteractions] = useState<UserInteraction[]>([]);
  const [categoryWeights, setCategoryWeights] = useState<Record<string, number>>({});

  // Load user interactions from localStorage
  useEffect(() => {
    const savedInteractions = localStorage.getItem('dharmamind_user_interactions');
    const savedWeights = localStorage.getItem('dharmamind_category_weights');
    
    if (savedInteractions) {
      try {
        const interactions = JSON.parse(savedInteractions);
        setUserInteractions(interactions);
      } catch (error) {
        console.error('Error loading user interactions:', error);
      }
    }

    if (savedWeights) {
      try {
        const weights = JSON.parse(savedWeights);
        setCategoryWeights(weights);
      } catch (error) {
        console.error('Error loading category weights:', error);
      }
    }
  }, []);

  // Save interactions to localStorage
  const saveUserInteraction = useCallback((suggestion: string) => {
    const timeOfDay = getTimeOfDay();
    const dayOfWeek = getDayOfWeek();
    const category = detectSuggestionCategory(suggestion);
    
    const newInteraction: UserInteraction = {
      suggestion,
      timestamp: new Date(),
      timeOfDay,
      dayOfWeek,
      category,
      frequency: 1
    };

    setUserInteractions(prev => {
      // Check if this exact suggestion was used before
      const existingIndex = prev.findIndex(
        interaction => interaction.suggestion === suggestion
      );

      let updatedInteractions;
      if (existingIndex >= 0) {
        // Increase frequency of existing suggestion
        updatedInteractions = [...prev];
        updatedInteractions[existingIndex] = {
          ...updatedInteractions[existingIndex],
          frequency: updatedInteractions[existingIndex].frequency + 1,
          timestamp: new Date()
        };
      } else {
        // Add new interaction
        updatedInteractions = [...prev, newInteraction];
      }

      // Keep only last 100 interactions to prevent memory bloat
      const trimmedInteractions = updatedInteractions.slice(-100);
      
      // Save to localStorage
      localStorage.setItem('dharmamind_user_interactions', JSON.stringify(trimmedInteractions));
      
      return trimmedInteractions;
    });

    // Update category weights based on usage
    setCategoryWeights(prev => {
      const newWeights = { ...prev };
      newWeights[category] = (newWeights[category] || 1.0) + 0.1;
      
      // Normalize weights to prevent infinite growth
      const maxWeight = Math.max(...Object.values(newWeights));
      if (maxWeight > 3.0) {
        Object.keys(newWeights).forEach(key => {
          newWeights[key] = newWeights[key] / maxWeight * 3.0;
        });
      }
      
      localStorage.setItem('dharmamind_category_weights', JSON.stringify(newWeights));
      return newWeights;
    });
  }, []);

  const getTimeOfDay = () => {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) return 'morning';
    if (hour >= 12 && hour < 17) return 'afternoon';
    if (hour >= 17 && hour < 21) return 'evening';
    return 'night';
  };

  const getDayOfWeek = () => {
    const days = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
    return days[new Date().getDay()];
  };

  const detectSuggestionCategory = (suggestion: string): string => {
    const lowerSuggestion = suggestion.toLowerCase();
    
    for (const category of suggestionCategories) {
      if (category.keywords.some(keyword => lowerSuggestion.includes(keyword))) {
        return category.name;
      }
    }
    
    return 'general';
  };

  const analyzeMessagePatterns = () => {
    if (messages.length <= 1) return {};
    
    const patterns: Record<string, number> = {};
    
    // Analyze recent user messages for keywords
    const recentUserMessages = messages
      .filter(msg => msg.sender === 'user')
      .slice(-10) // Last 10 user messages
      .map(msg => msg.content.toLowerCase());

    suggestionCategories.forEach(category => {
      let categoryScore = 0;
      category.keywords.forEach(keyword => {
        const mentions = recentUserMessages.filter(msg => msg.includes(keyword)).length;
        categoryScore += mentions;
      });
      
      if (categoryScore > 0) {
        patterns[category.name] = categoryScore;
      }
    });

    return patterns;
  };

  const generateSmartSuggestions = useCallback(() => {
    const timeOfDay = getTimeOfDay();
    const dayOfWeek = getDayOfWeek();
    const userName = user?.first_name || 'friend';
    const hasHistory = messages.length > 1;
    const isWeekend = dayOfWeek === 'saturday' || dayOfWeek === 'sunday';

    // Analyze current conversation patterns
    const messagePatterns = analyzeMessagePatterns();
    
    // Get user's preferred categories based on interaction history
    const preferredCategories = userInteractions.reduce((acc, interaction) => {
      acc[interaction.category] = (acc[interaction.category] || 0) + interaction.frequency;
      return acc;
    }, {} as Record<string, number>);

    // Get contextual preferences (time/day patterns)
    const contextualPreferences = userInteractions
      .filter(interaction => 
        interaction.timeOfDay === timeOfDay || 
        interaction.dayOfWeek === dayOfWeek
      )
      .reduce((acc, interaction) => {
        acc[interaction.category] = (acc[interaction.category] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

    let suggestions: string[] = [];

    // Calculate weighted scores for each category
    const categoryScores: Record<string, number> = {};
    
    suggestionCategories.forEach(category => {
      let score = category.weight;
      
      // Boost based on user interaction history
      score += (preferredCategories[category.name] || 0) * 0.3;
      
      // Boost based on current conversation patterns
      score += (messagePatterns[category.name] || 0) * 0.5;
      
      // Boost based on contextual preferences
      score += (contextualPreferences[category.name] || 0) * 0.2;
      
      // Boost based on saved category weights
      score *= (categoryWeights[category.name] || 1.0);
      
      categoryScores[category.name] = score;
    });

    // Sort categories by score and select top suggestions
    const sortedCategories = Object.entries(categoryScores)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3); // Top 3 categories

    // Generate suggestions from top categories
    sortedCategories.forEach(([categoryName, score]) => {
      const category = suggestionCategories.find(c => c.name === categoryName);
      if (category) {
        // Select suggestions based on user's previous choices
        const userFavoritesInCategory = userInteractions
          .filter(interaction => interaction.category === categoryName)
          .sort((a, b) => b.frequency - a.frequency)
          .slice(0, 2)
          .map(interaction => interaction.suggestion);

        // Mix user favorites with fresh suggestions
        const availableSuggestions = category.suggestions.filter(
          suggestion => !currentSuggestions.includes(suggestion)
        );
        
        const selectedSuggestions = [
          ...userFavoritesInCategory.slice(0, 1),
          ...availableSuggestions.slice(0, 1)
        ].filter(Boolean);

        suggestions.push(...selectedSuggestions);
      }
    });

    // If no smart suggestions available, provide contextual defaults
    if (suggestions.length === 0) {
      if (!isAuthenticated) {
        suggestions = [
          "Guide me through meditation",
          "What is dharma and how can it guide my life?",
          "Help me understand mindfulness practice"
        ];
      } else if (!hasHistory) {
        suggestions = [
          `Welcome ${userName}! How can I support your spiritual journey?`,
          "Guide me through meditation",
          "Help me find peace in daily life"
        ];
      } else {
        // Provide time-contextual suggestions
        const timeBasedSuggestions = {
          morning: ["Help me set intentions for today", "Guide me through morning meditation"],
          afternoon: ["Help me stay centered during busy times", "Guide me in mindful decision-making"],
          evening: ["Help me reflect on today's lessons", "Guide me through evening peace"],
          night: ["Help me prepare for restful sleep", "Guide me in releasing today's stress"]
        };
        
        suggestions = timeBasedSuggestions[timeOfDay] || timeBasedSuggestions.morning;
      }
    }

    // Ensure we have exactly 3 unique suggestions
    const uniqueSuggestions = Array.from(new Set(suggestions)).slice(0, 3);
    
    // If we need more suggestions, add from less preferred categories
    while (uniqueSuggestions.length < 3) {
      const remainingCategories = suggestionCategories.filter(
        category => !uniqueSuggestions.some(suggestion => 
          category.suggestions.includes(suggestion)
        )
      );
      
      if (remainingCategories.length > 0) {
        const randomCategory = remainingCategories[Math.floor(Math.random() * remainingCategories.length)];
        const availableSuggestions = randomCategory.suggestions.filter(
          suggestion => !uniqueSuggestions.includes(suggestion)
        );
        
        if (availableSuggestions.length > 0) {
          uniqueSuggestions.push(availableSuggestions[0]);
        } else {
          break;
        }
      } else {
        break;
      }
    }

    setCurrentSuggestions(uniqueSuggestions);
  }, [user, messages, userInteractions, categoryWeights, isAuthenticated]);

  // Enhanced suggestion click handler
  const handleSuggestionClick = (suggestion: string) => {
    saveUserInteraction(suggestion);
    onSuggestionClick(suggestion);
  };

  useEffect(() => {
    generateSmartSuggestions();
    
    // Refresh suggestions periodically and after interactions
    const interval = setInterval(generateSmartSuggestions, 60 * 1000); // Every minute
    return () => clearInterval(interval);
  }, [generateSmartSuggestions]);

  if (currentSuggestions.length === 0) return null;

  return (
    <div className={`mb-4 ${className}`}>
      <AnimatePresence mode="wait">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.3 }}
          className="space-y-2"
        >
          <div className="flex items-center space-x-2 mb-3">
            <span className="text-sm font-medium text-gray-600">
              � Smart suggestions based on your preferences:
            </span>
            <button 
              onClick={generateSmartSuggestions}
              className="text-xs text-emerald-500 hover:text-emerald-700 transition-colors"
              title="Refresh smart suggestions"
            >
              ↻ Refresh
            </button>
          </div>
          
          <div className="grid gap-2 md:grid-cols-1 lg:grid-cols-3">
            {currentSuggestions.map((suggestion, index) => (
              <motion.button
                key={`smart-${suggestion}-${index}`}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => handleSuggestionClick(suggestion)}
                className="text-left p-3 bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-100 rounded-lg hover:from-emerald-100 hover:to-green-100 hover:shadow-md transition-all duration-200 group relative"
              >
                <div className="flex items-start space-x-2">
                  <span className="text-emerald-500 group-hover:text-emerald-600 mt-0.5">🎯</span>
                  <span className="text-sm text-gray-700 group-hover:text-gray-900 line-clamp-2">
                    {suggestion}
                  </span>
                </div>
                
                {/* Learning indicator */}
                {userInteractions.some(interaction => interaction.suggestion === suggestion) && (
                  <div className="absolute top-1 right-1">
                    <span className="text-xs text-emerald-600" title="Learned from your preferences">
                      ✨
                    </span>
                  </div>
                )}
              </motion.button>
            ))}
          </div>
          
          {/* Learning status indicator */}
          <div className="flex items-center justify-center mt-4 text-xs text-gray-500">
            <span>
              💡 Learning from {userInteractions.length} interactions to personalize your experience
              {Object.keys(categoryWeights).length > 0 && (
                <span className="ml-2">
                  | Preferences: {Object.entries(categoryWeights)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 2)
                    .map(([category]) => category)
                    .join(', ')}
                </span>
              )}
            </span>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default PersonalizedSuggestions;
