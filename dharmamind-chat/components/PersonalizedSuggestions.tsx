/**
 * 🕉️ DharmaMind Personalized Suggestions
 * 
 * Smart, contextual suggestions that appear above the chat input
 * Replaces static sidebar examples with dynamic, personalized prompts
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';

interface PersonalizedSuggestionsProps {
  onSuggestionClick: (suggestion: string) => void;
  messages: any[];
  className?: string;
}

const PersonalizedSuggestions: React.FC<PersonalizedSuggestionsProps> = ({ 
  onSuggestionClick, 
  messages,
  className = ''
}) => {
  const { user, isAuthenticated } = useAuth();
  const [currentSuggestions, setCurrentSuggestions] = useState<string[]>([]);

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

  const generatePersonalizedSuggestions = () => {
    const timeOfDay = getTimeOfDay();
    const dayOfWeek = getDayOfWeek();
    const userName = user?.first_name || 'friend';
    const hasHistory = messages.length > 1;
    const isWeekend = dayOfWeek === 'saturday' || dayOfWeek === 'sunday';

    let suggestions: string[] = [];

    // Time-based suggestions
    const timeSuggestions = {
      morning: [
        `Good morning ${userName}! How can I help you start your day mindfully?`,
        "What intention would you like to set for today?",
        "Help me create a morning meditation practice",
        "How can I approach today's challenges with wisdom?"
      ],
      afternoon: [
        "How can I stay centered during this busy afternoon?",
        "Help me process the emotions I'm feeling right now",
        "What would a wise person do in my current situation?",
        "Guide me toward mindful decision-making"
      ],
      evening: [
        "Help me reflect on today's lessons and growth",
        "How can I release the stress from today?",
        "What insights can I gain from today's experiences?",
        "Guide me toward inner peace this evening"
      ],
      night: [
        "Help me prepare for restful, peaceful sleep",
        "How can I let go of today's worries?",
        "Share wisdom for quiet contemplation",
        "Guide me through gratitude for today"
      ]
    };

    // Day-specific suggestions
    const daySuggestions: Record<string, string[]> = {
      monday: [
        "How can I approach this new week with dharma?",
        "Help me set meaningful intentions for the week",
        "How to stay motivated when starting fresh?"
      ],
      friday: [
        "Help me reflect on this week's spiritual growth",
        "How can I transition mindfully into the weekend?",
        "What lessons did I learn this week?"
      ],
      weekend: [
        "How can I use this weekend for spiritual renewal?",
        "Help me balance rest with meaningful activities",
        "What practices can deepen my spiritual connection?"
      ]
    };

    // User-specific suggestions
    if (!isAuthenticated) {
      suggestions = [
        "What is dharma and how can it guide my life?",
        "How does DharmaMind work differently from other AI?",
        "Introduce me to mindful living practices",
        "What makes spiritual wisdom timeless?"
      ];
    } else if (!hasHistory) {
      suggestions = [
        `Welcome ${userName}! How can I support your spiritual journey?`,
        "What aspect of your life needs the most wisdom right now?",
        "Help me understand my life's purpose and direction",
        "How can I bring more peace into my daily routine?"
      ];
    } else {
      // For returning users, mix time and context
      suggestions = [
        ...timeSuggestions[timeOfDay].slice(0, 2),
        ...(isWeekend ? daySuggestions.weekend : daySuggestions[dayOfWeek] || []).slice(0, 1),
        "Continue our previous conversation about growth",
      ];
    }

    // Shuffle and take 3 suggestions
    const shuffled = suggestions.sort(() => Math.random() - 0.5);
    setCurrentSuggestions(shuffled.slice(0, 3));
  };

  useEffect(() => {
    generatePersonalizedSuggestions();
    
    // Refresh suggestions every 30 minutes
    const interval = setInterval(generatePersonalizedSuggestions, 30 * 60 * 1000);
    return () => clearInterval(interval);
  }, [user, messages.length, isAuthenticated]);

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
            <span className="text-sm font-medium text-gray-600">💭 Suggestions for you:</span>
            <button 
              onClick={generatePersonalizedSuggestions}
              className="text-xs text-blue-500 hover:text-blue-700 transition-colors"
              title="Refresh suggestions"
            >
              ↻ Refresh
            </button>
          </div>
          
          <div className="grid gap-2 md:grid-cols-1 lg:grid-cols-3">
            {currentSuggestions.map((suggestion, index) => (
              <motion.button
                key={`${suggestion}-${index}`}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => onSuggestionClick(suggestion)}
                className="text-left p-3 bg-gradient-to-r from-blue-50 to-gray-50 border border-blue-100 rounded-lg hover:from-blue-100 hover:to-gray-100 hover:shadow-md transition-all duration-200 group"
              >
                <div className="flex items-start space-x-2">
                  <span className="text-blue-500 group-hover:text-blue-600 mt-0.5">✨</span>
                  <span className="text-sm text-gray-700 group-hover:text-gray-900 line-clamp-2">
                    {suggestion}
                  </span>
                </div>
              </motion.button>
            ))}
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default PersonalizedSuggestions;
