import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Quote {
  text: string;
  source: string;
  category: 'Gita' | 'Upanishad' | 'Veda';
}

const miniQuotes: Quote[] = [
  {
    text: "You have the right to perform your actions, but you are not entitled to the fruits of action.",
    source: "Bhagavad Gita 2.47",
    category: "Gita"
  },
  {
    text: "Lead me from the unreal to the real, from darkness to light.",
    source: "Brihadaranyaka Upanishad",
    category: "Upanishad"
  },
  {
    text: "Truth is one, sages call it by many names.",
    source: "Rig Veda 1.164.46",
    category: "Veda"
  },
  {
    text: "The mind is restless but is subdued by practice.",
    source: "Bhagavad Gita 6.35",
    category: "Gita"
  },
  {
    text: "That art thou - you are that eternal Self.",
    source: "Chandogya Upanishad",
    category: "Upanishad"
  },
  {
    text: "Let noble thoughts come to us from all directions.",
    source: "Rig Veda 1.89.1",
    category: "Veda"
  }
];

const SidebarQuotes: React.FC = () => {
  const [currentQuote, setCurrentQuote] = useState<Quote>(miniQuotes[0]);
  const [quoteIndex, setQuoteIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setQuoteIndex((prevIndex) => {
        const newIndex = (prevIndex + 1) % miniQuotes.length;
        setCurrentQuote(miniQuotes[newIndex]);
        return newIndex;
      });
    }, 12000); // Change quote every 12 seconds

    return () => clearInterval(interval);
  }, []);

  const getCategoryColor = (category: Quote['category']) => {
    switch (category) {
      case 'Gita':
        return 'text-emerald-600';
      case 'Upanishad':
        return 'text-purple-600';
      case 'Veda':
        return 'text-green-600';
      default:
        return 'text-gray-600';
    }
  };

  const getCategoryIcon = (category: Quote['category']) => {
    switch (category) {
      case 'Gita':
        return '🕉️';
      case 'Upanishad':
        return '🪔';
      case 'Veda':
        return '📿';
      default:
        return '✨';
    }
  };

  return (
    <div className="bg-gradient-to-br from-gray-50 to-white border border-gray-200 rounded-lg p-4 mx-4 mb-4">
      <AnimatePresence mode="wait">
        <motion.div
          key={quoteIndex}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 mt-1">
              <span className="text-lg">{getCategoryIcon(currentQuote.category)}</span>
            </div>
            
            <div className="flex-1 min-w-0">
              <blockquote className="text-sm text-gray-700 leading-relaxed mb-2 italic">
                "{currentQuote.text}"
              </blockquote>
              
              <div className="flex items-center justify-between">
                <cite className={`text-xs font-medium not-italic ${getCategoryColor(currentQuote.category)}`}>
                  {currentQuote.source}
                </cite>
                
                <span className="text-xs text-gray-500 font-medium">
                  {currentQuote.category}
                </span>
              </div>
            </div>
          </div>
          
          {/* Simple progress bar */}
          <div className="mt-3 w-full bg-gray-200 rounded-full h-0.5">
            <motion.div
              className="h-0.5 rounded-full bg-gradient-to-r from-purple-500 to-blue-500"
              initial={{ width: "0%" }}
              animate={{ width: "100%" }}
              transition={{ duration: 12, ease: "linear" }}
            />
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default SidebarQuotes;
