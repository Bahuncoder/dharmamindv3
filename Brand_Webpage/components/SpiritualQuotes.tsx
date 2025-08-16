import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Quote {
  text: string;
  source: string;
  category: 'Gita' | 'Upanishad' | 'Veda';
}

const spiritualQuotes: Quote[] = [
  // Bhagavad Gita Quotes
  {
    text: "You have the right to perform your actions, but you are not entitled to the fruits of action.",
    source: "Bhagavad Gita 2.47",
    category: "Gita"
  },
  {
    text: "The soul is neither born, and nor does it die. It is not slain when the body is slain.",
    source: "Bhagavad Gita 2.20",
    category: "Gita"
  },
  {
    text: "Surrender all your works unto me, with a mind devoted to me, and free from attachment and selfishness.",
    source: "Bhagavad Gita 18.57",
    category: "Gita"
  },
  {
    text: "The mind is restless and difficult to restrain, but it is subdued by practice.",
    source: "Bhagavad Gita 6.35",
    category: "Gita"
  },
  {
    text: "When meditation is mastered, the mind is unwavering like the flame of a lamp in a windless place.",
    source: "Bhagavad Gita 6.19",
    category: "Gita"
  },
  
  // Upanishad Quotes
  {
    text: "That which is the finest essence—this whole world has that as its soul. That is Reality. That is Atman. That art thou.",
    source: "Chandogya Upanishad 6.8.7",
    category: "Upanishad"
  },
  {
    text: "Lead me from the unreal to the real, from darkness to light, from death to immortality.",
    source: "Brihadaranyaka Upanishad 1.3.28",
    category: "Upanishad"
  },
  {
    text: "The Self is one. Unmoving, it moves faster than the mind. The senses lag, but Self runs ahead.",
    source: "Isha Upanishad 4",
    category: "Upanishad"
  },
  {
    text: "In the city of Brahman is a secret dwelling, the lotus of the heart. Within this dwelling is a space, and within that space is the fulfillment of our desires.",
    source: "Chandogya Upanishad 8.1.1",
    category: "Upanishad"
  },
  {
    text: "The knower of Brahman becomes Brahman indeed.",
    source: "Mundaka Upanishad 3.2.9",
    category: "Upanishad"
  },
  
  // Vedic Quotes
  {
    text: "Truth is one, sages call it by many names.",
    source: "Rig Veda 1.164.46",
    category: "Veda"
  },
  {
    text: "Let noble thoughts come to us from all directions.",
    source: "Rig Veda 1.89.1",
    category: "Veda"
  },
  {
    text: "The whole universe is the creation of the Supreme Power meant for the benefit of creation. Each individual life form must learn to enjoy its benefits by forming a part of the system in close relation with other species. Let not any one species encroach upon others' rights.",
    source: "Isha Upanishad (Vedic)",
    category: "Veda"
  },
  {
    text: "What is not found in you, will not be found elsewhere. Seek in yourself, you are the universe.",
    source: "Atharva Veda",
    category: "Veda"
  },
  {
    text: "May all beings be happy and free, and may the thoughts, words, and actions of my own life contribute in some way to that happiness and to that freedom for all.",
    source: "Vedic Prayer",
    category: "Veda"
  }
];

const SpiritualQuotes: React.FC = () => {
  const [currentQuote, setCurrentQuote] = useState<Quote>(spiritualQuotes[0]);
  const [quoteIndex, setQuoteIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setQuoteIndex((prevIndex) => {
        const newIndex = (prevIndex + 1) % spiritualQuotes.length;
        setCurrentQuote(spiritualQuotes[newIndex]);
        return newIndex;
      });
    }, 8000); // Change quote every 8 seconds

    return () => clearInterval(interval);
  }, []);

  const getCategoryColor = (category: Quote['category']) => {
    switch (category) {
      case 'Gita':
        return 'from-orange-500 to-red-500';
      case 'Upanishad':
        return 'from-purple-500 to-blue-500';
      case 'Veda':
        return 'from-green-500 to-teal-500';
      default:
        return 'from-gray-500 to-gray-600';
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
    <div className="relative max-w-4xl mx-auto">
      <AnimatePresence mode="wait">
        <motion.div
          key={quoteIndex}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.6, ease: "easeInOut" }}
          className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-gray-200 p-8 mx-4"
        >
          <div className="flex items-start space-x-4">
            <div className="flex-shrink-0">
              <div className={`w-12 h-12 rounded-full bg-gradient-to-r ${getCategoryColor(currentQuote.category)} flex items-center justify-center text-white text-xl shadow-lg`}>
                {getCategoryIcon(currentQuote.category)}
              </div>
            </div>
            
            <div className="flex-1 min-w-0">
              <blockquote className="text-lg md:text-xl text-gray-800 font-medium leading-relaxed mb-4 italic">
                "{currentQuote.text}"
              </blockquote>
              
              <div className="flex items-center justify-between">
                <cite className="text-sm md:text-base text-gray-600 font-semibold not-italic">
                  — {currentQuote.source}
                </cite>
                
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium text-white bg-gradient-to-r ${getCategoryColor(currentQuote.category)} shadow-sm`}>
                  {currentQuote.category}
                </span>
              </div>
            </div>
          </div>
          
          {/* Progress indicator */}
          <div className="mt-6 w-full bg-gray-200 rounded-full h-1">
            <motion.div
              className={`h-1 rounded-full bg-gradient-to-r ${getCategoryColor(currentQuote.category)}`}
              initial={{ width: "0%" }}
              animate={{ width: "100%" }}
              transition={{ duration: 8, ease: "linear" }}
            />
          </div>
        </motion.div>
      </AnimatePresence>
      
      {/* Manual navigation dots */}
      <div className="flex justify-center mt-6 space-x-2">
        {spiritualQuotes.map((_, index) => (
          <button
            key={index}
            onClick={() => {
              setQuoteIndex(index);
              setCurrentQuote(spiritualQuotes[index]);
            }}
            className={`w-2 h-2 rounded-full transition-all duration-300 ${
              index === quoteIndex 
                ? `bg-gradient-to-r ${getCategoryColor(currentQuote.category)}` 
                : 'bg-gray-300 hover:bg-gray-400'
            }`}
          />
        ))}
      </div>
    </div>
  );
};

export default SpiritualQuotes;
