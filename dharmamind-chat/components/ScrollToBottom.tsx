import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDownIcon } from '@heroicons/react/24/outline';

interface ScrollToBottomProps {
  messagesContainerRef: React.RefObject<HTMLDivElement>;
  threshold?: number;
  className?: string;
}

const ScrollToBottom: React.FC<ScrollToBottomProps> = ({
  messagesContainerRef,
  threshold = 200,
  className = ''
}) => {
  const [showButton, setShowButton] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < threshold;
      setShowButton(!isNearBottom);
    };

    container.addEventListener('scroll', handleScroll);
    handleScroll(); // Check initial state

    return () => container.removeEventListener('scroll', handleScroll);
  }, [messagesContainerRef, threshold]);

  const scrollToBottom = () => {
    const container = messagesContainerRef.current;
    if (container) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
      setUnreadCount(0);
    }
  };

  return (
    <AnimatePresence>
      {showButton && (
        <motion.button
          initial={{ opacity: 0, scale: 0.8, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.8, y: 20 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={scrollToBottom}
          className={`
            fixed bottom-24 right-6 z-20
            w-12 h-12 bg-white rounded-full shadow-lg
            flex items-center justify-center
            text-gray-600 hover:text-gray-800
            border border-gray-200
            hover:shadow-xl transition-all duration-200
            ${className}
          `}
          title="Scroll to bottom"
        >
          <ChevronDownIcon className="w-5 h-5" />
          
          {/* Unread count badge */}
          {unreadCount > 0 && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="absolute -top-2 -right-2 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold"
            >
              {unreadCount > 9 ? '9+' : unreadCount}
            </motion.div>
          )}
        </motion.button>
      )}
    </AnimatePresence>
  );
};

export default ScrollToBottom;
