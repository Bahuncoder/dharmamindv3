import React, { useState } from 'react';
import { motion } from 'framer-motion';
import FeedbackModal from './FeedbackModal';

interface FeedbackButtonProps {
  conversationId?: string;
  messageId?: string;
  variant?: 'floating' | 'inline' | 'compact';
  className?: string;
}

const FeedbackButton: React.FC<FeedbackButtonProps> = ({ 
  conversationId, 
  messageId, 
  variant = 'floating',
  className = ''
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const baseButtonClasses = "flex items-center space-x-2 font-medium rounded-lg transition-all duration-300";
  
  const variantClasses = {
    floating: "fixed bottom-6 right-6 bg-primary-gradient text-white px-4 py-3 shadow-lg hover:shadow-xl hover:opacity-90 hover:scale-105 z-40",
    inline: "bg-primary-gradient text-white px-4 py-2 hover:opacity-90",
    compact: "bg-neutral-100 text-secondary px-3 py-2 text-sm hover:bg-neutral-200"
  };

  const iconSize = variant === 'compact' ? 'w-4 h-4' : 'w-5 h-5';

  return (
    <>
      <motion.button
        onClick={() => setIsModalOpen(true)}
        className={`${baseButtonClasses} ${variantClasses[variant]} ${className}`}
        whileHover={{ scale: variant === 'floating' ? 1.05 : 1.02 }}
        whileTap={{ scale: 0.98 }}
        initial={{ opacity: 0, y: variant === 'floating' ? 20 : 0 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: variant === 'floating' ? 1 : 0 }}
      >
        <svg 
          className={iconSize} 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" 
          />
        </svg>
        <span>
          {variant === 'compact' ? 'Feedback' : 'Share Feedback'}
        </span>
      </motion.button>

      <FeedbackModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        conversationId={conversationId}
        messageId={messageId}
      />
    </>
  );
};

export default FeedbackButton;
