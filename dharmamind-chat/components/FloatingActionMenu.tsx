import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PlusIcon,
  BookOpenIcon,
  ChatBubbleLeftRightIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  Cog6ToothIcon,
  HeartIcon,
  SparklesIcon,
  GlobeAltIcon
} from '@heroicons/react/24/outline';

interface FloatingActionMenuProps {
  onNewChat: () => void;
  onOpenNotes: () => void;
  onSearchHistory: () => void;
  onOpenSettings: () => void;
  onOpenJournal: () => void;
  onOpenInsights: () => void;
  onOpenCommunity: () => void;
  onOpenContemplation: () => void;
  className?: string;
}

const FloatingActionMenu: React.FC<FloatingActionMenuProps> = ({
  onNewChat,
  onOpenNotes,
  onSearchHistory,
  onOpenSettings,
  onOpenJournal,
  onOpenInsights,
  onOpenCommunity,
  onOpenContemplation,
  className = ''
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const menuItems = [
    {
      icon: ChatBubbleLeftRightIcon,
      label: 'New Chat',
      onClick: onNewChat,
      color: 'from-gold-500 to-gold-600',
      hoverColor: 'hover:from-gold-600 hover:to-gold-700'
    },
    {
      icon: DocumentTextIcon,
      label: 'Notes',
      onClick: onOpenNotes,
      color: 'from-neutral-1000 to-gold-600',
      hoverColor: 'hover:from-gold-600 hover:to-gold-700'
    },
    {
      icon: BookOpenIcon,
      label: 'Journal',
      onClick: onOpenJournal,
      color: 'from-gold-500 to-gold-600',
      hoverColor: 'hover:from-gold-600 hover:to-purple-700'
    },
    {
      icon: SparklesIcon,
      label: 'Insights',
      onClick: onOpenInsights,
      color: 'from-yellow-500 to-yellow-600',
      hoverColor: 'hover:from-yellow-600 hover:to-yellow-700'
    },
    {
      icon: HeartIcon,
      label: 'Contemplation',
      onClick: onOpenContemplation,
      color: 'from-rose-500 to-rose-600',
      hoverColor: 'hover:from-rose-600 hover:to-rose-700'
    },
    {
      icon: MagnifyingGlassIcon,
      label: 'Search',
      onClick: onSearchHistory,
      color: 'from-gray-500 to-gray-600',
      hoverColor: 'hover:from-gray-600 hover:to-gray-700'
    },
    {
      icon: GlobeAltIcon,
      label: 'Community',
      onClick: onOpenCommunity,
      color: 'from-pink-500 to-pink-600',
      hoverColor: 'hover:from-pink-600 hover:to-pink-700'
    },
    {
      icon: Cog6ToothIcon,
      label: 'Settings',
      onClick: onOpenSettings,
      color: 'from-indigo-500 to-indigo-600',
      hoverColor: 'hover:from-indigo-600 hover:to-indigo-700'
    }
  ];

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className={`fixed bottom-6 left-6 z-50 ${className}`}>
      {/* Menu Items */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="absolute bottom-16 left-0 space-y-3"
          >
            {menuItems.map((item, index) => (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, x: -20, scale: 0.5 }}
                animate={{ 
                  opacity: 1, 
                  x: 0, 
                  scale: 1,
                  transition: { delay: index * 0.05 }
                }}
                exit={{ 
                  opacity: 0, 
                  x: -20, 
                  scale: 0.5,
                  transition: { delay: (menuItems.length - index) * 0.05 }
                }}
                className="flex items-center space-x-3"
              >
                {/* Label */}
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="bg-white/90 backdrop-blur-sm px-3 py-2 rounded-lg shadow-lg border border-gray-200/50 text-sm font-medium text-gray-700"
                >
                  {item.label}
                </motion.div>
                
                {/* Action Button */}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => {
                    item.onClick();
                    setIsOpen(false);
                  }}
                  className={`
                    w-12 h-12 rounded-full shadow-lg
                    bg-gradient-to-r ${item.color} ${item.hoverColor}
                    text-white flex items-center justify-center
                    transition-all duration-200 transform
                    hover:shadow-xl
                  `}
                  title={item.label}
                >
                  <item.icon className="w-6 h-6" />
                </motion.button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main FAB */}
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={toggleMenu}
        className={`
          w-14 h-14 rounded-full shadow-lg
          bg-gradient-to-r from-gold-500 to-gold-600
          hover:from-gold-600 hover:to-gold-700
          text-white flex items-center justify-center
          transition-all duration-300 transform
          hover:shadow-xl relative overflow-hidden
          ${isOpen ? 'rotate-45' : 'rotate-0'}
        `}
        title={isOpen ? 'Close menu' : 'Open quick actions'}
      >
        {/* Background Animation */}
        <motion.div
          animate={{ 
            scale: isOpen ? [1, 1.2, 1] : 1,
            opacity: isOpen ? [0.5, 0.8, 0.5] : 0
          }}
          transition={{ 
            duration: 2, 
            repeat: isOpen ? Infinity : 0,
            ease: "easeInOut"
          }}
          className="absolute inset-0 bg-white rounded-full"
        />
        
        <PlusIcon className="w-7 h-7 relative z-10" />
        
        {/* Ripple Effect */}
        <motion.div
          animate={{ 
            scale: [1, 1.5, 1],
            opacity: [0.3, 0, 0.3]
          }}
          transition={{ 
            duration: 2, 
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute inset-0 bg-gold-400 rounded-full"
        />
      </motion.button>

      {/* Background Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsOpen(false)}
            className="fixed inset-0 bg-black/20 backdrop-blur-sm -z-10"
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default FloatingActionMenu;
