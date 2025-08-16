import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircleIcon, 
  ExclamationTriangleIcon, 
  InformationCircleIcon,
  XMarkIcon,
  StarIcon,
  BookmarkIcon,
  SpeakerWaveIcon,
  ClipboardDocumentIcon
} from '@heroicons/react/24/outline';

interface Toast {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  title: string;
  message: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
  icon?: 'copy' | 'favorite' | 'save' | 'speak' | 'share';
}

interface ToastProps {
  toasts: Toast[];
  onDismiss: (id: string) => void;
}

const ToastContainer: React.FC<ToastProps> = ({ toasts, onDismiss }) => {
  const getIcon = (type: Toast['type'], customIcon?: Toast['icon']) => {
    if (customIcon) {
      switch (customIcon) {
        case 'copy':
          return <ClipboardDocumentIcon className="w-5 h-5" />;
        case 'favorite':
          return <StarIcon className="w-5 h-5 text-emerald-500" />;
        case 'save':
          return <BookmarkIcon className="w-5 h-5 text-green-500" />;
        case 'speak':
          return <SpeakerWaveIcon className="w-5 h-5 text-blue-500" />;
        case 'share':
          return <div className="w-5 h-5 text-purple-500">🔗</div>;
      }
    }

    switch (type) {
      case 'success':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'error':
        return <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-5 h-5 text-gray-500" />;
      default:
        return <InformationCircleIcon className="w-5 h-5 text-blue-500" />;
    }
  };

  const getBackgroundColor = (type: Toast['type']) => {
    switch (type) {
      case 'success':
        return 'bg-green-50 border-green-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      case 'warning':
        return 'bg-gray-50 border-gray-200';
      default:
        return 'bg-blue-50 border-blue-200';
    }
  };

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm">
      <AnimatePresence>
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            initial={{ opacity: 0, x: 300, scale: 0.3 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 300, scale: 0.3 }}
            transition={{ duration: 0.3 }}
            className={`
              ${getBackgroundColor(toast.type)}
              border rounded-lg shadow-lg p-4 backdrop-blur-sm
              max-w-sm overflow-hidden
            `}
          >
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                {getIcon(toast.type, toast.icon)}
              </div>
              
              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-semibold text-gray-900 mb-1">
                  {toast.title}
                </h4>
                <p className="text-sm text-gray-700">
                  {toast.message}
                </p>
                
                {toast.action && (
                  <button
                    onClick={toast.action.onClick}
                    className="mt-2 text-sm font-medium text-blue-600 hover:text-blue-800 transition-colors"
                  >
                    {toast.action.label}
                  </button>
                )}
              </div>
              
              <button
                onClick={() => onDismiss(toast.id)}
                className="flex-shrink-0 text-gray-400 hover:text-gray-600 transition-colors"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};

// Hook for managing toasts
export const useToast = () => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = (toast: Omit<Toast, 'id'>) => {
    const id = Date.now().toString();
    const newToast = { ...toast, id };
    
    setToasts(prev => [...prev, newToast]);

    // Auto-dismiss after duration
    const duration = toast.duration || 4000;
    setTimeout(() => {
      dismissToast(id);
    }, duration);

    return id;
  };

  const dismissToast = (id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  const dismissAll = () => {
    setToasts([]);
  };

  // Spiritual-themed toast helpers
  const showSuccess = (title: string, message: string, icon?: Toast['icon']) => {
    return addToast({
      type: 'success',
      title,
      message,
      icon
    });
  };

  const showError = (title: string, message: string) => {
    return addToast({
      type: 'error',
      title,
      message
    });
  };

  const showInfo = (title: string, message: string, icon?: Toast['icon']) => {
    return addToast({
      type: 'info',
      title,
      message,
      icon
    });
  };

  const showCopySuccess = () => {
    return showSuccess(
      'Wisdom Copied',
      'Message copied to your clipboard',
      'copy'
    );
  };

  const showFavoriteAdded = () => {
    return showSuccess(
      'Added to Favorites',
      'This wisdom has been saved to your favorites',
      'favorite'
    );
  };

  const showFavoriteRemoved = () => {
    return showInfo(
      'Removed from Favorites',
      'This message is no longer in your favorites',
      'favorite'
    );
  };

  const showSavedToJournal = () => {
    return showSuccess(
      'Saved to Wisdom Journal',
      'This insight has been preserved in your spiritual journal',
      'save'
    );
  };

  const showRemovedFromJournal = () => {
    return showInfo(
      'Removed from Journal',
      'This message is no longer in your wisdom journal',
      'save'
    );
  };

  const showSpeaking = () => {
    return showInfo(
      'Speaking Wisdom',
      'Listen as the AI shares this spiritual insight',
      'speak'
    );
  };

  const showShared = () => {
    return showSuccess(
      'Wisdom Shared',
      'Your spiritual insight has been shared successfully',
      'share'
    );
  };

  return {
    toasts,
    addToast,
    dismissToast,
    dismissAll,
    showSuccess,
    showError,
    showInfo,
    showCopySuccess,
    showFavoriteAdded,
    showFavoriteRemoved,
    showSavedToJournal,
    showRemovedFromJournal,
    showSpeaking,
    showShared
  };
};

export default ToastContainer;
