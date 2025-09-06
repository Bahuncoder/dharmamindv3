import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface ToastContextType {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearAllToasts: () => void;
  success: (title: string, message?: string) => void;
  error: (title: string, message?: string) => void;
  warning: (title: string, message?: string) => void;
  info: (title: string, message?: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const generateId = useCallback(() => {
    return `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = generateId();
    const duration = toast.duration || 5000;
    const newToast: Toast = {
      ...toast,
      id,
      duration
    };

    setToasts(prev => [...prev, newToast]);

    // Auto remove toast after duration
    if (duration > 0) {
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== id));
      }, duration);
    }
  }, [generateId]);

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  }, []);

  const clearAllToasts = useCallback(() => {
    setToasts([]);
  }, []);

  const success = useCallback((title: string, message?: string) => {
    addToast({ type: 'success', title, message });
  }, [addToast]);

  const error = useCallback((title: string, message?: string) => {
    addToast({ type: 'error', title, message, duration: 7000 });
  }, [addToast]);

  const warning = useCallback((title: string, message?: string) => {
    addToast({ type: 'warning', title, message, duration: 6000 });
  }, [addToast]);

  const info = useCallback((title: string, message?: string) => {
    addToast({ type: 'info', title, message });
  }, [addToast]);

  const value: ToastContextType = {
    toasts,
    addToast,
    removeToast,
    clearAllToasts,
    success,
    error,
    warning,
    info
  };

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastContainer />
    </ToastContext.Provider>
  );
};

const ToastContainer: React.FC = () => {
  const { toasts, removeToast } = useToast();

  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 space-y-3 max-w-md">
      {toasts.map((toast) => (
        <ToastItem
          key={toast.id}
          toast={toast}
          onRemove={() => removeToast(toast.id)}
        />
      ))}
    </div>
  );
};

interface ToastItemProps {
  toast: Toast;
  onRemove: () => void;
}

const ToastItem: React.FC<ToastItemProps> = ({ toast, onRemove }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Trigger animation on mount
    setTimeout(() => setIsVisible(true), 100);
  }, []);

  const handleRemove = () => {
    setIsVisible(false);
    setTimeout(onRemove, 300); // Wait for animation
  };

  const getToastStyles = () => {
    const baseStyles = "transform transition-all duration-300 ease-in-out";
    const visibilityStyles = isVisible 
      ? "translate-x-0 opacity-100" 
      : "translate-x-full opacity-0";

    const typeStyles = {
      success: "border-l-4 border-success bg-success-light",
      error: "border-l-4 border-error bg-error-light",
      warning: "border-l-4 border-warning bg-warning-light",
      info: "border-l-4 border-info bg-info-light"
    };

    return `${baseStyles} ${visibilityStyles} ${typeStyles[toast.type]}`;
  };

  const getIcon = () => {
    const icons = {
      success: "✅",
      error: "❌", 
      warning: "⚠️",
      info: "ℹ️"
    };
    return icons[toast.type];
  };

  return (
    <div className={`${getToastStyles()} card-elevated p-4 shadow-lg max-w-md`}>
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 text-xl">
          {getIcon()}
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-bold text-primary text-sm">
            {toast.title}
          </div>
          {toast.message && (
            <div className="text-secondary text-sm mt-1 leading-relaxed">
              {toast.message}
            </div>
          )}
          {toast.action && (
            <button
              onClick={toast.action.onClick}
              className="text-primary font-bold text-sm mt-2 hover:underline"
            >
              {toast.action.label}
            </button>
          )}
        </div>
        <button
          onClick={handleRemove}
          className="flex-shrink-0 text-muted hover:text-primary transition-colors p-1 rounded-full hover:bg-neutral-200"
          aria-label="Close notification"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export const useToast = (): ToastContextType => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};

// Utility functions for common toast patterns
export const toastUtils = {
  saveSuccess: (item = 'Item') => ({
    type: 'success' as const,
    title: `${item} Saved Successfully`,
    message: 'Your changes have been saved.'
  }),

  deleteSuccess: (item = 'Item') => ({
    type: 'success' as const,
    title: `${item} Deleted`,
    message: 'The item has been removed successfully.'
  }),

  networkError: () => ({
    type: 'error' as const,
    title: 'Connection Error',
    message: 'Please check your internet connection and try again.'
  }),

  validationError: (field: string) => ({
    type: 'error' as const,
    title: 'Validation Error',
    message: `Please check the ${field} field and try again.`
  }),

  loadingError: () => ({
    type: 'error' as const,
    title: 'Loading Error',
    message: 'Failed to load data. Please refresh the page.'
  }),

  comingSoon: (feature: string) => ({
    type: 'info' as const,
    title: 'Coming Soon',
    message: `${feature} will be available in a future update.`
  })
};

export default ToastProvider;
