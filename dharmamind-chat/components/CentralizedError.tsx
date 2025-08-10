/**
 * ⚠️ DharmaMind Centralized Error Components
 * 
 * Unified error display components across all pages
 * Replaces scattered error message implementations
 */

import React from 'react';
import { ErrorState } from '../hooks/useError';

interface ErrorMessageProps {
  error: string | ErrorState | null;
  className?: string;
  variant?: 'inline' | 'banner' | 'toast' | 'modal';
  onDismiss?: () => void;
  showIcon?: boolean;
}

/**
 * Generic error message component
 */
export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  error,
  className = '',
  variant = 'inline',
  onDismiss,
  showIcon = true
}) => {
  if (!error) return null;

  const errorMessage = typeof error === 'string' ? error : error.message;
  const errorType = typeof error === 'string' ? 'general' : (error.type || 'general');

  const baseClasses = 'rounded-lg p-3 text-sm';
  const variantClasses = {
    inline: 'bg-red-50 text-red-700 border border-red-200',
    banner: 'bg-red-100 text-red-800 border-l-4 border-red-500 p-4',
    toast: 'bg-red-500 text-white shadow-lg',
    modal: 'bg-red-50 text-red-700 border border-red-300 p-4'
  };

  const iconClasses = {
    general: '⚠️',
    validation: '❌',
    network: '🌐',
    server: '🔧',
    auth: '🔒'
  };

  return (
    <div className={`${baseClasses} ${variantClasses[variant]} ${className} flex items-start justify-between`}>
      <div className="flex items-start space-x-2">
        {showIcon && (
          <span className="flex-shrink-0 text-lg">
            {iconClasses[errorType] || iconClasses.general}
          </span>
        )}
        <span className="flex-1">{errorMessage}</span>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="flex-shrink-0 ml-2 text-red-500 hover:text-red-700 transition-colors"
          aria-label="Dismiss error"
        >
          ✕
        </button>
      )}
    </div>
  );
};

/**
 * Field error component for forms
 */
interface FieldErrorProps {
  error?: string;
  className?: string;
}

export const FieldError: React.FC<FieldErrorProps> = ({
  error,
  className = ''
}) => {
  if (!error) return null;

  return (
    <div className={`text-red-600 text-sm mt-1 flex items-center space-x-1 ${className}`}>
      <span className="text-xs">❌</span>
      <span>{error}</span>
    </div>
  );
};

/**
 * Form errors summary component
 */
interface FormErrorsProps {
  errors: Record<string, string>;
  className?: string;
  title?: string;
}

export const FormErrors: React.FC<FormErrorsProps> = ({
  errors,
  className = '',
  title = 'Please fix the following errors:'
}) => {
  const errorList = Object.entries(errors);
  
  if (errorList.length === 0) return null;

  return (
    <div className={`bg-red-50 border border-red-200 rounded-lg p-4 ${className}`}>
      <h4 className="text-red-800 font-medium mb-2 flex items-center space-x-2">
        <span>⚠️</span>
        <span>{title}</span>
      </h4>
      <ul className="space-y-1">
        {errorList.map(([field, message]) => (
          <li key={field} className="text-red-700 text-sm flex items-start space-x-2">
            <span className="text-xs mt-0.5">•</span>
            <span><strong>{field}:</strong> {message}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

/**
 * Error boundary fallback component
 */
interface ErrorFallbackProps {
  error: Error;
  resetError: () => void;
}

export const ErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  resetError
}) => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6 text-center">
        <div className="mb-4">
          <span className="text-6xl">😵</span>
        </div>
        <h1 className="text-xl font-bold text-gray-900 mb-2">
          Oops! Something went wrong
        </h1>
        <p className="text-gray-600 mb-4">
          An unexpected error occurred. Please try refreshing the page.
        </p>
        <details className="text-left mb-4 bg-gray-50 p-3 rounded border">
          <summary className="cursor-pointer text-sm font-medium text-gray-700 mb-2">
            Error Details
          </summary>
          <pre className="text-xs text-red-600 whitespace-pre-wrap">
            {error.message}
          </pre>
        </details>
        <div className="space-y-2">
          <button
            onClick={resetError}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
          <button
            onClick={() => window.location.reload()}
            className="w-full bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 transition-colors"
          >
            Refresh Page
          </button>
        </div>
      </div>
    </div>
  );
};

/**
 * Network error component
 */
interface NetworkErrorProps {
  onRetry?: () => void;
  className?: string;
}

export const NetworkError: React.FC<NetworkErrorProps> = ({
  onRetry,
  className = ''
}) => {
  return (
    <div className={`bg-orange-50 border border-orange-200 rounded-lg p-4 ${className}`}>
      <div className="flex items-center space-x-3">
        <span className="text-2xl">🌐</span>
        <div className="flex-1">
          <h3 className="font-medium text-orange-800">Connection Problem</h3>
          <p className="text-orange-700 text-sm">
            Unable to connect to the server. Please check your internet connection.
          </p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="bg-orange-600 text-white px-3 py-1 rounded text-sm hover:bg-orange-700 transition-colors"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
};

/**
 * API error component with specific handling
 */
interface ApiErrorProps {
  error: ErrorState;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
}

export const ApiError: React.FC<ApiErrorProps> = ({
  error,
  onRetry,
  onDismiss,
  className = ''
}) => {
  const getErrorIcon = (type: ErrorState['type']) => {
    switch (type) {
      case 'network': return '🌐';
      case 'server': return '🔧';
      case 'auth': return '🔒';
      case 'validation': return '❌';
      default: return '⚠️';
    }
  };

  const getErrorColor = (type: ErrorState['type']) => {
    switch (type) {
      case 'network': return 'orange';
      case 'server': return 'red';
      case 'auth': return 'yellow';
      case 'validation': return 'red';
      default: return 'red';
    }
  };

  const color = getErrorColor(error.type);
  const icon = getErrorIcon(error.type);

  return (
    <div className={`bg-${color}-50 border border-${color}-200 rounded-lg p-4 ${className}`}>
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3">
          <span className="text-xl flex-shrink-0">{icon}</span>
          <div>
            <p className={`text-${color}-800 font-medium`}>
              {error.message}
            </p>
            {error.code && (
              <p className={`text-${color}-600 text-sm mt-1`}>
                Error Code: {error.code}
              </p>
            )}
          </div>
        </div>
        <div className="flex space-x-2">
          {onRetry && (
            <button
              onClick={onRetry}
              className={`bg-${color}-600 text-white px-3 py-1 rounded text-sm hover:bg-${color}-700 transition-colors`}
            >
              Retry
            </button>
          )}
          {onDismiss && (
            <button
              onClick={onDismiss}
              className={`text-${color}-500 hover:text-${color}-700 transition-colors`}
            >
              ✕
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Empty state with error context
 */
interface EmptyStateErrorProps {
  title: string;
  description: string;
  icon?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export const EmptyStateError: React.FC<EmptyStateErrorProps> = ({
  title,
  description,
  icon = '📭',
  action,
  className = ''
}) => {
  return (
    <div className={`text-center py-12 ${className}`}>
      <div className="text-6xl mb-4">{icon}</div>
      <h3 className="text-lg font-medium text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600 mb-6">{description}</p>
      {action && (
        <button
          onClick={action.onClick}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
        >
          {action.label}
        </button>
      )}
    </div>
  );
};
