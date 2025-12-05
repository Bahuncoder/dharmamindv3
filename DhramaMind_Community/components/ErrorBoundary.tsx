import React, { Component, ErrorInfo, ReactNode } from 'react';
import { useToast } from '../contexts/ToastContext';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

class ErrorBoundaryClass extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error details
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    // Update state with error info
    this.setState({ errorInfo });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Send error to monitoring service (if configured)
    this.reportError(error, errorInfo);
  }

  private reportError = (error: Error, errorInfo: ErrorInfo) => {
    // In production, send to error monitoring service
    if (process.env.NODE_ENV === 'production') {
      // Example: Sentry, LogRocket, etc.
      console.warn('Error reported to monitoring service:', {
        error: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack
      });
    }
  };

  private handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  private handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <div className="min-h-screen bg-secondary-bg flex items-center justify-center p-4">
          <div className="card-elevated p-8 max-w-lg w-full text-center">
            <div className="text-6xl mb-6">😔</div>
            <h1 className="text-3xl font-black text-primary mb-4">
              Oops! Something went wrong
            </h1>
            <p className="text-secondary font-semibold mb-6">
              We're sorry for the inconvenience. Our team has been notified and is working on a fix.
            </p>
            
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="mb-6 text-left">
                <summary className="cursor-pointer font-bold text-primary mb-2">
                  Error Details (Development Only)
                </summary>
                <div className="bg-error-light p-4 rounded-lg text-sm overflow-auto">
                  <div className="font-bold text-error mb-2">Error:</div>
                  <div className="mb-4 font-mono">{this.state.error.message}</div>
                  {this.state.error.stack && (
                    <>
                      <div className="font-bold text-error mb-2">Stack Trace:</div>
                      <pre className="whitespace-pre-wrap text-xs">
                        {this.state.error.stack}
                      </pre>
                    </>
                  )}
                  {this.state.errorInfo?.componentStack && (
                    <>
                      <div className="font-bold text-error mb-2 mt-4">Component Stack:</div>
                      <pre className="whitespace-pre-wrap text-xs">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    </>
                  )}
                </div>
              </details>
            )}

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={this.handleRetry}
                className="btn-primary px-6 py-3 rounded-lg font-bold hover:opacity-90 transition-opacity"
              >
                🔄 Try Again
              </button>
              <button
                onClick={this.handleReload}
                className="btn-outline px-6 py-3 rounded-lg font-bold hover:bg-primary hover:text-white transition-all"
              >
                🔃 Reload Page
              </button>
            </div>

            <div className="mt-6 p-4 bg-info-light rounded-lg">
              <p className="text-sm text-info font-semibold">
                💡 If this problem persists, please contact support with the error details above.
              </p>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Hook version for functional components
export const useErrorHandler = () => {
  const { error: showToast } = useToast();

  const handleError = React.useCallback((error: Error, context?: string) => {
    console.error(`Error in ${context || 'application'}:`, error);
    
    showToast(
      'An Error Occurred',
      context 
        ? `Error in ${context}: ${error.message}`
        : error.message
    );
  }, [showToast]);

  return { handleError };
};

// HOC for wrapping components with error boundary
export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  fallback?: ReactNode,
  onError?: (error: Error, errorInfo: ErrorInfo) => void
) => {
  const WrappedComponent = (props: P) => (
    <ErrorBoundaryClass fallback={fallback} onError={onError}>
      <Component {...props} />
    </ErrorBoundaryClass>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
};

// Async error handler
export const handleAsyncError = (error: Error, context?: string) => {
  console.error(`Async error in ${context || 'application'}:`, error);
  
  // In development, show more details
  if (process.env.NODE_ENV === 'development') {
    console.error('Stack trace:', error.stack);
  }

  // Could integrate with error reporting service here
  if (process.env.NODE_ENV === 'production') {
    // Report to monitoring service
    console.warn('Async error reported to monitoring service');
  }
};

// Network error handler
export const handleNetworkError = (error: any, endpoint?: string) => {
  console.error(`Network error ${endpoint ? `at ${endpoint}` : ''}:`, error);
  
  const errorMessage = error.response?.data?.message || error.message || 'Network error occurred';
  
  return {
    type: 'network' as const,
    message: errorMessage,
    status: error.response?.status,
    endpoint
  };
};

export default ErrorBoundaryClass;
