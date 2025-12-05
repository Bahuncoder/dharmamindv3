/**
 * 🔄 DharmaMind Centralized Loading Hook
 * 
 * Unified loading state management across all components
 * Replaces scattered useState(false) loading implementations
 */

import { useState, useCallback } from 'react';

export interface LoadingState {
  isLoading: boolean;
  loadingMessage?: string;
  progress?: number; // 0-100 for progress bars
}

export interface UseLoadingReturn {
  isLoading: boolean;
  loadingMessage: string | undefined;
  progress: number | undefined;
  startLoading: (message?: string, progress?: number) => void;
  stopLoading: () => void;
  updateProgress: (progress: number, message?: string) => void;
  withLoading: <T>(
    asyncFn: () => Promise<T>,
    loadingMessage?: string
  ) => Promise<T>;
}

/**
 * Main loading hook for individual components
 */
export const useLoading = (initialMessage?: string): UseLoadingReturn => {
  const [state, setState] = useState<LoadingState>({
    isLoading: false,
    loadingMessage: initialMessage,
    progress: undefined
  });

  const startLoading = useCallback((message?: string, progress?: number) => {
    setState({
      isLoading: true,
      loadingMessage: message || state.loadingMessage,
      progress
    });
  }, [state.loadingMessage]);

  const stopLoading = useCallback(() => {
    setState(prev => ({
      ...prev,
      isLoading: false,
      progress: undefined
    }));
  }, []);

  const updateProgress = useCallback((progress: number, message?: string) => {
    setState(prev => ({
      ...prev,
      progress,
      loadingMessage: message || prev.loadingMessage
    }));
  }, []);

  const withLoading = useCallback(async <T>(
    asyncFn: () => Promise<T>,
    loadingMessage?: string
  ): Promise<T> => {
    try {
      startLoading(loadingMessage);
      const result = await asyncFn();
      return result;
    } finally {
      stopLoading();
    }
  }, [startLoading, stopLoading]);

  return {
    isLoading: state.isLoading,
    loadingMessage: state.loadingMessage,
    progress: state.progress,
    startLoading,
    stopLoading,
    updateProgress,
    withLoading
  };
};

/**
 * Global loading hook for app-wide loading states
 */
export const useGlobalLoading = () => {
  const [globalStates, setGlobalStates] = useState<Map<string, LoadingState>>(new Map());

  const setGlobalLoading = useCallback((key: string, loading: boolean, message?: string) => {
    setGlobalStates(prev => {
      const newMap = new Map(prev);
      if (loading) {
        newMap.set(key, { isLoading: true, loadingMessage: message });
      } else {
        newMap.delete(key);
      }
      return newMap;
    });
  }, []);

  const isGlobalLoading = globalStates.size > 0;
  const globalLoadingMessage = Array.from(globalStates.values())[0]?.loadingMessage;

  return {
    isGlobalLoading,
    globalLoadingMessage,
    setGlobalLoading
  };
};

/**
 * Loading hook for forms with validation
 */
export const useFormLoading = () => {
  const loading = useLoading();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const submitWithLoading = useCallback(async <T>(
    submitFn: () => Promise<T>,
    loadingMessage = "Submitting..."
  ): Promise<T> => {
    setIsSubmitting(true);
    try {
      return await loading.withLoading(submitFn, loadingMessage);
    } finally {
      setIsSubmitting(false);
    }
  }, [loading]);

  return {
    ...loading,
    isSubmitting,
    submitWithLoading
  };
};

/**
 * Multi-step loading hook for complex operations
 */
export const useMultiStepLoading = (steps: string[]) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  const startMultiStep = useCallback(() => {
    setIsLoading(true);
    setCurrentStep(0);
  }, []);

  const nextStep = useCallback(() => {
    setCurrentStep(prev => Math.min(prev + 1, steps.length - 1));
  }, [steps.length]);

  const completeMultiStep = useCallback(() => {
    setIsLoading(false);
    setCurrentStep(0);
  }, []);

  const progress = ((currentStep + 1) / steps.length) * 100;
  const currentStepMessage = steps[currentStep];

  return {
    isLoading,
    currentStep,
    progress,
    currentStepMessage,
    startMultiStep,
    nextStep,
    completeMultiStep
  };
};
