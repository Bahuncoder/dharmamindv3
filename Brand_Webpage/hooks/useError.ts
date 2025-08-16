/**
 * ⚠️ DharmaMind Centralized Error Handling Hook
 * 
 * Unified error state management across all components
 * Replaces scattered useState('') error implementations
 */

import { useState, useCallback } from 'react';

export interface ErrorState {
  message: string;
  type?: 'validation' | 'network' | 'server' | 'auth' | 'general';
  field?: string; // For form field errors
  code?: string | number;
  timestamp?: Date;
}

export interface UseErrorReturn {
  error: ErrorState | null;
  errors: Record<string, string>; // For form field errors
  hasError: boolean;
  setError: (error: string | ErrorState) => void;
  setFieldError: (field: string, message: string) => void;
  clearError: () => void;
  clearFieldError: (field: string) => void;
  clearAllErrors: () => void;
  handleApiError: (error: any) => void;
  withErrorHandling: <T>(
    asyncFn: () => Promise<T>,
    errorMessage?: string
  ) => Promise<T | null>;
}

/**
 * Main error hook for individual components
 */
export const useError = (): UseErrorReturn => {
  const [error, setErrorState] = useState<ErrorState | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const setError = useCallback((errorInput: string | ErrorState) => {
    if (typeof errorInput === 'string') {
      setErrorState({
        message: errorInput,
        type: 'general',
        timestamp: new Date()
      });
    } else {
      setErrorState({
        ...errorInput,
        timestamp: new Date()
      });
    }
  }, []);

  const setFieldError = useCallback((field: string, message: string) => {
    setFieldErrors(prev => ({
      ...prev,
      [field]: message
    }));
  }, []);

  const clearError = useCallback(() => {
    setErrorState(null);
  }, []);

  const clearFieldError = useCallback((field: string) => {
    setFieldErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[field];
      return newErrors;
    });
  }, []);

  const clearAllErrors = useCallback(() => {
    setErrorState(null);
    setFieldErrors({});
  }, []);

  const handleApiError = useCallback((apiError: any) => {
    console.error('API Error:', apiError);
    
    if (apiError.response) {
      // Server responded with error status
      const status = apiError.response.status;
      const data = apiError.response.data;
      
      setError({
        message: data.message || getDefaultErrorMessage(status),
        type: getErrorType(status),
        code: status,
        timestamp: new Date()
      });
    } else if (apiError.request) {
      // Network error
      setError({
        message: 'Network error. Please check your connection and try again.',
        type: 'network',
        timestamp: new Date()
      });
    } else {
      // Other error
      setError({
        message: apiError.message || 'An unexpected error occurred.',
        type: 'general',
        timestamp: new Date()
      });
    }
  }, [setError]);

  const withErrorHandling = useCallback(async <T>(
    asyncFn: () => Promise<T>,
    errorMessage = 'An error occurred'
  ): Promise<T | null> => {
    try {
      clearError();
      return await asyncFn();
    } catch (error) {
      if (typeof errorMessage === 'string') {
        setError(errorMessage);
      } else {
        handleApiError(error);
      }
      return null;
    }
  }, [clearError, setError, handleApiError]);

  return {
    error,
    errors: fieldErrors,
    hasError: error !== null || Object.keys(fieldErrors).length > 0,
    setError,
    setFieldError,
    clearError,
    clearFieldError,
    clearAllErrors,
    handleApiError,
    withErrorHandling
  };
};

/**
 * Form validation hook with error handling
 */
export const useFormError = () => {
  const error = useError();

  const validateField = useCallback((
    field: string,
    value: string,
    rules: ValidationRule[]
  ): boolean => {
    error.clearFieldError(field);
    
    for (const rule of rules) {
      const errorMessage = rule(value);
      if (errorMessage) {
        error.setFieldError(field, errorMessage);
        return false;
      }
    }
    return true;
  }, [error]);

  const validateForm = useCallback((
    formData: Record<string, string>,
    validationRules: Record<string, ValidationRule[]>
  ): boolean => {
    error.clearAllErrors();
    let isValid = true;

    Object.entries(validationRules).forEach(([field, rules]) => {
      const value = formData[field] || '';
      if (!validateField(field, value, rules)) {
        isValid = false;
      }
    });

    return isValid;
  }, [error, validateField]);

  return {
    ...error,
    validateField,
    validateForm
  };
};

// ===============================
// VALIDATION RULES
// ===============================

export type ValidationRule = (value: string) => string | null;

export const validationRules = {
  required: (fieldName = 'This field'): ValidationRule => 
    (value: string) => !value.trim() ? `${fieldName} is required` : null,

  email: (): ValidationRule => 
    (value: string) => {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      return value && !emailRegex.test(value) ? 'Please enter a valid email address' : null;
    },

  minLength: (min: number): ValidationRule => 
    (value: string) => value && value.length < min ? `Must be at least ${min} characters` : null,

  maxLength: (max: number): ValidationRule => 
    (value: string) => value && value.length > max ? `Must be no more than ${max} characters` : null,

  password: (): ValidationRule => 
    (value: string) => {
      if (!value) return null;
      if (value.length < 8) return 'Password must be at least 8 characters';
      if (!/(?=.*[a-z])/.test(value)) return 'Password must contain at least one lowercase letter';
      if (!/(?=.*[A-Z])/.test(value)) return 'Password must contain at least one uppercase letter';
      if (!/(?=.*\d)/.test(value)) return 'Password must contain at least one number';
      return null;
    },

  confirmPassword: (originalPassword: string): ValidationRule =>
    (value: string) => value !== originalPassword ? 'Passwords do not match' : null,

  phone: (): ValidationRule =>
    (value: string) => {
      const phoneRegex = /^\+?[\d\s\-\(\)]+$/;
      return value && !phoneRegex.test(value) ? 'Please enter a valid phone number' : null;
    }
};

// ===============================
// HELPER FUNCTIONS
// ===============================

function getErrorType(status: number): ErrorState['type'] {
  if (status >= 400 && status < 500) return 'auth';
  if (status >= 500) return 'server';
  return 'general';
}

function getDefaultErrorMessage(status: number): string {
  switch (status) {
    case 400: return 'Invalid request. Please check your input.';
    case 401: return 'Authentication required. Please log in.';
    case 403: return 'Access denied. You do not have permission.';
    case 404: return 'Resource not found.';
    case 429: return 'Too many requests. Please try again later.';
    case 500: return 'Server error. Please try again later.';
    case 503: return 'Service unavailable. Please try again later.';
    default: return 'An unexpected error occurred.';
  }
}
