/**
 * 🕉️ DharmaMind Centralized Subscription Hooks
 * 
 * React hooks for subscription functionality
 * Replaces scattered subscription logic across components
 */

import { useState, useEffect, useCallback } from 'react';
import { subscriptionService, SubscriptionPlan, Subscription, PaymentMethod, CreateSubscriptionRequest, CreatePaymentMethodRequest } from '../services/subscriptionService';

// ===============================
// MAIN SUBSCRIPTION HOOK
// ===============================

export const useSubscription = () => {
  const [currentSubscription, setCurrentSubscription] = useState<Subscription | null>(null);
  const [subscriptionPlans, setSubscriptionPlans] = useState<SubscriptionPlan[]>([]);
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initialize subscription data
  useEffect(() => {
    const initializeSubscription = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Load all subscription data
        const [plans, subscription, methods] = await Promise.all([
          subscriptionService.loadSubscriptionPlans(),
          subscriptionService.loadCurrentSubscription(),
          subscriptionService.loadPaymentMethods(),
        ]);

        setSubscriptionPlans(plans);
        setCurrentSubscription(subscription);
        setPaymentMethods(methods);
      } catch (err: any) {
        console.error('Failed to initialize subscription:', err);
        setError(err.message || 'Failed to load subscription data');
      } finally {
        setIsLoading(false);
      }
    };

    initializeSubscription();
  }, []);

  // Event listeners for real-time updates
  useEffect(() => {
    const handleSubscriptionChange = (subscription: Subscription) => {
      setCurrentSubscription(subscription);
    };

    const handlePaymentMethodsChange = (methods: PaymentMethod[]) => {
      setPaymentMethods(methods);
    };

    const handleError = (error: any) => {
      setError(error.message || 'Subscription error occurred');
    };

    subscriptionService.on('subscriptionCreated', handleSubscriptionChange);
    subscriptionService.on('subscriptionUpdated', handleSubscriptionChange);
    subscriptionService.on('subscriptionCancelled', handleSubscriptionChange);
    subscriptionService.on('paymentMethodAdded', (method: PaymentMethod) => {
      setPaymentMethods(prev => [...prev, method]);
    });
    subscriptionService.on('error', handleError);

    return () => {
      // Cleanup event listeners if needed
      // Note: In a real implementation, you'd want to implement removeListener
    };
  }, []);

  // ===============================
  // SUBSCRIPTION ACTIONS
  // ===============================

  const createSubscription = useCallback(async (request: CreateSubscriptionRequest): Promise<Subscription> => {
    try {
      setError(null);
      const subscription = await subscriptionService.createSubscription(request);
      setCurrentSubscription(subscription);
      return subscription;
    } catch (err: any) {
      setError(err.message);
      throw err;
    }
  }, []);

  const updateSubscription = useCallback(async (subscriptionId: string, updates: { plan_id?: string; billing_interval?: 'monthly' | 'yearly' }): Promise<Subscription> => {
    try {
      setError(null);
      const subscription = await subscriptionService.updateSubscription(subscriptionId, updates);
      setCurrentSubscription(subscription);
      return subscription;
    } catch (err: any) {
      setError(err.message);
      throw err;
    }
  }, []);

  const cancelSubscription = useCallback(async (subscriptionId: string, immediate: boolean = false, reason?: string): Promise<Subscription> => {
    try {
      setError(null);
      const subscription = await subscriptionService.cancelSubscription(subscriptionId, immediate, reason);
      setCurrentSubscription(subscription);
      return subscription;
    } catch (err: any) {
      setError(err.message);
      throw err;
    }
  }, []);

  const createPaymentMethod = useCallback(async (paymentMethodData: CreatePaymentMethodRequest): Promise<PaymentMethod> => {
    try {
      setError(null);
      const paymentMethod = await subscriptionService.createPaymentMethod(paymentMethodData);
      setPaymentMethods(prev => [...prev, paymentMethod]);
      return paymentMethod;
    } catch (err: any) {
      setError(err.message);
      throw err;
    }
  }, []);

  // ===============================
  // COMPUTED VALUES
  // ===============================

  const currentPlan = subscriptionService.getCurrentPlan();
  const hasActiveSubscription = subscriptionService.hasActiveSubscription();
  const isInitialized = subscriptionService.isInitialized();

  // ===============================
  // UTILITY FUNCTIONS
  // ===============================

  const canUseFeature = useCallback((featureId: string): boolean => {
    return subscriptionService.canUseFeature(featureId);
  }, [currentSubscription]);

  const formatPrice = useCallback((amount: number, currency: string = 'USD', interval: string = 'month'): string => {
    return subscriptionService.formatPrice(amount, currency, interval);
  }, []);

  const calculateYearlySavings = useCallback((monthlyPrice: number): number => {
    return subscriptionService.calculateYearlySavings(monthlyPrice);
  }, []);

  return {
    // State
    currentSubscription,
    subscriptionPlans,
    paymentMethods,
    currentPlan,
    isLoading,
    error,
    hasActiveSubscription,
    isInitialized,

    // Actions
    createSubscription,
    updateSubscription,
    cancelSubscription,
    createPaymentMethod,

    // Utilities
    canUseFeature,
    formatPrice,
    calculateYearlySavings,

    // Clear error
    clearError: () => setError(null),
  };
};

// ===============================
// SPECIALIZED HOOKS
// ===============================

export const useSubscriptionPlans = () => {
  const { subscriptionPlans, isLoading, error } = useSubscription();
  return { plans: subscriptionPlans, isLoading, error };
};

export const useCurrentSubscription = () => {
  const { currentSubscription, currentPlan, hasActiveSubscription, isLoading } = useSubscription();
  return { 
    subscription: currentSubscription, 
    plan: currentPlan, 
    isActive: hasActiveSubscription,
    isLoading 
  };
};

export const usePaymentMethods = () => {
  const { paymentMethods, createPaymentMethod, isLoading, error } = useSubscription();
  return { 
    paymentMethods, 
    createPaymentMethod, 
    isLoading, 
    error 
  };
};

export const useFeatureAccess = () => {
  const { canUseFeature, currentPlan, hasActiveSubscription } = useSubscription();
  
  const checkFeature = useCallback((featureId: string) => {
    return {
      canUse: canUseFeature(featureId),
      requiresUpgrade: !canUseFeature(featureId),
      currentTier: currentPlan?.tier || 'free',
      isActive: hasActiveSubscription,
    };
  }, [canUseFeature, currentPlan, hasActiveSubscription]);

  return { checkFeature, currentPlan };
};

// ===============================
// SUBSCRIPTION MODAL HOOK
// ===============================

export const useSubscriptionModal = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null);
  const [showPaymentForm, setShowPaymentForm] = useState(false);
  const [billingInterval, setBillingInterval] = useState<'monthly' | 'yearly'>('monthly');

  const openModal = useCallback((planId?: string) => {
    setSelectedPlan(planId || null);
    setIsOpen(true);
  }, []);

  const closeModal = useCallback(() => {
    setIsOpen(false);
    setSelectedPlan(null);
    setShowPaymentForm(false);
  }, []);

  const selectPlan = useCallback((planId: string) => {
    setSelectedPlan(planId);
  }, []);

  const togglePaymentForm = useCallback((show?: boolean) => {
    setShowPaymentForm(show ?? !showPaymentForm);
  }, [showPaymentForm]);

  return {
    isOpen,
    selectedPlan,
    showPaymentForm,
    billingInterval,
    openModal,
    closeModal,
    selectPlan,
    togglePaymentForm,
    setBillingInterval,
  };
};

export default useSubscription;
