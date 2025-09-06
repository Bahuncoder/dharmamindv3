/**
 * 🕉️ DharmaMind Enhanced Subscription Hooks
 * 
 * Comprehensive subscription management with usage tracking,
 * feature gating, and real-time limit enforcement
 */

import { useState, useEffect, useCallback } from 'react';
import { useSession } from 'next-auth/react';
import { subscriptionService, SubscriptionPlan, Subscription, PaymentMethod, CreateSubscriptionRequest, CreatePaymentMethodRequest } from '../services/subscriptionService';

// ===============================
// ENHANCED TYPES
// ===============================

export interface UsageStats {
  messagesUsed: number;
  messagesLimit: number;
  wisdomModulesUsed: number;
  wisdomModulesLimit: number;
  apiRequestsUsed: number;
  apiRequestsLimit: number;
  resetDate: string;
  periodStart: string;
  periodEnd: string;
}

export interface FeatureCheck {
  canUse: boolean;
  requiresUpgrade: boolean;
  hasReachedLimit: boolean;
  usage: number;
  limit: number;
  percentage: number;
  remaining: number;
}

// ===============================
// MAIN ENHANCED SUBSCRIPTION HOOK
// ===============================

export const useSubscription = () => {
  const { data: session } = useSession();
  const [currentSubscription, setCurrentSubscription] = useState<Subscription | null>(null);
  const [subscriptionPlans, setSubscriptionPlans] = useState<SubscriptionPlan[]>([]);
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [usage, setUsage] = useState<UsageStats | null>(null);
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
        
        // Load usage data
        await loadUsageData(subscription);
      } catch (err: any) {
        console.error('Failed to initialize subscription:', err);
        setError(err.message || 'Failed to load subscription data');
        
        // Set defaults for demo/fallback
        setUsage(getDefaultUsage());
      } finally {
        setIsLoading(false);
      }
    };

    if (session?.user || !session) {
      initializeSubscription();
    }
  }, [session]);

  // Load usage statistics
  const loadUsageData = async (subscription: Subscription | null) => {
    try {
      // Check localStorage first
      const storageKey = session?.user ? 'dharma_usage_stats' : 'dharma_demo_usage';
      const savedUsage = localStorage.getItem(storageKey);
      
      if (savedUsage) {
        const parsed = JSON.parse(savedUsage);
        
        // Check if usage period has reset
        const resetDate = new Date(parsed.resetDate);
        if (new Date() > resetDate) {
          const newUsage = resetUsageStats();
          setUsage(newUsage);
          return;
        }
        
        setUsage(parsed);
        return;
      }

      // Fallback to default
      setUsage(getDefaultUsage());
    } catch (error) {
      console.warn('Failed to load usage data:', error);
      setUsage(getDefaultUsage());
    }
  };

  const getDefaultUsage = (): UsageStats => {
    const now = new Date();
    const resetDate = new Date(now.getFullYear(), now.getMonth() + 1, 1); // Next month
    
    return {
      messagesUsed: 0,
      messagesLimit: 50, // Free plan default
      wisdomModulesUsed: 0,
      wisdomModulesLimit: 5,
      apiRequestsUsed: 0,
      apiRequestsLimit: 0,
      resetDate: resetDate.toISOString(),
      periodStart: now.toISOString(),
      periodEnd: resetDate.toISOString(),
    };
  };

  const resetUsageStats = (): UsageStats => {
    const newUsage = getDefaultUsage();
    const storageKey = session?.user ? 'dharma_usage_stats' : 'dharma_demo_usage';
    localStorage.setItem(storageKey, JSON.stringify(newUsage));
    return newUsage;
  };

  // Event listeners for real-time updates
  useEffect(() => {
    const handleSubscriptionChange = (subscription: Subscription) => {
      setCurrentSubscription(subscription);
      loadUsageData(subscription);
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
    };
  }, [session]);

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
  // USAGE TRACKING & FEATURE GATING
  // ===============================

  const trackUsage = useCallback(async (feature: string, amount: number = 1): Promise<boolean> => {
    if (!usage) return false;
    
    let newUsage = { ...usage };
    let canProceed = true;
    
    switch (feature) {
      case 'messages':
        if (newUsage.messagesLimit !== -1 && newUsage.messagesUsed >= newUsage.messagesLimit) {
          canProceed = false;
        } else {
          newUsage.messagesUsed += amount;
        }
        break;
      case 'wisdom_modules':
        if (newUsage.wisdomModulesLimit !== -1 && newUsage.wisdomModulesUsed >= newUsage.wisdomModulesLimit) {
          canProceed = false;
        } else {
          newUsage.wisdomModulesUsed += amount;
        }
        break;
      case 'api_requests':
        if (newUsage.apiRequestsLimit !== -1 && newUsage.apiRequestsUsed >= newUsage.apiRequestsLimit) {
          canProceed = false;
        } else {
          newUsage.apiRequestsUsed += amount;
        }
        break;
    }
    
    if (canProceed) {
      setUsage(newUsage);
      
      // Save to localStorage
      const storageKey = session?.user ? 'dharma_usage_stats' : 'dharma_demo_usage';
      localStorage.setItem(storageKey, JSON.stringify(newUsage));
      
      // Sync with backend (if available)
      try {
        await fetch('/api/subscription/usage', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ feature, amount, usage: newUsage })
        });
      } catch (error) {
        console.warn('Failed to sync usage with backend:', error);
      }
    }
    
    return canProceed;
  }, [usage, session]);

  const checkFeatureUsage = useCallback((feature: string): FeatureCheck => {
    if (!usage) {
      return {
        canUse: false,
        requiresUpgrade: true,
        hasReachedLimit: true,
        usage: 0,
        limit: 0,
        percentage: 100,
        remaining: 0
      };
    }
    
    let used = 0;
    let limit = 0;
    
    switch (feature) {
      case 'messages':
        used = usage.messagesUsed;
        limit = usage.messagesLimit;
        break;
      case 'wisdom_modules':
        used = usage.wisdomModulesUsed;
        limit = usage.wisdomModulesLimit;
        break;
      case 'api_requests':
        used = usage.apiRequestsUsed;
        limit = usage.apiRequestsLimit;
        break;
    }
    
    const hasReachedLimit = limit !== -1 && used >= limit;
    const canUse = !hasReachedLimit;
    const requiresUpgrade = hasReachedLimit && !isPremiumPlan();
    const percentage = limit === -1 ? 0 : Math.min(100, (used / limit) * 100);
    const remaining = limit === -1 ? -1 : Math.max(0, limit - used);
    
    return {
      canUse,
      requiresUpgrade,
      hasReachedLimit,
      usage: used,
      limit,
      percentage,
      remaining
    };
  }, [usage]);

  // ===============================
  // COMPUTED VALUES
  // ===============================

  const currentPlan = subscriptionService.getCurrentPlan();
  const hasActiveSubscription = subscriptionService.hasActiveSubscription();
  const isInitialized = subscriptionService.isInitialized();

  const isFreePlan = useCallback((): boolean => {
    return currentPlan?.tier === 'basic' || !currentPlan;
  }, [currentPlan]);

  const isPremiumPlan = useCallback((): boolean => {
    return currentPlan?.tier === 'pro' || currentPlan?.tier === 'max';
  }, [currentPlan]);

  const isEnterprisePlan = useCallback((): boolean => {
    return currentPlan?.tier === 'enterprise';
  }, [currentPlan]);

  // ===============================
  // UPGRADE PROMPTS
  // ===============================

  const shouldShowUpgradePrompt = useCallback((feature?: string): boolean => {
    if (!isFreePlan()) return false;
    
    if (feature) {
      const check = checkFeatureUsage(feature);
      return check.hasReachedLimit || check.percentage >= 80;
    }
    
    // Check any feature approaching limit
    const messageCheck = checkFeatureUsage('messages');
    return messageCheck.percentage >= 80 || messageCheck.hasReachedLimit;
  }, [isFreePlan, checkFeatureUsage]);

  const getUpgradeMessage = useCallback((feature?: string): string => {
    if (!usage) return 'Upgrade to unlock unlimited access!';
    
    switch (feature) {
      case 'messages':
        const messageCheck = checkFeatureUsage('messages');
        if (messageCheck.hasReachedLimit) {
          return `You've reached your monthly limit of ${messageCheck.limit} messages. Upgrade to Professional for unlimited conversations!`;
        }
        return `Only ${messageCheck.remaining} messages left this month. Upgrade for unlimited spiritual guidance!`;
      
      case 'wisdom_modules':
        const moduleCheck = checkFeatureUsage('wisdom_modules');
        if (moduleCheck.hasReachedLimit) {
          return `You've accessed all ${moduleCheck.limit} basic modules. Upgrade to unlock all 32 wisdom modules!`;
        }
        return 'Upgrade to access advanced wisdom modules and deeper spiritual insights!';
      
      default:
        return 'Upgrade to DharmaMind Professional for unlimited access to all features!';
    }
  }, [usage, checkFeatureUsage]);

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

  const upgradePlan = useCallback(async (planId: string, billingInterval: 'monthly' | 'yearly' = 'monthly') => {
    try {
      setError(null);
      const result = await subscriptionService.upgradePlan(planId, billingInterval);
      if (result.success && result.subscription) {
        setCurrentSubscription(result.subscription);
      }
      return result;
    } catch (err: any) {
      setError(err.message);
      throw err;
    }
  }, []);

  const isPlanActive = useCallback((planId: string): boolean => {
    return currentSubscription?.plan_id === planId && currentSubscription?.status === 'active';
  }, [currentSubscription]);

  return {
    // State
    currentSubscription,
    subscriptionPlans,
    availablePlans: subscriptionPlans, // Alias for compatibility
    paymentMethods,
    currentPlan,
    usage,
    isLoading,
    error,
    hasActiveSubscription,
    isInitialized,

    // Actions
    createSubscription,
    updateSubscription,
    cancelSubscription,
    createPaymentMethod,
    upgradePlan,

    // Usage & Feature Gating
    trackUsage,
    checkFeatureUsage,
    isFreePlan,
    isPremiumPlan,
    isEnterprisePlan,
    isPlanActive,

    // Upgrade Prompts
    shouldShowUpgradePrompt,
    getUpgradeMessage,

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
