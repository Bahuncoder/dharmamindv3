/**
 * 🕉️ DharmaMind Subscription Context
 * 
 * Centralized subscription state management
 * Ensures consistent subscription data across the entire application
 */

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useAuth } from './AuthContext';
import { subscriptionService, Subscription, SubscriptionPlan } from '../services/subscriptionService';

interface SubscriptionContextType {
  // Current subscription state
  currentSubscription: Subscription | null;
  currentPlan: SubscriptionPlan | null;
  isLoading: boolean;
  
  // Plan information
  availablePlans: SubscriptionPlan[];
  
  // Methods
  refreshSubscription: () => Promise<void>;
  upgradePlan: (planId: string, billingInterval: 'monthly' | 'yearly') => Promise<boolean>;
  cancelSubscription: () => Promise<boolean>;
  
  // Helper methods
  hasFeature: (featureId: string) => boolean;
  isFeatureLimited: (featureId: string) => boolean;
  getUsageLimit: (featureId: string) => number | null;
  isPlanActive: () => boolean;
  canUpgrade: () => boolean;
}

const SubscriptionContext = createContext<SubscriptionContextType | undefined>(undefined);

interface SubscriptionProviderProps {
  children: ReactNode;
}

export const SubscriptionProvider: React.FC<SubscriptionProviderProps> = ({ children }) => {
  const { user, isAuthenticated, updateUserPlan } = useAuth();
  const [currentSubscription, setCurrentSubscription] = useState<Subscription | null>(null);
  const [currentPlan, setCurrentPlan] = useState<SubscriptionPlan | null>(null);
  const [availablePlans, setAvailablePlans] = useState<SubscriptionPlan[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Load subscription data when user changes
  useEffect(() => {
    if (isAuthenticated && user) {
      loadSubscriptionData();
    } else {
      // Reset subscription data for non-authenticated users
      setCurrentSubscription(null);
      setCurrentPlan(getBasicPlan());
    }
  }, [user, isAuthenticated]);

  // Load available plans on mount
  useEffect(() => {
    loadAvailablePlans();
    
    // Also ensure service is initialized
    if (typeof window !== 'undefined') {
      // Force load default plans immediately as fallback
      const defaultPlans = subscriptionService.getAvailablePlans();
      if (defaultPlans && defaultPlans.length > 0) {
        setAvailablePlans(defaultPlans);
      }
    }
  }, []);

  const loadSubscriptionData = async () => {
    if (!user) return;
    
    setIsLoading(true);
    try {
      // Get user's current subscription
      const subscription = await subscriptionService.getCurrentSubscription();
      setCurrentSubscription(subscription);
      
      // Get the plan details based on user's current plan or subscription
      const userPlan = user.subscription_plan;
      if (userPlan) {
        // Find plan by tier since user.subscription_plan is the tier
        const plan = availablePlans.find(p => p.tier === userPlan) || 
                   (await subscriptionService.getPlans()).find(p => p.tier === userPlan);
        if (plan) {
          setCurrentPlan(plan);
        } else {
          setCurrentPlan(getBasicPlan());
        }
      } else {
        // Default to basic plan if no subscription
        setCurrentPlan(getBasicPlan());
      }
    } catch (error) {
      console.error('Failed to load subscription data:', error);
      // Fallback to basic plan
      setCurrentPlan(getBasicPlan());
    } finally {
      setIsLoading(false);
    }
  };

  const loadAvailablePlans = async () => {
    try {
      const plans = await subscriptionService.getPlans();
      console.log('Loaded plans from service:', plans);
      if (plans && plans.length > 0) {
        setAvailablePlans(plans);
      } else {
        // If no plans loaded, use default plans
        const defaultPlans = subscriptionService.getAvailablePlans();
        console.log('Using default plans:', defaultPlans);
        setAvailablePlans(defaultPlans);
      }
    } catch (error) {
      console.error('Failed to load plans:', error);
      // Fallback to default plans
      const defaultPlans = subscriptionService.getAvailablePlans();
      console.log('Error fallback - using default plans:', defaultPlans);
      setAvailablePlans(defaultPlans);
    }
  };

  const getBasicPlan = (): SubscriptionPlan => {
    return availablePlans.find(plan => plan.tier === 'basic') || 
           subscriptionService.getAvailablePlans().find(plan => plan.tier === 'basic') ||
           subscriptionService.getAvailablePlans()[0];
  };

  const refreshSubscription = async () => {
    await loadSubscriptionData();
  };

  const upgradePlan = async (planId: string, billingInterval: 'monthly' | 'yearly'): Promise<boolean> => {
    if (!user) return false;
    
    setIsLoading(true);
    try {
      const result = await subscriptionService.upgradePlan(planId, billingInterval);
      
      if (result.success && result.subscription) {
        // Refresh subscription data to get the latest state
        await loadSubscriptionData();
        
        // Update user's plan in AuthContext
        const plan = await subscriptionService.getPlan(planId);
        if (plan) {
          updateUserPlan(plan.tier);
        }
        
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to upgrade plan:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const cancelSubscription = async (): Promise<boolean> => {
    if (!currentSubscription) return false;
    
    setIsLoading(true);
    try {
      const subscription = await subscriptionService.cancelSubscription(currentSubscription.subscription_id);
      
      if (subscription) {
        await loadSubscriptionData();
        // Update user to basic plan
        updateUserPlan('basic');
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to cancel subscription:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const hasFeature = (featureId: string): boolean => {
    if (!currentPlan) return false;
    
    const feature = currentPlan.features.find(f => f.feature_id === featureId);
    return feature?.included || false;
  };

  const isFeatureLimited = (featureId: string): boolean => {
    if (!currentPlan) return true;
    
    const feature = currentPlan.features.find(f => f.feature_id === featureId);
    return feature ? feature.usage_limit !== undefined && feature.usage_limit > 0 : true;
  };

  const getUsageLimit = (featureId: string): number | null => {
    if (!currentPlan) return null;
    
    const feature = currentPlan.features.find(f => f.feature_id === featureId);
    return feature?.usage_limit || null;
  };

  const isPlanActive = (): boolean => {
    if (!currentSubscription) return currentPlan?.tier === 'basic'; // Basic is always "active"
    
    return ['active', 'trialing'].includes(currentSubscription.status);
  };

  const canUpgrade = (): boolean => {
    if (!currentPlan) return true;
    
    // Can upgrade if not on the highest tier
    return currentPlan.tier !== 'enterprise';
  };

  const contextValue: SubscriptionContextType = {
    currentSubscription,
    currentPlan,
    isLoading,
    availablePlans,
    refreshSubscription,
    upgradePlan,
    cancelSubscription,
    hasFeature,
    isFeatureLimited,
    getUsageLimit,
    isPlanActive,
    canUpgrade,
  };

  return (
    <SubscriptionContext.Provider value={contextValue}>
      {children}
    </SubscriptionContext.Provider>
  );
};

export const useSubscription = () => {
  const context = useContext(SubscriptionContext);
  if (context === undefined) {
    throw new Error('useSubscription must be used within a SubscriptionProvider');
  }
  return context;
};

export default SubscriptionContext;
