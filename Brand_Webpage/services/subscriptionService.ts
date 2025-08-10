/**
 * 🕉️ DharmaMind Centralized Subscription Service
 * 
 * Single source of truth for all subscription functionality
 * Consolidates: SubscriptionModal, SubscriptionManager, subscription-api.js, paymentAPI.ts
 */

import { authService } from './authService';

// ===============================
// UNIFIED TYPES & INTERFACES
// ===============================

export interface SubscriptionPlan {
  id: string;
  name: string;
  description: string;
  tier: 'basic' | 'pro' | 'max' | 'enterprise';
  price: {
    monthly: number;
    yearly: number;
  };
  currency: string;
  features: PlanFeature[];
  limits: PlanLimits;
  popular?: boolean;
  trial_days?: number;
}

export interface PlanFeature {
  feature_id: string;
  name: string;
  description: string;
  included: boolean;
  usage_limit?: number;
}

export interface PlanLimits {
  messages_per_month: number;
  wisdom_modules: number;
  api_requests_per_month: number;
  team_members?: number;
}

export interface Subscription {
  id: string;
  subscription_id: string;
  user_id: string;
  plan_id: string;
  plan_name?: string;
  status: 'active' | 'inactive' | 'cancelled' | 'suspended' | 'past_due' | 'trialing';
  billing_interval: 'monthly' | 'yearly';
  amount?: number;
  currency?: string;
  current_period_start: string;
  current_period_end: string;
  trial_start?: string;
  trial_end?: string;
  cancel_at_period_end: boolean;
  cancelled_at?: string;
  created_at: string;
  updated_at: string;
}

export interface PaymentMethod {
  id: string;
  type: 'card' | 'bank_account';
  last_four: string;
  brand?: string;
  exp_month?: number;
  exp_year?: number;
  default: boolean;
}

export interface CreateSubscriptionRequest {
  plan_id: string;
  payment_method_id: string;
  billing_interval: 'monthly' | 'yearly';
  trial_period_days?: number;
  metadata?: Record<string, any>;
}

export interface CreatePaymentMethodRequest {
  method_type: 'credit_card' | 'bank_account' | 'paypal';
  card_details?: {
    number: string;
    exp_month: number;
    exp_year: number;
    cvc: string;
    cardholder_name: string;
  };
  billing_address: {
    line1: string;
    line2?: string;
    city: string;
    state: string;
    postal_code: string;
    country: string;
  };
  set_as_default?: boolean;
}

// ===============================
// CENTRALIZED SUBSCRIPTION SERVICE
// ===============================

class CentralizedSubscriptionService {
  private baseURL: string;
  private currentSubscription: Subscription | null = null;
  private subscriptionPlans: SubscriptionPlan[] = [];
  private paymentMethods: PaymentMethod[] = [];
  private isLoading = false;
  private eventListeners = new Map<string, Function[]>();

  constructor() {
    this.baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    this.initialize();
  }

  // ===============================
  // INITIALIZATION & EVENT MANAGEMENT
  // ===============================

  private async initialize() {
    try {
      this.isLoading = true;
      await this.loadSubscriptionPlans();
      await this.loadCurrentSubscription();
      await this.loadPaymentMethods();
      this.emit('initialized');
    } catch (error) {
      console.error('Failed to initialize subscription service:', error);
      this.emit('error', error);
    } finally {
      this.isLoading = false;
    }
  }

  public on(event: string, callback: Function) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event)!.push(callback);
  }

  private emit(event: string, data?: any) {
    const listeners = this.eventListeners.get(event) || [];
    listeners.forEach(callback => callback(data));
  }

  // ===============================
  // AUTHENTICATION HELPERS
  // ===============================

  private async getAuthHeaders(): Promise<Record<string, string>> {
    const token = authService.getAccessToken();
    return {
      'Content-Type': 'application/json',
      'Authorization': token ? `Bearer ${token}` : '',
    };
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
      throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  }

  // ===============================
  // SUBSCRIPTION PLANS
  // ===============================

  async loadSubscriptionPlans(): Promise<SubscriptionPlan[]> {
    try {
      const response = await fetch(`${this.baseURL}/api/subscription/plans`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      this.subscriptionPlans = await this.handleResponse<SubscriptionPlan[]>(response);
      this.emit('plansLoaded', this.subscriptionPlans);
      return this.subscriptionPlans;
    } catch (error) {
      console.error('Failed to load subscription plans:', error);
      // Fallback to default plans
      this.subscriptionPlans = this.getDefaultPlans();
      return this.subscriptionPlans;
    }
  }

  getSubscriptionPlans(): SubscriptionPlan[] {
    return this.subscriptionPlans;
  }

  private getDefaultPlans(): SubscriptionPlan[] {
    return [
      {
        id: 'dharma_basic',
        name: 'Basic',
        description: 'Essential spiritual guidance for beginners',
        tier: 'basic',
        price: { monthly: 0, yearly: 0 },
        currency: 'USD',
        features: [
          { feature_id: 'basic_chat', name: 'Basic Conversations', description: '50 messages per month', included: true, usage_limit: 50 },
          { feature_id: 'basic_modules', name: 'Basic Modules', description: '5 wisdom modules', included: true, usage_limit: 5 },
          { feature_id: 'email_support', name: 'Email Support', description: 'Standard support via email', included: true },
        ],
        limits: {
          messages_per_month: 50,
          wisdom_modules: 5,
          api_requests_per_month: 0,
        },
      },
      {
        id: 'dharma_pro',
        name: 'Pro',
        description: 'Advanced spiritual guidance for dedicated practitioners',
        tier: 'pro',
        price: { monthly: 19.99, yearly: 199.99 },
        currency: 'USD',
        popular: true,
        trial_days: 14,
        features: [
          { feature_id: 'unlimited_chat', name: 'Unlimited Conversations', description: 'Unlimited spiritual guidance', included: true },
          { feature_id: 'all_modules', name: 'All Wisdom Modules', description: 'Access to all 32 modules', included: true },
          { feature_id: 'priority_support', name: 'Priority Support', description: '24/7 priority assistance', included: true },
          { feature_id: 'voice_guidance', name: 'Voice Guidance', description: 'Audio spiritual sessions', included: true },
        ],
        limits: {
          messages_per_month: -1,
          wisdom_modules: -1,
          api_requests_per_month: 1000,
        },
      },
      {
        id: 'dharma_max',
        name: 'Max',
        description: 'Ultimate spiritual transformation experience',
        tier: 'max',
        price: { monthly: 39.99, yearly: 399.99 },
        currency: 'USD',
        trial_days: 7,
        features: [
          { feature_id: 'unlimited_chat', name: 'Unlimited Conversations', description: 'Unlimited spiritual guidance', included: true },
          { feature_id: 'all_modules', name: 'All Wisdom Modules', description: 'Access to all 32+ modules', included: true },
          { feature_id: 'priority_support', name: 'Priority Support', description: '24/7 priority assistance', included: true },
          { feature_id: 'voice_guidance', name: 'Voice Guidance', description: 'Audio spiritual sessions', included: true },
          { feature_id: 'personal_mentor', name: 'Personal Spiritual Mentor', description: '1-on-1 guidance sessions', included: true },
          { feature_id: 'custom_programs', name: 'Custom Programs', description: 'Personalized spiritual programs', included: true },
        ],
        limits: {
          messages_per_month: -1,
          wisdom_modules: -1,
          api_requests_per_month: 5000,
        },
      },
    ];
  }

  // ===============================
  // SUBSCRIPTION MANAGEMENT
  // ===============================

  async loadCurrentSubscription(): Promise<Subscription | null> {
    try {
      const response = await fetch(`${this.baseURL}/api/subscription/my-subscriptions`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      const subscriptions = await this.handleResponse<{ subscription: Subscription }[]>(response);
      const activeSubscription = subscriptions.find(sub => 
        sub.subscription.status === 'active' || sub.subscription.status === 'trialing'
      );

      this.currentSubscription = activeSubscription?.subscription || null;
      this.emit('subscriptionLoaded', this.currentSubscription);
      return this.currentSubscription;
    } catch (error) {
      console.error('Failed to load current subscription:', error);
      this.currentSubscription = null;
      return null;
    }
  }

  getCurrentSubscription(): Subscription | null {
    return this.currentSubscription;
  }

  async createSubscription(request: CreateSubscriptionRequest): Promise<Subscription> {
    try {
      const response = await fetch(`${this.baseURL}/api/subscription/create`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(request),
      });

      const result = await this.handleResponse<{ subscription: Subscription }>(response);
      this.currentSubscription = result.subscription;
      this.emit('subscriptionCreated', this.currentSubscription);
      return this.currentSubscription;
    } catch (error) {
      console.error('Failed to create subscription:', error);
      throw error;
    }
  }

  async updateSubscription(subscriptionId: string, updates: { plan_id?: string; billing_interval?: 'monthly' | 'yearly' }): Promise<Subscription> {
    try {
      const response = await fetch(`${this.baseURL}/api/subscription/${subscriptionId}`, {
        method: 'PUT',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(updates),
      });

      const result = await this.handleResponse<{ subscription: Subscription }>(response);
      this.currentSubscription = result.subscription;
      this.emit('subscriptionUpdated', this.currentSubscription);
      return this.currentSubscription;
    } catch (error) {
      console.error('Failed to update subscription:', error);
      throw error;
    }
  }

  async cancelSubscription(subscriptionId: string, immediate: boolean = false, reason?: string): Promise<Subscription> {
    try {
      const response = await fetch(`${this.baseURL}/api/subscription/${subscriptionId}/cancel`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify({ immediate, reason }),
      });

      const result = await this.handleResponse<{ subscription: Subscription }>(response);
      this.currentSubscription = result.subscription;
      this.emit('subscriptionCancelled', this.currentSubscription);
      return this.currentSubscription;
    } catch (error) {
      console.error('Failed to cancel subscription:', error);
      throw error;
    }
  }

  // ===============================
  // PAYMENT METHODS
  // ===============================

  async loadPaymentMethods(): Promise<PaymentMethod[]> {
    try {
      const response = await fetch(`${this.baseURL}/api/payment/methods`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      this.paymentMethods = await this.handleResponse<PaymentMethod[]>(response);
      this.emit('paymentMethodsLoaded', this.paymentMethods);
      return this.paymentMethods;
    } catch (error) {
      console.error('Failed to load payment methods:', error);
      this.paymentMethods = [];
      return this.paymentMethods;
    }
  }

  getPaymentMethods(): PaymentMethod[] {
    return this.paymentMethods;
  }

  async createPaymentMethod(paymentMethodData: CreatePaymentMethodRequest): Promise<PaymentMethod> {
    try {
      const response = await fetch(`${this.baseURL}/api/payment/methods`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(paymentMethodData),
      });

      const paymentMethod = await this.handleResponse<PaymentMethod>(response);
      this.paymentMethods.push(paymentMethod);
      this.emit('paymentMethodAdded', paymentMethod);
      return paymentMethod;
    } catch (error) {
      console.error('Failed to create payment method:', error);
      throw error;
    }
  }

  // ===============================
  // UTILITY METHODS
  // ===============================

  getCurrentPlan(): SubscriptionPlan | null {
    if (!this.currentSubscription) {
      return this.subscriptionPlans.find(plan => plan.tier === 'basic') || null;
    }

    return this.subscriptionPlans.find(plan => plan.id === this.currentSubscription!.plan_id) || null;
  }

  hasActiveSubscription(): boolean {
    return this.currentSubscription?.status === 'active' || this.currentSubscription?.status === 'trialing';
  }

  canUseFeature(featureId: string): boolean {
    const currentPlan = this.getCurrentPlan();
    if (!currentPlan) return false;

    const feature = currentPlan.features.find(f => f.feature_id === featureId);
    return feature?.included || false;
  }

  formatPrice(amount: number, currency: string = 'USD', interval: string = 'month'): string {
    const formatted = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(amount);
    
    return interval === 'year' ? `${formatted}/year` : `${formatted}/month`;
  }

  calculateYearlySavings(monthlyPrice: number): number {
    const yearlyPrice = monthlyPrice * 10; // 20% discount
    const monthlyCost = monthlyPrice * 12;
    return monthlyCost - yearlyPrice;
  }

  // ===============================
  // STATE GETTERS
  // ===============================

  isInitialized(): boolean {
    return !this.isLoading && this.subscriptionPlans.length > 0;
  }

  getLoadingState(): boolean {
    return this.isLoading;
  }
}

// ===============================
// SINGLETON EXPORT
// ===============================

export const subscriptionService = new CentralizedSubscriptionService();
export default subscriptionService;
