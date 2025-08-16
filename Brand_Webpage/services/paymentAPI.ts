/**
 * 🕉️ DharmaMind Payment API Service
 * 
 * Professional payment integration with secure backend connectivity
 * Features: Payment methods, subscriptions, billing, security compliance
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ===============================
// TYPES & INTERFACES
// ===============================

export interface PaymentMethod {
  id: string;
  type: 'credit_card' | 'debit_card' | 'bank_transfer' | 'paypal' | 'apple_pay' | 'google_pay' | 'stripe';
  last_four: string;
  brand?: string;
  exp_month?: number;
  exp_year?: number;
  default: boolean;
  created_at: string;
  verified: boolean;
  // Bank transfer specific fields
  bank_name?: string;
  account_type?: 'checking' | 'savings';
  account_holder_name?: string;
  // PayPal specific fields
  paypal_email?: string;
  // Stripe specific fields
  stripe_account_id?: string;
}

export interface CreatePaymentMethodRequest {
  type: PaymentMethod['type'];
  // Credit/Debit card fields
  card_number?: string;
  exp_month?: number;
  exp_year?: number;
  cvc?: string;
  cardholder_name?: string;
  // Bank transfer fields
  bank_name?: string;
  account_number?: string;
  routing_number?: string;
  account_holder_name?: string;
  account_type?: 'checking' | 'savings';
  // PayPal fields
  paypal_email?: string;
  // Stripe fields
  stripe_account_id?: string;
  // Common fields
  billing_address?: BillingAddress;
  save_for_future?: boolean;
  set_as_default?: boolean;
}

export interface BillingAddress {
  line1: string;
  line2?: string;
  city: string;
  state: string;
  postal_code: string;
  country: string;
}

export interface SubscriptionPlan {
  id: string;
  name: string;
  description: string;
  tier: 'free' | 'pro' | 'max' | 'enterprise';
  price: {
    monthly: number;
    yearly: number;
  };
  currency: string;
  features: PlanFeature[];
  limits: PlanLimits;
  popular?: boolean;
}

export interface Plan {
  id: string;
  name: string;
  description?: string;
  price: number;
  currency: string;
  billing_interval: 'monthly' | 'yearly';
  features?: string[];
  trial_period_days?: number;
  created_at: string;
  updated_at: string;
}

export interface CreateSubscriptionRequest {
  plan_id: string;
  payment_method_id: string;
  billing_interval: 'monthly' | 'yearly';
  trial_period_days?: number;
  metadata?: Record<string, any>;
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
  custom_modules?: number;
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
  usage_data?: {
    sessions_used?: number;
    sessions_limit?: number;
    tokens_used?: number;
    tokens_limit?: number;
  };
  created_at: string;
  updated_at: string;
}

export interface Invoice {
  invoice_id: string;
  user_id: string;
  subscription_id: string;
  amount: number;
  currency: string;
  status: 'pending' | 'paid' | 'failed' | 'cancelled';
  created_at: string;
  due_date: string;
  paid_at?: string;
  invoice_url?: string;
  line_items: InvoiceLineItem[];
}

export interface InvoiceLineItem {
  description: string;
  amount: number;
  quantity: number;
  period_start: string;
  period_end: string;
}

export interface UsageSummary {
  subscription_id: string;
  period_start: string;
  period_end: string;
  usage_records: UsageRecord[];
  total_usage: Record<string, number>;
  billing_summary: {
    subtotal: number;
    tax: number;
    total: number;
  };
}

export interface UsageRecord {
  feature_id: string;
  usage_count: number;
  limit: number;
  overage: number;
  cost: number;
}

export interface PaymentIntent {
  intent_id: string;
  amount: number;
  currency: string;
  status: 'requires_payment_method' | 'requires_confirmation' | 'requires_action' | 'processing' | 'succeeded' | 'cancelled';
  client_secret: string;
  next_action?: any;
}

// ===============================
// API SERVICE CLASS
// ===============================

class PaymentAPIService {
  private baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // ===============================
  // AUTHENTICATION HELPERS
  // ===============================

  private async getAuthHeaders(): Promise<Record<string, string>> {
    const token = await auth.getAccessToken();
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    };
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  }

  // ===============================
  // SUBSCRIPTION PLANS
  // ===============================

  async getSubscriptionPlans(): Promise<SubscriptionPlan[]> {
    try {
      const response = await fetch(`${this.baseURL}/subscription/plans`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      return this.handleResponse<SubscriptionPlan[]>(response);
    } catch (error) {
      console.error('Failed to get subscription plans:', error);
      throw error;
    }
  }

  async getPlans(): Promise<Plan[]> {
    try {
      const response = await fetch(`${this.baseURL}/subscription/plans`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      return this.handleResponse<Plan[]>(response);
    } catch (error) {
      console.error('Failed to get plans:', error);
      throw error;
    }
  }

  // ===============================
  // SUBSCRIPTION MANAGEMENT
  // ===============================

  async createSubscription(request: CreateSubscriptionRequest): Promise<Subscription>;
  async createSubscription(planId: string, paymentMethodId: string, billingInterval: 'monthly' | 'yearly'): Promise<Subscription>;
  async createSubscription(
    requestOrPlanId: CreateSubscriptionRequest | string, 
    paymentMethodId?: string, 
    billingInterval?: 'monthly' | 'yearly'
  ): Promise<Subscription> {
    try {
      let requestBody: CreateSubscriptionRequest;
      
      if (typeof requestOrPlanId === 'string') {
        requestBody = {
          plan_id: requestOrPlanId,
          payment_method_id: paymentMethodId!,
          billing_interval: billingInterval!,
        };
      } else {
        requestBody = requestOrPlanId;
      }

      const response = await fetch(`${this.baseURL}/subscription/create`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(requestBody),
      });

      const result = await this.handleResponse<{ subscription: Subscription }>(response);
      return result.subscription;
    } catch (error) {
      console.error('Failed to create subscription:', error);
      throw error;
    }
  }

  async getCurrentSubscription(): Promise<Subscription> {
    try {
      const subscriptions = await this.getMySubscriptions();
      const activeSubscription = subscriptions.find(sub => 
        sub.status === 'active' || sub.status === 'trialing'
      );
      
      if (!activeSubscription) {
        throw new Error('No active subscription found');
      }
      
      return activeSubscription;
    } catch (error) {
      console.error('Failed to get current subscription:', error);
      throw error;
    }
  }

  async getMySubscriptions(): Promise<Subscription[]> {
    try {
      const response = await fetch(`${this.baseURL}/subscription/my-subscriptions`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      const result = await this.handleResponse<{ subscription: Subscription }[]>(response);
      return result.map(r => r.subscription);
    } catch (error) {
      console.error('Failed to get subscriptions:', error);
      throw error;
    }
  }

  async updateSubscription(subscriptionId: string, updates: { plan_id?: string; billing_interval?: 'monthly' | 'yearly' }): Promise<Subscription> {
    try {
      const response = await fetch(`${this.baseURL}/subscription/${subscriptionId}`, {
        method: 'PUT',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(updates),
      });

      const result = await this.handleResponse<{ subscription: Subscription }>(response);
      return result.subscription;
    } catch (error) {
      console.error('Failed to update subscription:', error);
      throw error;
    }
  }

  async cancelSubscription(subscriptionId: string, immediate: boolean = false, reason?: string): Promise<Subscription> {
    try {
      const params = new URLSearchParams();
      if (immediate) params.append('immediate', 'true');
      if (reason) params.append('reason', reason);

      const response = await fetch(`${this.baseURL}/subscription/${subscriptionId}?${params}`, {
        method: 'DELETE',
        headers: await this.getAuthHeaders(),
      });

      const result = await this.handleResponse<{ subscription: Subscription }>(response);
      return result.subscription;
    } catch (error) {
      console.error('Failed to cancel subscription:', error);
      throw error;
    }
  }

  // ===============================
  // PAYMENT METHODS
  // ===============================

  async getPaymentMethods(): Promise<PaymentMethod[]> {
    try {
      const response = await fetch(`${this.baseURL}/payment/methods`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      return this.handleResponse<PaymentMethod[]>(response);
    } catch (error) {
      console.error('Failed to get payment methods:', error);
      throw error;
    }
  }

  async createPaymentMethod(paymentMethodData: CreatePaymentMethodRequest): Promise<PaymentMethod> {
    try {
      const response = await fetch(`${this.baseURL}/payment/methods`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(paymentMethodData),
      });

      return this.handleResponse<PaymentMethod>(response);
    } catch (error) {
      console.error('Failed to create payment method:', error);
      throw error;
    }
  }

  async updatePaymentMethod(paymentMethodId: string, updates: Partial<PaymentMethod>): Promise<PaymentMethod> {
    try {
      const response = await fetch(`${this.baseURL}/payment/methods/${paymentMethodId}`, {
        method: 'PUT',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(updates),
      });

      return this.handleResponse<PaymentMethod>(response);
    } catch (error) {
      console.error('Failed to update payment method:', error);
      throw error;
    }
  }

  async deletePaymentMethod(paymentMethodId: string): Promise<void> {
    try {
      const response = await fetch(`${this.baseURL}/payment/methods/${paymentMethodId}`, {
        method: 'DELETE',
        headers: await this.getAuthHeaders(),
      });

      await this.handleResponse<void>(response);
    } catch (error) {
      console.error('Failed to delete payment method:', error);
      throw error;
    }
  }

  async setDefaultPaymentMethod(paymentMethodId: string): Promise<PaymentMethod> {
    try {
      const response = await fetch(`${this.baseURL}/payment/methods/${paymentMethodId}/set-default`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
      });

      return this.handleResponse<PaymentMethod>(response);
    } catch (error) {
      console.error('Failed to set default payment method:', error);
      throw error;
    }
  }

  // ===============================
  // BILLING & INVOICES
  // ===============================

  async getInvoices(limit: number = 10, offset: number = 0): Promise<Invoice[]> {
    try {
      const params = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString(),
      });

      const response = await fetch(`${this.baseURL}/billing/invoices?${params}`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      return this.handleResponse<Invoice[]>(response);
    } catch (error) {
      console.error('Failed to get invoices:', error);
      throw error;
    }
  }

  async getInvoice(invoiceId: string): Promise<Invoice> {
    try {
      const response = await fetch(`${this.baseURL}/billing/invoices/${invoiceId}`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      return this.handleResponse<Invoice>(response);
    } catch (error) {
      console.error('Failed to get invoice:', error);
      throw error;
    }
  }

  async downloadInvoice(invoiceId: string): Promise<Blob> {
    try {
      const response = await fetch(`${this.baseURL}/billing/invoices/${invoiceId}/download`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      if (!response.ok) {
        throw new Error(`Failed to download invoice: ${response.statusText}`);
      }

      return response.blob();
    } catch (error) {
      console.error('Failed to download invoice:', error);
      throw error;
    }
  }

  // ===============================
  // USAGE TRACKING
  // ===============================

  async getSubscriptionUsage(subscriptionId: string): Promise<UsageSummary> {
    try {
      const response = await fetch(`${this.baseURL}/subscription/${subscriptionId}/usage`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      return this.handleResponse<UsageSummary>(response);
    } catch (error) {
      console.error('Failed to get subscription usage:', error);
      throw error;
    }
  }

  async getUsageSummary(subscriptionId: string, periodStart?: string, periodEnd?: string): Promise<UsageSummary> {
    try {
      const params = new URLSearchParams();
      if (periodStart) params.append('period_start', periodStart);
      if (periodEnd) params.append('period_end', periodEnd);

      const response = await fetch(`${this.baseURL}/subscription/${subscriptionId}/usage/summary?${params}`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      return this.handleResponse<UsageSummary>(response);
    } catch (error) {
      console.error('Failed to get usage summary:', error);
      throw error;
    }
  }

  // ===============================
  // PAYMENT INTENTS (for one-time payments)
  // ===============================

  async createPaymentIntent(amount: number, currency: string = 'USD', paymentMethodId?: string): Promise<PaymentIntent> {
    try {
      const response = await fetch(`${this.baseURL}/payment/intents`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify({
          amount,
          currency,
          payment_method_id: paymentMethodId,
        }),
      });

      return this.handleResponse<PaymentIntent>(response);
    } catch (error) {
      console.error('Failed to create payment intent:', error);
      throw error;
    }
  }

  async confirmPaymentIntent(intentId: string, paymentMethodId?: string): Promise<PaymentIntent> {
    try {
      const response = await fetch(`${this.baseURL}/payment/intents/${intentId}/confirm`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify({
          payment_method_id: paymentMethodId,
        }),
      });

      return this.handleResponse<PaymentIntent>(response);
    } catch (error) {
      console.error('Failed to confirm payment intent:', error);
      throw error;
    }
  }

  // ===============================
  // PAYMENT METHOD SPECIFIC METHODS
  // ===============================

  // Stripe Connect Integration
  async createStripeConnectAccount(email: string, country: string = 'US'): Promise<{ account_id: string; onboarding_url: string }> {
    try {
      const response = await fetch(`${this.baseURL}/payment/stripe/connect/create`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify({ email, country }),
      });

      return this.handleResponse<{ account_id: string; onboarding_url: string }>(response);
    } catch (error) {
      console.error('Failed to create Stripe Connect account:', error);
      throw error;
    }
  }

  async getStripeConnectAccountStatus(accountId: string): Promise<{ status: string; requirements: any }> {
    try {
      const response = await fetch(`${this.baseURL}/payment/stripe/connect/${accountId}/status`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      return this.handleResponse<{ status: string; requirements: any }>(response);
    } catch (error) {
      console.error('Failed to get Stripe Connect account status:', error);
      throw error;
    }
  }

  // PayPal Integration
  async initiatePayPalSetup(email: string): Promise<{ setup_url: string; reference_id: string }> {
    try {
      const response = await fetch(`${this.baseURL}/payment/paypal/setup`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify({ email }),
      });

      return this.handleResponse<{ setup_url: string; reference_id: string }>(response);
    } catch (error) {
      console.error('Failed to initiate PayPal setup:', error);
      throw error;
    }
  }

  async verifyPayPalAccount(referenceId: string): Promise<{ verified: boolean; account_info: any }> {
    try {
      const response = await fetch(`${this.baseURL}/payment/paypal/verify/${referenceId}`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
      });

      return this.handleResponse<{ verified: boolean; account_info: any }>(response);
    } catch (error) {
      console.error('Failed to verify PayPal account:', error);
      throw error;
    }
  }

  // Bank Transfer Integration
  async verifyBankAccount(bankData: {
    bank_name: string;
    account_number: string;
    routing_number: string;
    account_holder_name: string;
    account_type: 'checking' | 'savings';
  }): Promise<{ verification_id: string; micro_deposits: boolean }> {
    try {
      const response = await fetch(`${this.baseURL}/payment/bank/verify`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(bankData),
      });

      return this.handleResponse<{ verification_id: string; micro_deposits: boolean }>(response);
    } catch (error) {
      console.error('Failed to verify bank account:', error);
      throw error;
    }
  }

  async confirmMicroDeposits(verificationId: string, amounts: number[]): Promise<{ verified: boolean }> {
    try {
      const response = await fetch(`${this.baseURL}/payment/bank/verify/${verificationId}/confirm`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify({ amounts }),
      });

      return this.handleResponse<{ verified: boolean }>(response);
    } catch (error) {
      console.error('Failed to confirm micro deposits:', error);
      throw error;
    }
  }

  // ===============================
  // PAYMENT METHOD ICONS & METADATA
  // ===============================

  getPaymentMethodIcon(type: PaymentMethod['type']): string {
    const icons: Record<PaymentMethod['type'], string> = {
      credit_card: '💳',
      debit_card: '💳',
      bank_transfer: '🏦',
      paypal: '🅿️',
      apple_pay: '🍎',
      google_pay: '🟢',
      stripe: '💰'
    };
    return icons[type] || '💳';
  }

  getPaymentMethodDisplayName(type: PaymentMethod['type']): string {
    const names: Record<PaymentMethod['type'], string> = {
      credit_card: 'Credit Card',
      debit_card: 'Debit Card',
      bank_transfer: 'Bank Transfer',
      paypal: 'PayPal',
      apple_pay: 'Apple Pay',
      google_pay: 'Google Pay',
      stripe: 'Stripe Connect'
    };
    return names[type] || 'Payment Method';
  }

  getPaymentMethodDescription(type: PaymentMethod['type']): string {
    const descriptions: Record<PaymentMethod['type'], string> = {
      credit_card: 'Secure credit card processing with instant authorization',
      debit_card: 'Direct debit from your bank account with instant processing',
      bank_transfer: 'ACH bank transfer - takes 3-5 business days to process',
      paypal: 'Pay through your PayPal account with buyer protection',
      apple_pay: 'Quick and secure payments using Touch ID or Face ID',
      google_pay: 'Fast checkout with your Google account credentials',
      stripe: 'Professional payment processing through Stripe Connect'
    };
    return descriptions[type] || 'Secure payment processing';
  }

  // ===============================
  // UTILITY METHODS
  // ===============================

  async validatePaymentMethod(paymentMethodData: CreatePaymentMethodRequest): Promise<{ valid: boolean; errors: string[] }> {
    try {
      const response = await fetch(`${this.baseURL}/payment/methods/validate`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(paymentMethodData),
      });

      return this.handleResponse<{ valid: boolean; errors: string[] }>(response);
    } catch (error) {
      console.error('Failed to validate payment method:', error);
      return { valid: false, errors: ['Validation service unavailable'] };
    }
  }

  formatPrice(amount: number, currency: string = 'USD'): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
    }).format(amount);
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  }

  isCardExpired(expMonth: number, expYear: number): boolean {
    const now = new Date();
    const expiry = new Date(expYear, expMonth - 1, 1);
    return expiry < now;
  }
}

// ===============================
// SINGLETON EXPORT
// ===============================

export const paymentAPI = new PaymentAPIService();

// ===============================
// AUTHENTICATION STUB
// ===============================

// Simple auth stub - in production, this would be a proper auth service
const auth = {
  async getAccessToken(): Promise<string> {
    // In production, get from secure storage or auth provider
    return localStorage.getItem('auth_token') || '';
  }
};

export default paymentAPI;
