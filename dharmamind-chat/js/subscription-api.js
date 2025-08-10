/**
 * DharmaMind Frontend Subscription API
 * ===================================
 * 
 * Frontend API handlers for subscription and payment functionality.
 * Provides secure communication between frontend and backend subscription services.
 */

// Base API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// API client with authentication
class DharmaSubscriptionAPI {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // Get authentication headers
  getHeaders() {
    const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` })
    };
  }

  // Handle API errors
  async handleResponse(response) {
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  }

  // ===============================
  // SUBSCRIPTION PLANS
  // ===============================

  /**
   * Get all available subscription plans
   */
  async getSubscriptionPlans() {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/subscription/plans`, {
        method: 'GET',
        headers: this.getHeaders()
      });
      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to get subscription plans:', error);
      throw error;
    }
  }

  // ===============================
  // SUBSCRIPTION MANAGEMENT
  // ===============================

  /**
   * Create new subscription
   */
  async createSubscription(planId, paymentMethodId, options = {}) {
    try {
      const requestData = {
        plan_id: planId,
        payment_method_id: paymentMethodId,
        ...options
      };

      const response = await fetch(`${this.baseURL}/api/v1/subscription/create`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(requestData)
      });

      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to create subscription:', error);
      throw error;
    }
  }

  /**
   * Get user's subscriptions
   */
  async getMySubscriptions() {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/subscription/my-subscriptions`, {
        method: 'GET',
        headers: this.getHeaders()
      });
      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to get subscriptions:', error);
      throw error;
    }
  }

  /**
   * Get specific subscription details
   */
  async getSubscription(subscriptionId) {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/subscription/${subscriptionId}`, {
        method: 'GET',
        headers: this.getHeaders()
      });
      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to get subscription:', error);
      throw error;
    }
  }

  /**
   * Update subscription
   */
  async updateSubscription(subscriptionId, updates) {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/subscription/${subscriptionId}`, {
        method: 'PUT',
        headers: this.getHeaders(),
        body: JSON.stringify(updates)
      });
      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to update subscription:', error);
      throw error;
    }
  }

  /**
   * Cancel subscription
   */
  async cancelSubscription(subscriptionId, immediate = false, reason = null) {
    try {
      const params = new URLSearchParams({
        immediate: immediate.toString(),
        ...(reason && { reason })
      });

      const response = await fetch(`${this.baseURL}/api/v1/subscription/${subscriptionId}?${params}`, {
        method: 'DELETE',
        headers: this.getHeaders()
      });

      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to cancel subscription:', error);
      throw error;
    }
  }

  // ===============================
  // PAYMENT METHODS
  // ===============================

  /**
   * Create payment method
   */
  async createPaymentMethod(paymentData) {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/payment/methods`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(paymentData)
      });
      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to create payment method:', error);
      throw error;
    }
  }

  /**
   * Get user's payment methods
   */
  async getPaymentMethods() {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/payment/methods`, {
        method: 'GET',
        headers: this.getHeaders()
      });
      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to get payment methods:', error);
      throw error;
    }
  }

  // ===============================
  // USAGE TRACKING
  // ===============================

  /**
   * Get subscription usage
   */
  async getSubscriptionUsage(subscriptionId) {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/subscription/${subscriptionId}/usage`, {
        method: 'GET',
        headers: this.getHeaders()
      });
      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to get subscription usage:', error);
      throw error;
    }
  }

  /**
   * Get usage summary
   */
  async getUsageSummary(subscriptionId, periodStart = null, periodEnd = null) {
    try {
      const params = new URLSearchParams();
      if (periodStart) params.append('period_start', periodStart);
      if (periodEnd) params.append('period_end', periodEnd);

      const response = await fetch(
        `${this.baseURL}/api/v1/subscription/${subscriptionId}/usage/summary?${params}`,
        {
          method: 'GET',
          headers: this.getHeaders()
        }
      );
      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to get usage summary:', error);
      throw error;
    }
  }

  // ===============================
  // BILLING & INVOICES
  // ===============================

  /**
   * Get user's invoices
   */
  async getInvoices(limit = 10, offset = 0) {
    try {
      const params = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString()
      });

      const response = await fetch(`${this.baseURL}/api/v1/billing/invoices?${params}`, {
        method: 'GET',
        headers: this.getHeaders()
      });
      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to get invoices:', error);
      throw error;
    }
  }

  /**
   * Get specific invoice
   */
  async getInvoice(invoiceId) {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/billing/invoices/${invoiceId}`, {
        method: 'GET',
        headers: this.getHeaders()
      });
      return this.handleResponse(response);
    } catch (error) {
      console.error('Failed to get invoice:', error);
      throw error;
    }
  }

  // ===============================
  // UTILITY METHODS
  // ===============================

  /**
   * Format currency amount
   */
  formatCurrency(amount, currency = 'USD') {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(amount);
  }

  /**
   * Format date
   */
  formatDate(dateString) {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  }

  /**
   * Get plan features description
   */
  getPlanFeatures(planTier) {
    const features = {
      free: [
        '50 monthly conversations',
        'Basic spiritual guidance',
        'Community support',
        '5 essential modules'
      ],
      pro: [
        'Unlimited conversations',
        'All 32 wisdom modules',
        'Personal spiritual insights',
        'Priority support',
        'Advanced meditation guidance'
      ],
      max: [
        'Everything in Professional',
        'API access (10K calls/month)',
        'Custom integrations',
        'Advanced analytics',
        'White-label options'
      ],
      enterprise: [
        'Everything in Max',
        'Unlimited API access',
        'SSO integration',
        'Custom deployment',
        '24/7 dedicated support'
      ]
    };

    return features[planTier] || [];
  }

  /**
   * Calculate savings for yearly billing
   */
  calculateYearlySavings(monthlyPrice) {
    const yearlyPrice = monthlyPrice * 10; // 20% discount
    const monthlyCost = monthlyPrice * 12;
    return monthlyCost - yearlyPrice;
  }
}

// Export singleton instance
const dharmaSubscriptionAPI = new DharmaSubscriptionAPI();
export default dharmaSubscriptionAPI;

// Named exports for specific functions
export const {
  getSubscriptionPlans,
  createSubscription,
  getMySubscriptions,
  getSubscription,
  updateSubscription,
  cancelSubscription,
  createPaymentMethod,
  getPaymentMethods,
  getSubscriptionUsage,
  getUsageSummary,
  getInvoices,
  getInvoice,
  formatCurrency,
  formatDate,
  getPlanFeatures,
  calculateYearlySavings
} = dharmaSubscriptionAPI;
