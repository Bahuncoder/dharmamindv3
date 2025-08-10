/**
 * DharmaMind Subscription Integration Module
 * =========================================
 * 
 * Complete integration for subscription functionality including
 * UI components, payment processing, and state management.
 */

import dharmaSubscriptionAPI from './subscription-api.js';

class DharmaSubscriptionManager {
  constructor() {
    this.currentSubscription = null;
    this.subscriptionPlans = [];
    this.paymentMethods = [];
    this.isLoading = false;
    this.isInitialized = false;
    
    // Event listeners for subscription updates
    this.listeners = new Map();
    
    // Initialize on construction
    this.initialize();
  }

  /**
   * Initialize subscription manager
   */
  async initialize() {
    try {
      this.isLoading = true;
      await this.loadSubscriptionPlans();
      await this.loadCurrentSubscription();
      await this.loadPaymentMethods();
      this.isInitialized = true;
      this.emit('initialized');
    } catch (error) {
      console.error('Failed to initialize subscription manager:', error);
      this.emit('error', error);
    } finally {
      this.isLoading = false;
    }
  }

  // ===============================
  // EVENT MANAGEMENT
  // ===============================

  /**
   * Add event listener
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  /**
   * Remove event listener
   */
  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Emit event
   */
  emit(event, data = null) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }
  }

  // ===============================
  // SUBSCRIPTION MANAGEMENT
  // ===============================

  /**
   * Load available subscription plans
   */
  async loadSubscriptionPlans() {
    try {
      this.subscriptionPlans = await dharmaSubscriptionAPI.getSubscriptionPlans();
      this.emit('plansLoaded', this.subscriptionPlans);
      return this.subscriptionPlans;
    } catch (error) {
      console.error('Failed to load subscription plans:', error);
      throw error;
    }
  }

  /**
   * Load current user subscription
   */
  async loadCurrentSubscription() {
    try {
      const subscriptions = await dharmaSubscriptionAPI.getMySubscriptions();
      this.currentSubscription = subscriptions.find(sub => sub.status === 'active') || null;
      this.emit('subscriptionLoaded', this.currentSubscription);
      return this.currentSubscription;
    } catch (error) {
      console.error('Failed to load current subscription:', error);
      // Don't throw error if user is not authenticated
      this.currentSubscription = null;
      return null;
    }
  }

  /**
   * Load user payment methods
   */
  async loadPaymentMethods() {
    try {
      this.paymentMethods = await dharmaSubscriptionAPI.getPaymentMethods();
      this.emit('paymentMethodsLoaded', this.paymentMethods);
      return this.paymentMethods;
    } catch (error) {
      console.error('Failed to load payment methods:', error);
      this.paymentMethods = [];
      return [];
    }
  }

  /**
   * Subscribe to a plan
   */
  async subscribeToPlan(planId, paymentMethodId, options = {}) {
    try {
      this.isLoading = true;
      this.emit('subscriptionStarting', { planId, paymentMethodId });

      const subscription = await dharmaSubscriptionAPI.createSubscription(
        planId, 
        paymentMethodId, 
        options
      );

      this.currentSubscription = subscription;
      this.emit('subscriptionCreated', subscription);
      this.emit('subscriptionChanged', subscription);

      return subscription;
    } catch (error) {
      console.error('Failed to subscribe to plan:', error);
      this.emit('subscriptionError', error);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Upgrade subscription
   */
  async upgradeSubscription(newPlanId) {
    if (!this.currentSubscription) {
      throw new Error('No active subscription to upgrade');
    }

    try {
      this.isLoading = true;
      this.emit('upgradeStarting', { newPlanId });

      const updatedSubscription = await dharmaSubscriptionAPI.updateSubscription(
        this.currentSubscription.id,
        { plan_id: newPlanId }
      );

      this.currentSubscription = updatedSubscription;
      this.emit('subscriptionUpgraded', updatedSubscription);
      this.emit('subscriptionChanged', updatedSubscription);

      return updatedSubscription;
    } catch (error) {
      console.error('Failed to upgrade subscription:', error);
      this.emit('upgradeError', error);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Cancel subscription
   */
  async cancelSubscription(immediate = false, reason = null) {
    if (!this.currentSubscription) {
      throw new Error('No active subscription to cancel');
    }

    try {
      this.isLoading = true;
      this.emit('cancellationStarting', { immediate, reason });

      await dharmaSubscriptionAPI.cancelSubscription(
        this.currentSubscription.id,
        immediate,
        reason
      );

      if (immediate) {
        this.currentSubscription = null;
      } else {
        // Update status to indicate pending cancellation
        this.currentSubscription.status = 'pending_cancellation';
      }

      this.emit('subscriptionCancelled', { immediate, reason });
      this.emit('subscriptionChanged', this.currentSubscription);

      return true;
    } catch (error) {
      console.error('Failed to cancel subscription:', error);
      this.emit('cancellationError', error);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  // ===============================
  // PAYMENT METHODS
  // ===============================

  /**
   * Add payment method
   */
  async addPaymentMethod(paymentData) {
    try {
      this.isLoading = true;
      
      const paymentMethod = await dharmaSubscriptionAPI.createPaymentMethod(paymentData);
      this.paymentMethods.push(paymentMethod);
      
      this.emit('paymentMethodAdded', paymentMethod);
      return paymentMethod;
    } catch (error) {
      console.error('Failed to add payment method:', error);
      this.emit('paymentMethodError', error);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  // ===============================
  // USAGE TRACKING
  // ===============================

  /**
   * Get current usage
   */
  async getCurrentUsage() {
    if (!this.currentSubscription) {
      return null;
    }

    try {
      const usage = await dharmaSubscriptionAPI.getSubscriptionUsage(
        this.currentSubscription.id
      );
      this.emit('usageLoaded', usage);
      return usage;
    } catch (error) {
      console.error('Failed to get current usage:', error);
      throw error;
    }
  }

  /**
   * Check if feature is available
   */
  canUseFeature(featureName) {
    if (!this.currentSubscription) {
      // Check free tier limits
      return this.checkFreeTierFeature(featureName);
    }

    const plan = this.subscriptionPlans.find(
      p => p.id === this.currentSubscription.plan_id
    );

    if (!plan) {
      return false;
    }

    // Check plan features
    return plan.features && plan.features.includes(featureName);
  }

  /**
   * Check free tier feature availability
   */
  checkFreeTierFeature(featureName) {
    const freeTierFeatures = [
      'basic_chat',
      'community_support',
      'essential_modules'
    ];

    return freeTierFeatures.includes(featureName);
  }

  // ===============================
  // UI INTEGRATION
  // ===============================

  /**
   * Show subscription modal
   */
  showSubscriptionModal(targetPlan = null) {
    const modal = document.getElementById('subscription-modal');
    if (modal) {
      if (targetPlan) {
        this.emit('modalOpening', { targetPlan });
        // Pre-select the target plan
        const planSelector = modal.querySelector(`[data-plan="${targetPlan}"]`);
        if (planSelector) {
          planSelector.click();
        }
      }
      modal.style.display = 'flex';
      this.emit('modalOpened', { targetPlan });
    }
  }

  /**
   * Hide subscription modal
   */
  hideSubscriptionModal() {
    const modal = document.getElementById('subscription-modal');
    if (modal) {
      modal.style.display = 'none';
      this.emit('modalClosed');
    }
  }

  /**
   * Update UI elements with subscription info
   */
  updateUI() {
    this.updateSubscriptionStatus();
    this.updatePlanFeatures();
    this.updateUsageLimits();
  }

  /**
   * Update subscription status in UI
   */
  updateSubscriptionStatus() {
    const statusElements = document.querySelectorAll('[data-subscription-status]');
    statusElements.forEach(element => {
      if (this.currentSubscription) {
        element.textContent = this.currentSubscription.plan_name || 'Active';
        element.classList.add('active');
      } else {
        element.textContent = 'Free';
        element.classList.remove('active');
      }
    });
  }

  /**
   * Update plan features in UI
   */
  updatePlanFeatures() {
    const featureElements = document.querySelectorAll('[data-feature]');
    featureElements.forEach(element => {
      const feature = element.dataset.feature;
      const isAvailable = this.canUseFeature(feature);
      
      element.classList.toggle('available', isAvailable);
      element.classList.toggle('unavailable', !isAvailable);
    });
  }

  /**
   * Update usage limits in UI
   */
  async updateUsageLimits() {
    try {
      const usage = await this.getCurrentUsage();
      if (usage) {
        const usageElements = document.querySelectorAll('[data-usage-type]');
        usageElements.forEach(element => {
          const usageType = element.dataset.usageType;
          if (usage[usageType]) {
            element.textContent = `${usage[usageType].used}/${usage[usageType].limit}`;
          }
        });
      }
    } catch (error) {
      console.error('Failed to update usage limits:', error);
    }
  }

  // ===============================
  // UTILITY METHODS
  // ===============================

  /**
   * Get current plan info
   */
  getCurrentPlan() {
    if (!this.currentSubscription) {
      return {
        id: 'free',
        name: 'Free',
        tier: 'free',
        price: 0,
        features: dharmaSubscriptionAPI.getPlanFeatures('free')
      };
    }

    return this.subscriptionPlans.find(
      p => p.id === this.currentSubscription.plan_id
    );
  }

  /**
   * Check if user has active subscription
   */
  hasActiveSubscription() {
    return this.currentSubscription && this.currentSubscription.status === 'active';
  }

  /**
   * Get subscription expiry date
   */
  getSubscriptionExpiry() {
    if (!this.currentSubscription) {
      return null;
    }
    return new Date(this.currentSubscription.current_period_end);
  }

  /**
   * Format price display
   */
  formatPrice(amount, currency = 'USD', interval = 'month') {
    const formatted = dharmaSubscriptionAPI.formatCurrency(amount, currency);
    return interval === 'year' ? `${formatted}/year` : `${formatted}/month`;
  }
}

// Create global instance
window.dharmaSubscriptionManager = new DharmaSubscriptionManager();

// Initialize subscription UI when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initializeSubscriptionUI();
});

/**
 * Initialize subscription UI components
 */
function initializeSubscriptionUI() {
  const subscriptionManager = window.dharmaSubscriptionManager;

  // Add event listeners for subscription events
  subscriptionManager.on('initialized', () => {
    console.log('Subscription manager initialized');
    subscriptionManager.updateUI();
  });

  subscriptionManager.on('subscriptionChanged', (subscription) => {
    console.log('Subscription changed:', subscription);
    subscriptionManager.updateUI();
  });

  // Add click handlers for subscription buttons
  document.addEventListener('click', (event) => {
    const target = event.target;

    // Subscription modal trigger
    if (target.matches('[data-open-subscription]')) {
      event.preventDefault();
      const targetPlan = target.dataset.targetPlan;
      subscriptionManager.showSubscriptionModal(targetPlan);
    }

    // Close subscription modal
    if (target.matches('[data-close-subscription]')) {
      event.preventDefault();
      subscriptionManager.hideSubscriptionModal();
    }

    // Feature upgrade prompts
    if (target.matches('[data-require-upgrade]')) {
      const feature = target.dataset.requireUpgrade;
      if (!subscriptionManager.canUseFeature(feature)) {
        event.preventDefault();
        showUpgradePrompt(feature);
      }
    }
  });
}

/**
 * Show upgrade prompt for locked features
 */
function showUpgradePrompt(feature) {
  const modal = document.createElement('div');
  modal.className = 'upgrade-prompt-modal';
  modal.innerHTML = `
    <div class="upgrade-prompt-content">
      <h3>🚀 Unlock Advanced Features</h3>
      <p>This feature requires a subscription to unlock its full potential.</p>
      <div class="upgrade-actions">
        <button class="btn-primary" onclick="window.dharmaSubscriptionManager.showSubscriptionModal()">
          View Plans
        </button>
        <button class="btn-secondary" onclick="this.closest('.upgrade-prompt-modal').remove()">
          Maybe Later
        </button>
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  
  // Auto-remove after 10 seconds
  setTimeout(() => {
    if (modal.parentNode) {
      modal.remove();
    }
  }, 10000);
}

// Export for module usage
export default window.dharmaSubscriptionManager;
