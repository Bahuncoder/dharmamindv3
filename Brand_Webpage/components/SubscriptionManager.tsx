import React, { useState, useEffect } from 'react';
import { paymentAPI, Subscription, Plan, PaymentMethod, CreateSubscriptionRequest } from '../services/paymentAPI';
import PaymentMethodManager from './PaymentMethodManager';

interface SubscriptionManagerProps {
  onSubscriptionUpdated?: (subscription: Subscription) => void;
  currentSubscription?: Subscription | null;
}

const SubscriptionManager: React.FC<SubscriptionManagerProps> = ({
  onSubscriptionUpdated,
  currentSubscription
}) => {
  const [subscription, setSubscription] = useState<Subscription | null>(currentSubscription || null);
  const [plans, setPlans] = useState<Plan[]>([]);
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [showPlanSelection, setShowPlanSelection] = useState(false);
  const [showPaymentMethods, setShowPaymentMethods] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadSubscription(),
        loadPlans(),
        loadPaymentMethods()
      ]);
    } catch (error) {
      console.error('Failed to load data:', error);
      setError('Failed to load subscription data');
    } finally {
      setLoading(false);
    }
  };

  const loadSubscription = async () => {
    try {
      const sub = await paymentAPI.getCurrentSubscription();
      setSubscription(sub);
    } catch (error) {
      // User might not have a subscription yet
      setSubscription(null);
    }
  };

  const loadPlans = async () => {
    try {
      const plansData = await paymentAPI.getPlans();
      setPlans(plansData);
    } catch (error) {
      console.error('Failed to load plans:', error);
    }
  };

  const loadPaymentMethods = async () => {
    try {
      const methods = await paymentAPI.getPaymentMethods();
      setPaymentMethods(methods);
    } catch (error) {
      console.error('Failed to load payment methods:', error);
    }
  };

  const handleSubscribe = async (planId: string, paymentMethodId?: string) => {
    setSubmitting(true);
    setError(null);

    try {
      // If no payment method provided, use default
      let pmId = paymentMethodId;
      if (!pmId) {
        const defaultMethod = paymentMethods.find(pm => pm.default);
        if (defaultMethod) {
          pmId = defaultMethod.id;
        } else if (paymentMethods.length > 0) {
          pmId = paymentMethods[0].id;
        } else {
          throw new Error('No payment method available. Please add a payment method first.');
        }
      }

      const subscriptionRequest: CreateSubscriptionRequest = {
        plan_id: planId,
        payment_method_id: pmId!,
        billing_interval: 'monthly' // Default to monthly
      };

      const newSubscription = await paymentAPI.createSubscription(subscriptionRequest);
      setSubscription(newSubscription);
      setSuccess('Subscription created successfully!');
      setShowPlanSelection(false);

      if (onSubscriptionUpdated) {
        onSubscriptionUpdated(newSubscription);
      }
    } catch (error) {
      console.error('Failed to create subscription:', error);
      setError(error instanceof Error ? error.message : 'Failed to create subscription');
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancelSubscription = async () => {
    if (!subscription) return;

    if (!confirm('Are you sure you want to cancel your subscription? This action cannot be undone.')) {
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      await paymentAPI.cancelSubscription(subscription.id);
      
      // Reload subscription to get updated status
      await loadSubscription();
      
      setSuccess('Subscription cancelled successfully');

      if (onSubscriptionUpdated && subscription) {
        onSubscriptionUpdated({ ...subscription, status: 'cancelled' });
      }
    } catch (error) {
      console.error('Failed to cancel subscription:', error);
      setError('Failed to cancel subscription');
    } finally {
      setSubmitting(false);
    }
  };

  const handleUpgrade = async (newPlanId: string) => {
    if (!subscription) return;

    setSubmitting(true);
    setError(null);

    try {
      const updatedSubscription = await paymentAPI.updateSubscription(subscription.id, {
        plan_id: newPlanId
      });
      
      setSubscription(updatedSubscription);
      setSuccess('Subscription upgraded successfully!');
      setShowPlanSelection(false);

      if (onSubscriptionUpdated) {
        onSubscriptionUpdated(updatedSubscription);
      }
    } catch (error) {
      console.error('Failed to upgrade subscription:', error);
      setError('Failed to upgrade subscription');
    } finally {
      setSubmitting(false);
    }
  };

  const formatPrice = (amount: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(amount / 100); // Convert cents to dollars
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'trialing':
        return 'bg-blue-100 text-blue-800';
      case 'past_due':
        return 'bg-emerald-100 text-emerald-800';
      case 'cancelled':
      case 'canceled':
        return 'bg-red-100 text-red-800';
      case 'unpaid':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getPlanRecommendation = (plan: Plan) => {
    if (plan.name.toLowerCase().includes('premium') || plan.name.toLowerCase().includes('pro')) {
      return 'Most Popular';
    }
    if (plan.name.toLowerCase().includes('enterprise')) {
      return 'Best Value';
    }
    return null;
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-600 mx-auto"></div>
        <p className="text-gray-600 mt-4">Loading subscription information...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Alerts */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <svg className="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <p className="ml-3 text-sm text-red-800">{error}</p>
          </div>
        </div>
      )}

      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex">
            <svg className="w-5 h-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <p className="ml-3 text-sm text-green-800">{success}</p>
          </div>
        </div>
      )}

      {/* Current Subscription */}
      {subscription ? (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Current Subscription</h2>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(subscription.status)}`}>
              {subscription.status.charAt(0).toUpperCase() + subscription.status.slice(1)}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Plan Details</h3>
              <div className="space-y-2 text-sm">
                <p><span className="text-gray-600">Plan:</span> <span className="font-medium">{subscription.plan_name}</span></p>
                <p><span className="text-gray-600">Price:</span> <span className="font-medium">{formatPrice(subscription.amount || 0)} / {subscription.billing_interval}</span></p>
                <p><span className="text-gray-600">Started:</span> {formatDate(subscription.created_at)}</p>
                {subscription.current_period_end && (
                  <p><span className="text-gray-600">Next billing:</span> {formatDate(subscription.current_period_end)}</p>
                )}
              </div>
            </div>

            <div>
              <h3 className="font-medium text-gray-900 mb-2">Usage & Limits</h3>
              <div className="space-y-2 text-sm">
                {subscription.usage_data && (
                  <>
                    <p><span className="text-gray-600">Sessions used:</span> <span className="font-medium">{subscription.usage_data.sessions_used || 0}</span></p>
                    <p><span className="text-gray-600">Sessions limit:</span> <span className="font-medium">{subscription.usage_data.sessions_limit || 'Unlimited'}</span></p>
                  </>
                )}
                {subscription.trial_end && new Date(subscription.trial_end) > new Date() && (
                  <p><span className="text-gray-600">Trial ends:</span> {formatDate(subscription.trial_end)}</p>
                )}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-3 mt-6 pt-6 border-t border-gray-200">
            <button
              onClick={() => setShowPlanSelection(true)}
              className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-4 py-2 rounded-lg hover:from-amber-700 hover:to-emerald-700 transition-colors text-sm font-medium"
            >
              Upgrade Plan
            </button>
            <button
              onClick={() => setShowPaymentMethods(true)}
              className="border border-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium"
            >
              Manage Payment Methods
            </button>
            {subscription.status !== 'cancelled' && (
              <button
                onClick={handleCancelSubscription}
                disabled={submitting}
                className="border border-red-300 text-red-700 px-4 py-2 rounded-lg hover:bg-red-50 transition-colors text-sm font-medium disabled:opacity-50"
              >
                {submitting ? 'Cancelling...' : 'Cancel Subscription'}
              </button>
            )}
          </div>
        </div>
      ) : (
        /* No Subscription - Show Plans */
        <div className="text-center py-12">
          <div className="max-w-md mx-auto">
            <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
            </svg>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">No Active Subscription</h2>
            <p className="text-gray-600 mb-6">Choose a plan to unlock all features and start your spiritual journey</p>
            <button
              onClick={() => setShowPlanSelection(true)}
              className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-6 py-3 rounded-lg hover:from-amber-700 hover:to-emerald-700 transition-colors font-medium"
            >
              View Plans
            </button>
          </div>
        </div>
      )}

      {/* Plan Selection Modal */}
      {showPlanSelection && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">
                  {subscription ? 'Upgrade Your Plan' : 'Choose Your Plan'}
                </h2>
                <button
                  onClick={() => setShowPlanSelection(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {plans.map((plan) => {
                  const isCurrentPlan = subscription?.plan_id === plan.id;
                  const recommendation = getPlanRecommendation(plan);

                  return (
                    <div
                      key={plan.id}
                      className={`relative border rounded-lg p-6 ${
                        recommendation ? 'border-emerald-300 bg-emerald-50' : 'border-gray-200 bg-white'
                      } ${isCurrentPlan ? 'ring-2 ring-emerald-500' : ''}`}
                    >
                      {recommendation && (
                        <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                          <span className="bg-emerald-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                            {recommendation}
                          </span>
                        </div>
                      )}
                      
                      {isCurrentPlan && (
                        <div className="absolute -top-3 right-4">
                          <span className="bg-emerald-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                            Current Plan
                          </span>
                        </div>
                      )}

                      <div className="text-center mb-6">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">{plan.name}</h3>
                        <div className="text-3xl font-bold text-gray-900 mb-1">
                          {formatPrice(plan.price)}
                        </div>
                        <p className="text-gray-600">per {plan.billing_interval}</p>
                      </div>

                      {plan.description && (
                        <p className="text-gray-600 text-sm mb-4">{plan.description}</p>
                      )}

                      {plan.features && plan.features.length > 0 && (
                        <ul className="space-y-2 mb-6">
                          {plan.features && plan.features.map((feature: string, index: number) => (
                            <li key={index} className="flex items-center text-sm">
                              <svg className="w-4 h-4 text-emerald-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                              </svg>
                              {feature}
                            </li>
                          ))}
                        </ul>
                      )}

                      <button
                        onClick={() => {
                          if (subscription) {
                            handleUpgrade(plan.id);
                          } else {
                            handleSubscribe(plan.id);
                          }
                        }}
                        disabled={submitting || isCurrentPlan}
                        className={`w-full py-2 px-4 rounded-lg font-medium transition-colors ${
                          isCurrentPlan
                            ? 'bg-gray-100 text-gray-500 cursor-not-allowed'
                            : recommendation
                            ? 'bg-gradient-to-r from-amber-600 to-emerald-600 text-white hover:from-amber-700 hover:to-emerald-700'
                            : 'border border-gray-300 text-gray-700 hover:bg-gray-50'
                        } disabled:opacity-50`}
                      >
                        {submitting 
                          ? 'Processing...' 
                          : isCurrentPlan 
                          ? 'Current Plan' 
                          : subscription 
                          ? 'Upgrade' 
                          : 'Subscribe'
                        }
                      </button>
                    </div>
                  );
                })}
              </div>

              {!subscription && paymentMethods.length === 0 && (
                <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-blue-800 text-sm">
                    <strong>Note:</strong> You'll need to add a payment method before subscribing to a plan.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Payment Methods Modal */}
      {showPaymentMethods && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">Manage Payment Methods</h2>
                <button
                  onClick={() => setShowPaymentMethods(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <div className="p-6">
              <PaymentMethodManager
                onPaymentMethodAdded={(method) => {
                  setPaymentMethods(prev => [...prev, method]);
                  setSuccess('Payment method added successfully');
                }}
                onPaymentMethodDeleted={(methodId) => {
                  setPaymentMethods(prev => prev.filter(pm => pm.id !== methodId));
                  setSuccess('Payment method deleted successfully');
                }}
                onPaymentMethodSetDefault={(method) => {
                  setPaymentMethods(prev => 
                    prev.map(pm => ({
                      ...pm,
                      default: pm.id === method.id
                    }))
                  );
                  setSuccess('Default payment method updated');
                }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SubscriptionManager;
