/**
 * 🕉️ DharmaMind Centralized Subscription Modal
 * 
 * Unified subscription modal using centralized subscription service
 * Replaces complex scattered subscription logic
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useColors } from '../contexts/ColorContext';
import { useSubscription, useSubscriptionModal } from '../hooks/useSubscription';
import { useCentralizedSystem } from './CentralizedSystem';

interface SubscriptionModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentPlan?: 'free' | 'basic' | 'pro' | 'max' | 'professional' | 'enterprise';
  targetPlan?: string;
}

const CentralizedSubscriptionModal: React.FC<SubscriptionModalProps> = ({ 
  isOpen, 
  onClose, 
  currentPlan = 'free',
  targetPlan 
}) => {
  const { user, isAuthenticated } = useAuth();
  const { currentTheme } = useColors();
  const { toggleAuthModal } = useCentralizedSystem();
  const {
    subscriptionPlans,
    paymentMethods,
    currentSubscription,
    isLoading,
    error,
    createSubscription,
    createPaymentMethod,
    clearError
  } = useSubscription();

  const {
    selectedPlan,
    showPaymentForm,
    billingInterval,
    selectPlan,
    togglePaymentForm,
    setBillingInterval
  } = useSubscriptionModal();

  const [processing, setProcessing] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);

  // Map current plan to our new plan names
  const normalizedCurrentPlan = React.useMemo(() => {
    switch (currentPlan) {
      case 'professional':
        return 'pro';
      case 'enterprise':
        return 'max';
      case 'free':
      case 'basic':
      case 'pro':
      case 'max':
        return currentPlan;
      default:
        return 'free';
    }
  }, [currentPlan]);

  // Payment form state
  const [cardholderName, setCardholderName] = useState('');
  const [cardNumber, setCardNumber] = useState('');
  const [expiryDate, setExpiryDate] = useState('');
  const [cvv, setCvv] = useState('');
  const [billingAddress, setBillingAddress] = useState({
    line1: '',
    city: '',
    state: '',
    postal_code: '',
    country: 'US'
  });

  // Initialize with target plan
  useEffect(() => {
    if (targetPlan && isOpen) {
      selectPlan(targetPlan);
    }
  }, [targetPlan, isOpen, selectPlan]);

  // Clear states when modal closes
  useEffect(() => {
    if (!isOpen) {
      clearError();
      setSuccess(null);
      setProcessing(false);
      togglePaymentForm(false);
    }
  }, [isOpen, clearError, togglePaymentForm]);

  const handleUpgradeClick = (planId: string) => {
    if (!isAuthenticated) {
      toggleAuthModal(true);
      return;
    }

    selectPlan(planId);
    
    if (planId === 'dharma_free') {
      handleSubscriptionChange(planId);
    } else if (paymentMethods.length === 0) {
      togglePaymentForm(true);
    } else {
      handleSubscriptionChange(planId);
    }
  };

  const handleSubscriptionChange = async (planId: string) => {
    if (!isAuthenticated) {
      toggleAuthModal(true);
      return;
    }

    if (planId !== 'dharma_basic' && !selectedPlan && !showPaymentForm) {
      clearError();
      return;
    }

    setProcessing(true);
    clearError();
    setSuccess(null);

    try {
      let paymentMethodId = null;

      // Create payment method if adding new card
      if (showPaymentForm && planId !== 'dharma_free') {
        const paymentMethod = await createPaymentMethod({
          method_type: 'credit_card',
          card_details: {
            number: cardNumber.replace(/\s/g, ''),
            exp_month: parseInt(expiryDate.split('/')[0]),
            exp_year: parseInt('20' + expiryDate.split('/')[1]),
            cvc: cvv,
            cardholder_name: cardholderName
          },
          billing_address: billingAddress,
          set_as_default: true
        });
        paymentMethodId = paymentMethod.id;
      } else if (paymentMethods.length > 0) {
        paymentMethodId = paymentMethods.find(pm => pm.default)?.id || paymentMethods[0].id;
      }

      // Create subscription
      const subscription = await createSubscription({
        plan_id: planId,
        payment_method_id: paymentMethodId!,
        billing_interval: billingInterval,
        metadata: {
          source: 'centralized_subscription_modal',
          user_email: user?.email
        }
      });

      const planName = subscriptionPlans.find(p => p.id === planId)?.name;
      setSuccess(`Successfully ${planId === 'dharma_basic' ? 'downgraded to' : 'upgraded to'} ${planName}! 🎉`);
      
      // Close modal after success
      setTimeout(() => {
        onClose();
        window.location.reload(); // Refresh to update user state
      }, 2000);

    } catch (err: any) {
      console.error('Subscription change error:', err);
    } finally {
      setProcessing(false);
    }
  };

  const formatCardNumber = (value: string) => {
    return value.replace(/\s/g, '').replace(/(.{4})/g, '$1 ').trim();
  };

  const formatExpiryDate = (value: string) => {
    return value.replace(/\D/g, '').replace(/(\d{2})(\d)/, '$1/$2').slice(0, 5);
  };

  if (!isOpen) return null;

  // Loading state
  if (isLoading && subscriptionPlans.length === 0) {
    return (
<<<<<<< HEAD
      <div className="fixed inset-0 bg-neutral-900/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
        <div className="bg-neutral-100 rounded-xl p-8 max-w-md w-full text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-600 mx-auto mb-4"></div>
          <p className="text-neutral-600">Loading subscription plans...</p>
=======
      <div className="fixed inset-0 bg-stone-900/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-xl p-8 max-w-md w-full text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading subscription plans...</p>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-overlay flex items-center justify-center p-4 z-50">
      <div className="modal-content p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
<<<<<<< HEAD
          <h2 className="text-2xl font-semibold text-neutral-900">
=======
          <h2 className="text-2xl font-semibold text-primary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            🕉️ Upgrade Your Spiritual Journey
          </h2>
          <button
            onClick={onClose}
<<<<<<< HEAD
            className="text-neutral-600 hover:text-gold-600 transition-colors"
=======
            className="text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Error/Success Messages */}
        {error && (
          <div className="mb-4 p-4 bg-error-light border border-red-200 rounded-lg">
            <p className="text-error">{error}</p>
          </div>
        )}

        {success && (
          <div className="mb-4 p-4 bg-success-light border border-green-200 rounded-lg">
            <p className="text-success">{success}</p>
          </div>
        )}

        {/* Current Plan */}
        <div className="mb-6 p-4 bg-primary-gradient-light border border-medium rounded-lg">
<<<<<<< HEAD
          <p className="text-sm text-neutral-600 mb-1">Current Plan</p>
          <p className="font-semibold text-neutral-900 capitalize">{normalizedCurrentPlan}</p>
=======
          <p className="text-sm text-secondary mb-1">Current Plan</p>
          <p className="font-semibold text-primary capitalize">{normalizedCurrentPlan}</p>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
          <p className="text-sm text-success">May your spiritual growth continue with wisdom 🌱</p>
        </div>

        {/* Billing Interval Toggle */}
        <div className="mb-6 flex justify-center">
          <div className="bg-tertiary-bg p-1 rounded-lg flex">
            <button
              onClick={() => setBillingInterval('monthly')}
              className={`px-4 py-2 rounded-md transition-colors ${
                billingInterval === 'monthly' 
<<<<<<< HEAD
                  ? 'bg-primary-bg text-neutral-900 shadow-sm' 
                  : 'text-neutral-600'
=======
                  ? 'bg-primary-bg text-primary shadow-sm' 
                  : 'text-secondary'
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              }`}
            >
              Monthly
            </button>
            <button
              onClick={() => setBillingInterval('yearly')}
              className={`px-4 py-2 rounded-md transition-colors ${
                billingInterval === 'yearly' 
<<<<<<< HEAD
                  ? 'bg-primary-bg text-neutral-900 shadow-sm' 
                  : 'text-neutral-600'
=======
                  ? 'bg-primary-bg text-primary shadow-sm' 
                  : 'text-secondary'
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              }`}
            >
              Yearly
              <span className="ml-1 text-success font-medium">Save 20%</span>
            </button>
          </div>
        </div>

        {/* Plans Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Basic Plan */}
          <div className="card-primary hover:shadow-lg transition-all">
            <div className="text-center mb-4">
<<<<<<< HEAD
              <h3 className="text-xl font-bold text-neutral-900 mb-2">Basic</h3>
              <p className="text-neutral-600 text-sm mb-4">Perfect for personal spiritual growth</p>
              
              <div className="mb-4">
                <span className="text-3xl font-bold text-neutral-900">
                  ${billingInterval === 'monthly' ? '9' : '90'}
                </span>
                <span className="text-neutral-600">/{billingInterval === 'monthly' ? 'month' : 'year'}</span>
=======
              <h3 className="text-xl font-bold text-primary mb-2">Basic</h3>
              <p className="text-secondary text-sm mb-4">Perfect for personal spiritual growth</p>
              
              <div className="mb-4">
                <span className="text-3xl font-bold text-primary">
                  ${billingInterval === 'monthly' ? '9' : '90'}
                </span>
                <span className="text-secondary">/{billingInterval === 'monthly' ? 'month' : 'year'}</span>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                {billingInterval === 'yearly' && (
                  <p className="text-success text-sm mt-1">
                    Save $18/year
                  </p>
                )}
              </div>
            </div>
            
            <ul className="space-y-3 mb-6">
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                100 AI conversations/month
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Basic dharmic guidance
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Email support
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Basic meditation guidance
              </li>
            </ul>

            {normalizedCurrentPlan !== 'basic' && (
              <button
                onClick={() => handleUpgradeClick('dharma_basic')}
                disabled={processing}
                className="btn-outline w-full px-4 py-3 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {processing ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </span>
                ) : (
                  'Start Basic Plan'
                )}
              </button>
            )}
          </div>

          {/* Pro Plan */}
          <div className="card-elevated bg-primary-gradient-light border-2 border-medium hover:shadow-lg transition-all relative">
            {/* Most Popular Badge */}
            <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
              <span className="bg-primary-gradient text-inverse px-3 py-1 rounded-full text-sm font-medium">
                Most Popular
              </span>
            </div>
            
            <div className="text-center mb-4">
<<<<<<< HEAD
              <h3 className="text-xl font-bold text-neutral-900 mb-2">Pro</h3>
              <p className="text-neutral-600 text-sm mb-4">Advanced spiritual guidance and insights</p>
              
              <div className="mb-4">
                <span className="text-3xl font-bold text-neutral-900">
                  ${billingInterval === 'monthly' ? '19' : '190'}
                </span>
                <span className="text-neutral-600">/{billingInterval === 'monthly' ? 'month' : 'year'}</span>
=======
              <h3 className="text-xl font-bold text-primary mb-2">Pro</h3>
              <p className="text-secondary text-sm mb-4">Advanced spiritual guidance and insights</p>
              
              <div className="mb-4">
                <span className="text-3xl font-bold text-primary">
                  ${billingInterval === 'monthly' ? '19' : '190'}
                </span>
                <span className="text-secondary">/{billingInterval === 'monthly' ? 'month' : 'year'}</span>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                {billingInterval === 'yearly' && (
                  <p className="text-success text-sm mt-1">
                    Save $38/year
                  </p>
                )}
              </div>
            </div>
            
            <ul className="space-y-3 mb-6">
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Unlimited AI conversations
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Advanced dharmic wisdom
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Priority support
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Personalized meditations
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Voice interaction
              </li>
            </ul>

            {normalizedCurrentPlan !== 'pro' && (
              <button
                onClick={() => handleUpgradeClick('dharma_pro')}
                disabled={processing}
                className="btn-primary w-full px-4 py-3 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {processing ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </span>
                ) : (
                  'Start Pro Plan'
                )}
              </button>
            )}
          </div>

          {/* Max Plan */}
          <div className="card-elevated bg-secondary-gradient-light border-2 border-medium hover:shadow-lg transition-all relative">
            <div className="text-center mb-4">
<<<<<<< HEAD
              <h3 className="text-xl font-bold text-neutral-900 mb-2">Max</h3>
              <p className="text-neutral-600 text-sm mb-4">Ultimate spiritual transformation</p>
              
              <div className="mb-4">
                <span className="text-3xl font-bold text-neutral-900">
                  ${billingInterval === 'monthly' ? '39' : '390'}
                </span>
                <span className="text-neutral-600">/{billingInterval === 'monthly' ? 'month' : 'year'}</span>
=======
              <h3 className="text-xl font-bold text-primary mb-2">Max</h3>
              <p className="text-secondary text-sm mb-4">Ultimate spiritual transformation</p>
              
              <div className="mb-4">
                <span className="text-3xl font-bold text-primary">
                  ${billingInterval === 'monthly' ? '39' : '390'}
                </span>
                <span className="text-secondary">/{billingInterval === 'monthly' ? 'month' : 'year'}</span>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                {billingInterval === 'yearly' && (
                  <p className="text-success text-sm mt-1">
                    Save $78/year
                  </p>
                )}
              </div>
            </div>
            
            <ul className="space-y-3 mb-6">
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Everything in Pro
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                1-on-1 spiritual mentoring
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Custom dharmic programs
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                24/7 premium support
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Early access to features
              </li>
<<<<<<< HEAD
              <li className="flex items-start text-sm text-neutral-600">
=======
              <li className="flex items-start text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <svg className="w-4 h-4 text-success mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Exclusive masterclasses
              </li>
            </ul>

            {currentPlan !== 'max' && (
              <button
                onClick={() => handleUpgradeClick('dharma_max')}
                disabled={processing}
                className="btn-secondary w-full px-4 py-3 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {processing ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </span>
                ) : (
                  'Start Max Plan'
                )}
              </button>
            )}
          </div>
        </div>

        {/* Payment Form */}
        {showPaymentForm && (
          <div className="border-t border-medium pt-6">
<<<<<<< HEAD
            <h3 className="text-lg font-semibold text-neutral-900 mb-4">
=======
            <h3 className="text-lg font-semibold text-primary mb-4">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              🔐 Secure Payment Information
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Card Information */}
              <div>
<<<<<<< HEAD
                <h4 className="font-medium text-neutral-900 mb-3">Card Details</h4>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-neutral-600 mb-1">
=======
                <h4 className="font-medium text-primary mb-3">Card Details</h4>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-secondary mb-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                      Cardholder Name
                    </label>
                    <input
                      type="text"
                      value={cardholderName}
                      onChange={(e) => setCardholderName(e.target.value)}
                      className="input-primary w-full p-3 border rounded-lg"
                      placeholder="John Doe"
                    />
                  </div>
                  
                  <div>
<<<<<<< HEAD
                    <label className="block text-sm font-medium text-neutral-600 mb-1">
=======
                    <label className="block text-sm font-medium text-secondary mb-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                      Card Number
                    </label>
                    <input
                      type="text"
                      value={formatCardNumber(cardNumber)}
                      onChange={(e) => setCardNumber(e.target.value.replace(/\s/g, ''))}
                      className="input-primary w-full p-3 border rounded-lg"
                      placeholder="1234 5678 9012 3456"
                      maxLength={19}
                    />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
<<<<<<< HEAD
                      <label className="block text-sm font-medium text-neutral-600 mb-1">
=======
                      <label className="block text-sm font-medium text-secondary mb-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        Expiry Date
                      </label>
                      <input
                        type="text"
                        value={formatExpiryDate(expiryDate)}
                        onChange={(e) => setExpiryDate(e.target.value)}
                        className="input-primary w-full p-3 border rounded-lg"
                        placeholder="MM/YY"
                        maxLength={5}
                      />
                    </div>
                    
                    <div>
<<<<<<< HEAD
                      <label className="block text-sm font-medium text-neutral-600 mb-1">
=======
                      <label className="block text-sm font-medium text-secondary mb-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        CVV
                      </label>
                      <input
                        type="text"
                        value={cvv}
                        onChange={(e) => setCvv(e.target.value.replace(/\D/g, ''))}
                        className="input-primary w-full p-3 border rounded-lg"
                        placeholder="123"
                        maxLength={4}
                      />
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Billing Address */}
              <div>
<<<<<<< HEAD
                <h4 className="font-medium text-neutral-900 mb-3">Billing Address</h4>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-neutral-600 mb-1">
=======
                <h4 className="font-medium text-primary mb-3">Billing Address</h4>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-secondary mb-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                      Street Address
                    </label>
                    <input
                      type="text"
                      value={billingAddress.line1}
                      onChange={(e) => setBillingAddress({...billingAddress, line1: e.target.value})}
                      className="input-primary w-full p-3 border rounded-lg"
                      placeholder="123 Main St"
                    />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
<<<<<<< HEAD
                      <label className="block text-sm font-medium text-neutral-600 mb-1">
=======
                      <label className="block text-sm font-medium text-secondary mb-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        City
                      </label>
                      <input
                        type="text"
                        value={billingAddress.city}
                        onChange={(e) => setBillingAddress({...billingAddress, city: e.target.value})}
                        className="input-primary w-full p-3 border rounded-lg"
                        placeholder="New York"
                      />
                    </div>
                    
                    <div>
<<<<<<< HEAD
                      <label className="block text-sm font-medium text-neutral-600 mb-1">
=======
                      <label className="block text-sm font-medium text-secondary mb-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        State
                      </label>
                      <input
                        type="text"
                        value={billingAddress.state}
                        onChange={(e) => setBillingAddress({...billingAddress, state: e.target.value})}
                        className="input-primary w-full p-3 border rounded-lg"
                        placeholder="NY"
                      />
                    </div>
                  </div>
                  
                  <div>
<<<<<<< HEAD
                    <label className="block text-sm font-medium text-neutral-600 mb-1">
=======
                    <label className="block text-sm font-medium text-secondary mb-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                      ZIP Code
                    </label>
                    <input
                      type="text"
                      value={billingAddress.postal_code}
                      onChange={(e) => setBillingAddress({...billingAddress, postal_code: e.target.value})}
                      className="input-primary w-full p-3 border rounded-lg"
                      placeholder="10001"
                    />
                  </div>
                </div>
              </div>
            </div>
            
            {/* Payment Action */}
            <div className="mt-6 flex justify-end space-x-3">
              <button
                onClick={() => togglePaymentForm(false)}
                className="btn-outline px-6 py-3 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={() => selectedPlan && handleSubscriptionChange(selectedPlan)}
                disabled={processing || !cardholderName || !cardNumber || !expiryDate || !cvv}
                className="btn-primary px-6 py-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {processing ? 'Processing...' : 'Complete Payment'}
              </button>
            </div>
          </div>
        )}

        {/* Security Footer */}
        <div className="mt-8 pt-6 border-t border-medium">
<<<<<<< HEAD
          <div className="flex items-center justify-center space-x-6 text-sm text-neutral-500">
=======
          <div className="flex items-center justify-center space-x-6 text-sm text-tertiary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            <div className="flex items-center">
              <svg className="w-4 h-4 mr-2 text-success" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
              </svg>
              256-bit SSL Encryption
            </div>
            <div className="flex items-center">
              <svg className="w-4 h-4 mr-2 text-success" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              PCI DSS Compliant
            </div>
          </div>
          
<<<<<<< HEAD
          <p className="text-center text-xs text-neutral-500 mt-4">
=======
          <p className="text-center text-xs text-tertiary mt-4">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            All subscriptions include dharmic AI guidance, secure data handling, and cancel anytime.
          </p>
        </div>
      </div>
    </div>
  );
};

export default CentralizedSubscriptionModal;
