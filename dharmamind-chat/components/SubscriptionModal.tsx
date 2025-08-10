import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import { useSubscription, useSubscriptionModal } from '../hooks/useSubscription';
import AuthComponent from './AuthComponent';
import { authService } from '../services/authService';

interface SubscriptionModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentPlan: 'free' | 'professional' | 'enterprise';
}

interface PaymentMethod {
  id: string;
  type: 'card' | 'bank_account';
  last_four: string;
  brand?: string;
  exp_month?: number;
  exp_year?: number;
  default: boolean;
}

interface BillingAddress {
  line1: string;
  line2?: string;
  city: string;
  state: string;
  postal_code: string;
  country: string;
}

const SubscriptionModal: React.FC<SubscriptionModalProps> = ({ isOpen, onClose, currentPlan }) => {
  const router = useRouter();
  const { user, isAuthenticated, isLoading } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [showPaymentForm, setShowPaymentForm] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null);
  const [billingInterval, setBillingInterval] = useState<'monthly' | 'yearly'>('monthly');
  
  // Payment form state
  const [cardholderName, setCardholderName] = useState('');
  const [cardNumber, setCardNumber] = useState('');
  const [expiryDate, setExpiryDate] = useState('');
  const [cvv, setCvv] = useState('');
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [selectedPaymentMethod, setSelectedPaymentMethod] = useState<string | null>(null);
  const [billingAddress, setBillingAddress] = useState<BillingAddress>({
    line1: '',
    city: '',
    state: '',
    postal_code: '',
    country: 'US'
  });

  // Load payment methods on mount - MOVED BEFORE EARLY RETURN
  const loadPaymentMethods = async () => {
    if (!isAuthenticated) return;
    
    try {
      const token = authService.getAccessToken();
      const response = await fetch('/api/payment/methods', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const methods = await response.json();
        setPaymentMethods(methods);
        if (methods.length > 0) {
          setSelectedPaymentMethod(methods[0].id);
        }
      } else if (response.status === 401) {
        // Token expired or invalid, show auth modal
        setShowAuthModal(true);
      }
    } catch (err) {
      console.error('Failed to load payment methods:', err);
      setError('Failed to load payment methods');
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadPaymentMethods();
    }
  }, [isOpen]);

  // Early return AFTER all hooks
  if (!isOpen) return null;

  // Subscription plans with enhanced security and features
  const subscriptionPlans = {
    free: {
      id: 'dharma_free',
      name: 'Dharma Free',
      description: 'Basic spiritual guidance for everyone',
      price: 0,
      currency: 'USD',
      interval: billingInterval,
      trial_days: 0,
      features: [
        'Limited spiritual conversations (10/month)',
        'Access to basic wisdom modules',
        'Community support',
        'Basic meditation guidance'
      ]
    },
    professional: {
      id: 'dharma_pro',
      name: 'Dharma Professional',
      description: 'Advanced spiritual guidance for seekers',
      price: billingInterval === 'monthly' ? 29.99 : 299.99,
      currency: 'USD',
      interval: billingInterval,
      trial_days: 14,
      features: [
        'Unlimited spiritual conversations',
        'Access to all 32 wisdom modules',
        'Personal spiritual insights & tracking',
        'Priority customer support',
        'Advanced meditation guidance',
        'Personalized dharmic practices',
        'Spiritual progress analytics',
        'Mobile app access'
      ]
    },
    enterprise: {
      id: 'dharma_enterprise',
      name: 'Dharma Enterprise',
      description: 'Full-scale dharmic AI for organizations',
      price: billingInterval === 'monthly' ? 499.99 : 4999.99,
      currency: 'USD',
      interval: billingInterval,
      trial_days: 30,
      features: [
        'Everything in Professional',
        'Unlimited API access',
        'Unlimited team members',
        'SSO integration',
        'GDPR/HIPAA compliance',
        'Custom deployment options',
        '24/7 dedicated support',
        'Custom AI model training',
        'On-premise installation',
        'Advanced security controls'
      ]
    }
  };

  const handleUpgradeClick = (planId: string) => {
    // Check authentication first
    if (!isAuthenticated) {
      setShowAuthModal(true);
      return;
    }

    setSelectedPlan(planId);
    if (planId === 'dharma_free') {
      // Handle downgrade to free
      handleSubscriptionChange(planId);
    } else if (paymentMethods.length === 0) {
      setShowPaymentForm(true);
    } else {
      handleSubscriptionChange(planId);
    }
  };

  const handleSubscriptionChange = async (planId: string) => {
    if (planId !== 'dharma_free' && !selectedPaymentMethod && !showPaymentForm) {
      setError('Please select a payment method');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // Create payment method if adding new card
      let paymentMethodId = selectedPaymentMethod;
      
      if (showPaymentForm && planId !== 'dharma_free') {
        paymentMethodId = await createPaymentMethod();
      }

      // Create or update subscription
      const subscriptionResponse = await createSubscription(planId, paymentMethodId);
      
      const planName = subscriptionPlans[planId as keyof typeof subscriptionPlans]?.name;
      setSuccess(`Successfully ${planId === 'dharma_free' ? 'downgraded to' : 'upgraded to'} ${planName}! 🎉`);
      
      // Close modal after success
      setTimeout(() => {
        onClose();
        // Refresh page to update user state
        window.location.reload();
      }, 2000);

    } catch (err: any) {
      console.error('Subscription change error:', err);
      setError(err.message || 'Failed to process subscription change');
    } finally {
      setLoading(false);
    }
  };

  const createPaymentMethod = async (): Promise<string> => {
    // Validate payment form
    if (!cardholderName || !cardNumber || !expiryDate || !cvv || !billingAddress.line1) {
      throw new Error('Please fill in all payment details');
    }

    const token = authService.getAccessToken();
    if (!token) {
      throw new Error('Authentication required');
    }

    const response = await authService.authenticatedFetch('/api/payment/methods', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
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
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('Payment method creation failed:', errorData);
      throw new Error(errorData.detail || 'Failed to save payment method');
    }

    const paymentMethod = await response.json();
    return paymentMethod.payment_method_id;
  };

  const createSubscription = async (planId: string, paymentMethodId: string | null) => {
    const token = authService.getAccessToken();
    if (!token) {
      throw new Error('Authentication required');
    }

    const response = await authService.authenticatedFetch('/api/subscription/create', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        plan_id: planId,
        payment_method_id: paymentMethodId,
        metadata: {
          billing_interval: billingInterval,
          source: 'web_subscription_modal',
          user_email: user?.email
        }
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to process subscription');
    }

    return response.json();
  };

  const formatCardNumber = (value: string) => {
    return value.replace(/\s/g, '').replace(/(.{4})/g, '$1 ').trim();
  };

  const formatExpiryDate = (value: string) => {
    return value.replace(/\D/g, '').replace(/(\d{2})(\d)/, '$1/$2').slice(0, 5);
  };

  return (
    <div className="fixed inset-0 bg-stone-900/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-xl p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-semibold text-gray-900">
            🕉️ Upgrade Your Spiritual Journey
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Error/Success Messages */}
        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {success && (
          <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-green-800">{success}</p>
          </div>
        )}

        {/* Current Plan */}
        <div className="mb-6 p-4 bg-gradient-to-r from-gray-50 to-emerald-50 border border-emerald-200 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Current Plan</p>
          <p className="font-semibold text-gray-900 capitalize">{currentPlan}</p>
          <p className="text-sm text-emerald-600">May your spiritual growth continue with wisdom 🌱</p>
        </div>

        {/* Billing Interval Toggle */}
        <div className="mb-6 flex justify-center">
          <div className="bg-gray-100 p-1 rounded-lg flex">
            <button
              onClick={() => setBillingInterval('monthly')}
              className={`px-4 py-2 rounded-md transition-colors ${
                billingInterval === 'monthly' 
                  ? 'bg-white text-gray-900 shadow-sm' 
                  : 'text-gray-600'
              }`}
            >
              Monthly
            </button>
            <button
              onClick={() => setBillingInterval('yearly')}
              className={`px-4 py-2 rounded-md transition-colors ${
                billingInterval === 'yearly' 
                  ? 'bg-white text-gray-900 shadow-sm' 
                  : 'text-gray-600'
              }`}
            >
              Yearly
              <span className="ml-1 text-green-600 font-medium">Save 20%</span>
            </button>
          </div>
        </div>

        {/* Plans Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          
          {Object.entries(subscriptionPlans).map(([planKey, plan]) => (
            <div 
              key={planKey}
              className={`border rounded-xl p-6 relative ${
                currentPlan === planKey
                  ? 'border-green-500 bg-green-50' 
                  : 'border-gray-200 hover:border-gray-300 hover:shadow-lg transition-all'
              }`}
            >
              {/* Recommended Badge */}
              {planKey === 'professional' && (
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <span className="bg-gradient-to-r from-gray-500 to-emerald-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                    Most Popular
                  </span>
                </div>
              )}

              <div className="text-center mb-4">
                <h3 className="text-xl font-bold text-gray-900 mb-2">{plan.name}</h3>
                <p className="text-gray-600 text-sm mb-4">{plan.description}</p>
                
                <div className="mb-4">
                  <span className="text-3xl font-bold text-gray-900">
                    ${plan.price}
                  </span>
                  <span className="text-gray-600">/{plan.interval}</span>
                  {plan.trial_days > 0 && (
                    <p className="text-green-600 text-sm mt-1">
                      {plan.trial_days}-day free trial
                    </p>
                  )}
                </div>
              </div>
              
              <ul className="space-y-3 mb-6">
                {plan.features.map((feature, index) => (
                  <li key={index} className="flex items-start text-sm text-gray-700">
                    <svg className="w-4 h-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>

              {currentPlan !== planKey && (
                <button
                  onClick={() => handleUpgradeClick(plan.id)}
                  disabled={loading}
                  className={`w-full px-4 py-3 rounded-lg font-medium transition-colors ${
                    planKey === 'professional'
                      ? 'bg-gradient-to-r from-gray-600 to-emerald-600 text-white hover:from-gray-700 hover:to-emerald-700'
                      : planKey === 'free'
                      ? 'bg-stone-500 text-white hover:bg-stone-600'
                      : 'bg-gradient-to-r from-gray-600 to-emerald-600 text-white hover:from-gray-700 hover:to-emerald-700'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Processing...
                    </span>
                  ) : (
                    planKey === 'free' ? 'Downgrade to Free' : `Start ${plan.trial_days > 0 ? 'Free Trial' : 'Subscription'}`
                  )}
                </button>
              )}
            </div>
          ))}
        </div>

        {/* Payment Form */}
        {showPaymentForm && (
          <div className="border-t pt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              🔐 Secure Payment Information
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Card Information */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Card Details</h4>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Cardholder Name
                    </label>
                    <input
                      type="text"
                      value={cardholderName}
                      onChange={(e) => setCardholderName(e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                      placeholder="John Doe"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Card Number
                    </label>
                    <input
                      type="text"
                      value={formatCardNumber(cardNumber)}
                      onChange={(e) => setCardNumber(e.target.value.replace(/\s/g, ''))}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                      placeholder="1234 5678 9012 3456"
                      maxLength={19}
                    />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Expiry Date
                      </label>
                      <input
                        type="text"
                        value={formatExpiryDate(expiryDate)}
                        onChange={(e) => setExpiryDate(e.target.value)}
                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                        placeholder="MM/YY"
                        maxLength={5}
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        CVV
                      </label>
                      <input
                        type="text"
                        value={cvv}
                        onChange={(e) => setCvv(e.target.value.replace(/\D/g, ''))}
                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                        placeholder="123"
                        maxLength={4}
                      />
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Billing Address */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Billing Address</h4>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Street Address
                    </label>
                    <input
                      type="text"
                      value={billingAddress.line1}
                      onChange={(e) => setBillingAddress({...billingAddress, line1: e.target.value})}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                      placeholder="123 Main St"
                    />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        City
                      </label>
                      <input
                        type="text"
                        value={billingAddress.city}
                        onChange={(e) => setBillingAddress({...billingAddress, city: e.target.value})}
                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                        placeholder="New York"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        State
                      </label>
                      <input
                        type="text"
                        value={billingAddress.state}
                        onChange={(e) => setBillingAddress({...billingAddress, state: e.target.value})}
                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                        placeholder="NY"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      ZIP Code
                    </label>
                    <input
                      type="text"
                      value={billingAddress.postal_code}
                      onChange={(e) => setBillingAddress({...billingAddress, postal_code: e.target.value})}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                      placeholder="10001"
                    />
                  </div>
                </div>
              </div>
            </div>
            
            {/* Payment Action */}
            <div className="mt-6 flex justify-end space-x-3">
              <button
                onClick={() => setShowPaymentForm(false)}
                className="px-6 py-3 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => selectedPlan && handleSubscriptionChange(selectedPlan)}
                disabled={loading || !cardholderName || !cardNumber || !expiryDate || !cvv}
                className="px-6 py-3 bg-gradient-to-r from-emerald-600 to-blue-600 text-white rounded-lg hover:from-emerald-700 hover:to-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : 'Complete Payment'}
              </button>
            </div>
          </div>
        )}

        {/* Security & Compliance Footer */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <div className="flex items-center justify-center space-x-6 text-sm text-gray-500">
            <div className="flex items-center">
              <svg className="w-4 h-4 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
              </svg>
              256-bit SSL Encryption
            </div>
            <div className="flex items-center">
              <svg className="w-4 h-4 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              PCI DSS Compliant
            </div>
            <div className="flex items-center">
              <svg className="w-4 h-4 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Money-back Guarantee
            </div>
          </div>
          
          <p className="text-center text-xs text-gray-500 mt-4">
            All subscriptions include dharmic AI guidance, secure data handling, and cancel anytime.
            By subscribing, you agree to our Terms of Service and Privacy Policy.
          </p>
        </div>
      </div>

      {/* Authentication Modal */}
      {showAuthModal && (
        <div className="fixed inset-0 bg-stone-900/75 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-md w-full max-h-[90vh] overflow-auto">
            <AuthComponent 
              onClose={() => {
                setShowAuthModal(false);
                // Reload payment methods after successful auth
                setTimeout(() => {
                  loadPaymentMethods();
                }, 500);
              }}
              mode="login"
              hideDemo={true}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default SubscriptionModal;
