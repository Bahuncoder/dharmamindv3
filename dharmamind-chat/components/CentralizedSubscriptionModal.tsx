/**
 * 🕉️ DharmaMind Centralized Subscription Modal
 * 
 * Unified subscription modal using centralized subscription service
 * Replaces complex scattered subscription logic
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useSubscription } from '../contexts/SubscriptionContext';
import { siteConfig } from '../config/shared.config';

interface SubscriptionModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentPlan?: 'basic' | 'pro' | 'max' | 'enterprise';
  targetPlan?: string;
}

const CentralizedSubscriptionModal: React.FC<SubscriptionModalProps> = ({
  isOpen,
  onClose,
  targetPlan
}) => {
  const { user, isAuthenticated } = useAuth();
  const {
    currentPlan,
    availablePlans,
    isLoading,
    upgradePlan,
    isPlanActive
  } = useSubscription();

  // Debug logging
  useEffect(() => {
    console.log('Subscription Modal - availablePlans:', availablePlans);
    console.log('Subscription Modal - currentPlan:', currentPlan);
    console.log('Subscription Modal - isLoading:', isLoading);
  }, [availablePlans, currentPlan, isLoading]);

  const [selectedPlan, setSelectedPlan] = useState<string | null>(targetPlan || null);
  const [billingInterval, setBillingInterval] = useState<'monthly' | 'yearly'>('monthly');
  const [processing, setProcessing] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Clear states when modal closes
  useEffect(() => {
    if (!isOpen) {
      setError(null);
      setSuccess(null);
      setProcessing(false);
      setSelectedPlan(null);
    }
  }, [isOpen]);

  const handleUpgradeClick = async (planId: string) => {
    if (!isAuthenticated) {
      // For demo users, show sign up message
      setError('Please sign up to upgrade your plan');
      return;
    }

    setProcessing(true);
    setError(null);
    setSuccess(null);

    try {
      const result = await upgradePlan(planId, billingInterval);

      if (result) {
        const planName = availablePlans.find(p => p.id === planId)?.name;
        setSuccess(`Successfully upgraded to ${planName}! 🎉`);

        // Close modal after success
        setTimeout(() => {
          onClose();
        }, 2000);
      } else {
        setError('Failed to update subscription. Please try again.');
      }
    } catch (error) {
      console.error('Subscription change error:', error);
      setError('An error occurred. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">Choose Your Plan</h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Billing Interval Toggle */}
          <div className="flex justify-center mb-8">
            <div className="flex items-center bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setBillingInterval('monthly')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${billingInterval === 'monthly'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                  }`}
              >
                Monthly
              </button>
              <button
                onClick={() => setBillingInterval('yearly')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${billingInterval === 'yearly'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                  }`}
              >
                Yearly
                <span className="ml-2 bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">
                  Save {siteConfig.pricing.annualDiscount}%
                </span>
              </button>
            </div>
          </div>

          {/* Success/Error Messages */}
          {success && (
            <div className="mb-6 bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex">
                <svg className="w-5 h-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <p className="ml-3 text-sm text-green-800">{success}</p>
              </div>
            </div>
          )}

          {error && (
            <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex">
                <svg className="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <p className="ml-3 text-sm text-red-800">{error}</p>
              </div>
            </div>
          )}

          {/* Plans Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {availablePlans.map((plan) => {
              const isCurrentPlan = currentPlan?.id === plan.id;
              const isPopular = plan.tier === 'pro';

              return (
                <div
                  key={plan.id}
                  className={`relative bg-white rounded-lg border-2 transition-all ${isCurrentPlan
                      ? 'border-green-500 ring-2 ring-green-200'
                      : isPopular
                        ? 'border-amber-300'
                        : 'border-gray-200'
                    } ${isPopular ? 'transform scale-105' : ''}`}
                >
                  {isPopular && (
                    <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                      <span className="bg-gradient-to-r from-amber-500 to-emerald-500 text-white px-4 py-1 rounded-full text-sm font-medium">
                        Most Popular
                      </span>
                    </div>
                  )}

                  <div className="p-6">
                    <div className="text-center mb-6">
                      <h3 className="text-xl font-semibold text-gray-900">{plan.name}</h3>
                      <p className="text-gray-600 mt-2">{plan.description}</p>
                      <div className="mt-4">
                        <span className="text-3xl font-bold text-gray-900">
                          ${plan.price[billingInterval]}
                        </span>
                        <span className="text-gray-600">
                          /{billingInterval === 'monthly' ? 'month' : 'year'}
                        </span>
                      </div>
                    </div>

                    <ul className="space-y-3 mb-6">
                      {plan.features.map((feature, index) => (
                        <li key={index} className="flex items-center">
                          <svg
                            className={`w-5 h-5 mr-3 ${feature.included ? 'text-green-500' : 'text-gray-300'
                              }`}
                            fill="currentColor"
                            viewBox="0 0 20 20"
                          >
                            <path
                              fillRule="evenodd"
                              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                              clipRule="evenodd"
                            />
                          </svg>
                          <span className={feature.included ? 'text-gray-900' : 'text-gray-400'}>
                            {feature.name}
                          </span>
                        </li>
                      ))}
                    </ul>

                    <button
                      onClick={() => handleUpgradeClick(plan.id)}
                      disabled={processing || isCurrentPlan}
                      className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${isCurrentPlan
                          ? 'bg-green-100 text-green-700 cursor-not-allowed'
                          : processing
                            ? 'bg-gray-100 text-gray-500 cursor-not-allowed'
                            : isPopular
                              ? 'bg-gradient-to-r from-amber-600 to-emerald-600 text-white hover:from-amber-700 hover:to-emerald-700'
                              : 'bg-gray-900 text-white hover:bg-gray-800'
                        }`}
                    >
                      {processing
                        ? 'Processing...'
                        : isCurrentPlan
                          ? 'Current Plan'
                          : plan.tier === 'basic'
                            ? 'Start Free'
                            : 'Upgrade'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Footer */}
          <div className="mt-8 text-center text-sm text-gray-500">
            <p>{siteConfig.pricing.guarantees.moneyBack.description}</p>
            <p className="mt-1">{siteConfig.pricing.guarantees.cancelAnytime ? 'Cancel or change your plan anytime' : ''}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CentralizedSubscriptionModal;
