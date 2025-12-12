/**
 * DharmaMind Subscription Management Page
 * =======================================
 * Central subscription management for all DharmaMind apps
 */

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import Layout from '../components/layout/Layout';
import Link from 'next/link';

interface Subscription {
  plan: string;
  status: 'active' | 'canceled' | 'past_due';
  currentPeriodEnd: string;
  cancelAtPeriodEnd: boolean;
}

export default function SubscriptionPage() {
  const router = useRouter();
  const { success, plan: upgradedPlan, action } = router.query;

  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading subscription data
    setTimeout(() => {
      setSubscription({
        plan: 'Pro',
        status: 'active',
        currentPeriodEnd: '2025-01-15',
        cancelAtPeriodEnd: false,
      });
      setLoading(false);
    }, 500);
  }, []);

  // Show success message if just upgraded
  const showSuccessMessage = success === 'true' && upgradedPlan;

  const planFeatures: Record<string, string[]> = {
    Free: ['10 conversations/month', 'Basic AI guidance', 'Community access'],
    Pro: ['Unlimited conversations', 'All spiritual guides', 'Priority support', 'Advanced insights'],
    Max: ['Everything in Pro', 'Custom AI training', 'API access', 'White-label options'],
  };

  if (loading) {
    return (
      <Layout title="Subscription" description="Manage your DharmaMind subscription">
        <div className="min-h-screen bg-neutral-100 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gold-600 mx-auto mb-4"></div>
            <p className="text-neutral-600">Loading subscription...</p>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="Subscription" description="Manage your DharmaMind subscription">
      <div className="min-h-screen bg-neutral-100 py-12 px-6">
        <div className="max-w-3xl mx-auto">
          {/* Success Message */}
          {showSuccessMessage && (
            <div className="bg-green-50 border border-green-200 text-green-700 p-4 rounded-xl mb-8 flex items-center gap-3">
              <svg className="w-6 h-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="font-semibold">Welcome to {upgradedPlan}!</p>
                <p className="text-sm">Your subscription has been activated successfully.</p>
              </div>
            </div>
          )}

          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-semibold text-neutral-900 mb-2">Subscription</h1>
            <p className="text-neutral-600">Manage your DharmaMind subscription and access.</p>
          </div>

          {/* Current Plan */}
          <div className="bg-white p-8 rounded-2xl border border-neutral-200 mb-8">
            <div className="flex items-start justify-between mb-6">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <h2 className="text-2xl font-semibold text-neutral-900">{subscription?.plan} Plan</h2>
                  {subscription?.status === 'active' && (
                    <span className="px-3 py-1 text-sm font-medium bg-green-100 text-green-700 rounded-full">
                      Active
                    </span>
                  )}
                </div>
                <p className="text-neutral-600">
                  {subscription?.cancelAtPeriodEnd
                    ? `Cancels on ${new Date(subscription.currentPeriodEnd).toLocaleDateString()}`
                    : `Renews on ${new Date(subscription?.currentPeriodEnd || '').toLocaleDateString()}`}
                </p>
              </div>

              <Link
                href="/pricing"
                className="px-5 py-2.5 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors"
              >
                {subscription?.plan === 'Free' ? 'Upgrade' : 'Change Plan'}
              </Link>
            </div>

            {/* Plan Features */}
            <div className="border-t border-neutral-200 pt-6">
              <h3 className="text-sm font-medium text-neutral-500 mb-4">Your plan includes:</h3>
              <ul className="grid md:grid-cols-2 gap-3">
                {(planFeatures[subscription?.plan || 'Free'] || []).map((feature, i) => (
                  <li key={i} className="flex items-center gap-2 text-neutral-700">
                    <svg className="w-5 h-5 text-gold-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="grid md:grid-cols-3 gap-4 mb-8">
            <Link
              href="/billing"
              className="bg-white p-6 rounded-xl border border-neutral-200 hover:border-neutral-300 hover:shadow-sm transition-all text-center"
            >
              <svg className="w-8 h-8 text-gold-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
              </svg>
              <h3 className="font-semibold text-neutral-900 mb-1">Billing</h3>
              <p className="text-sm text-neutral-500">View invoices & payment methods</p>
            </Link>

            <Link
              href="https://dharmamind.ai"
              className="bg-white p-6 rounded-xl border border-neutral-200 hover:border-neutral-300 hover:shadow-sm transition-all text-center"
            >
              <svg className="w-8 h-8 text-gold-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              <h3 className="font-semibold text-neutral-900 mb-1">Start Chatting</h3>
              <p className="text-sm text-neutral-500">Open DharmaMind AI</p>
            </Link>

            <Link
              href="/help"
              className="bg-white p-6 rounded-xl border border-neutral-200 hover:border-neutral-300 hover:shadow-sm transition-all text-center"
            >
              <svg className="w-8 h-8 text-gold-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <h3 className="font-semibold text-neutral-900 mb-1">Help</h3>
              <p className="text-sm text-neutral-500">FAQs & support</p>
            </Link>
          </div>

          {/* Cancel Option */}
          {subscription?.plan !== 'Free' && !subscription?.cancelAtPeriodEnd && (
            <div className="text-center">
              <button className="text-sm text-neutral-500 hover:text-red-600 transition-colors">
                Cancel subscription
              </button>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
