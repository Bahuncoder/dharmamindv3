/**
 * DharmaMind Checkout Page
 * ========================
 * Central checkout page for all payment processing
 * Handles plan upgrades from Chat, Community, and direct visitors
 */

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Layout from '../components/layout/Layout';
import Link from 'next/link';

interface Plan {
    id: string;
    name: string;
    price: { monthly: number; yearly: number };
    description: string;
    features: string[];
}

const plans: Record<string, Plan> = {
    pro: {
        id: 'pro',
        name: 'Pro',
        price: { monthly: 19, yearly: 180 },
        description: 'For dedicated practitioners seeking deeper guidance.',
        features: [
            'Unlimited conversations',
            'All specialized spiritual guides',
            'Full conversation history',
            'Personalized practice recommendations',
            'Priority support',
        ],
    },
    max: {
        id: 'max',
        name: 'Max',
        price: { monthly: 49, yearly: 468 },
        description: 'For spiritual leaders and serious practitioners.',
        features: [
            'Everything in Pro',
            'Advanced AI training',
            'Custom meditation creation',
            'API access',
            'White-label options',
        ],
    },
};

export default function CheckoutPage() {
    const router = useRouter();
    const { plan: planId, billing = 'monthly', source, returnUrl } = router.query;

    const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null);
    const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'yearly'>('monthly');
    const [isProcessing, setIsProcessing] = useState(false);
    const [formData, setFormData] = useState({
        email: '',
        name: '',
        cardNumber: '',
        expiry: '',
        cvc: '',
    });

    useEffect(() => {
        if (planId && typeof planId === 'string') {
            setSelectedPlan(plans[planId.toLowerCase()] || plans.pro);
        }
        if (billing === 'yearly') {
            setBillingPeriod('yearly');
        }
    }, [planId, billing]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsProcessing(true);

        try {
            // In production, this would call Stripe API
            // For now, simulate processing
            await new Promise(resolve => setTimeout(resolve, 2000));

            // Redirect to success page or return URL
            const successUrl = returnUrl
                ? `${returnUrl}?success=true&plan=${selectedPlan?.id}`
                : `/subscription?success=true&plan=${selectedPlan?.id}`;

            router.push(successUrl as string);
        } catch (error) {
            console.error('Payment error:', error);
            setIsProcessing(false);
        }
    };

    const currentPrice = selectedPlan?.price[billingPeriod] || 0;
    const savings = selectedPlan
        ? Math.round((selectedPlan.price.monthly * 12 - selectedPlan.price.yearly) / 12)
        : 0;

    return (
        <Layout
            title="Checkout"
            description="Complete your DharmaMind subscription"
        >
            <div className="min-h-screen bg-neutral-100 py-12 px-6">
                <div className="max-w-5xl mx-auto">
                    {/* Header */}
                    <div className="text-center mb-12">
                        <h1 className="text-4xl font-semibold text-neutral-900 mb-4">
                            Complete Your Subscription
                        </h1>
                        <p className="text-lg text-neutral-600">
                            {source === 'chat' && 'Upgrading from DharmaMind Chat'}
                            {source === 'community' && 'Upgrading from DharmaMind Community'}
                            {!source && 'Secure checkout powered by Stripe'}
                        </p>
                    </div>

                    <div className="grid lg:grid-cols-2 gap-12">
                        {/* Order Summary */}
                        <div className="bg-white p-8 rounded-2xl border border-neutral-200 h-fit">
                            <h2 className="text-xl font-semibold text-neutral-900 mb-6">Order Summary</h2>

                            {/* Plan Selection */}
                            <div className="space-y-4 mb-8">
                                {Object.values(plans).map((plan) => (
                                    <button
                                        key={plan.id}
                                        onClick={() => setSelectedPlan(plan)}
                                        className={`w-full p-4 rounded-xl border-2 text-left transition-all ${selectedPlan?.id === plan.id
                                                ? 'border-gold-600 bg-gold-50'
                                                : 'border-neutral-200 hover:border-neutral-300'
                                            }`}
                                    >
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <h3 className="font-semibold text-neutral-900">{plan.name}</h3>
                                                <p className="text-sm text-neutral-600">{plan.description}</p>
                                            </div>
                                            <div className="text-right">
                                                <p className="font-semibold text-neutral-900">
                                                    ${plan.price[billingPeriod]}
                                                </p>
                                                <p className="text-xs text-neutral-500">
                                                    /{billingPeriod === 'monthly' ? 'mo' : 'yr'}
                                                </p>
                                            </div>
                                        </div>
                                    </button>
                                ))}
                            </div>

                            {/* Billing Period Toggle */}
                            <div className="mb-8">
                                <p className="text-sm font-medium text-neutral-700 mb-3">Billing Period</p>
                                <div className="flex gap-3">
                                    <button
                                        onClick={() => setBillingPeriod('monthly')}
                                        className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${billingPeriod === 'monthly'
                                                ? 'bg-gold-600 text-white'
                                                : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
                                            }`}
                                    >
                                        Monthly
                                    </button>
                                    <button
                                        onClick={() => setBillingPeriod('yearly')}
                                        className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${billingPeriod === 'yearly'
                                                ? 'bg-gold-600 text-white'
                                                : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
                                            }`}
                                    >
                                        Yearly
                                        {savings > 0 && (
                                            <span className="ml-2 text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
                                                Save ${savings}/mo
                                            </span>
                                        )}
                                    </button>
                                </div>
                            </div>

                            {/* Total */}
                            <div className="border-t border-neutral-200 pt-6">
                                <div className="flex justify-between items-center mb-2">
                                    <span className="text-neutral-600">Subtotal</span>
                                    <span className="font-medium">${currentPrice}</span>
                                </div>
                                <div className="flex justify-between items-center text-lg font-semibold">
                                    <span>Total</span>
                                    <span className="text-gold-600">${currentPrice}</span>
                                </div>
                                <p className="text-xs text-neutral-500 mt-2">
                                    {billingPeriod === 'monthly'
                                        ? 'Billed monthly. Cancel anytime.'
                                        : 'Billed annually. 14-day money-back guarantee.'}
                                </p>
                            </div>

                            {/* Features */}
                            {selectedPlan && (
                                <div className="mt-8 pt-6 border-t border-neutral-200">
                                    <h3 className="text-sm font-medium text-neutral-700 mb-3">Included features:</h3>
                                    <ul className="space-y-2">
                                        {selectedPlan.features.map((feature, i) => (
                                            <li key={i} className="flex items-center gap-2 text-sm text-neutral-600">
                                                <svg className="w-4 h-4 text-gold-600" fill="currentColor" viewBox="0 0 20 20">
                                                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                </svg>
                                                {feature}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>

                        {/* Payment Form */}
                        <div className="bg-white p-8 rounded-2xl border border-neutral-200">
                            <h2 className="text-xl font-semibold text-neutral-900 mb-6">Payment Details</h2>

                            <form onSubmit={handleSubmit} className="space-y-6">
                                <div>
                                    <label className="block text-sm font-medium text-neutral-700 mb-2">
                                        Email
                                    </label>
                                    <input
                                        type="email"
                                        required
                                        value={formData.email}
                                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                        className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-600 focus:border-transparent"
                                        placeholder="you@example.com"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-neutral-700 mb-2">
                                        Name on card
                                    </label>
                                    <input
                                        type="text"
                                        required
                                        value={formData.name}
                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                        className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-600 focus:border-transparent"
                                        placeholder="Full name"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-neutral-700 mb-2">
                                        Card number
                                    </label>
                                    <input
                                        type="text"
                                        required
                                        value={formData.cardNumber}
                                        onChange={(e) => setFormData({ ...formData, cardNumber: e.target.value })}
                                        className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-600 focus:border-transparent"
                                        placeholder="1234 5678 9012 3456"
                                        maxLength={19}
                                    />
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-neutral-700 mb-2">
                                            Expiry
                                        </label>
                                        <input
                                            type="text"
                                            required
                                            value={formData.expiry}
                                            onChange={(e) => setFormData({ ...formData, expiry: e.target.value })}
                                            className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-600 focus:border-transparent"
                                            placeholder="MM/YY"
                                            maxLength={5}
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-neutral-700 mb-2">
                                            CVC
                                        </label>
                                        <input
                                            type="text"
                                            required
                                            value={formData.cvc}
                                            onChange={(e) => setFormData({ ...formData, cvc: e.target.value })}
                                            className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-600 focus:border-transparent"
                                            placeholder="123"
                                            maxLength={4}
                                        />
                                    </div>
                                </div>

                                <button
                                    type="submit"
                                    disabled={isProcessing || !selectedPlan}
                                    className="w-full py-4 bg-gold-600 text-white font-semibold rounded-lg hover:bg-gold-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {isProcessing ? (
                                        <span className="flex items-center justify-center gap-2">
                                            <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            Processing...
                                        </span>
                                    ) : (
                                        `Pay $${currentPrice}`
                                    )}
                                </button>

                                <p className="text-xs text-neutral-500 text-center">
                                    By completing your purchase, you agree to our{' '}
                                    <Link href="/terms" className="text-gold-600 hover:underline">Terms</Link>
                                    {' '}and{' '}
                                    <Link href="/privacy" className="text-gold-600 hover:underline">Privacy Policy</Link>.
                                </p>
                            </form>

                            {/* Trust Badges */}
                            <div className="mt-8 pt-6 border-t border-neutral-200">
                                <div className="flex items-center justify-center gap-6 text-neutral-400">
                                    <div className="flex items-center gap-2">
                                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                                        </svg>
                                        <span className="text-xs">SSL Encrypted</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                        </svg>
                                        <span className="text-xs">Secure Payment</span>
                                    </div>
                                </div>
                                <p className="text-xs text-neutral-400 text-center mt-4">
                                    Powered by Stripe
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Back Link */}
                    <div className="text-center mt-8">
                        <Link href="/pricing" className="text-neutral-600 hover:text-neutral-900 transition-colors">
                            ← Back to pricing
                        </Link>
                    </div>
                </div>
            </div>
        </Layout>
    );
}
