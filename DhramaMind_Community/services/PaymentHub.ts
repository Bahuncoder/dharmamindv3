/**
 * DharmaMind Payment Redirect
 * ===========================
 * Redirects all payment operations to Brand Webpage (dharmamind.com)
 */

// Brand Webpage URL - single payment hub (.com handles payments, .ai is for chat)
const PAYMENT_HUB = process.env.NEXT_PUBLIC_BRAND_URL || 'https://dharmamind.com';

export const PaymentHub = {
    // URLs for payment pages
    urls: {
        pricing: `${PAYMENT_HUB}/pricing`,
        checkout: (plan: string) => `${PAYMENT_HUB}/checkout?plan=${plan}`,
        subscription: `${PAYMENT_HUB}/subscription`,
        billing: `${PAYMENT_HUB}/billing`,
        upgrade: (plan: string) => `${PAYMENT_HUB}/pricing?upgrade=${plan}`,
    },

    // Redirect functions
    goToPricing: () => {
        window.location.href = `${PAYMENT_HUB}/pricing`;
    },

    goToCheckout: (planId: string, billing: 'monthly' | 'yearly' = 'monthly') => {
        window.location.href = `${PAYMENT_HUB}/checkout?plan=${planId}&billing=${billing}`;
    },

    goToSubscription: () => {
        window.location.href = `${PAYMENT_HUB}/subscription`;
    },

    goToBilling: () => {
        window.location.href = `${PAYMENT_HUB}/billing`;
    },

    goToUpgrade: (currentPlan: string) => {
        window.location.href = `${PAYMENT_HUB}/pricing?current=${currentPlan}`;
    },
};

export default PaymentHub;
