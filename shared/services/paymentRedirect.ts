/**
 * DharmaMind Unified Payment Redirect Service
 * ============================================
 * All payment operations redirect to Brand Webpage (dharmamind.com)
 * This ensures a single payment entry point across all apps.
 * 
 * Domain Structure:
 * - dharmamind.com = Brand Webpage (payments, auth, marketing)
 * - dharmamind.ai = Chat App (AI chat functionality)
 * - dharmamind.org = Community platform
 */

// Brand Webpage URL - handles all payments (.com, not .ai)
const BRAND_PAYMENT_URL = process.env.NEXT_PUBLIC_BRAND_URL || 'https://dharmamind.com';

export interface PaymentRedirectParams {
    plan?: string;
    billing?: 'monthly' | 'yearly';
    returnUrl?: string;
    userId?: string;
    source?: 'chat' | 'community' | 'brand';
}

export interface SubscriptionRedirectParams {
    action: 'upgrade' | 'manage' | 'cancel' | 'billing';
    currentPlan?: string;
    returnUrl?: string;
}

/**
 * Redirect to Brand Webpage for payment/subscription
 */
export const paymentRedirect = {
    /**
     * Redirect to pricing page
     */
    toPricing: (params?: PaymentRedirectParams) => {
        const url = new URL(`${BRAND_PAYMENT_URL}/pricing`);
        if (params?.plan) url.searchParams.set('plan', params.plan);
        if (params?.billing) url.searchParams.set('billing', params.billing);
        if (params?.returnUrl) url.searchParams.set('returnUrl', params.returnUrl);
        if (params?.source) url.searchParams.set('source', params.source);

        if (typeof window !== 'undefined') {
            window.location.href = url.toString();
        }
        return url.toString();
    },

    /**
     * Redirect to checkout page for specific plan
     */
    toCheckout: (planId: string, params?: PaymentRedirectParams) => {
        const url = new URL(`${BRAND_PAYMENT_URL}/checkout`);
        url.searchParams.set('plan', planId);
        if (params?.billing) url.searchParams.set('billing', params.billing);
        if (params?.returnUrl) url.searchParams.set('returnUrl', params.returnUrl);
        if (params?.userId) url.searchParams.set('uid', params.userId);
        if (params?.source) url.searchParams.set('source', params.source);

        if (typeof window !== 'undefined') {
            window.location.href = url.toString();
        }
        return url.toString();
    },

    /**
     * Redirect to subscription management
     */
    toSubscription: (params?: SubscriptionRedirectParams) => {
        const url = new URL(`${BRAND_PAYMENT_URL}/subscription`);
        if (params?.action) url.searchParams.set('action', params.action);
        if (params?.currentPlan) url.searchParams.set('current', params.currentPlan);
        if (params?.returnUrl) url.searchParams.set('returnUrl', params.returnUrl);

        if (typeof window !== 'undefined') {
            window.location.href = url.toString();
        }
        return url.toString();
    },

    /**
     * Redirect to billing history/invoices
     */
    toBilling: (returnUrl?: string) => {
        const url = new URL(`${BRAND_PAYMENT_URL}/billing`);
        if (returnUrl) url.searchParams.set('returnUrl', returnUrl);

        if (typeof window !== 'undefined') {
            window.location.href = url.toString();
        }
        return url.toString();
    },

    /**
     * Redirect to add payment method
     */
    toAddPaymentMethod: (returnUrl?: string) => {
        const url = new URL(`${BRAND_PAYMENT_URL}/payment-methods`);
        if (returnUrl) url.searchParams.set('returnUrl', returnUrl);

        if (typeof window !== 'undefined') {
            window.location.href = url.toString();
        }
        return url.toString();
    },

    /**
     * Get URLs without redirecting (for links)
     */
    urls: {
        pricing: (params?: PaymentRedirectParams) => {
            const url = new URL(`${BRAND_PAYMENT_URL}/pricing`);
            if (params?.plan) url.searchParams.set('plan', params.plan);
            if (params?.billing) url.searchParams.set('billing', params.billing);
            return url.toString();
        },
        checkout: (planId: string, billing?: 'monthly' | 'yearly') => {
            const url = new URL(`${BRAND_PAYMENT_URL}/checkout`);
            url.searchParams.set('plan', planId);
            if (billing) url.searchParams.set('billing', billing);
            return url.toString();
        },
        subscription: `${BRAND_PAYMENT_URL}/subscription`,
        billing: `${BRAND_PAYMENT_URL}/billing`,
        paymentMethods: `${BRAND_PAYMENT_URL}/payment-methods`,
    }
};

/**
 * Check if user should be redirected to payment
 * (e.g., when trying to access premium features)
 */
export const shouldRedirectToPayment = (
    userPlan: string | undefined,
    requiredPlan: 'pro' | 'max' | 'enterprise'
): boolean => {
    const planHierarchy = ['free', 'basic', 'pro', 'max', 'enterprise'];
    const userPlanIndex = planHierarchy.indexOf(userPlan || 'free');
    const requiredPlanIndex = planHierarchy.indexOf(requiredPlan);
    return userPlanIndex < requiredPlanIndex;
};

export default paymentRedirect;
