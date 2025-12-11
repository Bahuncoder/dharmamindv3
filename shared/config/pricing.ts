/**
 * DharmaMind Pricing Configuration
 * Standalone module for pricing-related imports
 */

import { siteConfig } from '../../Brand_Webpage/config/site.config';

export const pricing = siteConfig.pricing;
export const plans = siteConfig.pricing.plans;
export const comparison = siteConfig.pricing.comparison;
export const guarantees = siteConfig.pricing.guarantees;
export const addOns = siteConfig.pricing.addOns;

// Plan IDs for easy reference
export const PLAN_IDS = {
    FREE: 'free',
    PRO: 'pro',
    ENTERPRISE: 'enterprise',
} as const;

export type PlanId = typeof PLAN_IDS[keyof typeof PLAN_IDS];

// Get plan by ID
export const getPlan = (planId: PlanId) =>
    plans.find(p => p.id === planId);

// Get plan price formatted
export const getPlanPrice = (planId: PlanId, period: 'monthly' | 'annual' = 'monthly'): string => {
    const plan = getPlan(planId);
    if (!plan) return 'N/A';

    const price = plan.price[period];
    if (price === 0) return 'Free';
    if (price === -1) return 'Custom';
    return `$${price}`;
};

// Check if plan has feature
export const planHasFeature = (planId: PlanId, featureIndex: number): boolean => {
    const compareValue = comparison[featureIndex]?.[planId];
    if (compareValue === undefined) return false;
    if (typeof compareValue === 'boolean') return compareValue;
    if (compareValue === '—' || compareValue === false) return false;
    return true;
};

// Get annual savings for a plan
export const getAnnualSavings = (planId: PlanId): number => {
    const plan = getPlan(planId);
    if (!plan) return 0;

    const { monthly, annual } = plan.price;
    if (monthly <= 0 || annual <= 0) return 0;

    return (monthly * 12) - (annual * 12);
};

// Check if user can upgrade from one plan to another
export const canUpgrade = (fromPlanId: PlanId, toPlanId: PlanId): boolean => {
    const planOrder = [PLAN_IDS.FREE, PLAN_IDS.PRO, PLAN_IDS.ENTERPRISE];
    return planOrder.indexOf(toPlanId) > planOrder.indexOf(fromPlanId);
};

export default pricing;
