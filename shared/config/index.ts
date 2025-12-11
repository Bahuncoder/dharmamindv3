/**
 * DharmaMind Shared Configuration
 * ================================
 * Single source of truth for all DharmaMind platforms
 * 
 * This package exports configuration from Brand_Webpage/config/site.config.ts
 * to be used across all DharmaMind products:
 * - Brand Website (Brand_Webpage)
 * - Chat Application (dharmamind-chat)
 * - Backend Services (backend)
 * 
 * Usage:
 * ```typescript
 * import { siteConfig, pricing, support, company } from '@dharmamind/shared-config';
 * ```
 */

// Re-export everything from the central config
export * from '../../Brand_Webpage/config/site.config';

// Import for convenience exports
import { siteConfig } from '../../Brand_Webpage/config/site.config';

// Convenience exports for common use cases
export const company = siteConfig.company;
export const social = siteConfig.social;
export const navigation = siteConfig.navigation;
export const stats = siteConfig.stats;
export const team = siteConfig.team;
export const products = siteConfig.products;
export const research = siteConfig.research;
export const safety = siteConfig.safety;
export const careers = siteConfig.careers;
export const blog = siteConfig.blog;
export const charter = siteConfig.charter;
export const testimonials = siteConfig.testimonials;
export const timeline = siteConfig.timeline;
export const press = siteConfig.press;
export const dharmaFaq = siteConfig.dharmaFaq;
export const pricing = siteConfig.pricing;
export const support = siteConfig.support;
export const legal = siteConfig.legal;

// Helper functions
export const getPlanById = (planId: string) =>
    siteConfig.pricing.plans.find(p => p.id === planId);

export const getFaqByCategory = (category: string) =>
    siteConfig.support.faq.filter(f => f.category === category);

export const getSupportCategory = (categoryId: string) =>
    siteConfig.support.categories.find(c => c.id === categoryId);

export const formatPrice = (price: number, period: 'monthly' | 'annual' = 'monthly') => {
    if (price === 0) return 'Free';
    if (price === -1) return 'Custom';
    return `$${price}${period === 'monthly' ? '/mo' : '/mo (billed annually)'}`;
};

export const getAnnualSavings = (monthlyPrice: number, annualPrice: number) => {
    if (monthlyPrice <= 0 || annualPrice <= 0) return 0;
    const annualCost = annualPrice * 12;
    const monthlyCost = monthlyPrice * 12;
    return monthlyCost - annualCost;
};

// Default export
export default siteConfig;
