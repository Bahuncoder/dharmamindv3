/**
 * DharmaMind Support Configuration
 * Standalone module for support/help-related imports
 */

import { siteConfig } from '../../Brand_Webpage/config/site.config';

export const support = siteConfig.support;
export const channels = siteConfig.support.channels;
export const responseTime = siteConfig.support.responseTime;
export const categories = siteConfig.support.categories;
export const faq = siteConfig.support.faq;
export const troubleshooting = siteConfig.support.troubleshooting;

// Category IDs for easy reference
export const SUPPORT_CATEGORIES = {
    GETTING_STARTED: 'getting-started',
    ACCOUNT: 'account',
    BILLING: 'billing',
    FEATURES: 'features',
    TECHNICAL: 'technical',
    API: 'api',
} as const;

export type SupportCategoryId = typeof SUPPORT_CATEGORIES[keyof typeof SUPPORT_CATEGORIES];

// Get category by ID
export const getCategory = (categoryId: SupportCategoryId) =>
    categories.find(c => c.id === categoryId);

// Get FAQs by category
export const getFaqsByCategory = (categoryId: SupportCategoryId) =>
    faq.filter(f => f.category === categoryId);

// Get all FAQ categories with counts
export const getFaqCategoriesWithCounts = () =>
    categories.map(cat => ({
        ...cat,
        count: faq.filter(f => f.category === cat.id).length,
    }));

// Search FAQs
export const searchFaqs = (query: string) => {
    const lowerQuery = query.toLowerCase();
    return faq.filter(f =>
        f.question.toLowerCase().includes(lowerQuery) ||
        f.answer.toLowerCase().includes(lowerQuery)
    );
};

// Get support email by type
export const getSupportEmail = (type: 'general' | 'enterprise' | 'billing' | 'security' = 'general') =>
    channels.email[type];

// Get response time by plan
export const getResponseTime = (planId: 'free' | 'pro' | 'enterprise') =>
    responseTime[planId];

// Get troubleshooting solutions for an issue
export const getTroubleshootingSolutions = (issue: string) => {
    const item = troubleshooting.find(t =>
        t.issue.toLowerCase().includes(issue.toLowerCase())
    );
    return item?.solutions || [];
};

// Check if live chat is available
export const isLiveChatAvailable = () => channels.chat.enabled;

// Get Discord invite link
export const getDiscordLink = () => channels.discord;

// Get documentation URLs
export const getDocUrls = () => ({
    main: support.documentation,
    api: support.apiDocs,
    status: support.statusPage,
});

export default support;
