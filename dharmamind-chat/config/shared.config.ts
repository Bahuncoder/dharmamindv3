/**
 * DharmaMind Chat - Shared Configuration
 * ======================================
 * This file imports configuration from the central Brand_Webpage config
 * to ensure consistency across all DharmaMind platforms.
 * 
 * Single source of truth: Brand_Webpage/config/site.config.ts
 */

// Import from the central config
// Note: In production, this would be an npm package or workspace dependency
// For now, we use relative imports to the shared config

// Since we can't use TypeScript path aliases without build config changes,
// we'll create local copies that reference the central config

export const siteConfig = {
    // ==========================================================================
    // MINIMALIST: Dark gray for TEXT, light grays for UI, gold for accents
    // ==========================================================================
    colors: {
        brand: {
            primary: '#1c1917',    // Near Black - TEXT ONLY
            secondary: '#d4a854',  // Warm Gold - buttons, accents
            accent: '#d4a854',     // Warm Gold - highlights
        },
        // Text colors (dark grays - ONLY for text)
        text: {
            primary: '#1c1917',    // Main text
            secondary: '#57534e',  // Secondary text
            muted: '#78716c',      // Muted text
            light: '#a8a29e',      // Very light text
        },
        // UI colors (light grays for backgrounds, borders)
        ui: {
            bg: '#fafaf9',         // Page background
            surface: '#f5f5f4',    // Card background
            border: '#e7e5e4',     // Borders
            hover: '#d6d3d1',      // Hover states
        },
        // Gold accent
        gold: {
            DEFAULT: '#d4a854',
            light: '#f4e4bd',
            dark: '#b8860b',
        },
    },
    company: {
        name: "DharmaMind",
        tagline: "AI with Soul. Powered by Dharma.",
        description: "Building AI systems that honor ancient wisdom while advancing human potential.",
        email: "hello@dharmamind.ai",
        supportEmail: "support@dharmamind.ai",
        pressEmail: "press@dharmamind.ai",
        careersEmail: "careers@dharmamind.ai",
    },
    social: {
        twitter: "https://twitter.com/dharmamind",
        linkedin: "https://linkedin.com/company/dharmamind",
        github: "https://github.com/dharmamind",
        youtube: "https://youtube.com/@dharmamind",
        discord: "https://discord.gg/dharmamind",
    },
    // Import pricing from central config (sync with Brand_Webpage/config/site.config.ts)
    pricing: {
        currency: "USD",
        billingCycles: ["monthly", "annual"] as const,
        annualDiscount: 20,
        plans: [
            {
                id: "free",
                name: "Free",
                tagline: "Begin your spiritual journey",
                price: { monthly: 0, annual: 0 },
                description: "For individuals exploring AI-guided spiritual growth.",
                features: [
                    "10 conversations per month",
                    "Basic AI guidance",
                    "Access to core wisdom traditions",
                    "Community Discord access",
                    "Email support (48hr response)",
                ],
                limits: {
                    conversationsPerMonth: 10,
                    historyDays: 7,
                    maxContextLength: 4000,
                    apiAccess: false,
                },
                cta: "Get Started Free",
                href: "/signup",
                highlighted: false,
                badge: null,
            },
            {
                id: "pro",
                name: "Pro",
                tagline: "Deepen your practice",
                price: { monthly: 19, annual: 15 },
                description: "For dedicated practitioners seeking deeper guidance.",
                features: [
                    "Unlimited conversations",
                    "All specialized spiritual guides",
                    "Full conversation history",
                    "Personalized practice recommendations",
                    "Priority support (24hr response)",
                    "Advanced insights & analytics",
                    "Guided meditation library",
                    "Export conversations",
                ],
                limits: {
                    conversationsPerMonth: -1,
                    historyDays: -1,
                    maxContextLength: 16000,
                    apiAccess: false,
                },
                cta: "Start 14-Day Free Trial",
                href: "/signup?plan=pro",
                highlighted: true,
                badge: "Most Popular",
            },
            {
                id: "enterprise",
                name: "Enterprise",
                tagline: "For organizations",
                price: { monthly: -1, annual: -1 },
                description: "For teams, temples, and wellness organizations.",
                features: [
                    "Everything in Pro",
                    "Unlimited team members",
                    "Custom AI training on your teachings",
                    "SSO/SAML authentication",
                    "Private deployment options",
                    "Dedicated account manager",
                    "Custom integrations & API access",
                    "SLA guarantee (99.9% uptime)",
                    "Compliance support (HIPAA, SOC2)",
                    "Priority phone support",
                ],
                limits: {
                    conversationsPerMonth: -1,
                    historyDays: -1,
                    maxContextLength: 32000,
                    apiAccess: true,
                },
                cta: "Contact Sales",
                href: "/contact?type=enterprise",
                highlighted: false,
                badge: null,
            },
        ],
        guarantees: {
            moneyBack: { days: 30, description: "30-day money back guarantee" },
            freeTrial: { days: 14, description: "14-day free trial for Pro" },
            cancelAnytime: true,
        },
    },
    support: {
        channels: {
            email: {
                general: "support@dharmamind.ai",
                enterprise: "enterprise@dharmamind.ai",
                billing: "billing@dharmamind.ai",
                security: "security@dharmamind.ai",
            },
            phone: {
                enterprise: "+1 (888) DHARMA-1",
                hours: "Mon-Fri 9am-6pm PST",
            },
            chat: {
                enabled: true,
                url: "/support",
                hours: "24/7 for Pro & Enterprise",
            },
            discord: "https://discord.gg/dharmamind",
        },
        responseTime: {
            free: "48 hours",
            pro: "24 hours",
            enterprise: "4 hours",
        },
        categories: [
            { id: "getting-started", name: "Getting Started", icon: "🚀", description: "New to DharmaMind? Start here." },
            { id: "account", name: "Account & Profile", icon: "👤", description: "Manage your account settings." },
            { id: "billing", name: "Billing & Subscription", icon: "💳", description: "Payment and subscription help." },
            { id: "features", name: "Features & Usage", icon: "✨", description: "Learn about features." },
            { id: "technical", name: "Technical Support", icon: "🔧", description: "Technical troubleshooting." },
            { id: "api", name: "API & Developers", icon: "⚡", description: "Developer resources." },
        ],
        statusPage: "https://status.dharmamind.ai",
        documentation: "https://docs.dharmamind.ai",
        apiDocs: "https://api.dharmamind.ai/docs",
    },
    legal: {
        privacyPolicy: "/privacy",
        termsOfService: "/terms",
        cookiePolicy: "/privacy#cookies",
    },
};

// Helper functions
export const getPlanById = (planId: string) =>
    siteConfig.pricing.plans.find(p => p.id === planId);

export const getPlanPrice = (planId: string, period: 'monthly' | 'annual' = 'monthly'): string => {
    const plan = getPlanById(planId);
    if (!plan) return 'N/A';
    const price = plan.price[period];
    if (price === 0) return 'Free';
    if (price === -1) return 'Custom';
    return `$${price}/mo`;
};

export const getPlanLimits = (planId: string) => {
    const plan = getPlanById(planId);
    return plan?.limits || null;
};

export type PricingPlan = typeof siteConfig.pricing.plans[0];
export type SupportCategory = typeof siteConfig.support.categories[0];

export default siteConfig;
