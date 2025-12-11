/**
 * DharmaMind Community - Shared Configuration
 * ============================================
 * This file mirrors the configuration from the central Brand_Webpage config
 * to ensure consistency across all DharmaMind platforms.
 * 
 * Single source of truth: Brand_Webpage/config/site.config.ts
 */

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

    // ==========================================================================
    // COMPANY INFORMATION
    // ==========================================================================
    company: {
        name: "DharmaMind",
        tagline: "AI with Soul. Powered by Dharma.",
        description: "Building AI systems that honor ancient wisdom while advancing human potential.",
        email: "hello@dharmamind.ai",
        supportEmail: "support@dharmamind.ai",
    },

    // ==========================================================================
    // COMMUNITY SPECIFIC
    // ==========================================================================
    community: {
        name: "DharmaMind Community",
        tagline: "Connect. Learn. Grow Together.",
        description: "A sacred space for seekers to connect, share wisdom, and support each other on the spiritual journey.",
        url: "https://dharmamind.org",
    },

    // ==========================================================================
    // SOCIAL LINKS
    // ==========================================================================
    social: {
        twitter: "https://twitter.com/sudippaudel01",
        linkedin: "https://linkedin.com/in/sudippaudel01",
        github: "https://github.com/dharmamind",
        youtube: "https://youtube.com/@dharmamind",
        discord: "https://discord.gg/dharmamind",
    },

    // ==========================================================================
    // NAVIGATION
    // ==========================================================================
    navigation: {
        main: [
            { name: "Home", href: "/" },
            { name: "Discussions", href: "/discussions" },
            { name: "Groups", href: "/groups" },
            { name: "Events", href: "/events" },
            { name: "Resources", href: "/resources" },
        ],
        footer: {
            community: [
                { name: "Guidelines", href: "/guidelines" },
                { name: "FAQ", href: "/faq" },
                { name: "Support", href: "/support" },
            ],
            legal: [
                { name: "Privacy Policy", href: "/privacy" },
                { name: "Terms of Service", href: "/terms" },
                { name: "Code of Conduct", href: "/conduct" },
            ],
        },
    },

    // ==========================================================================
    // PRODUCT LINKS
    // ==========================================================================
    products: {
        chat: {
            name: "DharmaMind Chat",
            url: "https://dharmamind.ai",
            description: "AI-powered spiritual guidance",
        },
        brand: {
            name: "DharmaMind",
            url: "https://dharmamind.com",
            description: "Learn about our mission",
        },
    },
};

export default siteConfig;
