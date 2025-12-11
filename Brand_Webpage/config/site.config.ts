// ============================================================================
// DHARMAMIND CENTRAL SITE CONFIGURATION
// ============================================================================
// All website content is managed from this single source of truth.
// Edit this file to update content across the entire website.
// ============================================================================

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
        description: "Building AI systems that honor ancient wisdom while advancing human potential. We create technology that serves the highest good.",
        mission: "To democratize access to spiritual wisdom through ethical AI, helping every individual discover their path to enlightenment and self-realization.",
        vision: "A world where technology and spirituality work in harmony, where AI serves as a bridge to ancient wisdom, and where every person has access to guidance for their spiritual journey.",
        founded: 2023,
        headquarters: "San Francisco, CA",
        email: "hello@dharmamind.ai",
        supportEmail: "support@dharmamind.ai",
        pressEmail: "press@dharmamind.ai",
        careersEmail: "careers@dharmamind.ai",
    },

    // ==========================================================================
    // SOCIAL LINKS
    // ==========================================================================
    social: {
        twitter: "https://twitter.com/dharmamind",
        linkedin: "https://linkedin.com/company/dharmamind",
        github: "https://github.com/dharmamind",
        youtube: "https://youtube.com/@dharmamind",
        discord: "https://discord.gg/dharmamind",
    },

    // ==========================================================================
    // NAVIGATION
    // ==========================================================================
    navigation: {
        main: [
            { name: "Research", href: "/research" },
            { name: "Products", href: "/products" },
            { name: "Safety", href: "/safety" },
            { name: "Company", href: "/company" },
        ],
        footer: {
            research: [
                { name: "Overview", href: "/research" },
                { name: "DharmaLLM", href: "/research#dharmallm" },
                { name: "Publications", href: "/research#publications" },
                { name: "Open Source", href: "https://github.com/dharmamind" },
            ],
            products: [
                { name: "DharmaMind Chat", href: "https://dharmamind.ai" },
                { name: "DharmaMind API", href: "/api-docs" },
                { name: "DharmaMind Vision", href: "/vision" },
                { name: "Enterprise", href: "/enterprise" },
                { name: "Pricing", href: "/pricing" },
                { name: "Community", href: "https://dharmamind.org" },
            ],
            company: [
                { name: "About", href: "/about" },
                { name: "Team", href: "/about#team" },
                { name: "Careers", href: "/careers" },
                { name: "Blog", href: "/blog" },
                { name: "Press", href: "/press" },
                { name: "Contact", href: "/contact" },
            ],
            legal: [
                { name: "Privacy Policy", href: "/privacy" },
                { name: "Terms of Service", href: "/terms" },
                { name: "Security", href: "/security" },
                { name: "Charter", href: "/charter" },
            ],
        },
    },

    // ==========================================================================
    // STATISTICS & METRICS
    // ==========================================================================
    stats: {
        hero: [
            { value: "50M+", label: "Conversations", description: "Meaningful spiritual dialogues" },
            { value: "180+", label: "Countries", description: "Global reach" },
            { value: "99.9%", label: "Uptime", description: "Reliable service" },
            { value: "4.9/5", label: "Rating", description: "User satisfaction" },
        ],
        company: [
            { value: "2023", label: "Founded" },
            { value: "75+", label: "Team Members" },
            { value: "6", label: "Global Offices" },
            { value: "$50M+", label: "Funding Raised" },
        ],
        waitlist: {
            members: 48,
            betaTesters: 43,
            rating: 4.9,
        },
    },

    // ==========================================================================
    // TEAM MEMBERS
    // ==========================================================================
    team: {
        leadership: [
            {
                name: "Sudip Paudel",
                role: "Founder & CEO",
                bio: "Visionary entrepreneur passionate about bridging ancient wisdom with modern technology. Dedicated to making spiritual guidance accessible through ethical AI.",
                image: "/team/sudip-paudel.jpg",
                linkedin: "https://linkedin.com/in/sudippaudel01",
                twitter: "https://twitter.com/sudippaudel01",
            },
        ],
        advisors: [
            {
                name: "Dr. David Kim",
                role: "AI Ethics Advisor",
                bio: "Director of AI Ethics at MIT. Leading researcher in responsible AI development and author of 'Algorithmic Morality'.",
                image: "/team/david-kim.jpg",
                organization: "MIT",
            },
            {
                name: "Swami Sarvananda",
                role: "Spiritual Advisor",
                bio: "Head of Ramakrishna Mission, Belur Math. 40+ years of teaching Vedanta and guiding spiritual seekers worldwide.",
                image: "/team/swami-sarvananda.jpg",
                organization: "Ramakrishna Mission",
            },
            {
                name: "Prof. Priya Shah",
                role: "Sanskrit & Philosophy Advisor",
                bio: "Professor of Sanskrit Studies at Oxford. Expert in translating and interpreting ancient Dharmic texts for modern audiences.",
                image: "/team/priya-shah.jpg",
                organization: "Oxford University",
            },
            {
                name: "Dr. Maria Santos",
                role: "Psychology Advisor",
                bio: "Clinical psychologist specializing in mindfulness-based therapies. Pioneer in integrating Eastern wisdom with Western psychology.",
                image: "/team/maria-santos.jpg",
                organization: "Stanford Medical",
            },
        ],
        departments: [
            { name: "Engineering", count: 35 },
            { name: "Research", count: 15 },
            { name: "Design", count: 8 },
            { name: "Philosophy & Content", count: 10 },
            { name: "Operations", count: 7 },
        ],
    },

    // ==========================================================================
    // PRODUCTS
    // ==========================================================================
    products: {
        featured: [
            {
                id: "chat",
                name: "DharmaMind Chat",
                tagline: "Your personal spiritual guide",
                description: "Experience profound conversations about life, purpose, and spiritual growth. Our AI understands the depth of Dharmic philosophy and meets you where you are on your journey.",
                icon: "💬",
                href: "/products#chat",
                features: [
                    "Personalized spiritual guidance",
                    "24/7 availability",
                    "Multi-tradition knowledge",
                    "Progress tracking",
                ],
                cta: "Start Chatting",
            },
            {
                id: "api",
                name: "DharmaMind API",
                tagline: "Integrate wisdom into your apps",
                description: "Build spiritually-aware applications with our powerful API. Access DharmaLLM's capabilities for your wellness apps, meditation platforms, or educational tools.",
                icon: "⚡",
                href: "/products#api",
                features: [
                    "RESTful API",
                    "Real-time streaming",
                    "Multi-language support",
                    "Enterprise SLAs",
                ],
                cta: "View Documentation",
            },
            {
                id: "vision",
                name: "DharmaMind Vision",
                tagline: "Sacred imagery analysis",
                description: "Analyze and understand religious iconography, sacred art, and spiritual symbols. Perfect for museums, educational institutions, and cultural preservation.",
                icon: "👁️",
                href: "/products#vision",
                features: [
                    "Symbol recognition",
                    "Art analysis",
                    "Cultural context",
                    "Historical insights",
                ],
                cta: "Explore Vision",
            },
            {
                id: "enterprise",
                name: "DharmaMind Enterprise",
                tagline: "For organizations that care",
                description: "Comprehensive solutions for wellness organizations, spiritual centers, educational institutions, and healthcare providers seeking to integrate ethical AI.",
                icon: "🏛️",
                href: "/products#enterprise",
                features: [
                    "Custom deployment",
                    "Dedicated support",
                    "Training & onboarding",
                    "Compliance ready",
                ],
                cta: "Contact Sales",
            },
        ],
        useCases: [
            {
                title: "Personal Growth",
                description: "Daily guidance for meditation, mindfulness, and spiritual development.",
                icon: "🌱",
            },
            {
                title: "Wellness Apps",
                description: "Integrate spiritual guidance into mental health and wellness platforms.",
                icon: "💚",
            },
            {
                title: "Education",
                description: "Teach philosophy, comparative religion, and ethical thinking.",
                icon: "📚",
            },
            {
                title: "Cultural Preservation",
                description: "Document and analyze sacred texts, art, and traditions.",
                icon: "🏺",
            },
            {
                title: "Healthcare",
                description: "Support holistic patient care with spiritual wellness resources.",
                icon: "🏥",
            },
            {
                title: "Research",
                description: "Advance understanding of consciousness, philosophy, and human flourishing.",
                icon: "🔬",
            },
        ],
    },

    // ==========================================================================
    // RESEARCH
    // ==========================================================================
    research: {
        areas: [
            {
                id: "dharmallm",
                title: "DharmaLLM",
                description: "Our foundational language model trained on curated spiritual and philosophical texts from multiple traditions, designed to provide wise and contextually appropriate guidance.",
                icon: "🧠",
                status: "Active",
                papers: 12,
            },
            {
                id: "emotional-intelligence",
                title: "Emotional Intelligence",
                description: "Research into AI systems that can recognize, understand, and respond appropriately to human emotional states during spiritual conversations.",
                icon: "💝",
                status: "Active",
                papers: 8,
            },
            {
                id: "sacred-vision",
                title: "Sacred Vision",
                description: "Computer vision systems specialized in understanding religious iconography, sacred geometry, and spiritual symbolism across cultures.",
                icon: "👁️",
                status: "Active",
                papers: 5,
            },
            {
                id: "knowledge-graph",
                title: "Dharma Knowledge Graph",
                description: "A comprehensive knowledge graph connecting concepts across Buddhist, Hindu, Jain, and other Dharmic traditions with scholarly sources.",
                icon: "🕸️",
                status: "Active",
                papers: 6,
            },
        ],
        publications: [
            {
                title: "DharmaLLM: A Language Model for Spiritual Guidance",
                authors: ["S. Chen", "N. Sharva", "R. Patel"],
                venue: "NeurIPS 2024",
                year: 2024,
                abstract: "We present DharmaLLM, a large language model specifically designed for spiritual and philosophical conversations...",
                link: "/papers/dharmallm-2024.pdf",
                featured: true,
            },
            {
                title: "Ethical Considerations in AI-Assisted Spiritual Guidance",
                authors: ["D. Kim", "N. Sharva", "M. Santos"],
                venue: "AAAI 2024",
                year: 2024,
                abstract: "This paper explores the ethical dimensions of using artificial intelligence in spiritual contexts...",
                link: "/papers/ethics-spiritual-ai-2024.pdf",
                featured: true,
            },
            {
                title: "Cross-Cultural Understanding in Dharmic AI Systems",
                authors: ["P. Shah", "R. Patel", "S. Chen"],
                venue: "ACL 2024",
                year: 2024,
                abstract: "We investigate methods for training AI systems that can navigate the nuances of different spiritual traditions...",
                link: "/papers/cross-cultural-2024.pdf",
                featured: false,
            },
            {
                title: "Emotion-Aware Spiritual Dialogue Systems",
                authors: ["A. Rodriguez", "S. Chen", "M. Santos"],
                venue: "EMNLP 2024",
                year: 2024,
                abstract: "We present a novel approach to building dialogue systems that recognize and respond to emotional states...",
                link: "/papers/emotion-dialogue-2024.pdf",
                featured: false,
            },
            {
                title: "Sacred Symbol Recognition with Deep Learning",
                authors: ["S. Chen", "P. Shah"],
                venue: "CVPR 2024",
                year: 2024,
                abstract: "A comprehensive approach to recognizing and contextualizing sacred symbols from multiple traditions...",
                link: "/papers/sacred-symbols-2024.pdf",
                featured: false,
            },
        ],
        openSource: [
            {
                name: "dharma-tokenizer",
                description: "Multilingual tokenizer optimized for Sanskrit, Pali, and spiritual texts",
                stars: 1200,
                language: "Python",
                link: "https://github.com/dharmamind/dharma-tokenizer",
            },
            {
                name: "sacred-symbols-dataset",
                description: "Curated dataset of 50K+ labeled sacred symbols for vision research",
                stars: 890,
                language: "Dataset",
                link: "https://github.com/dharmamind/sacred-symbols-dataset",
            },
            {
                name: "dharma-eval",
                description: "Evaluation framework for spiritual AI systems",
                stars: 650,
                language: "Python",
                link: "https://github.com/dharmamind/dharma-eval",
            },
        ],
    },

    // ==========================================================================
    // SAFETY & ETHICS
    // ==========================================================================
    safety: {
        principles: [
            {
                title: "Do No Harm (Ahimsa)",
                description: "Our AI is designed to never cause psychological, emotional, or spiritual harm. We prioritize user wellbeing above engagement metrics.",
                icon: "🛡️",
            },
            {
                title: "Truthfulness (Satya)",
                description: "We commit to honesty about AI limitations, never claiming divine authority, and always distinguishing AI guidance from human wisdom.",
                icon: "✨",
            },
            {
                title: "Respect for Traditions",
                description: "We honor the integrity of spiritual traditions, avoiding appropriation and working with authentic scholars and practitioners.",
                icon: "🙏",
            },
            {
                title: "User Autonomy",
                description: "We empower users to make their own spiritual choices, never creating dependency or replacing human spiritual relationships.",
                icon: "🦋",
            },
            {
                title: "Privacy as Sacred",
                description: "Spiritual conversations are deeply personal. We treat user data as sacred, with industry-leading privacy protections.",
                icon: "🔒",
            },
            {
                title: "Continuous Improvement",
                description: "We actively seek feedback from spiritual leaders, users, and ethicists to continuously improve our systems.",
                icon: "🔄",
            },
        ],
        practices: [
            "Red teaming by religious scholars and ethicists",
            "Regular bias audits across spiritual traditions",
            "User feedback integration in model updates",
            "Transparent documentation of training data sources",
            "Crisis intervention protocols with human escalation",
            "Regular third-party safety audits",
        ],
        commitments: [
            {
                title: "Never Replace Human Connection",
                description: "Our AI complements, never replaces, human spiritual teachers, counselors, and community.",
            },
            {
                title: "Crisis Safety Net",
                description: "When users show signs of crisis, we immediately connect them with human support resources.",
            },
            {
                title: "Tradition Accuracy",
                description: "We work with authentic scholars to ensure accurate representation of all traditions.",
            },
        ],
    },

    // ==========================================================================
    // CAREERS
    // ==========================================================================
    careers: {
        values: [
            {
                title: "Purpose-Driven Work",
                description: "Join a team building technology that genuinely helps people grow spiritually and find meaning.",
                icon: "🎯",
            },
            {
                title: "Diverse Perspectives",
                description: "We bring together engineers, philosophers, scholars, and practitioners from all backgrounds.",
                icon: "🌍",
            },
            {
                title: "Continuous Learning",
                description: "We encourage exploration, provide learning stipends, and support personal growth.",
                icon: "📖",
            },
            {
                title: "Mindful Culture",
                description: "Daily meditation sessions, flexible hours, and a focus on work-life harmony.",
                icon: "🧘",
            },
        ],
        benefits: [
            "Competitive salary and equity",
            "Health, dental, and vision insurance",
            "Unlimited PTO",
            "$5,000 annual learning stipend",
            "Daily meditation sessions",
            "Spiritual retreat program",
            "Remote-first culture",
            "Parental leave",
        ],
        departments: [
            {
                name: "Engineering",
                description: "Build the infrastructure and models powering spiritual AI",
                openRoles: 8,
            },
            {
                name: "Research",
                description: "Advance the science of ethical and spiritual AI",
                openRoles: 4,
            },
            {
                name: "Philosophy & Content",
                description: "Curate wisdom and ensure authentic representation",
                openRoles: 3,
            },
            {
                name: "Design",
                description: "Create mindful, beautiful user experiences",
                openRoles: 2,
            },
            {
                name: "Operations",
                description: "Keep our mission running smoothly",
                openRoles: 2,
            },
        ],
        openings: [
            {
                id: "senior-ml-engineer",
                title: "Senior ML Engineer",
                department: "Engineering",
                location: "Remote / San Francisco",
                type: "Full-time",
                description: "Lead development of DharmaLLM training infrastructure and optimization.",
            },
            {
                id: "nlp-researcher",
                title: "NLP Research Scientist",
                department: "Research",
                location: "Remote / San Francisco",
                type: "Full-time",
                description: "Advance our language understanding capabilities for spiritual dialogue.",
            },
            {
                id: "sanskrit-scholar",
                title: "Sanskrit Scholar & Content Lead",
                department: "Philosophy & Content",
                location: "Remote",
                type: "Full-time",
                description: "Lead curation and verification of Dharmic content and translations.",
            },
            {
                id: "product-designer",
                title: "Senior Product Designer",
                department: "Design",
                location: "Remote / San Francisco",
                type: "Full-time",
                description: "Design mindful, accessible experiences for spiritual seekers.",
            },
            {
                id: "safety-engineer",
                title: "AI Safety Engineer",
                department: "Engineering",
                location: "Remote / San Francisco",
                type: "Full-time",
                description: "Ensure our AI systems remain safe, ethical, and beneficial.",
            },
            {
                id: "frontend-engineer",
                title: "Senior Frontend Engineer",
                department: "Engineering",
                location: "Remote / San Francisco",
                type: "Full-time",
                description: "Build beautiful, accessible web experiences with React/Next.js.",
            },
        ],
    },

    // ==========================================================================
    // BLOG & NEWS
    // ==========================================================================
    blog: {
        categories: ["Engineering", "Research", "Philosophy", "Company", "Product Updates"],
        featured: [
            {
                slug: "introducing-dharmallm-2",
                title: "Introducing DharmaLLM 2.0: A New Era of Spiritual AI",
                excerpt: "Today we announce DharmaLLM 2.0, our most capable and ethical language model yet, trained with deeper understanding of Dharmic traditions.",
                author: "Neel Sharva",
                date: "2024-12-01",
                category: "Product Updates",
                image: "/blog/dharmallm-2.jpg",
                readTime: "8 min read",
            },
            {
                slug: "ethics-spiritual-ai",
                title: "The Ethics of AI in Spiritual Guidance: Our Approach",
                excerpt: "How do we build AI that respects sacred traditions while providing meaningful guidance? A deep dive into our ethical framework.",
                author: "Dr. David Kim",
                date: "2024-11-15",
                category: "Philosophy",
                image: "/blog/ethics.jpg",
                readTime: "12 min read",
            },
            {
                slug: "sacred-vision-launch",
                title: "Launching Sacred Vision: AI That Sees the Divine",
                excerpt: "Our new computer vision system can recognize and explain sacred symbols from traditions around the world.",
                author: "Dr. Sarah Chen",
                date: "2024-11-01",
                category: "Research",
                image: "/blog/sacred-vision.jpg",
                readTime: "6 min read",
            },
        ],
        posts: [
            {
                slug: "building-dharma-tokenizer",
                title: "Building a Tokenizer for Sacred Languages",
                excerpt: "Technical deep-dive into creating a tokenizer that handles Sanskrit, Pali, and other sacred languages effectively.",
                author: "Dr. Sarah Chen",
                date: "2024-10-20",
                category: "Engineering",
                readTime: "10 min read",
            },
            {
                slug: "meditation-in-workplace",
                title: "Why We Start Every Day with Meditation",
                excerpt: "How daily meditation practice shapes our company culture and product development.",
                author: "Alex Rodriguez",
                date: "2024-10-10",
                category: "Company",
                readTime: "5 min read",
            },
            {
                slug: "knowledge-graph-dharma",
                title: "Mapping the Dharma: Our Knowledge Graph Journey",
                excerpt: "How we're building a comprehensive knowledge graph connecting concepts across spiritual traditions.",
                author: "Ravi Patel",
                date: "2024-09-25",
                category: "Research",
                readTime: "8 min read",
            },
        ],
    },

    // ==========================================================================
    // CHARTER & PRINCIPLES
    // ==========================================================================
    charter: {
        preamble: "DharmaMind exists to serve humanity's highest aspirations. We build AI that honors ancient wisdom, respects all beings, and helps individuals on their journey toward self-realization. This charter guides every decision we make.",
        principles: [
            {
                number: 1,
                title: "Service to All Beings",
                description: "Our technology exists to serve, not to exploit. We measure success by positive impact on human flourishing, not merely by profit or growth metrics.",
                commitment: "We will never optimize for engagement at the expense of user wellbeing.",
            },
            {
                number: 2,
                title: "Honoring Wisdom Traditions",
                description: "We approach spiritual traditions with deep respect, working alongside authentic teachers and scholars to ensure accurate, respectful representation.",
                commitment: "We will partner with recognized authorities from each tradition we represent.",
            },
            {
                number: 3,
                title: "Transparency and Truth",
                description: "We are honest about what our AI can and cannot do. We never claim divine authority or suggest our AI is a replacement for human wisdom.",
                commitment: "We will always clearly identify AI-generated content and acknowledge limitations.",
            },
            {
                number: 4,
                title: "Privacy as Sacred",
                description: "Spiritual conversations are among the most intimate human experiences. We treat user data with the reverence it deserves.",
                commitment: "We will never sell user data or use spiritual conversations for advertising.",
            },
            {
                number: 5,
                title: "Accessibility and Inclusion",
                description: "Spiritual wisdom should be available to all, regardless of economic status, location, or background.",
                commitment: "We will always maintain a free tier and work to serve underserved communities.",
            },
            {
                number: 6,
                title: "Continuous Ethical Evolution",
                description: "We recognize that ethical understanding evolves. We commit to ongoing dialogue with ethicists, spiritual leaders, and users.",
                commitment: "We will conduct annual ethics reviews with external advisors and publish findings.",
            },
        ],
        governance: {
            description: "Our ethics board includes religious scholars, AI ethicists, and community representatives who review major decisions.",
            members: [
                { name: "Dr. David Kim", role: "Chair, AI Ethics" },
                { name: "Swami Sarvananda", role: "Spiritual Traditions" },
                { name: "Prof. Priya Shah", role: "Philosophy & Sanskrit" },
                { name: "Dr. Maria Santos", role: "Psychology & Wellbeing" },
            ],
        },
    },

    // ==========================================================================
    // TESTIMONIALS
    // ==========================================================================
    testimonials: [
        {
            quote: "DharmaMind helped me understand concepts from the Bhagavad Gita that I'd struggled with for years. It's like having a patient teacher available anytime.",
            author: "Priya M.",
            role: "Yoga Teacher",
            location: "Mumbai, India",
            rating: 5,
        },
        {
            quote: "As a therapist, I recommend DharmaMind to clients exploring spiritual dimensions of their healing journey. It's thoughtful and never prescriptive.",
            author: "Dr. James Wilson",
            role: "Clinical Psychologist",
            location: "Toronto, Canada",
            rating: 5,
        },
        {
            quote: "The depth of knowledge about Buddhist philosophy is remarkable. It corrects misconceptions gently and points to authentic sources.",
            author: "Ven. Tenzin Norbu",
            role: "Buddhist Monk",
            location: "Dharamsala, India",
            rating: 5,
        },
        {
            quote: "I've been on a spiritual journey for decades. DharmaMind is the first AI that truly understands the nuances of different traditions.",
            author: "Sarah K.",
            role: "Meditation Practitioner",
            location: "San Francisco, USA",
            rating: 5,
        },
    ],

    // ==========================================================================
    // TIMELINE / MILESTONES
    // ==========================================================================
    timeline: [
        {
            year: "2023",
            quarter: "Q1",
            title: "Company Founded",
            description: "DharmaMind founded by Neel Sharva with a vision to bridge AI and spiritual wisdom.",
        },
        {
            year: "2023",
            quarter: "Q3",
            title: "Seed Funding",
            description: "Raised $5M seed round to build initial team and begin DharmaLLM development.",
        },
        {
            year: "2024",
            quarter: "Q1",
            title: "DharmaLLM Alpha",
            description: "Released first version of DharmaLLM to select beta testers.",
        },
        {
            year: "2024",
            quarter: "Q2",
            title: "Series A Funding",
            description: "Raised $25M Series A to scale team and infrastructure.",
        },
        {
            year: "2024",
            quarter: "Q3",
            title: "Public Beta Launch",
            description: "Opened DharmaMind Chat to public beta users worldwide.",
        },
        {
            year: "2024",
            quarter: "Q4",
            title: "DharmaLLM 2.0",
            description: "Released major model update with improved understanding and safety.",
        },
        {
            year: "2025",
            quarter: "Q1",
            title: "Enterprise Launch",
            description: "Launched DharmaMind Enterprise for organizations.",
        },
        {
            year: "2026",
            quarter: "Q2",
            title: "Full Platform",
            description: "Target date for comprehensive platform launch with all features.",
        },
    ],

    // ==========================================================================
    // PRESS & MEDIA
    // ==========================================================================
    press: {
        kit: "/press/dharmamind-press-kit.zip",
        logos: {
            primary: "/press/logo-primary.svg",
            white: "/press/logo-white.svg",
            black: "/press/logo-black.svg",
        },
        coverage: [
            {
                outlet: "TechCrunch",
                title: "DharmaMind raises $25M to bring spiritual AI to the masses",
                date: "2024-06-15",
                link: "https://techcrunch.com/dharmamind-series-a",
            },
            {
                outlet: "The Verge",
                title: "Can AI help you find enlightenment?",
                date: "2024-08-20",
                link: "https://theverge.com/dharmamind-review",
            },
            {
                outlet: "Wired",
                title: "The startup building AI with a soul",
                date: "2024-09-10",
                link: "https://wired.com/dharmamind-profile",
            },
        ],
    },

    // ==========================================================================
    // DHARMA FAQ (from dharmamind.ai)
    // ==========================================================================
    dharmaFaq: [
        {
            term: "Dharma",
            question: "What is Dharma?",
            answer: "Dharma represents cosmic order, moral law, and righteous duty. It encompasses the eternal principles that govern the universe and guide individual conduct toward harmony and spiritual evolution.",
        },
        {
            term: "Karma",
            question: "What is Karma?",
            answer: "Karma is the universal law of cause and effect. Every action, thought, and intention creates ripples that shape our present experiences and future possibilities. DharmaMind helps you understand these patterns.",
        },
        {
            term: "Moksha",
            question: "What is Moksha?",
            answer: "Moksha is liberation from the cycle of birth and death (samsara). It represents the ultimate spiritual goal—the realization of one's true nature beyond ego and material attachments.",
        },
        {
            term: "Atman",
            question: "What is Atman?",
            answer: "Atman is the eternal self or soul that exists beyond the physical body and mind. It is pure consciousness, unchanging and immortal—your true essence beyond all temporary identities.",
        },
        {
            term: "Samskaras",
            question: "What are Samskaras?",
            answer: "Samskaras are mental impressions and patterns formed by past experiences. They influence our thoughts, behaviors, and reactions. Understanding them is key to personal transformation.",
        },
        {
            term: "Purusharthas",
            question: "What are the Purusharthas?",
            answer: "The four Purusharthas are the aims of human life: Dharma (righteousness), Artha (prosperity), Kama (pleasure), and Moksha (liberation). DharmaMind helps you balance all four.",
        },
    ],

    // ==========================================================================
    // UNIFIED PRICING & SUBSCRIPTION
    // ==========================================================================
    // Single source of truth for all pricing across platforms
    pricing: {
        currency: "USD",
        billingCycles: ["monthly", "annual"],
        annualDiscount: 20, // percentage
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
                href: "https://chat.dharmamind.ai/signup",
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
                    conversationsPerMonth: -1, // unlimited
                    historyDays: -1, // unlimited
                    maxContextLength: 16000,
                    apiAccess: false,
                },
                cta: "Start 14-Day Free Trial",
                href: "https://chat.dharmamind.ai/signup?plan=pro",
                highlighted: true,
                badge: "Most Popular",
            },
            {
                id: "enterprise",
                name: "Enterprise",
                tagline: "For organizations",
                price: { monthly: -1, annual: -1 }, // custom pricing
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
        addOns: [
            {
                id: "api-access",
                name: "API Access",
                price: { monthly: 49, annual: 39 },
                description: "Access DharmaLLM API for custom integrations",
                availableFor: ["pro"],
            },
            {
                id: "extra-team-seats",
                name: "Additional Team Seats",
                price: { monthly: 9, annual: 7 },
                description: "Per additional team member",
                availableFor: ["enterprise"],
            },
        ],
        comparison: [
            { feature: "Monthly conversations", free: "10", pro: "Unlimited", enterprise: "Unlimited" },
            { feature: "AI guidance quality", free: "Basic", pro: "Advanced", enterprise: "Advanced + Custom" },
            { feature: "Wisdom traditions", free: "Core", pro: "All traditions", enterprise: "All + Custom" },
            { feature: "Conversation history", free: "7 days", pro: "Unlimited", enterprise: "Unlimited" },
            { feature: "Meditation library", free: false, pro: true, enterprise: true },
            { feature: "Practice recommendations", free: false, pro: true, enterprise: true },
            { feature: "Export conversations", free: false, pro: true, enterprise: true },
            { feature: "API access", free: false, pro: false, enterprise: true },
            { feature: "Custom AI training", free: false, pro: false, enterprise: true },
            { feature: "SSO/SAML", free: false, pro: false, enterprise: true },
            { feature: "Support level", free: "Community", pro: "Priority", enterprise: "Dedicated" },
            { feature: "Response time", free: "48 hours", pro: "24 hours", enterprise: "4 hours" },
            { feature: "SLA", free: false, pro: false, enterprise: "99.9%" },
        ],
        guarantees: {
            moneyBack: { days: 30, description: "30-day money back guarantee, no questions asked" },
            freeTrial: { days: 14, description: "14-day free trial for Pro plan" },
            cancelAnytime: true,
        },
    },

    // ==========================================================================
    // UNIFIED SUPPORT & HELP CENTER
    // ==========================================================================
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
                url: "https://chat.dharmamind.ai/support",
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
            {
                id: "getting-started",
                name: "Getting Started",
                icon: "🚀",
                description: "New to DharmaMind? Start here.",
            },
            {
                id: "account",
                name: "Account & Profile",
                icon: "👤",
                description: "Manage your account settings and profile.",
            },
            {
                id: "billing",
                name: "Billing & Subscription",
                icon: "💳",
                description: "Payment, invoices, and subscription management.",
            },
            {
                id: "features",
                name: "Features & Usage",
                icon: "✨",
                description: "Learn about DharmaMind features.",
            },
            {
                id: "technical",
                name: "Technical Support",
                icon: "🔧",
                description: "Technical issues and troubleshooting.",
            },
            {
                id: "api",
                name: "API & Developers",
                icon: "⚡",
                description: "API documentation and developer resources.",
            },
        ],
        faq: [
            // Getting Started
            {
                id: "gs-1",
                category: "getting-started",
                question: "How do I get started with DharmaMind?",
                answer: "Simply click 'Get Started' on our homepage to create a free account. You can also try our demo version without signing up to explore the platform's capabilities.",
            },
            {
                id: "gs-2",
                category: "getting-started",
                question: "What makes DharmaMind different from other AI assistants?",
                answer: "DharmaMind combines cutting-edge AI technology with ancient dharmic wisdom. Our 32-dimensional wisdom architecture ensures responses are not just intelligent, but ethically grounded and spiritually meaningful.",
            },
            {
                id: "gs-3",
                category: "getting-started",
                question: "What spiritual traditions does DharmaMind support?",
                answer: "DharmaMind has been trained on teachings from Buddhism, Hinduism, Yoga, Vedanta, Taoism, Zen, Christian mysticism, Sufism, and many other wisdom traditions. We continuously expand our knowledge base while ensuring accuracy and respect for each tradition.",
            },
            // Account
            {
                id: "acc-1",
                category: "account",
                question: "How do I reset my password?",
                answer: "Click 'Forgot Password' on the login page, enter your email address, and follow the instructions sent to your email to reset your password.",
            },
            {
                id: "acc-2",
                category: "account",
                question: "Can I change my email address?",
                answer: "Yes, go to Settings > Account > Email to update your email address. You'll need to verify the new email before the change takes effect.",
            },
            {
                id: "acc-3",
                category: "account",
                question: "How do I delete my account?",
                answer: "Go to Settings > Privacy & Data > Delete Account. Please note that this action is permanent and cannot be undone. Your data will be deleted within 30 days.",
            },
            // Billing
            {
                id: "bill-1",
                category: "billing",
                question: "What payment methods do you accept?",
                answer: "We accept all major credit cards (Visa, MasterCard, American Express, Discover) through Stripe. Enterprise customers can also pay via invoice or bank transfer.",
            },
            {
                id: "bill-2",
                category: "billing",
                question: "Can I cancel my subscription anytime?",
                answer: "Yes, you can cancel your subscription at any time from Settings > Subscription. Your access will continue until the end of your current billing period with no additional charges.",
            },
            {
                id: "bill-3",
                category: "billing",
                question: "Do you offer refunds?",
                answer: "Yes, we offer a 30-day money-back guarantee for all paid plans. If you're not satisfied, contact support within 30 days of purchase for a full refund.",
            },
            {
                id: "bill-4",
                category: "billing",
                question: "Is there a discount for annual billing?",
                answer: "Yes! Annual billing saves you 20% compared to monthly billing. That's like getting over 2 months free!",
            },
            {
                id: "bill-5",
                category: "billing",
                question: "Do you offer discounts for spiritual organizations?",
                answer: "Yes, we offer special pricing for temples, monasteries, ashrams, and non-profit spiritual organizations. Contact our sales team to learn more.",
            },
            // Features
            {
                id: "feat-1",
                category: "features",
                question: "What are the 32 wisdom modules?",
                answer: "Our 32-dimensional wisdom architecture includes dharmic concepts like Karma, Moksha, Viveka, Shakti, Ahimsa, and 27 others. Each conversation is processed through relevant philosophical frameworks to provide meaningful guidance.",
            },
            {
                id: "feat-2",
                category: "features",
                question: "How does the chat history work?",
                answer: "All your conversations are automatically saved and can be accessed from your chat interface. Pro users have unlimited history, while Free users can access the last 7 days. You can also export your chat history.",
            },
            {
                id: "feat-3",
                category: "features",
                question: "Can I use DharmaMind offline?",
                answer: "DharmaMind requires an internet connection as it relies on cloud-based AI processing. However, your recent chat history is cached locally for quick access when you reconnect.",
            },
            {
                id: "feat-4",
                category: "features",
                question: "Is there a mobile app?",
                answer: "We're developing native iOS and Android apps (coming Q2 2025). Currently, our web app is fully responsive and works great on mobile browsers.",
            },
            // Technical
            {
                id: "tech-1",
                category: "technical",
                question: "Is my data secure?",
                answer: "Yes, we use enterprise-grade security including AES-256 encryption, SOC 2 Type II compliance, and regular security audits. Your conversations are private and never shared or sold.",
            },
            {
                id: "tech-2",
                category: "technical",
                question: "What browsers are supported?",
                answer: "DharmaMind works on all modern browsers including Chrome, Firefox, Safari, Edge, and Opera. We recommend using the latest version for the best experience.",
            },
            {
                id: "tech-3",
                category: "technical",
                question: "Is my conversation data used to train the AI?",
                answer: "By default, your conversations are NOT used for training. You can opt-in to help improve our AI through Settings > Privacy, but this is entirely optional and data is always anonymized.",
            },
            // API
            {
                id: "api-1",
                category: "api",
                question: "How do I get API access?",
                answer: "API access is available for Enterprise customers and as an add-on for Pro users ($49/month). Visit our API documentation at docs.dharmamind.ai to learn more.",
            },
            {
                id: "api-2",
                category: "api",
                question: "What are the API rate limits?",
                answer: "Rate limits vary by plan: Pro API add-on gets 1000 requests/minute, Enterprise gets custom limits based on your needs. See our API documentation for details.",
            },
            {
                id: "api-3",
                category: "api",
                question: "Is there a sandbox environment?",
                answer: "Yes, we provide a sandbox environment for testing your API integration. You'll receive sandbox credentials when you sign up for API access.",
            },
        ],
        troubleshooting: [
            {
                issue: "Can't log in",
                solutions: [
                    "Check your email and password are correct",
                    "Try resetting your password",
                    "Clear browser cache and cookies",
                    "Try a different browser",
                    "Contact support if issue persists",
                ],
            },
            {
                issue: "Chat not loading",
                solutions: [
                    "Check your internet connection",
                    "Refresh the page",
                    "Clear browser cache",
                    "Disable browser extensions",
                    "Check our status page for outages",
                ],
            },
            {
                issue: "Payment failed",
                solutions: [
                    "Verify your card details are correct",
                    "Check if your card has sufficient funds",
                    "Contact your bank to authorize the transaction",
                    "Try a different payment method",
                    "Contact billing support",
                ],
            },
        ],
        statusPage: "https://status.dharmamind.ai",
        documentation: "https://docs.dharmamind.ai",
        apiDocs: "https://api.dharmamind.ai/docs",
    },

    // ==========================================================================
    // LEGAL & COMPLIANCE
    // ==========================================================================
    legal: {
        privacyPolicy: {
            lastUpdated: "2024-12-01",
            url: "/privacy",
        },
        termsOfService: {
            lastUpdated: "2024-12-01",
            url: "/terms",
        },
        cookiePolicy: {
            lastUpdated: "2024-12-01",
            url: "/privacy#cookies",
        },
        compliance: {
            gdpr: true,
            ccpa: true,
            soc2: true,
            hipaa: false, // available for enterprise
        },
        dataRetention: {
            free: "90 days",
            pro: "Indefinite",
            enterprise: "Custom",
        },
    },
};

// Type exports for TypeScript support
export type SiteConfig = typeof siteConfig;
export type TeamMember = typeof siteConfig.team.leadership[0];
export type Advisor = typeof siteConfig.team.advisors[0];
export type Product = typeof siteConfig.products.featured[0];
export type ResearchArea = typeof siteConfig.research.areas[0];
export type Publication = typeof siteConfig.research.publications[0];
export type SafetyPrinciple = typeof siteConfig.safety.principles[0];
export type JobOpening = typeof siteConfig.careers.openings[0];
export type BlogPost = typeof siteConfig.blog.posts[0];
export type Testimonial = typeof siteConfig.testimonials[0];
export type TimelineEvent = typeof siteConfig.timeline[0];
export type PricingPlan = typeof siteConfig.pricing.plans[0];
export type SupportCategory = typeof siteConfig.support.categories[0];
export type SupportFAQ = typeof siteConfig.support.faq[0];
