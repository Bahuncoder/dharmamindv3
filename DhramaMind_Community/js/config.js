// DharmaMind AI Configuration
// AI with Soul powered by Dharma
// ===============================

const CONFIG = {
    // Application Identity
    APP: {
        NAME: 'DharmaMind AI',
        VERSION: '1.0.0',
        DESCRIPTION: 'AI with Soul powered by Dharma - Inner clarity for modern minds',
        TAGLINE: 'AI with Soul | Inner Clarity for Modern Minds',
        MISSION: 'Providing inner clarity, spiritual growth, leadership wisdom, and mental clarity through ancient Hindu knowledge in modern, universal language',
        VISION: 'AI with soul for universal human growth and enlightenment'
    },

    // Core Principles (DharmaMind DNA)
    PRINCIPLES: {
        NO_JUDGMENT: true,
        NO_DISTRACTION: true, 
        NO_MANIPULATION: true,
        UNIVERSAL_LANGUAGE: true,
        HINDU_WISDOM_BASED: true,
        MODERN_EXPRESSION: true,
        INNER_CLARITY_FOCUSED: true,
        DHARMA_POWERED: true
    },

    // Target Audience & Use Cases
    AUDIENCE: {
        INNER_CLARITY_SEEKERS: 'Those seeking clear thinking and decision-making',
        SPIRITUAL_GROWTH_SEEKERS: 'Individuals on personal spiritual evolution journey',
        LEADERS: 'People in leadership roles seeking dharmic guidance',
        MENTAL_CLARITY_SEEKERS: 'Those wanting sharp mind and clear thoughts',
        UNIVERSAL_HUMANS: 'All humans seeking core growth elements'
    },

    // API Configuration
    API: {
        BASE_URL: 'http://localhost:8000',  // Fixed to match backend server
        ENDPOINTS: {
            CHAT: '/api/v1/chat',  // Correct endpoint path
            STATUS: '/',  // Root status endpoint
            MODULES: '/api/v1/models',  // Available models endpoint
            LEARNING: '/learning',  // New learning endpoint
            SYSTEM_STATUS: '/',
            HEALTH: '/health'
        },
        TIMEOUT: 30000, // 30 seconds
        RETRY_ATTEMPTS: 3,
        RETRY_DELAY: 1000 // 1 second
    },

    // Chat Configuration
    CHAT: {
        MAX_MESSAGE_LENGTH: 2000,
        TYPING_INDICATOR_DELAY: 500,
        AUTO_SCROLL_THRESHOLD: 100,
        MESSAGE_HISTORY_LIMIT: 100,
        RESPONSE_TIMEOUT: 30000,
        STREAMING_ENABLED: true,
        VOICE_INPUT_ENABLED: true
    },

    // UI Configuration
    UI: {
        THEME: {
            DEFAULT: 'light',
            STORAGE_KEY: 'dharmamind-theme',
            OPTIONS: ['light', 'dark', 'auto']
        },
        LANGUAGE: {
            DEFAULT: 'en',
            STORAGE_KEY: 'dharmamind-language',
            OPTIONS: {
                'en': 'English',
                'hi': 'हिन्दी',
                'sa': 'संस्कृत',
                'es': 'Español',
                'fr': 'Français',
                'de': 'Deutsch',
                'zh': '中文',
                'ja': '日本語'
            }
        },
        TRADITION: {
            DEFAULT: 'universal',
            STORAGE_KEY: 'dharmamind-tradition',
            OPTIONS: {
                'universal': 'Universal Wisdom',
                'vedanta': 'Vedanta',
                'yoga': 'Yoga',
                'ayurveda': 'Ayurveda',
                'bhakti': 'Bhakti',
                'karma_yoga': 'Karma Yoga',
                'dharmic': 'Dharmic',
                'vedic': 'Vedic',
                'advaita': 'Advaita',
                'tantra': 'Tantra'
            }
        },
        SPIRITUAL_LEVEL: {
            DEFAULT: 'beginner',
            STORAGE_KEY: 'dharmamind-level',
            OPTIONS: {
                'beginner': 'Beginner',
                'intermediate': 'Intermediate',
                'advanced': 'Advanced',
                'all': 'All Levels'
            }
        },
        ANIMATIONS: {
            ENABLED: true,
            STORAGE_KEY: 'dharmamind-animations',
            DURATION: {
                FAST: 150,
                NORMAL: 300,
                SLOW: 500
            }
        },
        SIDEBAR: {
            DEFAULT_STATE: 'open',
            STORAGE_KEY: 'dharmamind-sidebar',
            BREAKPOINT: 1024
        },
        NOTIFICATIONS: {
            DURATION: 5000,
            MAX_VISIBLE: 3,
            POSITION: 'top-right'
        }
    },

    // Spiritual Configuration
    SPIRITUAL: {
        RESPONSE_STYLES: {
            'compassionate': 'Compassionate & Gentle',
            'scholarly': 'Scholarly & Detailed',
            'practical': 'Practical & Direct',
            'poetic': 'Poetic & Inspirational'
        },
        DEFAULT_RESPONSE_STYLE: 'compassionate',
        INCLUDE_SOURCES: true,
        TRADITION_NEUTRAL: true,
        CULTURAL_SENSITIVITY: true
    },

    // Features Configuration
    FEATURES: {
        KNOWLEDGE_SEARCH: true,
        MODULE_SELECTION: true,
        VOICE_INPUT: false, // Disabled by default, requires user permission
        EXPORT_CHAT: true,
        SHARE_WISDOM: true,
        PROGRESS_TRACKING: true,
        PERSONALIZATION: true,
        OFFLINE_MODE: false,
        COMMUNITY_FEATURES: false
    },

    // Privacy Configuration
    PRIVACY: {
        SAVE_CONVERSATIONS: true,
        COLLECT_FEEDBACK: true,
        ANONYMOUS_MODE: false,
        DATA_RETENTION_DAYS: 90,
        STORAGE_KEY: 'dharmamind-privacy'
    },

    // Storage Keys
    STORAGE_KEYS: {
        SETTINGS: 'dharmamind-settings',
        CHAT_HISTORY: 'dharmamind-chat-history',
        USER_PREFERENCES: 'dharmamind-preferences',
        MODULE_FAVORITES: 'dharmamind-module-favorites',
        SEARCH_HISTORY: 'dharmamind-search-history'
    },

    // System Status
    SYSTEM: {
        STATUS_CHECK_INTERVAL: 60000, // 1 minute
        HEALTH_CHECK_INTERVAL: 30000, // 30 seconds
        RETRY_CONNECTION_INTERVAL: 5000, // 5 seconds
        MAX_CONNECTION_RETRIES: 10
    },

    // Dharma Modules Configuration
    MODULES: {
        CATEGORIES: {
            'foundational': {
                name: 'Foundational Teachings',
                icon: '🏛️',
                color: '#667eea'
            },
            'philosophical': {
                name: 'Philosophical Concepts',
                icon: '🤔',
                color: '#4facfe'
            },
            'practical': {
                name: 'Practical Wisdom',
                icon: '🛠️',
                color: '#fa709a'
            },
            'advanced': {
                name: 'Advanced Practices',
                icon: '🔮',
                color: '#764ba2'
            },
            'specialized': {
                name: 'Specialized Knowledge',
                icon: '⭐',
                color: '#f093fb'
            }
        },
        TRADITION_COLORS: {
            'universal': '#48bb78',
            'vedanta': '#764ba2',
            'yoga': '#4facfe',
            'ayurveda': '#48bb78',
            'bhakti': '#f093fb',
            'karma_yoga': '#fa709a',
            'dharmic': '#ed8936',
            'vedic': '#4299e1',
            'advaita': '#667eea',
            'tantra': '#f6d365'
        },
        REFRESH_INTERVAL: 300000 // 5 minutes
    },

    // Knowledge Search Configuration
    KNOWLEDGE: {
        MIN_QUERY_LENGTH: 3,
        MAX_RESULTS: 20,
        DEBOUNCE_DELAY: 300,
        SIMILARITY_THRESHOLD: 0.7,
        CACHE_DURATION: 300000, // 5 minutes
        HIGHLIGHT_MATCHES: true
    },

    // Error Messages
    ERRORS: {
        NETWORK: 'Unable to connect to DharmaMind. Please check your connection.',
        TIMEOUT: 'Request timed out. Please try again.',
        SERVER: 'Server error occurred. Our team has been notified.',
        VALIDATION: 'Please check your input and try again.',
        PERMISSION: 'Permission denied. Please check your settings.',
        RATE_LIMIT: 'Too many requests. Please wait a moment before trying again.',
        GENERIC: 'An unexpected error occurred. Please try again.'
    },

    // Success Messages
    MESSAGES: {
        CHAT_EXPORTED: 'Chat conversation exported successfully.',
        SETTINGS_SAVED: 'Settings saved successfully.',
        FEEDBACK_SENT: 'Thank you for your feedback!',
        WISDOM_SHARED: 'Wisdom shared successfully.',
        CONNECTION_RESTORED: 'Connection to DharmaMind restored.',
        MODULE_FAVORITED: 'Module added to favorites.',
        SEARCH_SAVED: 'Search saved to history.'
    },

    // Wisdom Quotes for Status Bar
    WISDOM_QUOTES: [
        "The mind is everything. What you think you become. — Buddha",
        "Be yourself; everyone else is already taken. — Oscar Wilde",
        "The only way to do great work is to love what you do. — Steve Jobs",
        "In the depth of winter, I finally learned that there was in me an invincible summer. — Albert Camus",
        "The journey of a thousand miles begins with a single step. — Lao Tzu",
        "Your task is not to seek for love, but merely to seek and find all the barriers within yourself that you have built against it. — Rumi",
        "The privilege of a lifetime is being who you are. — Joseph Campbell",
        "What lies behind us and what lies before us are tiny matters compared to what lies within us. — Ralph Waldo Emerson",
        "The best way to find yourself is to lose yourself in the service of others. — Mahatma Gandhi",
        "Darkness cannot drive out darkness; only light can do that. Hate cannot drive out hate; only love can do that. — Martin Luther King Jr.",
        "Be the change that you wish to see in the world. — Mahatma Gandhi",
        "Two things are infinite: the universe and human stupidity; and I'm not sure about the universe. — Albert Einstein",
        "Yesterday is history, tomorrow is a mystery, today is a gift of God, which is why we call it the present. — Bill Keane",
        "If you want to go fast, go alone. If you want to go far, go together. — African Proverb",
        "The only impossible journey is the one you never begin. — Tony Robbins"
    ],

    // Debug Configuration
    DEBUG: {
        ENABLED: false, // Set to true for development
        LOG_LEVEL: 'info', // 'debug', 'info', 'warn', 'error'
        LOG_API_CALLS: false,
        LOG_USER_ACTIONS: false,
        PERFORMANCE_MONITORING: false
    }
};

// Theme Detection
if (CONFIG.UI.THEME.DEFAULT === 'auto') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    CONFIG.UI.THEME.DEFAULT = prefersDark ? 'dark' : 'light';
}

// Feature Detection
CONFIG.FEATURES.VOICE_INPUT = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
CONFIG.FEATURES.OFFLINE_MODE = 'serviceWorker' in navigator;
CONFIG.FEATURES.CLIPBOARD_API = 'clipboard' in navigator;
CONFIG.FEATURES.SHARE_API = 'share' in navigator;

// Export configuration
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
} else {
    window.CONFIG = CONFIG;
}

// Debug logging
if (CONFIG.DEBUG.ENABLED) {
    console.log('🌟 DharmaMind Configuration Loaded:', CONFIG);
    console.log('🔧 Features Detected:', {
        voiceInput: CONFIG.FEATURES.VOICE_INPUT,
        offlineMode: CONFIG.FEATURES.OFFLINE_MODE,
        clipboardAPI: CONFIG.FEATURES.CLIPBOARD_API,
        shareAPI: CONFIG.FEATURES.SHARE_API
    });
}
