/**
 * DharmaMind Enterprise Configuration
 * Centralized configuration for enterprise features and settings
 */

export const ENTERPRISE_CONFIG = {
  // Application Information
  app: {
    name: 'DharmaMind Enterprise',
    version: '2.1.0-enterprise',
    buildNumber: process.env.BUILD_NUMBER || '20250809001',
    environment: process.env.NODE_ENV || 'development',
  },

  // Performance & Monitoring
  performance: {
    loadingTimeout: 10000, // 10 seconds max loading time
    healthCheckInterval: 30000, // 30 seconds
    performanceMetricsEnabled: true,
    responseTimeThreshold: 200, // milliseconds
    networkLatencyWarning: 1000, // milliseconds
  },

  // Security Configuration
  security: {
    encryptionStandard: 'AES-256',
    sessionTimeout: 3600000, // 1 hour
    maxLoginAttempts: 5,
    securityHeaders: {
      'X-Frame-Options': 'DENY',
      'X-Content-Type-Options': 'nosniff',
      'X-XSS-Protection': '1; mode=block',
      'Referrer-Policy': 'strict-origin-when-cross-origin',
    },
  },

  // Compliance Standards
  compliance: {
    standards: {
      soc2: {
        enabled: true,
        auditDate: '2024-12-15',
        nextReview: '2025-12-15',
      },
      iso27001: {
        enabled: true,
        certificationDate: '2024-11-20',
        nextAudit: '2025-11-20',
      },
      gdpr: {
        enabled: true,
        dataRetentionDays: 365,
        anonymizationEnabled: true,
      },
      hipaa: {
        enabled: false, // Enable for healthcare clients
        encryptionRequired: true,
      },
    },
    dataGovernance: {
      encryptionAtRest: true,
      encryptionInTransit: true,
      auditLogging: true,
      dataClassification: 'CONFIDENTIAL',
    },
  },

  // Service Level Agreement
  sla: {
    availability: 99.9, // 99.9% uptime
    responseTime: 200, // milliseconds
    support: {
      businessHours: '8:00-18:00 UTC',
      emergencySupport: '24/7',
      escalationLevels: ['L1', 'L2', 'L3', 'L4'],
    },
  },

  // AI & Consciousness Configuration
  ai: {
    models: {
      primary: 'dharma-gpt-4-enterprise',
      fallback: 'dharma-gpt-3.5-turbo',
      consciousness: 'dharmic-awareness-v2',
    },
    features: {
      multiModal: true,
      realTimeProcessing: true,
      contextAwareness: true,
      ethicalConstraints: true,
      spiritualWisdom: true,
    },
    limits: {
      tokensPerMinute: 10000,
      conversationsPerHour: 1000,
      maxContextLength: 32000,
    },
  },

  // Enterprise Integrations
  integrations: {
    sso: {
      enabled: true,
      providers: ['SAML', 'OAuth2', 'OpenID Connect'],
      autoProvisioning: true,
    },
    apis: {
      rateLimit: 1000, // requests per minute
      authentication: 'Bearer Token',
      versioning: 'v2',
    },
    webhooks: {
      enabled: true,
      retryAttempts: 3,
      timeout: 30000,
    },
  },

  // Loading Experience
  loading: {
    steps: [
      { key: 'init', text: 'Initializing DharmaMind Enterprise...', icon: '⚡', duration: 200 },
      { key: 'auth', text: 'Authenticating secure session...', icon: '🔐', duration: 400 },
      { key: 'consciousness', text: 'Loading consciousness modules...', icon: '🧠', duration: 600 },
      { key: 'neural', text: 'Establishing neural connections...', icon: '🔗', duration: 800 },
      { key: 'optimization', text: 'Optimizing response algorithms...', icon: '⚙️', duration: 1000 },
      { key: 'interface', text: 'Preparing enterprise interface...', icon: '💼', duration: 1200 },
      { key: 'security', text: 'Finalizing secure environment...', icon: '🛡️', duration: 1400 },
      { key: 'ready', text: 'Enterprise system ready!', icon: '✅', duration: 1600 },
    ],
    animations: {
      particleCount: 15,
      floatingSpeed: 'variable',
      glowIntensity: 'high',
    },
  },

  // Error Handling
  errorHandling: {
    retryAttempts: 3,
    backoffMultiplier: 2,
    timeoutMs: 30000,
    fallbackMode: 'graceful',
    userNotification: true,
  },

  // Analytics & Monitoring
  monitoring: {
    realUserMonitoring: true,
    performanceTracking: true,
    errorTracking: true,
    businessMetrics: true,
    dashboardUrl: '/enterprise/dashboard',
  },

  // Branding
  branding: {
    colors: {
      primary: '#10b981', // emerald-500
      secondary: '#f97316', // orange-500
      accent: '#3b82f6', // blue-500
      enterprise: '#8b5cf6', // purple-500
    },
    fonts: {
      primary: 'Inter, sans-serif',
      mono: 'JetBrains Mono, monospace',
    },
    logo: {
      main: '/logo-enterprise.svg',
      favicon: '/favicon-enterprise.ico',
      badge: 'Enterprise',
    },
  },
} as const;

// Environment-specific overrides
export const getEnterpriseConfig = () => {
  if (typeof window === 'undefined') {
    // Server-side configuration
    return {
      ...ENTERPRISE_CONFIG,
      server: {
        port: process.env.PORT || 3003,
        cors: {
          origin: process.env.CORS_ORIGIN || '*',
          credentials: true,
        },
      },
    };
  }

  // Client-side configuration
  return {
    ...ENTERPRISE_CONFIG,
    client: {
      apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3003/api',
      wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3003',
    },
  };
};

// Type definitions for TypeScript
export type EnterpriseConfig = typeof ENTERPRISE_CONFIG;
export type LoadingStep = typeof ENTERPRISE_CONFIG.loading.steps[0];
export type ComplianceStandard = keyof typeof ENTERPRISE_CONFIG.compliance.standards;
