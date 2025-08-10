// DharmaMind Application Controller
// =================================

class DharmaMindApp {
    constructor() {
        this.version = '1.0.0';
        this.initialized = false;
        this.components = {};
        this.eventListeners = [];
        this.healthCheckInterval = null;
        this.connectionStatus = 'disconnected';
        
        this.init();
    }

    async init() {
        try {
            console.log('🙏 Initializing DharmaMind Application...');
            
            // Show loading state
            this.showLoadingState();
            
            // Initialize core systems
            await this.initializeCore();
            
            // Initialize components
            await this.initializeComponents();
            
            // Setup global event listeners
            this.setupGlobalEvents();
            
            // Perform initial health check
            await this.performHealthCheck();
            
            // Start periodic health checks
            this.startHealthChecks();
            
            // Setup error handling
            this.setupErrorHandling();
            
            // Setup service worker
            await this.setupServiceWorker();
            
            // Initialize feature detection
            this.initializeFeatureDetection();
            
            // Setup analytics (if enabled)
            this.setupAnalytics();
            
            // Mark as initialized
            this.initialized = true;
            
            // Hide loading state
            this.hideLoadingState();
            
            // Show welcome if first time
            this.checkFirstTimeUser();
            
            // Trigger app ready event
            this.triggerAppReady();
            
            console.log('✨ DharmaMind Application initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize DharmaMind Application:', error);
            this.showErrorState(error);
        }
    }

    showLoadingState() {
        const loader = document.getElementById('app-loader');
        if (loader) {
            loader.style.display = 'flex';
        }
        
        // Show loading overlay with spiritual animation
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="lotus-loader">
                    <div class="lotus-petal"></div>
                    <div class="lotus-petal"></div>
                    <div class="lotus-petal"></div>
                    <div class="lotus-petal"></div>
                    <div class="lotus-petal"></div>
                    <div class="lotus-petal"></div>
                    <div class="lotus-petal"></div>
                    <div class="lotus-petal"></div>
                </div>
                <h2>Awakening DharmaMind...</h2>
                <p class="loading-message">Preparing your spiritual guidance system</p>
            </div>
        `;
        document.body.appendChild(overlay);
    }

    hideLoadingState() {
        const loader = document.getElementById('app-loader');
        if (loader) {
            loader.style.display = 'none';
        }
        
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('fade-out');
            setTimeout(() => overlay.remove(), 500);
        }
    }

    showErrorState(error) {
        const errorHtml = `
            <div class="error-overlay">
                <div class="error-content">
                    <div class="error-icon">⚠️</div>
                    <h2>Unable to Initialize DharmaMind</h2>
                    <p>We encountered an issue while starting the application:</p>
                    <div class="error-message">${error.message}</div>
                    <div class="error-actions">
                        <button onclick="location.reload()" class="btn btn-primary">
                            <i class="fas fa-redo"></i> Reload Application
                        </button>
                        <button onclick="app.reportError('${error.message}')" class="btn btn-secondary">
                            <i class="fas fa-bug"></i> Report Issue
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', errorHtml);
    }

    async initializeCore() {
        // Initialize configuration
        if (typeof CONFIG === 'undefined') {
            throw new Error('Configuration not loaded');
        }
        
        // Initialize API client
        if (typeof api === 'undefined' || !api) {
            throw new Error('API client not available');
        }
        
        // Wait for DOM to be ready
        if (document.readyState !== 'complete') {
            await new Promise(resolve => {
                document.addEventListener('DOMContentLoaded', resolve);
            });
        }
    }

    async initializeComponents() {
        const initPromises = [];
        
        // Wait for all component scripts to load
        await this.waitForComponents();
        
        // Initialize components in order
        const componentInitializers = [
            () => this.initializeSettings(),
            () => this.initializeNotifications(),
            () => this.initializeUI(),
            () => this.initializeModules(),
            () => this.initializeSearch(),
            () => this.initializeChat()
        ];
        
        for (const initializer of componentInitializers) {
            try {
                await initializer();
            } catch (error) {
                console.warn('Component initialization warning:', error);
            }
        }
    }

    async waitForComponents() {
        const requiredComponents = ['settings', 'notifications', 'ui', 'modules', 'search', 'chat'];
        const maxWaitTime = 10000; // 10 seconds
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWaitTime) {
            const allLoaded = requiredComponents.every(component => 
                window[component] && typeof window[component] === 'object'
            );
            
            if (allLoaded) {
                return;
            }
            
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        // Log which components are missing
        const missing = requiredComponents.filter(component => !window[component]);
        if (missing.length > 0) {
            console.warn('Some components failed to load:', missing);
        }
    }

    async initializeSettings() {
        if (window.settings) {
            this.components.settings = window.settings;
            console.log('⚙️ Settings component initialized');
        }
    }

    async initializeNotifications() {
        if (window.notifications) {
            this.components.notifications = window.notifications;
            console.log('🔔 Notifications component initialized');
        }
    }

    async initializeUI() {
        if (window.ui) {
            this.components.ui = window.ui;
            console.log('🎨 UI component initialized');
        }
    }

    async initializeModules() {
        if (window.modules) {
            this.components.modules = window.modules;
            console.log('📦 Modules component initialized');
        }
    }

    async initializeSearch() {
        if (window.search) {
            this.components.search = window.search;
            console.log('🔍 Search component initialized');
        }
    }

    async initializeChat() {
        if (window.chat) {
            this.components.chat = window.chat;
            console.log('💬 Chat component initialized');
        }
    }

    setupGlobalEvents() {
        // Window events
        window.addEventListener('beforeunload', (e) => {
            this.handleBeforeUnload(e);
        });

        window.addEventListener('online', () => {
            this.handleConnectionChange(true);
        });

        window.addEventListener('offline', () => {
            this.handleConnectionChange(false);
        });

        window.addEventListener('error', (e) => {
            this.handleGlobalError(e);
        });

        window.addEventListener('unhandledrejection', (e) => {
            this.handleUnhandledRejection(e);
        });

        // Visibility change
        document.addEventListener('visibilitychange', () => {
            this.handleVisibilityChange();
        });

        // Custom events
        document.addEventListener('settingsChanged', (e) => {
            this.handleSettingsChanged(e.detail);
        });

        document.addEventListener('modulesChanged', (e) => {
            this.handleModulesChanged(e.detail);
        });

        document.addEventListener('themeChanged', (e) => {
            this.handleThemeChanged(e.detail);
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleGlobalKeydown(e);
        });
    }

    async performHealthCheck() {
        try {
            const health = await api.checkHealth();
            this.updateConnectionStatus('connected');
            this.updateHealthStatus(health);
            
            if (this.components.notifications) {
                this.components.notifications.success('Connected', 'Successfully connected to DharmaMind services');
            }
        } catch (error) {
            console.warn('Health check failed:', error);
            this.updateConnectionStatus('disconnected');
            
            if (this.components.notifications) {
                this.components.notifications.warning('Connection', 'Some features may be limited in offline mode');
            }
        }
    }

    startHealthChecks() {
        this.healthCheckInterval = setInterval(async () => {
            try {
                await api.checkHealth();
                if (this.connectionStatus !== 'connected') {
                    this.updateConnectionStatus('connected');
                    if (this.components.notifications) {
                        this.components.notifications.success('Connected', 'Connection restored');
                    }
                }
            } catch (error) {
                if (this.connectionStatus !== 'disconnected') {
                    this.updateConnectionStatus('disconnected');
                    if (this.components.notifications) {
                        this.components.notifications.warning('Disconnected', 'Working in offline mode');
                    }
                }
            }
        }, CONFIG.API.HEALTH_CHECK_INTERVAL || 30000);
    }

    updateConnectionStatus(status) {
        this.connectionStatus = status;
        document.body.classList.toggle('offline', status === 'disconnected');
        
        const statusIndicator = document.getElementById('connection-status');
        if (statusIndicator) {
            statusIndicator.className = `connection-status ${status}`;
            statusIndicator.title = status === 'connected' ? 'Connected to DharmaMind' : 'Offline mode';
        }
    }

    updateHealthStatus(health) {
        // Update health indicators in UI
        const healthStatus = document.getElementById('health-status');
        if (healthStatus && health) {
            healthStatus.innerHTML = `
                <div class="health-indicator">
                    <span class="health-label">API:</span>
                    <span class="health-value ${health.api ? 'healthy' : 'unhealthy'}">
                        ${health.api ? '🟢' : '🔴'}
                    </span>
                </div>
                <div class="health-indicator">
                    <span class="health-label">Database:</span>
                    <span class="health-value ${health.database ? 'healthy' : 'unhealthy'}">
                        ${health.database ? '🟢' : '🔴'}
                    </span>
                </div>
                <div class="health-indicator">
                    <span class="health-label">AI:</span>
                    <span class="health-value ${health.ai ? 'healthy' : 'unhealthy'}">
                        ${health.ai ? '🟢' : '🔴'}
                    </span>
                </div>
            `;
        }
    }

    setupErrorHandling() {
        // Set up global error boundary
        this.originalConsoleError = console.error;
        console.error = (...args) => {
            this.originalConsoleError.apply(console, args);
            
            // Report critical errors
            if (args[0] && typeof args[0] === 'string' && args[0].includes('Uncaught')) {
                this.reportError(args.join(' '));
            }
        };
    }

    async setupServiceWorker() {
        if (!CONFIG.FEATURES.SERVICE_WORKER || !('serviceWorker' in navigator)) {
            console.log('Service Worker not supported or disabled');
            return;
        }

        try {
            const registration = await navigator.serviceWorker.register('/service-worker.js', {
                scope: '/'
            });
            
            console.log('Service Worker registered:', registration);
            
            // Listen for updates
            registration.addEventListener('updatefound', () => {
                const newWorker = registration.installing;
                newWorker.addEventListener('statechange', () => {
                    if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                        if (this.components.notifications) {
                            this.components.notifications.info('Update Available', 
                                'A new version of DharmaMind is available. Refresh to update.');
                        }
                    }
                });
            });
            
        } catch (error) {
            console.warn('Service Worker registration failed:', error);
        }
    }

    initializeFeatureDetection() {
        // Detect browser capabilities
        const features = {
            webgl: this.detectWebGL(),
            webrtc: this.detectWebRTC(),
            notifications: 'Notification' in window,
            clipboard: navigator.clipboard !== undefined,
            share: navigator.share !== undefined,
            speechRecognition: 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window,
            speechSynthesis: 'speechSynthesis' in window,
            storage: this.detectStorage(),
            offlineSupport: 'serviceWorker' in navigator,
            touch: 'ontouchstart' in window
        };

        // Update CONFIG with detected features
        Object.assign(CONFIG.FEATURES, features);
        
        // Store in global scope
        window.DETECTED_FEATURES = features;
        
        console.log('🔍 Feature detection complete:', features);
    }

    detectWebGL() {
        try {
            const canvas = document.createElement('canvas');
            return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
        } catch (e) {
            return false;
        }
    }

    detectWebRTC() {
        return !!(window.RTCPeerConnection || window.webkitRTCPeerConnection || window.mozRTCPeerConnection);
    }

    detectStorage() {
        try {
            const test = 'test';
            localStorage.setItem(test, test);
            localStorage.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    }

    setupAnalytics() {
        if (!CONFIG.PRIVACY.ANALYTICS_ENABLED) {
            console.log('📊 Analytics disabled by configuration');
            return;
        }

        // Initialize privacy-friendly analytics
        this.analytics = {
            pageView: (page) => {
                // Implement privacy-friendly page view tracking
                console.log('📊 Page view:', page);
            },
            event: (category, action, label) => {
                // Implement privacy-friendly event tracking
                console.log('📊 Event:', category, action, label);
            }
        };
    }

    checkFirstTimeUser() {
        const isFirstTime = !localStorage.getItem(CONFIG.STORAGE_KEYS.FIRST_VISIT);
        
        if (isFirstTime) {
            localStorage.setItem(CONFIG.STORAGE_KEYS.FIRST_VISIT, 'false');
            
            // Show welcome tour or onboarding
            setTimeout(() => {
                this.showWelcomeExperience();
            }, 1000);
        }
    }

    showWelcomeExperience() {
        if (this.components.notifications) {
            this.components.notifications.wisdom(
                'Welcome to DharmaMind! 🙏 I\'m here to guide you on your spiritual journey. Feel free to ask me anything about meditation, philosophy, or spiritual practices.',
                {
                    duration: 10000,
                    persistent: true,
                    actions: [
                        {
                            id: 'start_tour',
                            label: 'Take Tour',
                            callback: () => this.startTour()
                        },
                        {
                            id: 'start_chatting',
                            label: 'Start Chatting',
                            callback: () => this.focusChat()
                        }
                    ]
                }
            );
        }
    }

    startTour() {
        // Implement guided tour of features
        if (this.components.notifications) {
            this.components.notifications.info('Tour', 'Guided tour feature coming soon!');
        }
    }

    focusChat() {
        const messageInput = document.getElementById('message-input');
        if (messageInput) {
            messageInput.focus();
        }
    }

    triggerAppReady() {
        // Dispatch app ready event
        document.dispatchEvent(new CustomEvent('dharmamindReady', {
            detail: {
                version: this.version,
                components: Object.keys(this.components),
                features: window.DETECTED_FEATURES
            }
        }));
    }

    // Event handlers
    handleBeforeUnload(e) {
        // Check if there are unsaved changes
        if (this.components.settings && this.components.settings.isDirty) {
            e.preventDefault();
            e.returnValue = 'You have unsaved settings. Are you sure you want to leave?';
        }
    }

    handleConnectionChange(isOnline) {
        const status = isOnline ? 'connected' : 'disconnected';
        this.updateConnectionStatus(status);
        
        if (this.components.notifications) {
            const message = isOnline ? 'Connection restored' : 'Working in offline mode';
            const type = isOnline ? 'success' : 'warning';
            this.components.notifications.show(type, 'Connection', message);
        }
    }

    handleGlobalError(e) {
        console.error('Global error:', e.error);
        
        if (e.error && e.error.stack && this.components.notifications) {
            this.components.notifications.error('Application Error', 
                'An unexpected error occurred. Please refresh if the issue persists.');
        }
    }

    handleUnhandledRejection(e) {
        console.error('Unhandled promise rejection:', e.reason);
        
        if (this.components.notifications) {
            this.components.notifications.error('System Error', 
                'A system error occurred. Some features may not work correctly.');
        }
    }

    handleVisibilityChange() {
        if (document.hidden) {
            // Page is hidden
            this.pauseNonEssentialTasks();
        } else {
            // Page is visible
            this.resumeNonEssentialTasks();
            
            // Perform health check when returning
            this.performHealthCheck();
        }
    }

    handleSettingsChanged(detail) {
        console.log('Settings changed:', detail.settings);
        
        // Update components with new settings
        Object.values(this.components).forEach(component => {
            if (component.updateSettings) {
                component.updateSettings(detail.settings);
            }
        });
    }

    handleModulesChanged(detail) {
        console.log('Modules changed:', detail);
        
        // Update chat context with selected modules
        if (this.components.chat && detail.selected) {
            this.components.chat.updateModuleContext(detail.selected);
        }
    }

    handleThemeChanged(detail) {
        console.log('Theme changed:', detail.isDark ? 'dark' : 'light');
    }

    handleGlobalKeydown(e) {
        // Global keyboard shortcuts
        if (e.ctrlKey || e.metaKey) {
            switch (e.key) {
                case ',':
                    e.preventDefault();
                    if (this.components.settings) {
                        this.components.settings.openSettings();
                    }
                    break;
            }
        }
    }

    pauseNonEssentialTasks() {
        // Pause animations, reduce update frequency, etc.
        document.body.classList.add('page-hidden');
    }

    resumeNonEssentialTasks() {
        // Resume normal operation
        document.body.classList.remove('page-hidden');
    }

    reportError(errorMessage) {
        // Implement error reporting
        console.log('Reporting error:', errorMessage);
        
        if (this.components.notifications) {
            this.components.notifications.info('Error Reported', 
                'Thank you for reporting this issue. We\'ll investigate promptly.');
        }
    }

    // Public API methods
    getComponent(name) {
        return this.components[name];
    }

    isInitialized() {
        return this.initialized;
    }

    getVersion() {
        return this.version;
    }

    restart() {
        window.location.reload();
    }

    // Cleanup
    destroy() {
        // Clear intervals
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }

        // Cleanup components
        Object.values(this.components).forEach(component => {
            if (component.destroy) {
                component.destroy();
            }
        });

        // Remove event listeners
        this.eventListeners.forEach(({ element, event, handler }) => {
            element.removeEventListener(event, handler);
        });

        // Restore console
        if (this.originalConsoleError) {
            console.error = this.originalConsoleError;
        }
    }
}

// Initialize DharmaMind Application
window.app = new DharmaMindApp();

// Export for global access
window.DharmaMindApp = DharmaMindApp;
