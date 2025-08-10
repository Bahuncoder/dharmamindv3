// DharmaMind Notifications System
// ===============================

class NotificationSystem {
    constructor() {
        this.container = null;
        this.notifications = [];
        this.maxNotifications = 5;
        this.defaultDuration = 5000;
        this.soundEnabled = false;
        this.browserNotificationsEnabled = false;
        
        this.init();
    }

    init() {
        this.createContainer();
        this.setupNotificationSettings();
        this.setupBrowserNotifications();
        this.loadSoundSettings();
    }

    createContainer() {
        // Check if container already exists
        this.container = document.getElementById('notifications-container');
        
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'notifications-container';
            this.container.className = 'notifications-container';
            document.body.appendChild(this.container);
        }
    }

    setupNotificationSettings() {
        // Listen for settings changes
        document.addEventListener('settingsChanged', (e) => {
            const settings = e.detail.settings;
            this.soundEnabled = settings.sound_notifications || false;
            this.browserNotificationsEnabled = settings.browser_notifications || false;
        });
    }

    setupBrowserNotifications() {
        // Request permission if needed
        if ('Notification' in window && Notification.permission === 'default') {
            // Don't auto-request, wait for user settings
        }
    }

    loadSoundSettings() {
        // Load sound preferences from settings
        if (window.settings) {
            this.soundEnabled = window.settings.getSetting('sound_notifications') || false;
            this.browserNotificationsEnabled = window.settings.getSetting('browser_notifications') || false;
        }
    }

    show(type, title, message, options = {}) {
        const notification = this.createNotification(type, title, message, options);
        this.addNotification(notification);
        
        // Show browser notification if enabled
        if (this.browserNotificationsEnabled && options.browserNotification !== false) {
            this.showBrowserNotification(title, message, type);
        }
        
        // Play sound if enabled
        if (this.soundEnabled && options.silent !== true) {
            this.playNotificationSound(type);
        }
        
        return notification.id;
    }

    createNotification(type, title, message, options) {
        const id = 'notification-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        const duration = options.duration || this.defaultDuration;
        const persistent = options.persistent || false;
        const actions = options.actions || [];

        const notification = {
            id,
            type,
            title,
            message,
            duration,
            persistent,
            actions,
            timestamp: Date.now(),
            element: null
        };

        notification.element = this.createNotificationElement(notification);
        return notification;
    }

    createNotificationElement(notification) {
        const element = document.createElement('div');
        element.className = `notification notification-${notification.type}`;
        element.dataset.notificationId = notification.id;

        const icon = this.getNotificationIcon(notification.type);
        
        element.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">
                    ${icon}
                </div>
                <div class="notification-body">
                    <div class="notification-title">${notification.title}</div>
                    <div class="notification-message">${notification.message}</div>
                    ${notification.actions.length > 0 ? `
                        <div class="notification-actions">
                            ${notification.actions.map(action => `
                                <button class="notification-action" data-action="${action.id}">
                                    ${action.label}
                                </button>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
                <button class="notification-close" aria-label="Close notification">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            ${!notification.persistent ? `
                <div class="notification-progress">
                    <div class="notification-progress-bar" style="animation-duration: ${notification.duration}ms;"></div>
                </div>
            ` : ''}
        `;

        // Setup event listeners
        this.setupNotificationEvents(element, notification);

        return element;
    }

    getNotificationIcon(type) {
        const icons = {
            success: '<i class="fas fa-check-circle"></i>',
            error: '<i class="fas fa-exclamation-circle"></i>',
            warning: '<i class="fas fa-exclamation-triangle"></i>',
            info: '<i class="fas fa-info-circle"></i>',
            wisdom: '<i class="fas fa-lotus"></i>',
            meditation: '<i class="fas fa-om"></i>',
            reminder: '<i class="fas fa-bell"></i>'
        };
        return icons[type] || icons.info;
    }

    setupNotificationEvents(element, notification) {
        // Close button
        const closeButton = element.querySelector('.notification-close');
        closeButton.addEventListener('click', () => {
            this.remove(notification.id);
        });

        // Action buttons
        const actionButtons = element.querySelectorAll('.notification-action');
        actionButtons.forEach(button => {
            button.addEventListener('click', () => {
                const actionId = button.dataset.action;
                this.handleNotificationAction(notification.id, actionId);
            });
        });

        // Click to dismiss (optional)
        if (notification.type !== 'error' && !notification.persistent) {
            element.addEventListener('click', (e) => {
                if (!e.target.closest('.notification-action, .notification-close')) {
                    this.remove(notification.id);
                }
            });
        }

        // Auto-remove after duration
        if (!notification.persistent && notification.duration > 0) {
            setTimeout(() => {
                this.remove(notification.id);
            }, notification.duration);
        }
    }

    addNotification(notification) {
        // Add to notifications array
        this.notifications.push(notification);

        // Remove excess notifications
        while (this.notifications.length > this.maxNotifications) {
            const oldest = this.notifications.shift();
            if (oldest.element && oldest.element.parentNode) {
                oldest.element.remove();
            }
        }

        // Add to DOM
        this.container.appendChild(notification.element);

        // Trigger entrance animation
        setTimeout(() => {
            notification.element.classList.add('notification-show');
        }, 10);

        // Announce to screen readers
        this.announceToScreenReader(notification);
    }

    remove(notificationId) {
        const index = this.notifications.findIndex(n => n.id === notificationId);
        if (index === -1) return;

        const notification = this.notifications[index];
        
        // Remove from array
        this.notifications.splice(index, 1);

        // Remove from DOM with animation
        if (notification.element) {
            notification.element.classList.add('notification-exit');
            
            setTimeout(() => {
                if (notification.element.parentNode) {
                    notification.element.remove();
                }
            }, 300);
        }
    }

    removeAll() {
        this.notifications.forEach(notification => {
            if (notification.element && notification.element.parentNode) {
                notification.element.remove();
            }
        });
        this.notifications = [];
    }

    handleNotificationAction(notificationId, actionId) {
        const notification = this.notifications.find(n => n.id === notificationId);
        if (!notification) return;

        const action = notification.actions.find(a => a.id === actionId);
        if (action && action.callback) {
            action.callback();
        }

        // Remove notification after action
        this.remove(notificationId);

        // Trigger action event
        document.dispatchEvent(new CustomEvent('notificationAction', {
            detail: {
                notificationId,
                actionId,
                notification
            }
        }));
    }

    showBrowserNotification(title, message, type) {
        if ('Notification' in window && Notification.permission === 'granted') {
            try {
                const browserNotification = new Notification(title, {
                    body: message,
                    icon: this.getBrowserNotificationIcon(type),
                    badge: '/favicon.ico',
                    tag: 'dharmamind-notification',
                    requireInteraction: type === 'error',
                    silent: !this.soundEnabled
                });

                // Auto-close after 5 seconds (for non-persistent notifications)
                if (type !== 'error') {
                    setTimeout(() => {
                        browserNotification.close();
                    }, 5000);
                }

                browserNotification.onclick = () => {
                    window.focus();
                    browserNotification.close();
                };

            } catch (error) {
                console.error('Browser notification failed:', error);
            }
        }
    }

    getBrowserNotificationIcon(type) {
        // You would provide actual icon URLs here
        const baseUrl = window.location.origin;
        const icons = {
            success: `${baseUrl}/icons/success.png`,
            error: `${baseUrl}/icons/error.png`,
            warning: `${baseUrl}/icons/warning.png`,
            info: `${baseUrl}/icons/info.png`,
            wisdom: `${baseUrl}/icons/wisdom.png`,
            meditation: `${baseUrl}/icons/meditation.png`,
            reminder: `${baseUrl}/icons/reminder.png`
        };
        return icons[type] || icons.info;
    }

    playNotificationSound(type) {
        if (!this.soundEnabled) return;

        try {
            // Create audio context if it doesn't exist
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }

            // Generate different tones for different notification types
            const frequency = this.getNotificationTone(type);
            this.playTone(frequency, 0.1, 200); // frequency, volume, duration

        } catch (error) {
            console.error('Sound notification failed:', error);
        }
    }

    getNotificationTone(type) {
        const tones = {
            success: 523.25, // C5
            error: 311.13,   // E♭4
            warning: 415.30, // G♯4
            info: 440.00,    // A4
            wisdom: 587.33,  // D5
            meditation: 349.23, // F4
            reminder: 493.88 // B4
        };
        return tones[type] || tones.info;
    }

    playTone(frequency, volume, duration) {
        if (!this.audioContext) return;

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);

        oscillator.frequency.setValueAtTime(frequency, this.audioContext.currentTime);
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
        gainNode.gain.linearRampToValueAtTime(volume, this.audioContext.currentTime + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + duration / 1000);

        oscillator.start(this.audioContext.currentTime);
        oscillator.stop(this.audioContext.currentTime + duration / 1000);
    }

    announceToScreenReader(notification) {
        // Create screen reader announcement
        const announcement = document.createElement('div');
        announcement.className = 'sr-only';
        announcement.setAttribute('aria-live', notification.type === 'error' ? 'assertive' : 'polite');
        announcement.textContent = `${notification.type}: ${notification.title}. ${notification.message}`;
        
        document.body.appendChild(announcement);
        
        // Remove after announcement
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }

    // Predefined notification types for common scenarios
    success(title, message, options = {}) {
        return this.show('success', title, message, options);
    }

    error(title, message, options = {}) {
        return this.show('error', title, message, { persistent: true, ...options });
    }

    warning(title, message, options = {}) {
        return this.show('warning', title, message, options);
    }

    info(title, message, options = {}) {
        return this.show('info', title, message, options);
    }

    wisdom(message, options = {}) {
        return this.show('wisdom', 'Spiritual Wisdom', message, {
            duration: 8000,
            browserNotification: false,
            ...options
        });
    }

    meditation(title, message, options = {}) {
        return this.show('meditation', title, message, {
            duration: 10000,
            ...options
        });
    }

    reminder(title, message, options = {}) {
        return this.show('reminder', title, message, {
            persistent: true,
            actions: [
                { id: 'dismiss', label: 'Dismiss' },
                { id: 'remind_later', label: 'Remind Later' }
            ],
            ...options
        });
    }

    // API for scheduled notifications
    scheduleReminder(title, message, delay, options = {}) {
        return setTimeout(() => {
            this.reminder(title, message, options);
        }, delay);
    }

    scheduleMeditationReminder(delay = 3600000) { // Default 1 hour
        return this.scheduleReminder(
            'Meditation Time',
            'Take a moment to center yourself and breathe mindfully.',
            delay,
            {
                actions: [
                    { 
                        id: 'start_meditation', 
                        label: 'Start Meditation',
                        callback: () => {
                            // Start meditation guide
                            if (window.chat) {
                                window.chat.sendMessage('Guide me through a short meditation');
                            }
                        }
                    },
                    { id: 'remind_later', label: 'Remind Later' }
                ]
            }
        );
    }

    scheduleWisdomReminder(delay = 7200000) { // Default 2 hours
        const wisdomQuotes = [
            "The mind is everything. What you think you become. - Buddha",
            "Be yourself; everyone else is already taken. - Oscar Wilde",
            "The present moment is the only time over which we have dominion. - Thich Nhat Hanh",
            "Peace comes from within. Do not seek it without. - Buddha",
            "Your task is not to seek for love, but merely to seek and find all the barriers within yourself that you have built against it. - Rumi"
        ];

        const randomQuote = wisdomQuotes[Math.floor(Math.random() * wisdomQuotes.length)];
        
        return this.scheduleReminder(
            'Daily Wisdom',
            randomQuote,
            delay,
            {
                actions: [
                    { id: 'reflect', label: 'Reflect on This' },
                    { id: 'share', label: 'Share Wisdom' }
                ]
            }
        );
    }

    // Settings management
    updateSettings(newSettings) {
        this.soundEnabled = newSettings.sound_notifications || false;
        this.browserNotificationsEnabled = newSettings.browser_notifications || false;
        this.maxNotifications = newSettings.max_notifications || 5;
    }

    // Cleanup method
    destroy() {
        this.removeAll();
        if (this.container && this.container.parentNode) {
            this.container.remove();
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}

// Initialize notifications system
document.addEventListener('DOMContentLoaded', () => {
    window.notifications = new NotificationSystem();
    
    if (CONFIG.DEBUG.ENABLED) {
        console.log('🔔 DharmaMind Notifications System initialized');
    }
});
