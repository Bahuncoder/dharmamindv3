// DharmaMind Settings Interface
// =============================

class SettingsInterface {
    constructor() {
        this.settings = {};
        this.defaultSettings = this.getDefaultSettings();
        this.settingsModal = null;
        this.isDirty = false;
        
        this.init();
    }

    init() {
        this.loadSettings();
        this.setupSettingsModal();
        this.setupSettingsForm();
        this.setupSettingsNavigation();
        this.applySettings();
    }

    getDefaultSettings() {
        return {
            // Spiritual Preferences
            tradition: CONFIG.UI.TRADITION.DEFAULT,
            spiritual_level: CONFIG.UI.SPIRITUAL_LEVEL.DEFAULT,
            response_style: CONFIG.SPIRITUAL.DEFAULT_RESPONSE_STYLE,
            tradition_neutral: true,
            
            // UI Preferences
            theme: CONFIG.UI.THEME.DEFAULT,
            language: CONFIG.UI.LANGUAGE.DEFAULT,
            font_size: 'medium',
            animations: true,
            sound_effects: false,
            
            // Chat Settings
            streaming_enabled: CONFIG.CHAT.STREAMING_ENABLED,
            auto_save: true,
            message_history_limit: CONFIG.CHAT.MESSAGE_HISTORY_LIMIT,
            include_sources: true,
            typing_indicators: true,
            
            // Privacy Settings
            save_conversations: CONFIG.PRIVACY.SAVE_CONVERSATIONS,
            analytics_enabled: CONFIG.PRIVACY.ANALYTICS_ENABLED,
            personalization: true,
            data_retention: '30days',
            
            // Accessibility
            high_contrast: false,
            reduced_motion: false,
            screen_reader: false,
            keyboard_navigation: true,
            
            // Notifications
            browser_notifications: false,
            sound_notifications: false,
            wisdom_reminders: false,
            meditation_reminders: false,
            
            // Advanced
            debug_mode: CONFIG.DEBUG.ENABLED,
            beta_features: false,
            api_timeout: CONFIG.API.TIMEOUT,
            cache_enabled: true
        };
    }

    setupSettingsModal() {
        const settingsButton = document.getElementById('settings-btn');
        this.settingsModal = document.getElementById('settings-modal');

        if (settingsButton) {
            settingsButton.addEventListener('click', () => {
                this.openSettings();
            });
        }

        if (this.settingsModal) {
            // Setup close handlers
            const closeButtons = this.settingsModal.querySelectorAll('.modal-close, [data-close="settings-modal"]');
            closeButtons.forEach(btn => {
                btn.addEventListener('click', () => {
                    this.closeSettings();
                });
            });

            // Close on overlay click
            this.settingsModal.addEventListener('click', (e) => {
                if (e.target === this.settingsModal) {
                    this.closeSettings();
                }
            });
        }
    }

    setupSettingsForm() {
        if (!this.settingsModal) return;

        // Tab navigation
        const tabButtons = this.settingsModal.querySelectorAll('.settings-tab');
        const tabContents = this.settingsModal.querySelectorAll('.settings-tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabId = button.dataset.tab;
                this.switchSettingsTab(tabId, tabButtons, tabContents);
            });
        });

        // Form inputs
        const formInputs = this.settingsModal.querySelectorAll('input, select, textarea');
        formInputs.forEach(input => {
            input.addEventListener('change', () => {
                this.handleSettingChange(input);
            });
        });

        // Action buttons
        const saveButton = this.settingsModal.querySelector('.settings-save');
        const cancelButton = this.settingsModal.querySelector('.settings-cancel');
        const resetButton = this.settingsModal.querySelector('.settings-reset');

        if (saveButton) {
            saveButton.addEventListener('click', () => {
                this.saveSettings();
            });
        }

        if (cancelButton) {
            cancelButton.addEventListener('click', () => {
                this.cancelSettings();
            });
        }

        if (resetButton) {
            resetButton.addEventListener('click', () => {
                this.resetSettings();
            });
        }

        // Import/Export
        const exportButton = this.settingsModal.querySelector('.settings-export');
        const importButton = this.settingsModal.querySelector('.settings-import');
        const importFile = this.settingsModal.querySelector('#settings-import-file');

        if (exportButton) {
            exportButton.addEventListener('click', () => {
                this.exportSettings();
            });
        }

        if (importButton && importFile) {
            importButton.addEventListener('click', () => {
                importFile.click();
            });

            importFile.addEventListener('change', (e) => {
                this.importSettings(e.target.files[0]);
            });
        }
    }

    setupSettingsNavigation() {
        // Quick settings in header
        const quickSettings = document.querySelectorAll('.quick-setting');
        quickSettings.forEach(setting => {
            setting.addEventListener('click', () => {
                const settingType = setting.dataset.setting;
                this.toggleQuickSetting(settingType);
            });
        });
    }

    openSettings() {
        if (!this.settingsModal) return;

        // Populate form with current settings
        this.populateSettingsForm();
        
        // Show modal
        this.settingsModal.classList.add('active');
        
        // Focus first input
        setTimeout(() => {
            const firstInput = this.settingsModal.querySelector('input, select');
            if (firstInput) {
                firstInput.focus();
            }
        }, 100);
    }

    closeSettings() {
        if (!this.settingsModal) return;

        if (this.isDirty) {
            if (confirm('You have unsaved changes. Do you want to discard them?')) {
                this.isDirty = false;
                this.settingsModal.classList.remove('active');
            }
        } else {
            this.settingsModal.classList.remove('active');
        }
    }

    switchSettingsTab(tabId, tabButtons, tabContents) {
        // Update tab buttons
        tabButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });

        // Update tab contents
        tabContents.forEach(content => {
            content.classList.toggle('active', content.id === `${tabId}-settings`);
        });
    }

    populateSettingsForm() {
        if (!this.settingsModal) return;

        Object.entries(this.settings).forEach(([key, value]) => {
            const input = this.settingsModal.querySelector(`[name="${key}"]`);
            if (input) {
                if (input.type === 'checkbox') {
                    input.checked = Boolean(value);
                } else if (input.type === 'radio') {
                    const radioButton = this.settingsModal.querySelector(`[name="${key}"][value="${value}"]`);
                    if (radioButton) {
                        radioButton.checked = true;
                    }
                } else {
                    input.value = value;
                }
            }
        });

        this.isDirty = false;
    }

    handleSettingChange(input) {
        const key = input.name;
        let value;

        if (input.type === 'checkbox') {
            value = input.checked;
        } else if (input.type === 'radio') {
            value = input.value;
        } else if (input.type === 'number') {
            value = parseInt(input.value) || 0;
        } else {
            value = input.value;
        }

        // Update temporary settings
        this.settings[key] = value;
        this.isDirty = true;

        // Apply immediate changes for some settings
        this.applyImmediateSetting(key, value);

        // Update save button state
        this.updateSaveButton();
    }

    applyImmediateSetting(key, value) {
        switch (key) {
            case 'theme':
                document.body.classList.toggle('dark-mode', value === 'dark');
                break;

            case 'font_size':
                document.documentElement.style.setProperty('--font-size-multiplier', 
                    value === 'small' ? '0.9' : value === 'large' ? '1.1' : '1');
                break;

            case 'animations':
                document.body.classList.toggle('reduced-motion', !value);
                break;

            case 'high_contrast':
                document.body.classList.toggle('high-contrast', value);
                break;

            case 'reduced_motion':
                document.body.classList.toggle('reduced-motion', value);
                break;
        }
    }

    updateSaveButton() {
        const saveButton = this.settingsModal?.querySelector('.settings-save');
        if (saveButton) {
            saveButton.disabled = !this.isDirty;
        }
    }

    saveSettings() {
        try {
            // Validate settings
            const validation = this.validateSettings(this.settings);
            if (!validation.valid) {
                if (window.notifications) {
                    window.notifications.show('error', 'Invalid Settings', validation.message);
                }
                return;
            }

            // Save to localStorage
            localStorage.setItem(CONFIG.STORAGE_KEYS.USER_SETTINGS, JSON.stringify(this.settings));

            // Apply all settings
            this.applySettings();

            // Mark as clean
            this.isDirty = false;
            this.updateSaveButton();

            // Close modal
            this.settingsModal.classList.remove('active');

            // Trigger settings changed event
            document.dispatchEvent(new CustomEvent('settingsChanged', {
                detail: { settings: this.settings }
            }));

            if (window.notifications) {
                window.notifications.show('success', 'Settings Saved', 'Your preferences have been updated');
            }

        } catch (error) {
            console.error('Failed to save settings:', error);
            if (window.notifications) {
                window.notifications.show('error', 'Save Failed', 'Unable to save settings. Please try again.');
            }
        }
    }

    cancelSettings() {
        if (this.isDirty) {
            if (confirm('You have unsaved changes. Do you want to discard them?')) {
                this.loadSettings(); // Reload from storage
                this.isDirty = false;
                this.settingsModal.classList.remove('active');
            }
        } else {
            this.settingsModal.classList.remove('active');
        }
    }

    resetSettings() {
        if (confirm('Are you sure you want to reset all settings to default values?')) {
            this.settings = { ...this.defaultSettings };
            this.populateSettingsForm();
            this.applySettings();
            this.isDirty = true;
            this.updateSaveButton();

            if (window.notifications) {
                window.notifications.show('info', 'Settings Reset', 'All settings have been reset to defaults');
            }
        }
    }

    validateSettings(settings) {
        // Basic validation
        if (!settings.tradition || !CONFIG.UI.TRADITION.OPTIONS.includes(settings.tradition)) {
            return { valid: false, message: 'Invalid spiritual tradition selected' };
        }

        if (!settings.spiritual_level || !CONFIG.UI.SPIRITUAL_LEVEL.OPTIONS.includes(settings.spiritual_level)) {
            return { valid: false, message: 'Invalid spiritual level selected' };
        }

        if (!settings.language || !CONFIG.UI.LANGUAGE.OPTIONS.includes(settings.language)) {
            return { valid: false, message: 'Invalid language selected' };
        }

        if (typeof settings.message_history_limit !== 'number' || 
            settings.message_history_limit < 10 || settings.message_history_limit > 1000) {
            return { valid: false, message: 'Message history limit must be between 10 and 1000' };
        }

        if (typeof settings.api_timeout !== 'number' || 
            settings.api_timeout < 5000 || settings.api_timeout > 60000) {
            return { valid: false, message: 'API timeout must be between 5 and 60 seconds' };
        }

        return { valid: true };
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem(CONFIG.STORAGE_KEYS.USER_SETTINGS);
            if (saved) {
                const parsed = JSON.parse(saved);
                this.settings = { ...this.defaultSettings, ...parsed };
            } else {
                this.settings = { ...this.defaultSettings };
            }
        } catch (error) {
            console.error('Failed to load settings:', error);
            this.settings = { ...this.defaultSettings };
        }
    }

    applySettings() {
        // Apply theme
        document.body.classList.toggle('dark-mode', this.settings.theme === 'dark');

        // Apply font size
        const fontMultiplier = this.settings.font_size === 'small' ? '0.9' : 
                               this.settings.font_size === 'large' ? '1.1' : '1';
        document.documentElement.style.setProperty('--font-size-multiplier', fontMultiplier);

        // Apply accessibility settings
        document.body.classList.toggle('high-contrast', this.settings.high_contrast);
        document.body.classList.toggle('reduced-motion', this.settings.reduced_motion || !this.settings.animations);

        // Update UI components
        this.updateLanguage();
        this.updateNotificationPermissions();

        // Save individual settings to localStorage for quick access
        Object.entries(this.settings).forEach(([key, value]) => {
            const storageKey = CONFIG.STORAGE_KEYS[key.toUpperCase()];
            if (storageKey) {
                localStorage.setItem(storageKey, String(value));
            }
        });
    }

    updateLanguage() {
        // Update document language
        document.documentElement.lang = this.settings.language;

        // Trigger language change event
        document.dispatchEvent(new CustomEvent('languageChanged', {
            detail: { language: this.settings.language }
        }));
    }

    updateNotificationPermissions() {
        if (this.settings.browser_notifications && 'Notification' in window) {
            if (Notification.permission === 'default') {
                Notification.requestPermission().then(permission => {
                    if (permission !== 'granted') {
                        this.settings.browser_notifications = false;
                        this.saveSettings();
                    }
                });
            }
        }
    }

    toggleQuickSetting(settingType) {
        switch (settingType) {
            case 'theme':
                this.settings.theme = this.settings.theme === 'light' ? 'dark' : 'light';
                break;

            case 'animations':
                this.settings.animations = !this.settings.animations;
                break;

            case 'sounds':
                this.settings.sound_effects = !this.settings.sound_effects;
                break;

            case 'notifications':
                this.settings.browser_notifications = !this.settings.browser_notifications;
                break;
        }

        this.applyImmediateSetting(settingType, this.settings[settingType]);
        this.saveSettingsToStorage();

        if (window.notifications) {
            const status = this.settings[settingType] ? 'enabled' : 'disabled';
            window.notifications.show('info', 'Setting Updated', 
                `${settingType.charAt(0).toUpperCase() + settingType.slice(1)} ${status}`);
        }
    }

    saveSettingsToStorage() {
        try {
            localStorage.setItem(CONFIG.STORAGE_KEYS.USER_SETTINGS, JSON.stringify(this.settings));
        } catch (error) {
            console.error('Failed to save settings:', error);
        }
    }

    exportSettings() {
        try {
            const exportData = {
                version: '1.0',
                timestamp: new Date().toISOString(),
                settings: this.settings
            };

            const blob = new Blob([JSON.stringify(exportData, null, 2)], {
                type: 'application/json'
            });

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `dharmamind-settings-${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            if (window.notifications) {
                window.notifications.show('success', 'Settings Exported', 
                    'Your settings have been exported successfully');
            }

        } catch (error) {
            console.error('Export failed:', error);
            if (window.notifications) {
                window.notifications.show('error', 'Export Failed', 
                    'Unable to export settings. Please try again.');
            }
        }
    }

    async importSettings(file) {
        if (!file) return;

        try {
            const text = await file.text();
            const importData = JSON.parse(text);

            // Validate import data
            if (!importData.settings) {
                throw new Error('Invalid settings file format');
            }

            // Merge with defaults and validate
            const newSettings = { ...this.defaultSettings, ...importData.settings };
            const validation = this.validateSettings(newSettings);
            
            if (!validation.valid) {
                throw new Error(validation.message);
            }

            // Apply imported settings
            this.settings = newSettings;
            this.populateSettingsForm();
            this.applySettings();
            this.isDirty = true;
            this.updateSaveButton();

            if (window.notifications) {
                window.notifications.show('success', 'Settings Imported', 
                    'Settings have been imported successfully. Click Save to apply.');
            }

        } catch (error) {
            console.error('Import failed:', error);
            if (window.notifications) {
                window.notifications.show('error', 'Import Failed', 
                    `Unable to import settings: ${error.message}`);
            }
        }
    }

    // Public API methods
    getSetting(key) {
        return this.settings[key];
    }

    setSetting(key, value) {
        this.settings[key] = value;
        this.applyImmediateSetting(key, value);
        this.saveSettingsToStorage();
    }

    getSettings() {
        return { ...this.settings };
    }

    resetSetting(key) {
        if (this.defaultSettings.hasOwnProperty(key)) {
            this.settings[key] = this.defaultSettings[key];
            this.applyImmediateSetting(key, this.settings[key]);
            this.saveSettingsToStorage();
        }
    }
}

// Initialize settings interface
document.addEventListener('DOMContentLoaded', () => {
    window.settings = new SettingsInterface();
    
    if (CONFIG.DEBUG.ENABLED) {
        console.log('⚙️ DharmaMind Settings Interface initialized');
    }
});
