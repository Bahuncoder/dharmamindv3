// DharmaMind UI Controller
// ========================

class UIController {
    constructor() {
        this.sidebar = null;
        this.header = null;
        this.darkMode = localStorage.getItem(CONFIG.UI.THEME.STORAGE_KEY) === 'dark';
        this.isMenuOpen = false;
        this.currentPage = 'chat';
        
        this.init();
    }

    init() {
        this.setupDarkModeToggle();
        this.setupMobileMenu();
        this.setupTooltips();
        this.setupKeyboardShortcuts();
        this.setupResizeHandler();
        this.initializeTheme();
        this.setupAccessibility();
        this.setupPageNavigation();
    }

    setupDarkModeToggle() {
        const darkModeToggle = document.getElementById('dark-mode-toggle');
        if (darkModeToggle) {
            darkModeToggle.addEventListener('click', () => this.toggleDarkMode());
            this.updateDarkModeIcon();
        }
    }

    toggleDarkMode() {
        this.darkMode = !this.darkMode;
        document.body.classList.toggle('dark-mode', this.darkMode);
        
        // Save preference
        localStorage.setItem(CONFIG.UI.THEME.STORAGE_KEY, this.darkMode ? 'dark' : 'light');
        
        this.updateDarkModeIcon();
        
        // Trigger custom event for other components
        document.dispatchEvent(new CustomEvent('themeChanged', {
            detail: { isDark: this.darkMode }
        }));

        if (window.notifications) {
            window.notifications.show('info', 'Theme Changed', 
                `Switched to ${this.darkMode ? 'dark' : 'light'} mode`);
        }
    }

    updateDarkModeIcon() {
        const darkModeToggle = document.getElementById('dark-mode-toggle');
        if (darkModeToggle) {
            const icon = darkModeToggle.querySelector('i');
            if (icon) {
                icon.className = this.darkMode ? 'fas fa-sun' : 'fas fa-moon';
            }
        }
    }

    setupMobileMenu() {
        const menuToggle = document.getElementById('menu-toggle');
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('mobile-overlay');
        
        if (menuToggle && sidebar) {
            menuToggle.addEventListener('click', () => this.toggleMobileMenu());
        }

        if (overlay) {
            overlay.addEventListener('click', () => this.closeMobileMenu());
        }

        // Close menu on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isMenuOpen) {
                this.closeMobileMenu();
            }
        });

        // Close menu when clicking sidebar links on mobile
        const sidebarLinks = sidebar?.querySelectorAll('.sidebar-link, .module-card');
        sidebarLinks?.forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth <= 768) {
                    this.closeMobileMenu();
                }
            });
        });
    }

    toggleMobileMenu() {
        this.isMenuOpen = !this.isMenuOpen;
        this.updateMobileMenu();
    }

    closeMobileMenu() {
        this.isMenuOpen = false;
        this.updateMobileMenu();
    }

    updateMobileMenu() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('mobile-overlay');
        const menuToggle = document.getElementById('menu-toggle');

        if (sidebar) {
            sidebar.classList.toggle('mobile-open', this.isMenuOpen);
        }

        if (overlay) {
            overlay.classList.toggle('active', this.isMenuOpen);
        }

        if (menuToggle) {
            const icon = menuToggle.querySelector('i');
            if (icon) {
                icon.className = this.isMenuOpen ? 'fas fa-times' : 'fas fa-bars';
            }
        }

        // Prevent body scrolling when menu is open
        document.body.style.overflow = this.isMenuOpen ? 'hidden' : '';
    }

    setupTooltips() {
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        
        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', (e) => this.showTooltip(e));
            element.addEventListener('mouseleave', () => this.hideTooltip());
        });
    }

    showTooltip(event) {
        const element = event.currentTarget;
        const tooltipText = element.dataset.tooltip;
        
        if (!tooltipText) return;

        // Remove existing tooltip
        this.hideTooltip();

        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = tooltipText;
        tooltip.id = 'active-tooltip';

        document.body.appendChild(tooltip);

        // Position tooltip
        const rect = element.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        
        let left = rect.left + rect.width / 2 - tooltipRect.width / 2;
        let top = rect.top - tooltipRect.height - 8;

        // Adjust if tooltip goes off screen
        if (left < 8) left = 8;
        if (left + tooltipRect.width > window.innerWidth - 8) {
            left = window.innerWidth - tooltipRect.width - 8;
        }
        if (top < 8) {
            top = rect.bottom + 8;
            tooltip.classList.add('tooltip-bottom');
        }

        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;

        setTimeout(() => tooltip.classList.add('visible'), 10);
    }

    hideTooltip() {
        const tooltip = document.getElementById('active-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Don't trigger shortcuts when typing in inputs
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
                return;
            }

            // Ctrl/Cmd + / - Show shortcuts help
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                this.showKeyboardShortcuts();
            }

            // Ctrl/Cmd + K - Focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.focusSearch();
            }

            // Ctrl/Cmd + Enter - Send message (when in message input)
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const messageInput = document.getElementById('message-input');
                if (messageInput && document.activeElement === messageInput) {
                    e.preventDefault();
                    if (window.chat) {
                        window.chat.sendMessage();
                    }
                }
            }

            // Escape - Close modals/menus
            if (e.key === 'Escape') {
                this.closeModals();
                this.closeMobileMenu();
            }

            // Ctrl/Cmd + D - Toggle dark mode
            if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
                e.preventDefault();
                this.toggleDarkMode();
            }
        });
    }

    showKeyboardShortcuts() {
        const modal = document.getElementById('shortcuts-modal');
        if (modal) {
            this.showModal('shortcuts-modal');
        } else {
            // Create shortcuts modal dynamically
            this.createShortcutsModal();
        }
    }

    createShortcutsModal() {
        const shortcuts = [
            { key: 'Ctrl + /', description: 'Show keyboard shortcuts' },
            { key: 'Ctrl + K', description: 'Focus search' },
            { key: 'Ctrl + Enter', description: 'Send message' },
            { key: 'Ctrl + D', description: 'Toggle dark mode' },
            { key: 'Escape', description: 'Close modals/menus' },
            { key: 'Tab', description: 'Navigate elements' },
        ];

        const modalHtml = `
            <div class="modal-overlay" id="shortcuts-modal">
                <div class="modal-container">
                    <div class="modal-header">
                        <h3>Keyboard Shortcuts</h3>
                        <button class="modal-close" data-close="shortcuts-modal">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="modal-content">
                        <div class="shortcuts-list">
                            ${shortcuts.map(shortcut => `
                                <div class="shortcut-item">
                                    <div class="shortcut-key">${shortcut.key}</div>
                                    <div class="shortcut-description">${shortcut.description}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHtml);
        this.showModal('shortcuts-modal');
    }

    focusSearch() {
        const searchInput = document.getElementById('knowledge-search');
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        }
    }

    closeModals() {
        const openModals = document.querySelectorAll('.modal-overlay.active');
        openModals.forEach(modal => {
            modal.classList.remove('active');
        });
    }

    setupResizeHandler() {
        let resizeTimeout;
        
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.handleResize();
            }, 250);
        });
    }

    handleResize() {
        // Close mobile menu on desktop
        if (window.innerWidth > 768 && this.isMenuOpen) {
            this.closeMobileMenu();
        }

        // Update chat container height
        this.updateChatHeight();

        // Trigger resize event for other components
        document.dispatchEvent(new CustomEvent('uiResize', {
            detail: {
                width: window.innerWidth,
                height: window.innerHeight
            }
        }));
    }

    updateChatHeight() {
        const chatContainer = document.getElementById('chat-container');
        const header = document.querySelector('.header');
        const chatInput = document.querySelector('.chat-input-container');
        
        if (chatContainer && header && chatInput) {
            const headerHeight = header.offsetHeight;
            const inputHeight = chatInput.offsetHeight;
            const availableHeight = window.innerHeight - headerHeight - inputHeight - 32; // 32px for padding
            
            chatContainer.style.height = `${availableHeight}px`;
        }
    }

    initializeTheme() {
        // Apply saved theme
        document.body.classList.toggle('dark-mode', this.darkMode);
        
        // Set CSS custom properties based on theme
        this.updateThemeVariables();
        
        // Listen for system theme changes
        if (window.matchMedia) {
            const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
            mediaQuery.addEventListener('change', (e) => {
                if (!localStorage.getItem(CONFIG.UI.THEME.STORAGE_KEY)) {
                    this.darkMode = e.matches;
                    document.body.classList.toggle('dark-mode', this.darkMode);
                    this.updateDarkModeIcon();
                }
            });
        }
    }

    updateThemeVariables() {
        const root = document.documentElement;
        
        if (this.darkMode) {
            root.style.setProperty('--primary-bg', '#1a1a2e');
            root.style.setProperty('--secondary-bg', '#16213e');
            root.style.setProperty('--text-primary', '#ffffff');
            root.style.setProperty('--text-secondary', '#b0b0b0');
        } else {
            root.style.setProperty('--primary-bg', '#ffffff');
            root.style.setProperty('--secondary-bg', '#f8f9fa');
            root.style.setProperty('--text-primary', '#2c3e50');
            root.style.setProperty('--text-secondary', '#6c757d');
        }
    }

    setupAccessibility() {
        // Focus management
        this.setupFocusTrap();
        
        // ARIA labels and announcements
        this.updateAriaLabels();
        
        // High contrast mode detection
        if (window.matchMedia) {
            const highContrastQuery = window.matchMedia('(prefers-contrast: high)');
            highContrastQuery.addEventListener('change', (e) => {
                document.body.classList.toggle('high-contrast', e.matches);
            });
            
            if (highContrastQuery.matches) {
                document.body.classList.add('high-contrast');
            }
        }

        // Reduced motion preference
        if (window.matchMedia) {
            const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
            reducedMotionQuery.addEventListener('change', (e) => {
                document.body.classList.toggle('reduced-motion', e.matches);
            });
            
            if (reducedMotionQuery.matches) {
                document.body.classList.add('reduced-motion');
            }
        }
    }

    setupFocusTrap() {
        // Trap focus within modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                const activeModal = document.querySelector('.modal-overlay.active');
                if (activeModal) {
                    this.trapFocus(e, activeModal);
                }
            }
        });
    }

    trapFocus(event, container) {
        const focusableElements = container.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        if (event.shiftKey) {
            if (document.activeElement === firstElement) {
                event.preventDefault();
                lastElement.focus();
            }
        } else {
            if (document.activeElement === lastElement) {
                event.preventDefault();
                firstElement.focus();
            }
        }
    }

    updateAriaLabels() {
        // Update dynamic ARIA labels
        const darkModeToggle = document.getElementById('dark-mode-toggle');
        if (darkModeToggle) {
            darkModeToggle.setAttribute('aria-label', 
                `Switch to ${this.darkMode ? 'light' : 'dark'} mode`);
        }

        const menuToggle = document.getElementById('menu-toggle');
        if (menuToggle) {
            menuToggle.setAttribute('aria-label', 
                `${this.isMenuOpen ? 'Close' : 'Open'} navigation menu`);
        }
    }

    setupPageNavigation() {
        // Handle page navigation
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = link.dataset.page;
                if (page) {
                    this.navigateToPage(page);
                }
            });
        });

        // Handle browser back/forward
        window.addEventListener('popstate', (e) => {
            if (e.state && e.state.page) {
                this.navigateToPage(e.state.page, false);
            }
        });
    }

    navigateToPage(page, updateHistory = true) {
        if (page === this.currentPage) return;

        // Hide all pages
        const pages = document.querySelectorAll('.page-content');
        pages.forEach(p => p.style.display = 'none');

        // Show target page
        const targetPage = document.getElementById(`${page}-page`);
        if (targetPage) {
            targetPage.style.display = 'block';
            this.currentPage = page;

            // Update navigation state
            this.updateNavigation(page);

            // Update browser history
            if (updateHistory) {
                history.pushState({ page }, '', `#${page}`);
            }

            // Trigger page change event
            document.dispatchEvent(new CustomEvent('pageChanged', {
                detail: { page }
            }));
        }
    }

    updateNavigation(activePage) {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.classList.toggle('active', link.dataset.page === activePage);
        });
    }

    showModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('active');
            
            // Focus first focusable element
            setTimeout(() => {
                const firstFocusable = modal.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
                if (firstFocusable) {
                    firstFocusable.focus();
                }
            }, 100);

            // Setup close handlers
            const closeButtons = modal.querySelectorAll('[data-close]');
            closeButtons.forEach(btn => {
                btn.addEventListener('click', () => this.hideModal(modalId));
            });

            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideModal(modalId);
                }
            });
        }
    }

    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('active');
        }
    }

    // Utility methods
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // Public API methods
    toggleSidebar() {
        if (window.innerWidth <= 768) {
            this.toggleMobileMenu();
        }
    }

    showNotification(type, title, message) {
        if (window.notifications) {
            window.notifications.show(type, title, message);
        }
    }

    updateStatus(status, message) {
        const statusBar = document.getElementById('status-bar');
        if (statusBar) {
            statusBar.className = `status-bar status-${status}`;
            statusBar.textContent = message;
            statusBar.style.display = 'block';
            
            if (status === 'success' || status === 'info') {
                setTimeout(() => {
                    statusBar.style.display = 'none';
                }, 3000);
            }
        }
    }
}

// Initialize UI controller
document.addEventListener('DOMContentLoaded', () => {
    window.ui = new UIController();
    
    if (CONFIG.DEBUG.ENABLED) {
        console.log('🎨 DharmaMind UI Controller initialized');
    }
});
