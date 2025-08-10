// DharmaMind Modules Interface
// ============================

class ModulesInterface {
    constructor() {
        this.modules = [];
        this.selectedModules = new Set();
        this.availableModules = {};
        this.moduleStates = {};
        
        this.init();
    }

    async init() {
        this.setupModuleGrid();
        this.setupModuleFilters();
        this.setupModuleSearch();
        this.loadModulePreferences();
        await this.loadAvailableModules();
        this.renderModules();
    }

    setupModuleGrid() {
        const moduleGrid = document.getElementById('dharma-modules');
        if (moduleGrid) {
            moduleGrid.addEventListener('click', (e) => {
                const moduleCard = e.target.closest('.module-card');
                if (moduleCard) {
                    this.handleModuleClick(moduleCard);
                }
            });
        }
    }

    setupModuleFilters() {
        const filterButtons = document.querySelectorAll('.module-filter');
        filterButtons.forEach(button => {
            button.addEventListener('click', () => {
                this.setActiveFilter(button.dataset.filter);
            });
        });
    }

    setupModuleSearch() {
        const moduleSearch = document.getElementById('module-search');
        if (moduleSearch) {
            moduleSearch.addEventListener('input', this.debounce((e) => {
                this.filterModules(e.target.value);
            }, 300));
        }
    }

    async loadAvailableModules() {
        try {
            // Get modules from API
            const response = await api.getModules();
            this.availableModules = response.modules || {};
            
            // Transform to array for easier handling
            this.modules = Object.entries(this.availableModules).map(([key, module]) => ({
                id: key,
                name: module.name || key,
                description: module.description || '',
                category: module.category || 'general',
                tradition: module.tradition || 'universal',
                level: module.level || 'beginner',
                enabled: module.enabled !== false,
                icon: module.icon || this.getDefaultIcon(module.category),
                features: module.features || [],
                dependencies: module.dependencies || [],
                status: module.status || 'ready'
            }));

            if (CONFIG.DEBUG.ENABLED) {
                console.log('📦 Loaded modules:', this.modules);
            }

        } catch (error) {
            console.error('Failed to load modules:', error);
            
            // Fallback to default modules
            this.loadDefaultModules();
            
            if (window.notifications) {
                window.notifications.show('warning', 'Modules', 
                    'Some modules may not be available. Using default configuration.');
            }
        }
    }

    loadDefaultModules() {
        this.modules = [
            {
                id: 'vedanta_wisdom',
                name: 'Vedanta Wisdom',
                description: 'Non-dual philosophy and self-realization teachings from Upanishads',
                category: 'wisdom',
                tradition: 'vedanta',
                level: 'advanced',
                icon: '🕉️',
                enabled: true,
                features: ['self_inquiry', 'consciousness', 'brahman'],
                status: 'ready'
            },
            {
                id: 'yoga_practice',
                name: 'Yoga Practice',
                description: 'Complete yoga system - asana, pranayama, meditation from Patanjali',
                category: 'practice',
                tradition: 'yoga',
                level: 'beginner',
                icon: '🧘',
                enabled: true,
                features: ['asanas', 'pranayama', 'meditation'],
                status: 'ready'
            },
            {
                id: 'ayurveda_healing',
                name: 'Ayurveda Healing',
                description: 'Ancient Vedic healing science and holistic wellness system',
                category: 'health',
                tradition: 'ayurveda',
                level: 'intermediate',
                icon: '🌿',
                enabled: true,
                features: ['doshas', 'herbs', 'lifestyle'],
                status: 'ready'
            },
            {
                id: 'bhakti_devotion',
                name: 'Bhakti Yoga',
                description: 'Path of love and devotion to the Divine through surrender',
                category: 'devotion',
                tradition: 'bhakti',
                level: 'beginner',
                icon: '💖',
                enabled: true,
                features: ['kirtan', 'japa', 'surrender'],
                status: 'ready'
            },
            {
                id: 'karma_yoga',
                name: 'Karma Yoga',
                description: 'Path of selfless action and service as taught in Bhagavad Gita',
                category: 'action',
                tradition: 'karma_yoga',
                level: 'beginner',
                icon: '🤝',
                enabled: true,
                features: ['selfless_service', 'dharmic_action', 'detachment'],
                status: 'ready'
            },
            {
                id: 'dharmic_living',
                name: 'Dharmic Living',
                description: 'Living according to dharma and cosmic order (Rita)',
                category: 'lifestyle',
                tradition: 'dharmic',
                level: 'intermediate',
                icon: '⚖️',
                enabled: true,
                features: ['ethics', 'right_action', 'life_purpose'],
                status: 'ready'
            },
            {
                id: 'vedic_wisdom',
                name: 'Vedic Wisdom',
                description: 'Ancient Vedic knowledge from Rigveda, Samaveda, Yajurveda, Atharvaveda',
                category: 'wisdom',
                tradition: 'vedic',
                level: 'intermediate',
                icon: '�',
                enabled: true,
                features: ['vedic_mantras', 'rituals', 'cosmic_knowledge'],
                status: 'ready'
            },
            {
                id: 'advaita_philosophy',
                name: 'Advaita Vedanta',
                description: 'Non-dualistic philosophy as taught by Adi Shankaracharya',
                category: 'philosophy',
                tradition: 'advaita',
                level: 'advanced',
                icon: '🌟',
                enabled: true,
                features: ['non_duality', 'maya_understanding', 'self_realization'],
                status: 'ready'
            },
            {
                id: 'tantra_practice',
                name: 'Tantra Sadhana',
                description: 'Sacred tantric practices for spiritual transformation',
                category: 'practice',
                tradition: 'tantra',
                level: 'advanced',
                icon: '🔱',
                enabled: true,
                features: ['yantra', 'mantra', 'energy_work'],
                status: 'ready'
            },
            {
                id: 'universal_dharma',
                name: 'Universal Dharma',
                description: 'Universal principles accessible to all seekers worldwide',
                category: 'universal',
                tradition: 'universal',
                level: 'beginner',
                icon: '🌍',
                enabled: true,
                features: ['meditation', 'compassion', 'mindfulness'],
                status: 'ready'
            }
        ];
    }

    renderModules() {
        const moduleGrid = document.getElementById('dharma-modules');
        if (!moduleGrid) return;

        const activeFilter = document.querySelector('.module-filter.active')?.dataset.filter || 'all';
        const searchTerm = document.getElementById('module-search')?.value.toLowerCase() || '';

        let filteredModules = this.modules.filter(module => {
            // Filter by category
            if (activeFilter !== 'all' && module.category !== activeFilter) {
                return false;
            }

            // Filter by search term
            if (searchTerm && !module.name.toLowerCase().includes(searchTerm) && 
                !module.description.toLowerCase().includes(searchTerm)) {
                return false;
            }

            return true;
        });

        moduleGrid.innerHTML = filteredModules.map(module => this.createModuleCard(module)).join('');

        // Update module count
        this.updateModuleCount(filteredModules.length);
    }

    createModuleCard(module) {
        const isSelected = this.selectedModules.has(module.id);
        const isEnabled = module.enabled && module.status === 'ready';
        
        return `
            <div class="module-card ${isSelected ? 'selected' : ''} ${!isEnabled ? 'disabled' : ''}" 
                 data-module-id="${module.id}"
                 data-category="${module.category}"
                 data-tradition="${module.tradition}"
                 data-level="${module.level}">
                
                <div class="module-header">
                    <div class="module-icon">${module.icon}</div>
                    <div class="module-status">
                        ${this.getStatusIcon(module.status)}
                    </div>
                </div>
                
                <div class="module-content">
                    <h4 class="module-name">${module.name}</h4>
                    <p class="module-description">${module.description}</p>
                    
                    <div class="module-meta">
                        <span class="module-tradition">${module.tradition}</span>
                        <span class="module-level">${module.level}</span>
                    </div>
                    
                    ${module.features.length > 0 ? `
                        <div class="module-features">
                            ${module.features.slice(0, 3).map(feature => 
                                `<span class="feature-tag">${this.formatFeature(feature)}</span>`
                            ).join('')}
                            ${module.features.length > 3 ? `<span class="feature-more">+${module.features.length - 3}</span>` : ''}
                        </div>
                    ` : ''}
                </div>
                
                <div class="module-actions">
                    <button class="module-toggle ${isSelected ? 'active' : ''}" 
                            ${!isEnabled ? 'disabled' : ''}>
                        <i class="fas ${isSelected ? 'fa-check' : 'fa-plus'}"></i>
                        ${isSelected ? 'Active' : 'Activate'}
                    </button>
                    
                    <button class="module-info" data-module-info="${module.id}">
                        <i class="fas fa-info-circle"></i>
                    </button>
                </div>
            </div>
        `;
    }

    getDefaultIcon(category) {
        const icons = {
            practice: '🧘',
            wisdom: '📿',
            healing: '🌟',
            mysticism: '🔮',
            knowledge: '📚',
            divination: '🌙',
            general: '✨'
        };
        return icons[category] || icons.general;
    }

    getStatusIcon(status) {
        const icons = {
            ready: '<i class="fas fa-check-circle status-ready"></i>',
            loading: '<i class="fas fa-spinner fa-spin status-loading"></i>',
            error: '<i class="fas fa-exclamation-triangle status-error"></i>',
            disabled: '<i class="fas fa-pause-circle status-disabled"></i>'
        };
        return icons[status] || icons.ready;
    }

    formatFeature(feature) {
        return feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    handleModuleClick(moduleCard) {
        const moduleId = moduleCard.dataset.moduleId;
        const module = this.modules.find(m => m.id === moduleId);
        
        if (!module || !module.enabled || module.status !== 'ready') {
            return;
        }

        // Check if clicking info button
        if (event.target.closest('.module-info')) {
            this.showModuleInfo(module);
            return;
        }

        // Toggle module selection
        this.toggleModule(moduleId);
    }

    toggleModule(moduleId) {
        const module = this.modules.find(m => m.id === moduleId);
        if (!module) return;

        if (this.selectedModules.has(moduleId)) {
            this.selectedModules.delete(moduleId);
        } else {
            // Check dependencies
            if (this.checkDependencies(module)) {
                this.selectedModules.add(moduleId);
            } else {
                this.showDependencyWarning(module);
                return;
            }
        }

        // Update UI
        this.updateModuleCard(moduleId);
        this.saveModulePreferences();
        
        // Trigger module change event
        document.dispatchEvent(new CustomEvent('modulesChanged', {
            detail: {
                selected: Array.from(this.selectedModules),
                changed: moduleId,
                action: this.selectedModules.has(moduleId) ? 'added' : 'removed'
            }
        }));

        if (window.notifications) {
            const action = this.selectedModules.has(moduleId) ? 'activated' : 'deactivated';
            window.notifications.show('success', 'Module Updated', 
                `${module.name} has been ${action}`);
        }
    }

    checkDependencies(module) {
        if (!module.dependencies || module.dependencies.length === 0) {
            return true;
        }

        for (const dep of module.dependencies) {
            if (!this.selectedModules.has(dep)) {
                return false;
            }
        }
        return true;
    }

    showDependencyWarning(module) {
        const missing = module.dependencies.filter(dep => !this.selectedModules.has(dep));
        const missingNames = missing.map(dep => {
            const depModule = this.modules.find(m => m.id === dep);
            return depModule ? depModule.name : dep;
        });

        if (window.notifications) {
            window.notifications.show('warning', 'Dependencies Required', 
                `${module.name} requires: ${missingNames.join(', ')}`);
        }
    }

    updateModuleCard(moduleId) {
        const card = document.querySelector(`[data-module-id="${moduleId}"]`);
        if (!card) return;

        const isSelected = this.selectedModules.has(moduleId);
        const toggleBtn = card.querySelector('.module-toggle');
        const icon = toggleBtn.querySelector('i');

        card.classList.toggle('selected', isSelected);
        toggleBtn.classList.toggle('active', isSelected);
        
        icon.className = `fas ${isSelected ? 'fa-check' : 'fa-plus'}`;
        toggleBtn.innerHTML = `<i class="${icon.className}"></i> ${isSelected ? 'Active' : 'Activate'}`;
    }

    showModuleInfo(module) {
        const modalHtml = `
            <div class="modal-overlay" id="module-info-modal">
                <div class="modal-container">
                    <div class="modal-header">
                        <div class="module-info-header">
                            <span class="module-icon-large">${module.icon}</span>
                            <div>
                                <h3>${module.name}</h3>
                                <div class="module-meta">
                                    <span class="tradition-tag">${module.tradition}</span>
                                    <span class="level-tag">${module.level}</span>
                                </div>
                            </div>
                        </div>
                        <button class="modal-close" data-close="module-info-modal">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="modal-content">
                        <div class="module-details">
                            <p class="module-description-full">${module.description}</p>
                            
                            ${module.features.length > 0 ? `
                                <div class="features-section">
                                    <h4>Features</h4>
                                    <div class="features-grid">
                                        ${module.features.map(feature => `
                                            <div class="feature-item">
                                                <i class="fas fa-star"></i>
                                                ${this.formatFeature(feature)}
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${module.dependencies.length > 0 ? `
                                <div class="dependencies-section">
                                    <h4>Dependencies</h4>
                                    <div class="dependencies-list">
                                        ${module.dependencies.map(dep => {
                                            const depModule = this.modules.find(m => m.id === dep);
                                            const isActive = this.selectedModules.has(dep);
                                            return `
                                                <div class="dependency-item ${isActive ? 'active' : 'inactive'}">
                                                    <i class="fas ${isActive ? 'fa-check-circle' : 'fa-circle'}"></i>
                                                    ${depModule ? depModule.name : dep}
                                                </div>
                                            `;
                                        }).join('')}
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-primary" onclick="modules.toggleModule('${module.id}'); ui.hideModal('module-info-modal');">
                            ${this.selectedModules.has(module.id) ? 'Deactivate' : 'Activate'} Module
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal
        const existingModal = document.getElementById('module-info-modal');
        if (existingModal) {
            existingModal.remove();
        }

        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        if (window.ui) {
            window.ui.showModal('module-info-modal');
        }
    }

    setActiveFilter(filter) {
        // Update filter buttons
        document.querySelectorAll('.module-filter').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });

        // Re-render modules
        this.renderModules();
    }

    filterModules(searchTerm) {
        this.renderModules();
    }

    updateModuleCount(count) {
        const countElement = document.getElementById('module-count');
        if (countElement) {
            countElement.textContent = `${count} modules`;
        }
    }

    saveModulePreferences() {
        try {
            const preferences = {
                selected: Array.from(this.selectedModules),
                timestamp: Date.now()
            };
            localStorage.setItem(CONFIG.STORAGE_KEYS.MODULE_PREFERENCES, JSON.stringify(preferences));
        } catch (error) {
            console.error('Failed to save module preferences:', error);
        }
    }

    loadModulePreferences() {
        try {
            const saved = localStorage.getItem(CONFIG.STORAGE_KEYS.MODULE_PREFERENCES);
            if (saved) {
                const preferences = JSON.parse(saved);
                this.selectedModules = new Set(preferences.selected || []);
            }
        } catch (error) {
            console.error('Failed to load module preferences:', error);
        }
    }

    // Public API methods
    getSelectedModules() {
        return Array.from(this.selectedModules);
    }

    selectModule(moduleId) {
        if (this.modules.find(m => m.id === moduleId)) {
            this.toggleModule(moduleId);
        }
    }

    deselectModule(moduleId) {
        if (this.selectedModules.has(moduleId)) {
            this.toggleModule(moduleId);
        }
    }

    selectAllModules() {
        this.modules.forEach(module => {
            if (module.enabled && module.status === 'ready') {
                this.selectedModules.add(module.id);
            }
        });
        this.renderModules();
        this.saveModulePreferences();
    }

    deselectAllModules() {
        this.selectedModules.clear();
        this.renderModules();
        this.saveModulePreferences();
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
}

// Initialize modules interface
document.addEventListener('DOMContentLoaded', () => {
    window.modules = new ModulesInterface();
    
    if (CONFIG.DEBUG.ENABLED) {
        console.log('📦 DharmaMind Modules Interface initialized');
    }
});
