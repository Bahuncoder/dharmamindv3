// DharmaMind Search Interface
// ===========================

class SearchInterface {
    constructor() {
        this.searchInput = null;
        this.searchResults = null;
        this.searchSuggestions = null;
        this.currentQuery = '';
        this.searchHistory = [];
        this.recentSearches = [];
        this.isSearching = false;
        this.searchDebounce = null;
        this.searchCache = new Map();
        
        this.init();
    }

    init() {
        this.setupSearchInput();
        this.setupSearchFilters();
        this.setupSearchHistory();
        this.loadSearchHistory();
        this.setupKeyboardShortcuts();
        this.setupVoiceSearch();
    }

    setupSearchInput() {
        this.searchInput = document.getElementById('knowledge-search');
        this.searchResults = document.getElementById('search-results');
        this.searchSuggestions = document.getElementById('search-suggestions');

        if (!this.searchInput) return;

        // Input event listeners
        this.searchInput.addEventListener('input', (e) => {
            this.handleSearchInput(e.target.value);
        });

        this.searchInput.addEventListener('focus', () => {
            this.showSearchInterface();
        });

        this.searchInput.addEventListener('blur', (e) => {
            // Delay hiding to allow clicking on results
            setTimeout(() => {
                if (!e.relatedTarget?.closest('.search-container')) {
                    this.hideSearchInterface();
                }
            }, 150);
        });

        this.searchInput.addEventListener('keydown', (e) => {
            this.handleSearchKeydown(e);
        });

        // Search button
        const searchButton = document.getElementById('search-button');
        if (searchButton) {
            searchButton.addEventListener('click', () => {
                this.performSearch(this.searchInput.value);
            });
        }

        // Clear search button
        const clearButton = document.getElementById('clear-search');
        if (clearButton) {
            clearButton.addEventListener('click', () => {
                this.clearSearch();
            });
        }
    }

    setupSearchFilters() {
        const filterButtons = document.querySelectorAll('.search-filter');
        filterButtons.forEach(button => {
            button.addEventListener('click', () => {
                this.setSearchFilter(button.dataset.filter);
            });
        });

        // Tradition filter
        const traditionSelect = document.getElementById('tradition-filter');
        if (traditionSelect) {
            traditionSelect.addEventListener('change', (e) => {
                this.setTraditionFilter(e.target.value);
            });
        }

        // Content type filter
        const contentTypeSelect = document.getElementById('content-type-filter');
        if (contentTypeSelect) {
            contentTypeSelect.addEventListener('change', (e) => {
                this.setContentTypeFilter(e.target.value);
            });
        }
    }

    setupSearchHistory() {
        const historyContainer = document.getElementById('search-history');
        if (historyContainer) {
            historyContainer.addEventListener('click', (e) => {
                const historyItem = e.target.closest('.history-item');
                if (historyItem) {
                    const query = historyItem.dataset.query;
                    this.performSearch(query);
                }
            });
        }
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K - Focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.focusSearch();
            }

            // Escape - Clear search or hide interface
            if (e.key === 'Escape' && this.searchInput === document.activeElement) {
                if (this.currentQuery) {
                    this.clearSearch();
                } else {
                    this.hideSearchInterface();
                }
            }
        });
    }

    setupVoiceSearch() {
        if (!CONFIG.FEATURES.VOICE_INPUT) return;

        const voiceSearchButton = document.getElementById('voice-search');
        if (voiceSearchButton) {
            voiceSearchButton.addEventListener('click', () => {
                this.startVoiceSearch();
            });
        }
    }

    handleSearchInput(query) {
        this.currentQuery = query.trim();

        // Clear previous debounce
        if (this.searchDebounce) {
            clearTimeout(this.searchDebounce);
        }

        // Update UI
        this.updateSearchUI();

        if (this.currentQuery.length === 0) {
            this.showSearchSuggestions();
            return;
        }

        if (this.currentQuery.length < CONFIG.SEARCH.MIN_QUERY_LENGTH) {
            this.hideSearchResults();
            return;
        }

        // Debounce search
        this.searchDebounce = setTimeout(() => {
            this.performSearch(this.currentQuery);
        }, CONFIG.SEARCH.DEBOUNCE_DELAY);
    }

    handleSearchKeydown(e) {
        switch (e.key) {
            case 'Enter':
                e.preventDefault();
                if (this.currentQuery.length >= CONFIG.SEARCH.MIN_QUERY_LENGTH) {
                    this.performSearch(this.currentQuery);
                }
                break;

            case 'ArrowDown':
                e.preventDefault();
                this.navigateResults('down');
                break;

            case 'ArrowUp':
                e.preventDefault();
                this.navigateResults('up');
                break;

            case 'Escape':
                this.hideSearchInterface();
                break;
        }
    }

    async performSearch(query) {
        if (!query || query.length < CONFIG.SEARCH.MIN_QUERY_LENGTH) {
            return;
        }

        this.isSearching = true;
        this.updateSearchUI();

        try {
            // Check cache first
            const cacheKey = this.getCacheKey(query);
            if (this.searchCache.has(cacheKey)) {
                const cachedResults = this.searchCache.get(cacheKey);
                this.displaySearchResults(cachedResults, query);
                return;
            }

            // Perform API search
            const searchParams = {
                query: query,
                tradition: this.getSelectedTradition(),
                content_type: this.getSelectedContentType(),
                limit: CONFIG.SEARCH.MAX_RESULTS,
                include_sources: true
            };

            const results = await api.searchKnowledge(searchParams);
            
            // Cache results
            this.searchCache.set(cacheKey, results);
            
            // Display results
            this.displaySearchResults(results, query);
            
            // Add to search history
            this.addToSearchHistory(query);

        } catch (error) {
            console.error('Search failed:', error);
            this.displaySearchError(error);
        } finally {
            this.isSearching = false;
            this.updateSearchUI();
        }
    }

    displaySearchResults(results, query) {
        if (!this.searchResults) return;

        this.hideSearchSuggestions();
        this.searchResults.style.display = 'block';

        if (!results.results || results.results.length === 0) {
            this.displayNoResults(query);
            return;
        }

        const resultsHtml = `
            <div class="search-results-header">
                <div class="results-count">
                    Found ${results.results.length} results for "${query}"
                </div>
                <div class="search-time">
                    ${results.search_time ? `(${results.search_time}ms)` : ''}
                </div>
            </div>
            
            <div class="search-results-list">
                ${results.results.map((result, index) => this.createResultItem(result, index)).join('')}
            </div>
            
            ${results.related_queries && results.related_queries.length > 0 ? `
                <div class="related-queries">
                    <h4>Related Searches</h4>
                    <div class="related-queries-list">
                        ${results.related_queries.map(relatedQuery => `
                            <button class="related-query" data-query="${relatedQuery}">
                                ${relatedQuery}
                            </button>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;

        this.searchResults.innerHTML = resultsHtml;

        // Setup result interactions
        this.setupResultInteractions();
    }

    createResultItem(result, index) {
        const relevanceScore = Math.round((result.relevance || 0.5) * 100);
        
        return `
            <div class="search-result-item" data-result-index="${index}">
                <div class="result-header">
                    <div class="result-title">
                        <h4>${this.highlightQuery(result.title, this.currentQuery)}</h4>
                        <div class="result-meta">
                            <span class="result-tradition">${result.tradition || 'Universal'}</span>
                            <span class="result-type">${result.content_type || 'Teaching'}</span>
                            <span class="result-relevance">${relevanceScore}% match</span>
                        </div>
                    </div>
                    <div class="result-actions">
                        <button class="result-action" data-action="view" title="View full content">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="result-action" data-action="ask" title="Ask about this">
                            <i class="fas fa-comment"></i>
                        </button>
                        <button class="result-action" data-action="save" title="Save for later">
                            <i class="far fa-bookmark"></i>
                        </button>
                    </div>
                </div>
                
                <div class="result-content">
                    <p class="result-excerpt">
                        ${this.highlightQuery(result.content || result.excerpt, this.currentQuery)}
                    </p>
                    
                    ${result.source ? `
                        <div class="result-source">
                            <i class="fas fa-book"></i>
                            <span>${result.source}</span>
                        </div>
                    ` : ''}
                    
                    ${result.tags && result.tags.length > 0 ? `
                        <div class="result-tags">
                            ${result.tags.map(tag => `<span class="result-tag">${tag}</span>`).join('')}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    displayNoResults(query) {
        this.searchResults.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">🔍</div>
                <h3>No results found</h3>
                <p>We couldn't find any content matching "${query}"</p>
                
                <div class="search-suggestions">
                    <h4>Try:</h4>
                    <ul>
                        <li>Using different keywords</li>
                        <li>Checking your spelling</li>
                        <li>Using more general terms</li>
                        <li>Exploring different spiritual traditions</li>
                    </ul>
                </div>
                
                <div class="popular-topics">
                    <h4>Popular Topics</h4>
                    <div class="topic-buttons">
                        ${CONFIG.SEARCH.POPULAR_TOPICS.map(topic => `
                            <button class="topic-button" data-query="${topic}">${topic}</button>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;

        this.setupNoResultsInteractions();
    }

    displaySearchError(error) {
        if (!this.searchResults) return;

        const errorMessage = error instanceof APIError ? 
            error.getUserFriendlyMessage() : 
            'An error occurred while searching. Please try again.';

        this.searchResults.innerHTML = `
            <div class="search-error">
                <div class="error-icon">⚠️</div>
                <h3>Search Error</h3>
                <p>${errorMessage}</p>
                <button class="retry-search" onclick="search.performSearch('${this.currentQuery}')">
                    <i class="fas fa-redo"></i> Retry Search
                </button>
            </div>
        `;
    }

    setupResultInteractions() {
        if (!this.searchResults) return;

        // Result actions
        this.searchResults.addEventListener('click', (e) => {
            const action = e.target.closest('.result-action');
            if (action) {
                e.preventDefault();
                e.stopPropagation();
                
                const resultItem = action.closest('.search-result-item');
                const resultIndex = parseInt(resultItem.dataset.resultIndex);
                const actionType = action.dataset.action;
                
                this.handleResultAction(actionType, resultIndex);
                return;
            }

            // Related queries
            const relatedQuery = e.target.closest('.related-query');
            if (relatedQuery) {
                const query = relatedQuery.dataset.query;
                this.performSearch(query);
                return;
            }

            // Result item click
            const resultItem = e.target.closest('.search-result-item');
            if (resultItem) {
                const resultIndex = parseInt(resultItem.dataset.resultIndex);
                this.handleResultAction('view', resultIndex);
            }
        });
    }

    setupNoResultsInteractions() {
        if (!this.searchResults) return;

        this.searchResults.addEventListener('click', (e) => {
            const topicButton = e.target.closest('.topic-button');
            if (topicButton) {
                const query = topicButton.dataset.query;
                this.performSearch(query);
            }
        });
    }

    handleResultAction(action, resultIndex) {
        // Implementation would depend on the specific result data structure
        // This is a placeholder for the actual implementation
        
        switch (action) {
            case 'view':
                this.viewResult(resultIndex);
                break;
            case 'ask':
                this.askAboutResult(resultIndex);
                break;
            case 'save':
                this.saveResult(resultIndex);
                break;
        }
    }

    viewResult(resultIndex) {
        // Show result in a modal or navigate to detailed view
        if (window.notifications) {
            window.notifications.show('info', 'View Result', 'Opening detailed view...');
        }
    }

    askAboutResult(resultIndex) {
        // Add result context to chat and focus message input
        const messageInput = document.getElementById('message-input');
        if (messageInput && window.chat) {
            messageInput.value = `Tell me more about: ${this.currentQuery}`;
            messageInput.focus();
            this.hideSearchInterface();
            
            if (window.notifications) {
                window.notifications.show('success', 'Question Added', 'Question added to chat input');
            }
        }
    }

    saveResult(resultIndex) {
        // Save result to user's favorites/bookmarks
        if (window.notifications) {
            window.notifications.show('success', 'Saved', 'Result saved to your bookmarks');
        }
    }

    showSearchInterface() {
        const searchContainer = document.querySelector('.search-container');
        if (searchContainer) {
            searchContainer.classList.add('focused');
        }

        if (this.currentQuery.length === 0) {
            this.showSearchSuggestions();
        }
    }

    hideSearchInterface() {
        const searchContainer = document.querySelector('.search-container');
        if (searchContainer) {
            searchContainer.classList.remove('focused');
        }

        this.hideSearchResults();
        this.hideSearchSuggestions();
    }

    showSearchSuggestions() {
        if (!this.searchSuggestions) return;

        const suggestionsHtml = `
            <div class="search-suggestions-content">
                ${this.recentSearches.length > 0 ? `
                    <div class="recent-searches">
                        <h4><i class="fas fa-history"></i> Recent Searches</h4>
                        <div class="recent-searches-list">
                            ${this.recentSearches.slice(0, 5).map(search => `
                                <button class="recent-search-item" data-query="${search}">
                                    <i class="fas fa-search"></i>
                                    <span>${search}</span>
                                </button>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                <div class="popular-searches">
                    <h4><i class="fas fa-fire"></i> Popular Searches</h4>
                    <div class="popular-searches-list">
                        ${CONFIG.SEARCH.POPULAR_TOPICS.map(topic => `
                            <button class="popular-search-item" data-query="${topic}">
                                <i class="fas fa-star"></i>
                                <span>${topic}</span>
                            </button>
                        `).join('')}
                    </div>
                </div>
                
                <div class="search-tips">
                    <h4><i class="fas fa-lightbulb"></i> Search Tips</h4>
                    <ul>
                        <li>Use quotes for exact phrases: "four noble truths"</li>
                        <li>Filter by tradition or content type</li>
                        <li>Try different spellings or synonyms</li>
                        <li>Use voice search for natural queries</li>
                    </ul>
                </div>
            </div>
        `;

        this.searchSuggestions.innerHTML = suggestionsHtml;
        this.searchSuggestions.style.display = 'block';

        // Setup suggestion interactions
        this.searchSuggestions.addEventListener('click', (e) => {
            const suggestionItem = e.target.closest('[data-query]');
            if (suggestionItem) {
                const query = suggestionItem.dataset.query;
                this.searchInput.value = query;
                this.performSearch(query);
            }
        });
    }

    hideSearchSuggestions() {
        if (this.searchSuggestions) {
            this.searchSuggestions.style.display = 'none';
        }
    }

    hideSearchResults() {
        if (this.searchResults) {
            this.searchResults.style.display = 'none';
        }
    }

    updateSearchUI() {
        // Update clear button
        const clearButton = document.getElementById('clear-search');
        if (clearButton) {
            clearButton.style.display = this.currentQuery.length > 0 ? 'block' : 'none';
        }

        // Update search button
        const searchButton = document.getElementById('search-button');
        if (searchButton) {
            searchButton.disabled = this.isSearching || this.currentQuery.length < CONFIG.SEARCH.MIN_QUERY_LENGTH;
            
            const icon = searchButton.querySelector('i');
            if (icon) {
                icon.className = this.isSearching ? 'fas fa-spinner fa-spin' : 'fas fa-search';
            }
        }
    }

    clearSearch() {
        this.currentQuery = '';
        this.searchInput.value = '';
        this.hideSearchResults();
        this.updateSearchUI();
        this.showSearchSuggestions();
    }

    focusSearch() {
        if (this.searchInput) {
            this.searchInput.focus();
            this.searchInput.select();
        }
    }

    navigateResults(direction) {
        // Implement keyboard navigation through search results
        const resultItems = this.searchResults?.querySelectorAll('.search-result-item');
        if (!resultItems || resultItems.length === 0) return;

        // This would need more implementation for proper navigation
    }

    startVoiceSearch() {
        if (!CONFIG.FEATURES.VOICE_INPUT || !window.SpeechRecognition && !window.webkitSpeechRecognition) {
            if (window.notifications) {
                window.notifications.show('error', 'Voice Search', 'Voice search is not supported in this browser');
            }
            return;
        }

        // Voice search implementation would go here
        if (window.notifications) {
            window.notifications.show('info', 'Voice Search', 'Voice search feature coming soon!');
        }
    }

    // Helper methods
    highlightQuery(text, query) {
        if (!text || !query) return text;
        
        const regex = new RegExp(`(${query})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }

    getCacheKey(query) {
        const tradition = this.getSelectedTradition();
        const contentType = this.getSelectedContentType();
        return `${query}|${tradition}|${contentType}`;
    }

    getSelectedTradition() {
        const traditionSelect = document.getElementById('tradition-filter');
        return traditionSelect?.value || 'all';
    }

    getSelectedContentType() {
        const contentTypeSelect = document.getElementById('content-type-filter');
        return contentTypeSelect?.value || 'all';
    }

    setSearchFilter(filter) {
        document.querySelectorAll('.search-filter').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });
        
        // Re-perform search if there's a query
        if (this.currentQuery.length >= CONFIG.SEARCH.MIN_QUERY_LENGTH) {
            this.performSearch(this.currentQuery);
        }
    }

    setTraditionFilter(tradition) {
        if (this.currentQuery.length >= CONFIG.SEARCH.MIN_QUERY_LENGTH) {
            this.performSearch(this.currentQuery);
        }
    }

    setContentTypeFilter(contentType) {
        if (this.currentQuery.length >= CONFIG.SEARCH.MIN_QUERY_LENGTH) {
            this.performSearch(this.currentQuery);
        }
    }

    addToSearchHistory(query) {
        // Remove if already exists
        this.recentSearches = this.recentSearches.filter(search => search !== query);
        
        // Add to beginning
        this.recentSearches.unshift(query);
        
        // Limit size
        this.recentSearches = this.recentSearches.slice(0, CONFIG.SEARCH.HISTORY_LIMIT);
        
        // Save to localStorage
        this.saveSearchHistory();
    }

    saveSearchHistory() {
        try {
            localStorage.setItem(CONFIG.STORAGE_KEYS.SEARCH_HISTORY, JSON.stringify(this.recentSearches));
        } catch (error) {
            console.error('Failed to save search history:', error);
        }
    }

    loadSearchHistory() {
        try {
            const saved = localStorage.getItem(CONFIG.STORAGE_KEYS.SEARCH_HISTORY);
            if (saved) {
                this.recentSearches = JSON.parse(saved);
            }
        } catch (error) {
            console.error('Failed to load search history:', error);
            this.recentSearches = [];
        }
    }
}

// Initialize search interface
document.addEventListener('DOMContentLoaded', () => {
    window.search = new SearchInterface();
    
    if (CONFIG.DEBUG.ENABLED) {
        console.log('🔍 DharmaMind Search Interface initialized');
    }
});
