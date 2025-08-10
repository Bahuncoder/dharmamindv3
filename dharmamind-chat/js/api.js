// DharmaMind API Interface
// ========================

class DharmaMindAPI {
    constructor() {
        this.baseURL = CONFIG.API.BASE_URL;
        this.timeout = CONFIG.API.TIMEOUT;
        this.retryAttempts = CONFIG.API.RETRY_ATTEMPTS;
        this.retryDelay = CONFIG.API.RETRY_DELAY;
        
        // Request interceptors
        this.requestInterceptors = [];
        this.responseInterceptors = [];
        
        // Active requests for cancellation
        this.activeRequests = new Map();
        
        this.setupHealthCheck();
    }

    // Health Check System
    setupHealthCheck() {
        this.isOnline = true;
        this.lastHealthCheck = Date.now();
        
        setInterval(() => {
            this.checkSystemHealth();
        }, CONFIG.SYSTEM.HEALTH_CHECK_INTERVAL);
    }

    async checkSystemHealth() {
        try {
            const response = await this.request('GET', CONFIG.API.ENDPOINTS.HEALTH, null, {
                timeout: 5000,
                skipRetry: true
            });
            
            if (response.status === 'healthy') {
                if (!this.isOnline) {
                    this.isOnline = true;
                    this.notifyConnectionRestored();
                }
                this.lastHealthCheck = Date.now();
                return true;
            }
        } catch (error) {
            if (this.isOnline) {
                this.isOnline = false;
                this.notifyConnectionLost();
            }
            return false;
        }
    }

    notifyConnectionRestored() {
        if (window.notifications) {
            window.notifications.show('success', 'Connection Restored', CONFIG.MESSAGES.CONNECTION_RESTORED);
        }
        this.updateSystemStatus();
    }

    notifyConnectionLost() {
        if (window.notifications) {
            window.notifications.show('error', 'Connection Lost', CONFIG.ERRORS.NETWORK);
        }
        this.updateSystemStatus();
    }

    updateSystemStatus() {
        const statusBar = document.getElementById('status-bar');
        if (statusBar) {
            const indicators = statusBar.querySelectorAll('.status-item i');
            indicators.forEach(indicator => {
                indicator.className = this.isOnline ? 'fas fa-circle status-green' : 'fas fa-circle status-red';
            });
        }
    }

    // Core Request Method
    async request(method, endpoint, data = null, options = {}) {
        const requestId = this.generateRequestId();
        const url = `${this.baseURL}${endpoint}`;
        
        const config = {
            method: method.toUpperCase(),
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                ...options.headers
            },
            signal: options.signal,
            ...options
        };

        if (data && ['POST', 'PUT', 'PATCH'].includes(config.method)) {
            config.body = JSON.stringify(data);
        }

        // Add request interceptors
        for (const interceptor of this.requestInterceptors) {
            await interceptor(config);
        }

        const controller = new AbortController();
        config.signal = controller.signal;
        
        // Store active request
        this.activeRequests.set(requestId, controller);

        // Set timeout
        const timeoutId = setTimeout(() => {
            controller.abort();
        }, options.timeout || this.timeout);

        try {
            if (CONFIG.DEBUG.LOG_API_CALLS) {
                console.log(`🌐 API Request: ${method} ${endpoint}`, data);
            }

            let response;
            let retryCount = 0;
            const maxRetries = options.skipRetry ? 0 : this.retryAttempts;

            while (retryCount <= maxRetries) {
                try {
                    response = await fetch(url, config);
                    break;
                } catch (error) {
                    if (retryCount < maxRetries && this.shouldRetry(error)) {
                        retryCount++;
                        await this.delay(this.retryDelay * retryCount);
                        continue;
                    }
                    throw error;
                }
            }

            clearTimeout(timeoutId);
            this.activeRequests.delete(requestId);

            if (!response.ok) {
                throw new APIError(response.status, response.statusText, await response.text());
            }

            const result = await response.json();

            // Add response interceptors
            for (const interceptor of this.responseInterceptors) {
                await interceptor(result);
            }

            if (CONFIG.DEBUG.LOG_API_CALLS) {
                console.log(`✅ API Response: ${method} ${endpoint}`, result);
            }

            return result;

        } catch (error) {
            clearTimeout(timeoutId);
            this.activeRequests.delete(requestId);

            if (CONFIG.DEBUG.LOG_API_CALLS) {
                console.error(`❌ API Error: ${method} ${endpoint}`, error);
            }

            throw this.handleError(error);
        }
    }

    shouldRetry(error) {
        // Retry on network errors but not on 4xx client errors
        return !error.status || (error.status >= 500 && error.status < 600);
    }

    handleError(error) {
        if (error.name === 'AbortError') {
            return new APIError(408, 'Timeout', CONFIG.ERRORS.TIMEOUT);
        }
        
        if (error instanceof APIError) {
            return error;
        }

        if (!navigator.onLine) {
            return new APIError(0, 'Network Error', CONFIG.ERRORS.NETWORK);
        }

        return new APIError(500, 'Unknown Error', CONFIG.ERRORS.GENERIC);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    generateRequestId() {
        return Math.random().toString(36).substr(2, 9);
    }

    // Cancel all active requests
    cancelAllRequests() {
        for (const [id, controller] of this.activeRequests) {
            controller.abort();
        }
        this.activeRequests.clear();
    }

    // Chat API Methods
    async sendMessage(message, options = {}) {
        const payload = {
            message: message.trim(),
            context: {
                tradition: options.tradition || localStorage.getItem(CONFIG.UI.TRADITION.STORAGE_KEY) || CONFIG.UI.TRADITION.DEFAULT,
                spiritual_level: options.spiritual_level || localStorage.getItem(CONFIG.UI.SPIRITUAL_LEVEL.STORAGE_KEY) || CONFIG.UI.SPIRITUAL_LEVEL.DEFAULT,
                language: options.language || localStorage.getItem(CONFIG.UI.LANGUAGE.STORAGE_KEY) || CONFIG.UI.LANGUAGE.DEFAULT,
                response_style: options.response_style || CONFIG.SPIRITUAL.DEFAULT_RESPONSE_STYLE,
                include_sources: options.include_sources !== false,
                tradition_neutral: options.tradition_neutral !== false,
                session_id: options.session_id || this.getSessionId(),
                ...options.context
            },
            user_feedback: options.user_feedback || null
        };

        const response = await this.request('POST', CONFIG.API.ENDPOINTS.CHAT, payload, {
            timeout: CONFIG.CHAT.RESPONSE_TIMEOUT
        });

        // Handle the new continuous learning response format
        if (response.success && response.data) {
            const data = response.data;
            return {
                message: data.final_response || data.advanced_llm_response || "I apologize, but I couldn't generate a response.",
                dharmallm_response: data.dharmallm_response,
                advanced_llm_response: data.advanced_llm_response,
                spiritual_enhancement: data.spiritual_enhancement,
                modules_activated: data.modules_activated || [],
                learning_applied: data.learning_applied || false,
                timestamp: data.timestamp,
                metadata: {
                    response_time: Date.now() - this.requestStartTime,
                    user_message: data.user_message,
                    continuous_learning: true
                }
            };
        } else {
            throw new APIError(500, 'Invalid response format', response);
        }
    }

    async sendChatStream(message, options = {}, onChunk = null) {
        const url = `${this.baseURL}${CONFIG.API.ENDPOINTS.CHAT}/stream`;
        
        const payload = {
            message: message.trim(),
            tradition: options.tradition || localStorage.getItem(CONFIG.UI.TRADITION.STORAGE_KEY) || CONFIG.UI.TRADITION.DEFAULT,
            spiritual_level: options.spiritual_level || localStorage.getItem(CONFIG.UI.SPIRITUAL_LEVEL.STORAGE_KEY) || CONFIG.UI.SPIRITUAL_LEVEL.DEFAULT,
            language: options.language || localStorage.getItem(CONFIG.UI.LANGUAGE.STORAGE_KEY) || CONFIG.UI.LANGUAGE.DEFAULT,
            response_style: options.response_style || CONFIG.SPIRITUAL.DEFAULT_RESPONSE_STYLE,
            stream: true,
            ...options.additional_params
        };

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/stream'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new APIError(response.status, response.statusText, await response.text());
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') {
                            return;
                        }
                        
                        try {
                            const chunk = JSON.parse(data);
                            if (onChunk) {
                                onChunk(chunk);
                            }
                        } catch (error) {
                            console.error('Error parsing SSE chunk:', error);
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }

    // Knowledge Search API
    async searchKnowledge(query, options = {}) {
        const payload = {
            query: query.trim(),
            limit: options.limit || CONFIG.KNOWLEDGE.MAX_RESULTS,
            tradition: options.tradition,
            spiritual_level: options.spiritual_level,
            similarity_threshold: options.similarity_threshold || CONFIG.KNOWLEDGE.SIMILARITY_THRESHOLD,
            ...options.filters
        };

        return await this.request('POST', CONFIG.API.ENDPOINTS.KNOWLEDGE_SEARCH, payload);
    }

    // Modules API
    async getModules(options = {}) {
        const params = new URLSearchParams();
        
        if (options.category) params.append('category', options.category);
        if (options.tradition) params.append('tradition', options.tradition);
        if (options.spiritual_level) params.append('spiritual_level', options.spiritual_level);
        
        const url = CONFIG.API.ENDPOINTS.MODULES + (params.toString() ? `?${params.toString()}` : '');
        
        return await this.request('GET', url);
    }

    async getModuleDetails(moduleId) {
        return await this.request('GET', `${CONFIG.API.ENDPOINTS.MODULES}/${moduleId}`);
    }

    // System Status API
    async getSystemStatus() {
        return await this.request('GET', CONFIG.API.ENDPOINTS.SYSTEM_STATUS);
    }

    // Feedback API
    async sendFeedback(feedback) {
        const payload = {
            type: feedback.type || 'general',
            rating: feedback.rating,
            message: feedback.message,
            context: feedback.context || {},
            timestamp: new Date().toISOString(),
            session_id: this.getSessionId()
        };

        return await this.request('POST', CONFIG.API.ENDPOINTS.FEEDBACK, payload);
    }

    // Session Management
    getSessionId() {
        let sessionId = localStorage.getItem('dharmamind-session-id');
        if (!sessionId) {
            sessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('dharmamind-session-id', sessionId);
        }
        return sessionId;
    }

    // Interceptor Management
    addRequestInterceptor(interceptor) {
        this.requestInterceptors.push(interceptor);
    }

    addResponseInterceptor(interceptor) {
        this.responseInterceptors.push(interceptor);
    }

    removeRequestInterceptor(interceptor) {
        const index = this.requestInterceptors.indexOf(interceptor);
        if (index > -1) {
            this.requestInterceptors.splice(index, 1);
        }
    }

    removeResponseInterceptor(interceptor) {
        const index = this.responseInterceptors.indexOf(interceptor);
        if (index > -1) {
            this.responseInterceptors.splice(index, 1);
        }
    }
}

// Custom API Error Class
class APIError extends Error {
    constructor(status, statusText, message) {
        super(message || statusText);
        this.name = 'APIError';
        this.status = status;
        this.statusText = statusText;
    }

    getUserFriendlyMessage() {
        switch (this.status) {
            case 0:
                return CONFIG.ERRORS.NETWORK;
            case 408:
            case 504:
                return CONFIG.ERRORS.TIMEOUT;
            case 429:
                return CONFIG.ERRORS.RATE_LIMIT;
            case 400:
            case 422:
                return CONFIG.ERRORS.VALIDATION;
            case 401:
            case 403:
                return CONFIG.ERRORS.PERMISSION;
            case 500:
            case 502:
            case 503:
                return CONFIG.ERRORS.SERVER;
            default:
                return this.message || CONFIG.ERRORS.GENERIC;
        }
    }
}

// Create global API instance
const api = new DharmaMindAPI();

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DharmaMindAPI, APIError, api };
} else {
    window.api = api;
    window.APIError = APIError;
}

// Debug information
if (CONFIG.DEBUG.ENABLED) {
    console.log('🌐 DharmaMind API Interface initialized');
    
    // Add logging interceptors in debug mode
    api.addRequestInterceptor(async (config) => {
        if (CONFIG.DEBUG.LOG_API_CALLS) {
            console.log('📤 API Request:', config);
        }
    });
    
    api.addResponseInterceptor(async (response) => {
        if (CONFIG.DEBUG.LOG_API_CALLS) {
            console.log('📥 API Response:', response);
        }
    });
}
