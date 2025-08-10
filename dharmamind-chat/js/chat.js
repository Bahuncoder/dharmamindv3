// DharmaMind Chat Interface
// =========================

class ChatInterface {
    constructor() {
        this.messagesContainer = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-btn');
        this.voiceButton = document.getElementById('voice-btn');
        this.welcomeSection = document.getElementById('welcome-section');
        this.charCount = document.getElementById('char-count');
        
        this.messages = [];
        this.isTyping = false;
        this.currentStreamingMessage = null;
        this.voiceRecognition = null;
        this.isRecording = false;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupVoiceRecognition();
        this.loadChatHistory();
        this.updateCharacterCount();
        this.setupWelcomeSuggestions();
    }

    setupEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Send message on Enter (Shift+Enter for new line)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea();
            this.updateCharacterCount();
            this.updateSendButton();
        });

        // Voice recording
        this.voiceButton.addEventListener('click', () => this.toggleVoiceRecording());

        // Clear chat
        const clearBtn = document.getElementById('clear-chat');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearChat());
        }

        // Export chat
        const exportBtn = document.getElementById('export-chat');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportChat());
        }

        // Share chat
        const shareBtn = document.getElementById('share-chat');
        if (shareBtn) {
            shareBtn.addEventListener('click', () => this.shareWisdom());
        }
    }

    setupWelcomeSuggestions() {
        const suggestionCards = document.querySelectorAll('.suggestion-card');
        suggestionCards.forEach(card => {
            card.addEventListener('click', () => {
                const suggestion = card.dataset.suggestion;
                if (suggestion) {
                    this.messageInput.value = suggestion;
                    this.updateCharacterCount();
                    this.sendMessage();
                }
            });
        });
    }

    setupVoiceRecognition() {
        if (!CONFIG.FEATURES.VOICE_INPUT) {
            this.voiceButton.style.display = 'none';
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (SpeechRecognition) {
            this.voiceRecognition = new SpeechRecognition();
            this.voiceRecognition.continuous = false;
            this.voiceRecognition.interimResults = true;
            this.voiceRecognition.lang = localStorage.getItem(CONFIG.UI.LANGUAGE.STORAGE_KEY) || 'en-US';

            this.voiceRecognition.onstart = () => {
                this.isRecording = true;
                this.voiceButton.classList.add('recording');
                this.voiceButton.innerHTML = '<i class="fas fa-stop"></i>';
            };

            this.voiceRecognition.onresult = (event) => {
                let finalTranscript = '';
                let interimTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript;
                    } else {
                        interimTranscript += transcript;
                    }
                }

                this.messageInput.value = finalTranscript + interimTranscript;
                this.updateCharacterCount();
            };

            this.voiceRecognition.onend = () => {
                this.isRecording = false;
                this.voiceButton.classList.remove('recording');
                this.voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
            };

            this.voiceRecognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.isRecording = false;
                this.voiceButton.classList.remove('recording');
                this.voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
                
                if (window.notifications) {
                    window.notifications.show('error', 'Voice Input Error', 'Unable to access microphone. Please check permissions.');
                }
            };
        }
    }

    toggleVoiceRecording() {
        if (!this.voiceRecognition) return;

        if (this.isRecording) {
            this.voiceRecognition.stop();
        } else {
            this.voiceRecognition.start();
        }
    }

    autoResizeTextarea() {
        const textarea = this.messageInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    updateCharacterCount() {
        const currentLength = this.messageInput.value.length;
        const maxLength = CONFIG.CHAT.MAX_MESSAGE_LENGTH;
        
        this.charCount.textContent = currentLength;
        
        if (currentLength > maxLength * 0.9) {
            this.charCount.parentElement.classList.add('warning');
        } else {
            this.charCount.parentElement.classList.remove('warning');
        }
        
        if (currentLength > maxLength) {
            this.charCount.parentElement.classList.add('error');
        } else {
            this.charCount.parentElement.classList.remove('error');
        }
    }

    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        const isValid = this.messageInput.value.length <= CONFIG.CHAT.MAX_MESSAGE_LENGTH;
        
        this.sendButton.disabled = !hasText || !isValid || this.isTyping;
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message || message.length > CONFIG.CHAT.MAX_MESSAGE_LENGTH || this.isTyping) {
            return;
        }

        // Hide welcome section
        if (this.welcomeSection && !this.welcomeSection.classList.contains('hidden')) {
            this.welcomeSection.classList.add('hidden');
        }

        // Add user message
        this.addMessage('user', message);
        
        // Clear input
        this.messageInput.value = '';
        this.updateCharacterCount();
        this.updateSendButton();
        this.autoResizeTextarea();

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Get user preferences
            const preferences = this.getUserPreferences();
            
            // Send message via API
            if (CONFIG.CHAT.STREAMING_ENABLED) {
                await this.sendStreamingMessage(message, preferences);
            } else {
                await this.sendRegularMessage(message, preferences);
            }

        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            
            const errorMessage = error instanceof APIError ? 
                error.getUserFriendlyMessage() : 
                CONFIG.ERRORS.GENERIC;
                
            this.addMessage('assistant', `I apologize, but I encountered an error: ${errorMessage}. Please try again.`, {
                isError: true
            });
            
            if (window.notifications) {
                window.notifications.show('error', 'Chat Error', errorMessage);
            }
        }

        // Save chat history
        this.saveChatHistory();
    }

    async sendRegularMessage(message, preferences) {
        const response = await api.sendMessage(message, preferences);
        
        this.hideTypingIndicator();
        
        // Enhanced response with continuous learning information
        this.addMessage('assistant', response.message, {
            dharmallm_response: response.dharmallm_response,
            advanced_llm_response: response.advanced_llm_response,
            spiritual_enhancement: response.spiritual_enhancement,
            modules_activated: response.modules_activated,
            learning_applied: response.learning_applied,
            metadata: response.metadata,
            timestamp: response.timestamp
        });
    }

    async sendStreamingMessage(message, preferences) {
        let assistantMessage = null;
        let streamedContent = '';
        
        await api.sendChatStream(message, preferences, (chunk) => {
            if (chunk.type === 'start') {
                this.hideTypingIndicator();
                assistantMessage = this.addMessage('assistant', '', {
                    isStreaming: true
                });
            } else if (chunk.type === 'content') {
                streamedContent += chunk.content;
                if (assistantMessage) {
                    this.updateMessageContent(assistantMessage, streamedContent);
                }
            } else if (chunk.type === 'metadata') {
                if (assistantMessage) {
                    this.updateMessageMetadata(assistantMessage, {
                        sources: chunk.sources,
                        modules: chunk.modules_used,
                        metadata: chunk.metadata
                    });
                }
            } else if (chunk.type === 'done') {
                if (assistantMessage) {
                    assistantMessage.classList.remove('streaming');
                }
                this.currentStreamingMessage = null;
            }
        });
    }

    getUserPreferences() {
        return {
            tradition: localStorage.getItem(CONFIG.UI.TRADITION.STORAGE_KEY) || CONFIG.UI.TRADITION.DEFAULT,
            spiritual_level: localStorage.getItem(CONFIG.UI.SPIRITUAL_LEVEL.STORAGE_KEY) || CONFIG.UI.SPIRITUAL_LEVEL.DEFAULT,
            language: localStorage.getItem(CONFIG.UI.LANGUAGE.STORAGE_KEY) || CONFIG.UI.LANGUAGE.DEFAULT,
            response_style: localStorage.getItem('dharmamind-response-style') || CONFIG.SPIRITUAL.DEFAULT_RESPONSE_STYLE,
            include_sources: localStorage.getItem('dharmamind-include-sources') !== 'false',
            tradition_neutral: localStorage.getItem('dharmamind-tradition-neutral') !== 'false'
        };
    }

    addMessage(role, content, options = {}) {
        const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.dataset.messageId = messageId;
        
        if (options.isStreaming) {
            messageDiv.classList.add('streaming');
            this.currentStreamingMessage = messageDiv;
        }

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? '👤' : '🙏';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        
        if (options.isError) {
            textDiv.innerHTML = `<p style="color: var(--error-color);">${content}</p>`;
        } else {
            textDiv.innerHTML = this.formatMessageContent(content);
        }

        bubble.appendChild(textDiv);

        // Add sources if provided
        if (options.sources && options.sources.length > 0) {
            const sourcesDiv = this.createSourcesElement(options.sources);
            bubble.appendChild(sourcesDiv);
        }

        // Add modules if provided (updated for continuous learning)
        if (options.modules_activated && options.modules_activated.length > 0) {
            const modulesDiv = this.createModulesElement(options.modules_activated);
            bubble.appendChild(modulesDiv);
        }

        // Add continuous learning info
        if (role === 'assistant' && options.learning_applied) {
            const learningDiv = this.createLearningInfoElement(options);
            bubble.appendChild(learningDiv);
        }

        // Add Multi-LLM model info if provided
        if (role === 'assistant' && (options.advanced_llm_response || options.dharmallm_response)) {
            const modelDiv = this.createEnhancedModelInfoElement(options);
            bubble.appendChild(modelDiv);
        }

        // Add message metadata
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = new Date().toLocaleTimeString();
        
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        
        if (role === 'assistant') {
            // Copy button
            const copyBtn = document.createElement('button');
            copyBtn.className = 'message-action';
            copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
            copyBtn.title = 'Copy message';
            copyBtn.addEventListener('click', () => this.copyMessage(content));
            
            // Like button
            const likeBtn = document.createElement('button');
            likeBtn.className = 'message-action';
            likeBtn.innerHTML = '<i class="far fa-heart"></i>';
            likeBtn.title = 'Like this response';
            likeBtn.addEventListener('click', () => this.likeMessage(messageId, content));
            
            actionsDiv.appendChild(copyBtn);
            actionsDiv.appendChild(likeBtn);
        }
        
        metaDiv.appendChild(timeSpan);
        metaDiv.appendChild(actionsDiv);
        bubble.appendChild(metaDiv);

        messageContent.appendChild(bubble);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        // Store message
        this.messages.push({
            id: messageId,
            role,
            content,
            timestamp: Date.now(),
            ...options
        });

        return messageDiv;
    }

    updateMessageContent(messageElement, content) {
        const textDiv = messageElement.querySelector('.message-text');
        if (textDiv) {
            textDiv.innerHTML = this.formatMessageContent(content);
            this.scrollToBottom();
        }
    }

    updateMessageMetadata(messageElement, metadata) {
        const bubble = messageElement.querySelector('.message-bubble');
        if (!bubble) return;

        // Add sources
        if (metadata.sources && metadata.sources.length > 0) {
            const existingSources = bubble.querySelector('.message-sources');
            if (existingSources) {
                existingSources.remove();
            }
            const sourcesDiv = this.createSourcesElement(metadata.sources);
            bubble.insertBefore(sourcesDiv, bubble.querySelector('.message-meta'));
        }

        // Add modules
        if (metadata.modules && metadata.modules.length > 0) {
            const existingModules = bubble.querySelector('.message-modules');
            if (existingModules) {
                existingModules.remove();
            }
            const modulesDiv = this.createModulesElement(metadata.modules);
            bubble.insertBefore(modulesDiv, bubble.querySelector('.message-meta'));
        }
    }

    createSourcesElement(sources) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        
        const title = document.createElement('div');
        title.className = 'sources-title';
        title.innerHTML = '<i class="fas fa-book"></i> Sources & References';
        sourcesDiv.appendChild(title);

        sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            const sourceTitle = document.createElement('div');
            sourceTitle.className = 'source-title';
            sourceTitle.textContent = source.title || source.source;
            
            const sourceTradition = document.createElement('div');
            sourceTradition.className = 'source-tradition';
            sourceTradition.textContent = source.tradition || 'universal';
            
            sourceItem.appendChild(sourceTitle);
            sourceItem.appendChild(sourceTradition);
            sourcesDiv.appendChild(sourceItem);
        });

        return sourcesDiv;
    }

    createModulesElement(modules) {
        const modulesDiv = document.createElement('div');
        modulesDiv.className = 'message-modules';

        modules.forEach(module => {
            const moduleTag = document.createElement('span');
            moduleTag.className = 'module-tag';
            moduleTag.textContent = module.name || module;
            modulesDiv.appendChild(moduleTag);
        });

        return modulesDiv;
    }

    createModelInfoElement(options) {
        const modelDiv = document.createElement('div');
        modelDiv.className = 'message-model-info';

        const model_used = options.model_used || options.metadata?.model_used || 'dharmamind';
        const wisdom_level = options.wisdom_level || options.metadata?.wisdom_level || 0.5;
        const confidence = options.confidence || options.metadata?.confidence || 0.8;

        // Model badge
        const modelBadge = document.createElement('span');
        modelBadge.className = `model-badge model-${model_used.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase()}`;
        
        // Map model names to display names and emojis
        const modelInfo = {
            'gpt-4o': { name: 'GPT-4o', emoji: '🧠' },
            'gpt-4o-mini': { name: 'GPT-4o Mini', emoji: '⚡' },
            'claude-3-opus': { name: 'Claude Opus', emoji: '🎭' },
            'claude-3-sonnet': { name: 'Claude Sonnet', emoji: '🎪' },
            'gemini-pro': { name: 'Gemini Pro', emoji: '💎' },
            'gemini-pro-vision': { name: 'Gemini Vision', emoji: '👁️' },
            'dharmic_fallback': { name: 'DharmaMind', emoji: '🕉️' },
            'dharmamind-v1.0': { name: 'DharmaMind', emoji: '🕉️' }
        };

        const info = modelInfo[model_used] || { name: model_used, emoji: '🤖' };
        modelBadge.innerHTML = `${info.emoji} ${info.name}`;

        // Wisdom level indicator
        const wisdomDiv = document.createElement('div');
        wisdomDiv.className = 'wisdom-indicator';
        const wisdomLevel = Math.round(wisdom_level * 100);
        wisdomDiv.innerHTML = `<span class="wisdom-label">Wisdom:</span> <span class="wisdom-value">${wisdomLevel}%</span>`;

        modelDiv.appendChild(modelBadge);
        modelDiv.appendChild(wisdomDiv);

        return modelDiv;
    }

    formatMessageContent(content) {
        if (!content) return '';
        
        // Basic markdown-like formatting
        let formatted = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');

        // Wrap in paragraph tags
        if (!formatted.startsWith('<p>')) {
            formatted = '<p>' + formatted + '</p>';
        }

        return formatted;
    }

    showTypingIndicator() {
        if (this.isTyping) return;
        
        this.isTyping = true;
        this.updateSendButton();

        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typing-indicator';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '🙏';

        const dotsDiv = document.createElement('div');
        dotsDiv.className = 'typing-dots';
        dotsDiv.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

        typingDiv.appendChild(avatar);
        typingDiv.appendChild(dotsDiv);

        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        
        this.isTyping = false;
        this.updateSendButton();
    }

    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 50);
    }

    async copyMessage(content) {
        try {
            if (CONFIG.FEATURES.CLIPBOARD_API) {
                await navigator.clipboard.writeText(content);
            } else {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = content;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
            }
            
            if (window.notifications) {
                window.notifications.show('success', 'Copied', 'Message copied to clipboard');
            }
        } catch (error) {
            console.error('Copy failed:', error);
            if (window.notifications) {
                window.notifications.show('error', 'Copy Failed', 'Unable to copy message');
            }
        }
    }

    async likeMessage(messageId, content) {
        try {
            await api.sendFeedback({
                type: 'like',
                rating: 5,
                message: 'User liked this response',
                context: {
                    message_id: messageId,
                    content: content.substring(0, 100) // First 100 chars for context
                }
            });
            
            // Update button visual state
            const likeBtn = document.querySelector(`[data-message-id="${messageId}"] .message-action i.fa-heart`);
            if (likeBtn) {
                likeBtn.className = 'fas fa-heart';
                likeBtn.style.color = 'var(--lotus-color)';
            }
            
            if (window.notifications) {
                window.notifications.show('success', 'Thank You', 'Your feedback helps improve our responses');
            }
        } catch (error) {
            console.error('Feedback failed:', error);
        }
    }

    clearChat() {
        if (confirm('Are you sure you want to clear all chat messages?')) {
            this.messages = [];
            this.messagesContainer.innerHTML = '';
            this.welcomeSection.classList.remove('hidden');
            this.saveChatHistory();
            
            if (window.notifications) {
                window.notifications.show('info', 'Chat Cleared', 'All messages have been removed');
            }
        }
    }

    exportChat() {
        if (this.messages.length === 0) {
            if (window.notifications) {
                window.notifications.show('warning', 'No Messages', 'No chat messages to export');
            }
            return;
        }

        const chatData = {
            timestamp: new Date().toISOString(),
            messages: this.messages.map(msg => ({
                role: msg.role,
                content: msg.content,
                timestamp: new Date(msg.timestamp).toISOString()
            }))
        };

        const blob = new Blob([JSON.stringify(chatData, null, 2)], {
            type: 'application/json'
        });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dharmamind-chat-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        if (window.notifications) {
            window.notifications.show('success', 'Exported', CONFIG.MESSAGES.CHAT_EXPORTED);
        }
    }

    async shareWisdom() {
        const lastAssistantMessage = this.messages
            .filter(msg => msg.role === 'assistant')
            .pop();

        if (!lastAssistantMessage) {
            if (window.notifications) {
                window.notifications.show('warning', 'No Wisdom', 'No AI responses to share');
            }
            return;
        }

        const shareText = `Wisdom from DharmaMind:\n\n"${lastAssistantMessage.content}"\n\n🙏 Shared with compassion`;

        if (CONFIG.FEATURES.SHARE_API) {
            try {
                await navigator.share({
                    title: 'Spiritual Wisdom from DharmaMind',
                    text: shareText
                });
                
                if (window.notifications) {
                    window.notifications.show('success', 'Shared', CONFIG.MESSAGES.WISDOM_SHARED);
                }
                return;
            } catch (error) {
                // Fall through to clipboard sharing
            }
        }

        // Fallback to clipboard
        try {
            await this.copyMessage(shareText);
            if (window.notifications) {
                window.notifications.show('success', 'Copied for Sharing', 'Wisdom copied to clipboard for sharing');
            }
        } catch (error) {
            console.error('Share failed:', error);
            if (window.notifications) {
                window.notifications.show('error', 'Share Failed', 'Unable to share wisdom');
            }
        }
    }

    saveChatHistory() {
        if (!CONFIG.PRIVACY.SAVE_CONVERSATIONS) return;
        
        try {
            const historyData = {
                timestamp: Date.now(),
                messages: this.messages.slice(-CONFIG.CHAT.MESSAGE_HISTORY_LIMIT)
            };
            
            localStorage.setItem(CONFIG.STORAGE_KEYS.CHAT_HISTORY, JSON.stringify(historyData));
        } catch (error) {
            console.error('Failed to save chat history:', error);
        }
    }

    loadChatHistory() {
        if (!CONFIG.PRIVACY.SAVE_CONVERSATIONS) return;
        
        try {
            const historyData = localStorage.getItem(CONFIG.STORAGE_KEYS.CHAT_HISTORY);
            if (historyData) {
                const parsed = JSON.parse(historyData);
                
                // Load recent messages (last 24 hours)
                const dayAgo = Date.now() - (24 * 60 * 60 * 1000);
                if (parsed.timestamp > dayAgo && parsed.messages) {
                    this.messages = parsed.messages;
                    
                    // Render messages
                    parsed.messages.forEach(msg => {
                        if (this.welcomeSection && !this.welcomeSection.classList.contains('hidden')) {
                            this.welcomeSection.classList.add('hidden');
                        }
                        this.addMessage(msg.role, msg.content, msg);
                    });
                }
            }
        } catch (error) {
            console.error('Failed to load chat history:', error);
        }
    }

    // New methods for continuous learning display
    createLearningInfoElement(options) {
        const learningDiv = document.createElement('div');
        learningDiv.className = 'learning-info';
        learningDiv.innerHTML = `
            <div class="learning-header">
                <i class="fas fa-brain"></i>
                <span>Continuous Learning Applied</span>
            </div>
            <div class="learning-details">
                <small>This response incorporates learning from previous interactions to better serve your spiritual journey.</small>
            </div>
        `;
        return learningDiv;
    }

    createEnhancedModelInfoElement(options) {
        const modelDiv = document.createElement('div');
        modelDiv.className = 'model-info enhanced';
        
        let modelContent = '<div class="model-header"><i class="fas fa-robot"></i> AI Response Details</div>';
        
        // Show which models contributed
        if (options.advanced_llm_response) {
            modelContent += `
                <div class="model-source">
                    <span class="model-label">Advanced AI:</span>
                    <span class="model-badge">GPT/Gemini/Claude</span>
                </div>
            `;
        }
        
        if (options.dharmallm_response && options.dharmallm_response.trim()) {
            modelContent += `
                <div class="model-source">
                    <span class="model-label">DharmaLLM:</span>
                    <span class="model-badge dharma">Custom Spiritual AI</span>
                </div>
            `;
        }
        
        // Show spiritual enhancement
        if (options.spiritual_enhancement) {
            const enhancement = options.spiritual_enhancement;
            if (enhancement.activated_modules && enhancement.activated_modules.length > 0) {
                modelContent += `
                    <div class="spiritual-enhancement">
                        <span class="model-label">Spiritual Modules:</span>
                        <div class="modules-list">
                            ${enhancement.activated_modules.map(module => 
                                `<span class="module-badge">${module.replace('_', ' ')}</span>`
                            ).join('')}
                        </div>
                    </div>
                `;
            }
            
            if (enhancement.practices_suggested && enhancement.practices_suggested.length > 0) {
                modelContent += `
                    <div class="suggested-practices">
                        <span class="model-label">Suggested Practices:</span>
                        <div class="practices-list">
                            ${enhancement.practices_suggested.map(practice => 
                                `<span class="practice-badge">${practice}</span>`
                            ).join('')}
                        </div>
                    </div>
                `;
            }
        }
        
        modelDiv.innerHTML = modelContent;
        return modelDiv;
    }
}

// Initialize chat interface
document.addEventListener('DOMContentLoaded', () => {
    window.chat = new ChatInterface();
    
    if (CONFIG.DEBUG.ENABLED) {
        console.log('💬 DharmaMind Chat Interface initialized');
    }
});
