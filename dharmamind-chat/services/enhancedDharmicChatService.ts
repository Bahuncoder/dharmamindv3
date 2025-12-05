/**
 * Enhanced Dharmic Chat Service
 * Professional, modular service for spiritual guidance with comprehensive features
 */

import axios, { AxiosError, AxiosInstance } from 'axios';

// Configuration
const DHARMALLM_API_URL = process.env.NEXT_PUBLIC_DHARMALLM_API_URL || 'http://localhost:8001';
const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const REQUEST_TIMEOUT = 30000;
const MAX_RETRIES = 3;
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Types
export interface DharmicMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: Date;
  metadata?: DharmicMetadata;
}

export interface DharmicMetadata {
  dharmic_alignment?: number;
  wisdom_score?: number;
  tradition?: string[];
  category?: MessageCategory;
  spiritual_context?: SpiritualContext;
  insights?: DharmicInsight[];
  practices?: SuggestedPractice[];
  scripture_references?: ScriptureReference[];
  emotional_tone?: string;
  consciousness_level?: string;
  chakra_resonance?: ChakraResonance[];
  modules_used?: string[];
}

export type MessageCategory = 
  | 'wisdom' 
  | 'guidance' 
  | 'practice' 
  | 'teaching' 
  | 'contemplation'
  | 'support'
  | 'inquiry';

export interface SpiritualContext {
  life_aspect?: string;
  dharmic_principle?: string[];
  user_state?: string;
  session_theme?: string;
  journey_stage?: string;
}

export interface DharmicInsight {
  type: 'wisdom' | 'practice' | 'reflection' | 'warning' | 'encouragement';
  content: string;
  source?: string;
  relevance: number;
}

export interface SuggestedPractice {
  name: string;
  type: 'meditation' | 'contemplation' | 'breathing' | 'mindfulness' | 'ritual';
  duration?: number;
  description: string;
  benefits?: string[];
  instructions?: string[];
}

export interface ScriptureReference {
  text: string;
  source: string;
  tradition: string;
  context?: string;
  relevance: number;
}

export interface ChakraResonance {
  chakra: string;
  resonance: number;
  significance: string;
}

export interface DharmicChatOptions {
  conversationId?: string;
  userId?: string;
  tradition_preference?: string[];
  response_style?: 'conversational' | 'wise' | 'practical' | 'poetic';
  include_scripture?: boolean;
  include_practices?: boolean;
  include_insights?: boolean;
  context?: SpiritualContext;
  temperature?: number;
  max_tokens?: number;
}

export interface DharmicChatResponse {
  message: DharmicMessage;
  conversationId: string;
  processing_info: ProcessingInfo;
  suggestions?: string[];
  related_topics?: string[];
}

export interface ProcessingInfo {
  processing_time: number;
  model_used: string;
  service_used: string;
  cached: boolean;
  dharmic_enhancement: boolean;
  modules_activated?: string[];
}

// Cache Interface
interface CacheEntry {
  response: DharmicChatResponse;
  timestamp: number;
}

/**
 * Enhanced Dharmic Chat Service Class
 */
class EnhancedDharmicChatService {
  private authToken: string | null = null;
  private axiosInstance: AxiosInstance;
  private responseCache: Map<string, CacheEntry> = new Map();
  private requestQueue: Map<string, Promise<DharmicChatResponse>> = new Map();

  constructor() {
    this.axiosInstance = axios.create({
      timeout: REQUEST_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Load auth token from storage
    if (typeof window !== 'undefined') {
      this.authToken = localStorage.getItem('auth_token');
    }

    // Setup response interceptor for error handling
    this.setupInterceptors();
  }

  /**
   * Setup axios interceptors for consistent error handling
   */
  private setupInterceptors(): void {
    this.axiosInstance.interceptors.request.use(
      (config) => {
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    this.axiosInstance.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          this.clearAuthToken();
          // Could trigger a re-auth flow here
        }
        return Promise.reject(error);
      }
    );
  }

  /**
   * Set authentication token
   */
  setAuthToken(token: string): void {
    this.authToken = token;
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', token);
    }
  }

  /**
   * Clear authentication token
   */
  clearAuthToken(): void {
    this.authToken = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token');
    }
  }

  /**
   * Generate cache key for request
   */
  private getCacheKey(message: string, options: DharmicChatOptions): string {
    return `${message.toLowerCase().trim()}-${JSON.stringify(options)}`;
  }

  /**
   * Check if cache entry is valid
   */
  private isCacheValid(entry: CacheEntry): boolean {
    return Date.now() - entry.timestamp < CACHE_DURATION;
  }

  /**
   * Get cached response if available and valid
   */
  private getCachedResponse(cacheKey: string): DharmicChatResponse | null {
    const entry = this.responseCache.get(cacheKey);
    if (entry && this.isCacheValid(entry)) {
      return {
        ...entry.response,
        processing_info: {
          ...entry.response.processing_info,
          cached: true,
        },
      };
    }
    return null;
  }

  /**
   * Cache response
   */
  private cacheResponse(cacheKey: string, response: DharmicChatResponse): void {
    this.responseCache.set(cacheKey, {
      response,
      timestamp: Date.now(),
    });

    // Clean old cache entries
    if (this.responseCache.size > 100) {
      const oldestKey = this.responseCache.keys().next().value;
      this.responseCache.delete(oldestKey);
    }
  }

  /**
   * Send a dharmic chat message with enhanced features
   */
  async sendMessage(
    message: string,
    options: DharmicChatOptions = {}
  ): Promise<DharmicChatResponse> {
    // Input validation
    if (!message || message.trim().length === 0) {
      throw new Error('Message cannot be empty');
    }

    if (message.length > 4000) {
      throw new Error('Message too long. Please keep under 4000 characters.');
    }

    // Check cache
    const cacheKey = this.getCacheKey(message, options);
    const cached = this.getCachedResponse(cacheKey);
    if (cached) {
      console.log('✅ Returning cached response');
      return cached;
    }

    // Check if request is already in progress
    const inProgress = this.requestQueue.get(cacheKey);
    if (inProgress) {
      console.log('⏳ Request already in progress, waiting...');
      return inProgress;
    }

    // Create new request
    const requestPromise = this.executeRequest(message, options);
    this.requestQueue.set(cacheKey, requestPromise);

    try {
      const response = await requestPromise;
      this.cacheResponse(cacheKey, response);
      return response;
    } finally {
      this.requestQueue.delete(cacheKey);
    }
  }

  /**
   * Execute the actual API request with retry logic
   */
  private async executeRequest(
    message: string,
    options: DharmicChatOptions
  ): Promise<DharmicChatResponse> {
    const startTime = Date.now();

    // Try authenticated backend first
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        return await this.tryBackendAPI(message, options, startTime);
      } catch (error) {
        console.log(`Backend attempt ${attempt}/${MAX_RETRIES} failed:`, error);

        if (attempt === MAX_RETRIES) {
          // Try DharmaLLM direct
          try {
            return await this.tryDharmaLLMDirect(message, options, startTime);
          } catch (dharmaError) {
            console.log('DharmaLLM direct failed:', dharmaError);
            // Final fallback to enhanced local wisdom
            return await this.getEnhancedLocalWisdom(message, options, startTime);
          }
        }

        // Wait before retry (exponential backoff)
        await this.sleep(Math.pow(2, attempt) * 1000);
      }
    }

    // This should never be reached due to the throw in the loop
    throw new Error('All request attempts failed');
  }

  /**
   * Try authenticated backend API
   */
  private async tryBackendAPI(
    message: string,
    options: DharmicChatOptions,
    startTime: number
  ): Promise<DharmicChatResponse> {
    console.log('🔐 Trying authenticated backend...');

    const payload = {
      message,
      conversation_id: options.conversationId,
      user_id: options.userId || 'seeker',
      include_personal_growth: options.include_insights,
      include_spiritual_guidance: true,
      include_ethical_guidance: true,
      response_style: options.response_style || 'conversational',
      tradition_preference: options.tradition_preference,
      context: options.context,
      temperature: options.temperature || 0.8,
      max_tokens: options.max_tokens || 1024,
    };

    const response = await this.axiosInstance.post(
      `${BACKEND_URL}/api/v1/dharmic-chat`,
      payload
    );

    if (response.status === 200 && response.data) {
      const data = response.data;
      return this.formatBackendResponse(data, startTime);
    }

    throw new Error(`Backend returned status ${response.status}`);
  }

  /**
   * Try DharmaLLM direct connection
   */
  private async tryDharmaLLMDirect(
    message: string,
    options: DharmicChatOptions,
    startTime: number
  ): Promise<DharmicChatResponse> {
    console.log('⚡ Trying DharmaLLM direct...');

    const payload = {
      message,
      session_id: options.conversationId,
      user_id: options.userId || 'seeker',
      temperature: options.temperature || 0.8,
      max_tokens: options.max_tokens || 1024,
      include_dharmic_insights: true,
    };

    const response = await this.axiosInstance.post(
      `${DHARMALLM_API_URL}/api/v1/chat`,
      payload
    );

    if (response.status === 200 && response.data) {
      const data = response.data;
      return this.formatDharmaLLMResponse(data, startTime, options.conversationId);
    }

    throw new Error(`DharmaLLM returned status ${response.status}`);
  }

  /**
   * Format backend API response
   */
  private formatBackendResponse(
    data: any,
    startTime: number
  ): DharmicChatResponse {
    const processingTime = Date.now() - startTime;

    return {
      message: {
        id: data.message_id || `msg-${Date.now()}`,
        content: data.response || data.message,
        role: 'assistant',
        timestamp: new Date(data.timestamp || Date.now()),
        metadata: {
          dharmic_alignment: data.dharmic_alignment || data.metadata?.dharmic_alignment,
          wisdom_score: data.confidence_score,
          tradition: data.tradition_detected,
          category: this.detectMessageCategory(data.response),
          spiritual_context: data.spiritual_context,
          insights: this.formatInsights(data.dharmic_insights),
          practices: this.formatPractices(data.growth_suggestions),
          scripture_references: this.formatScriptures(data.metadata?.scripture_refs),
          emotional_tone: data.metadata?.emotional_tone,
          consciousness_level: data.metadata?.consciousness_level,
          modules_used: data.modules_used || data.processing_info?.chakra_modules_used,
        },
      },
      conversationId: data.conversation_id || 'backend-session',
      processing_info: {
        processing_time: data.processing_time || processingTime,
        model_used: data.model_used || 'Backend DharmaLLM',
        service_used: 'Authenticated Backend',
        cached: false,
        dharmic_enhancement: true,
        modules_activated: data.modules_used || data.processing_info?.chakra_modules_used,
      },
      suggestions: data.suggestions || data.growth_suggestions,
      related_topics: data.related_concepts || data.related_topics,
    };
  }

  /**
   * Format DharmaLLM direct response
   */
  private formatDharmaLLMResponse(
    data: any,
    startTime: number,
    conversationId?: string
  ): DharmicChatResponse {
    const processingTime = Date.now() - startTime;

    return {
      message: {
        id: `msg-${Date.now()}`,
        content: data.response,
        role: 'assistant',
        timestamp: new Date(data.timestamp || Date.now()),
        metadata: {
          dharmic_alignment: data.dharmic_alignment,
          wisdom_score: data.confidence,
          category: this.detectMessageCategory(data.response),
          insights: this.formatInsights(data.dharmic_insights),
          scripture_references: this.formatScriptures(data.sources),
          modules_used: data.sources || ['DharmaLLM'],
        },
      },
      conversationId: data.session_id || conversationId || 'direct-session',
      processing_info: {
        processing_time: data.processing_time || processingTime,
        model_used: 'DharmaLLM Direct',
        service_used: 'DharmaLLM Service',
        cached: false,
        dharmic_enhancement: true,
      },
      suggestions: [],
      related_topics: [],
    };
  }

  /**
   * Get enhanced local wisdom (fallback)
   */
  private async getEnhancedLocalWisdom(
    message: string,
    options: DharmicChatOptions,
    startTime: number
  ): Promise<DharmicChatResponse> {
    console.log('🕉️ Using enhanced local dharmic wisdom...');

    // Analyze message for spiritual themes
    const themes = this.detectSpiritualThemes(message);
    const category = this.detectMessageCategory(message);
    
    // Generate contextual response
    const response = this.generateContextualResponse(message, themes, category, options);
    const insights = this.generateInsights(themes, category);
    const practices = this.generatePractices(themes);
    const scriptures = this.generateScriptureReferences(themes);

    const processingTime = Date.now() - startTime;

    return {
      message: {
        id: `local-${Date.now()}`,
        content: response,
        role: 'assistant',
        timestamp: new Date(),
        metadata: {
          dharmic_alignment: 0.85,
          wisdom_score: 0.8,
          tradition: themes.traditions,
          category: category,
          spiritual_context: {
            life_aspect: themes.life_aspect,
            dharmic_principle: themes.principles,
            user_state: themes.emotional_state,
          },
          insights: insights,
          practices: practices,
          scripture_references: scriptures,
          emotional_tone: themes.tone,
          modules_used: ['Enhanced Local Wisdom'],
        },
      },
      conversationId: options.conversationId || 'local-session',
      processing_info: {
        processing_time: processingTime,
        model_used: 'Local Dharmic Wisdom Engine',
        service_used: 'Offline Dharmic Processing',
        cached: false,
        dharmic_enhancement: true,
        modules_activated: ['wisdom', 'scripture', 'practices'],
      },
      suggestions: this.generateSuggestions(themes),
      related_topics: this.generateRelatedTopics(themes),
    };
  }

  /**
   * Detect spiritual themes in message
   */
  private detectSpiritualThemes(message: string): any {
    const lower = message.toLowerCase();
    
    const themes = {
      traditions: [] as string[],
      principles: [] as string[],
      life_aspect: '',
      emotional_state: 'seeking',
      tone: 'inquiry',
      practices: [] as string[],
    };

    // Detect traditions
    if (/(buddha|buddhis|zen|vipassana|mindfulness)/i.test(lower)) themes.traditions.push('Buddhism');
    if (/(hindu|vedic|vedanta|yoga|krishna|rama|shiva)/i.test(lower)) themes.traditions.push('Hinduism');
    if (/(jain|ahimsa|mahavira)/i.test(lower)) themes.traditions.push('Jainism');
    if (/(sikh|guru nanak|waheguru)/i.test(lower)) themes.traditions.push('Sikhism');
    if (/(tao|dao|wu wei|yin yang)/i.test(lower)) themes.traditions.push('Taoism');

    // Detect dharmic principles
    if (/(compassion|loving.?kindness|metta|karuna)/i.test(lower)) themes.principles.push('Compassion');
    if (/(wisdom|prajna|discernment)/i.test(lower)) themes.principles.push('Wisdom');
    if (/(mindful|present|awareness|consciousness)/i.test(lower)) themes.principles.push('Mindfulness');
    if (/(impermanence|anicca|change)/i.test(lower)) themes.principles.push('Impermanence');
    if (/(non.?attachment|letting.?go|vairagya)/i.test(lower)) themes.principles.push('Non-attachment');
    if (/(dharma|duty|right.?action)/i.test(lower)) themes.principles.push('Dharma');
    if (/(karma|action|consequence)/i.test(lower)) themes.principles.push('Karma');
    if (/(meditation|dhyana|samadhi)/i.test(lower)) themes.practices.push('Meditation');
    if (/(breathing|breath|pranayama)/i.test(lower)) themes.practices.push('Breathing');

    // Detect life aspects
    if (/(relationship|love|family|friend)/i.test(lower)) themes.life_aspect = 'relationships';
    else if (/(work|career|job|purpose)/i.test(lower)) themes.life_aspect = 'career';
    else if (/(suffering|pain|difficult|struggle)/i.test(lower)) themes.life_aspect = 'challenges';
    else if (/(peace|calm|tranquil|serene)/i.test(lower)) themes.life_aspect = 'inner_peace';
    else if (/(growth|develop|improve|better)/i.test(lower)) themes.life_aspect = 'personal_growth';

    // Detect emotional tone
    if (/(anxious|worry|fear|afraid)/i.test(lower)) themes.emotional_state = 'anxious';
    else if (/(sad|depress|grief|loss)/i.test(lower)) themes.emotional_state = 'sorrowful';
    else if (/(angry|frustrat|annoy)/i.test(lower)) themes.emotional_state = 'agitated';
    else if (/(confus|lost|uncertain)/i.test(lower)) themes.emotional_state = 'confused';
    else if (/(grateful|thank|appreciate)/i.test(lower)) themes.emotional_state = 'grateful';

    return themes;
  }

  /**
   * Detect message category
   */
  private detectMessageCategory(message: string): MessageCategory {
    const lower = message.toLowerCase();

    if (/(how|what|why|explain|tell me about)/i.test(lower)) return 'inquiry';
    if (/(meditat|contemplat|practice|exercise)/i.test(lower)) return 'practice';
    if (/(guide|help|advice|should i)/i.test(lower)) return 'guidance';
    if (/(teach|learn|understand|wisdom)/i.test(lower)) return 'teaching';
    if (/(reflect|ponder|think about)/i.test(lower)) return 'contemplation';
    if (/(feel|emotion|difficult|struggle)/i.test(lower)) return 'support';
    
    return 'wisdom';
  }

  /**
   * Generate contextual response based on themes
   */
  private generateContextualResponse(
    message: string,
    themes: any,
    category: MessageCategory,
    options: DharmicChatOptions
  ): string {
    // This would be much more sophisticated in production
    const responses = {
      greeting: '🙏 Namaste, dear seeker. I am here to walk with you on your spiritual journey.',
      wisdom: '🕉️ In the ancient wisdom traditions, we learn that ',
      guidance: '🌸 The dharmic path suggests that ',
      practice: '🧘 For your spiritual practice, consider ',
      support: '💙 I hear your heart speaking. Remember that ',
    };

    // Generate a thoughtful response based on the detected themes
    let response = '';

    if (themes.emotional_state === 'anxious') {
      response = `${responses.support}anxiety is like a passing cloud - it comes and it goes. The Buddha taught that suffering arises from clinging, and peace comes from letting go.\n\n`;
    } else if (themes.emotional_state === 'sorrowful') {
      response = `${responses.support}grief is a sacred teacher. As the Bhagavad Gita reminds us, just as we change our clothes, the soul changes bodies. What seems like an ending is also a beginning.\n\n`;
    } else if (category === 'practice') {
      response = `${responses.practice}${themes.practices.length > 0 ? themes.practices.join(' and ') : 'mindful awareness'}. `;
    } else {
      response = `${responses.wisdom}the path to understanding begins with observing our mind with gentle awareness.\n\n`;
    }

    // Add dharmic principle reference if detected
    if (themes.principles.length > 0) {
      response += `Your question touches upon the principle of **${themes.principles[0]}**, which is central to the dharmic path.\n\n`;
    }

    // Add contextual wisdom
    response += this.getContextualWisdom(themes, category);

    return response;
  }

  /**
   * Get contextual wisdom quote or teaching
   */
  private getContextualWisdom(themes: any, category: MessageCategory): string {
    const wisdomQuotes = {
      compassion: '*"Just as a mother would protect her only child with her life, even so let one cultivate a boundless love towards all beings."* - Buddha\n\nTrue compassion flows when we recognize our interconnectedness.',
      wisdom: '*"The mind is everything. What you think you become."* - Buddha\n\nWisdom arises when we observe our thoughts without judgment.',
      mindfulness: '*"Do not dwell in the past, do not dream of the future, concentrate the mind on the present moment."* - Buddha\n\nThe present moment is where life unfolds.',
      impermanence: '*"Nothing is permanent in this wicked world, not even our troubles."* - Charlie Chaplin echoing ancient dharma\n\nUnderstanding impermanence brings both peace and urgency to live well.',
      dharma: '*"Do your duty, without attachment to results."* - Bhagavad Gita\n\nDharma is about right action, done with proper intention.',
    };

    // Select appropriate wisdom based on detected principles
    if (themes.principles.length > 0) {
      const principle = themes.principles[0].toLowerCase();
      return wisdomQuotes[principle] || wisdomQuotes.wisdom;
    }

    return '**Remember:** The spiritual journey is not about reaching a destination, but about walking the path with awareness and compassion.';
  }

  /**
   * Generate insights based on themes
   */
  private generateInsights(themes: any, category: MessageCategory): DharmicInsight[] {
    const insights: DharmicInsight[] = [];

    if (themes.emotional_state === 'anxious') {
      insights.push({
        type: 'practice',
        content: 'Practice grounding through mindful breathing to calm anxiety',
        relevance: 0.9,
      });
    }

    if (themes.practices.includes('Meditation')) {
      insights.push({
        type: 'encouragement',
        content: 'Regular meditation practice strengthens your inner stability',
        source: 'Dhammapada',
        relevance: 0.85,
      });
    }

    if (themes.principles.includes('Compassion')) {
      insights.push({
        type: 'wisdom',
        content: 'Compassion for yourself is the foundation of compassion for others',
        relevance: 0.88,
      });
    }

    return insights;
  }

  /**
   * Generate suggested practices
   */
  private generatePractices(themes: any): SuggestedPractice[] {
    const practices: SuggestedPractice[] = [];

    if (themes.emotional_state === 'anxious' || themes.practices.includes('Breathing')) {
      practices.push({
        name: '4-7-8 Breathing',
        type: 'breathing',
        duration: 5,
        description: 'A calming breath technique to reduce anxiety',
        benefits: ['Reduces anxiety', 'Calms nervous system', 'Improves focus'],
        instructions: [
          'Breathe in through nose for 4 counts',
          'Hold breath for 7 counts',
          'Exhale through mouth for 8 counts',
          'Repeat 4 times',
        ],
      });
    }

    if (themes.practices.includes('Meditation') || themes.life_aspect === 'inner_peace') {
      practices.push({
        name: 'Loving-Kindness Meditation',
        type: 'meditation',
        duration: 15,
        description: 'Cultivate compassion for yourself and others',
        benefits: ['Increases compassion', 'Reduces negative emotions', 'Enhances wellbeing'],
        instructions: [
          'Sit comfortably and close your eyes',
          'Repeat: "May I be happy, may I be healthy, may I be at peace"',
          'Extend these wishes to loved ones, then all beings',
        ],
      });
    }

    return practices;
  }

  /**
   * Format insights from API response
   */
  private formatInsights(insights: any): DharmicInsight[] {
    if (!insights || !Array.isArray(insights)) return [];

    return insights.map((insight, index) => ({
      type: 'wisdom',
      content: typeof insight === 'string' ? insight : insight.content,
      source: typeof insight === 'object' ? insight.source : undefined,
      relevance: typeof insight === 'object' ? insight.relevance : 0.8,
    }));
  }

  /**
   * Format practices from API response
   */
  private formatPractices(practices: any): SuggestedPractice[] {
    if (!practices || !Array.isArray(practices)) return [];

    return practices.map((practice) => {
      if (typeof practice === 'string') {
        return {
          name: practice,
          type: 'mindfulness',
          description: practice,
        };
      }
      return practice as SuggestedPractice;
    });
  }

  /**
   * Generate scripture references
   */
  private generateScriptureReferences(themes: any): ScriptureReference[] {
    const scriptures: ScriptureReference[] = [];

    if (themes.principles.includes('Compassion')) {
      scriptures.push({
        text: 'Hatred does not cease by hatred, but only by love; this is the eternal rule.',
        source: 'Dhammapada 1:5',
        tradition: 'Buddhism',
        relevance: 0.9,
      });
    }

    if (themes.principles.includes('Dharma')) {
      scriptures.push({
        text: 'You have the right to work, but never to the fruit of work.',
        source: 'Bhagavad Gita 2:47',
        tradition: 'Hinduism',
        relevance: 0.88,
      });
    }

    return scriptures;
  }

  /**
   * Format scriptures from API response
   */
  private formatScriptures(scriptures: any): ScriptureReference[] {
    if (!scriptures || !Array.isArray(scriptures)) return [];

    return scriptures.map((scripture) => {
      if (typeof scripture === 'string') {
        return {
          text: scripture,
          source: 'Ancient Wisdom',
          tradition: 'Universal',
          relevance: 0.75,
        };
      }
      return scripture as ScriptureReference;
    });
  }

  /**
   * Generate contextual suggestions
   */
  private generateSuggestions(themes: any): string[] {
    const suggestions: string[] = [];

    if (themes.practices.includes('Meditation')) {
      suggestions.push('Learn about different meditation techniques');
      suggestions.push('Start a daily meditation practice');
    }

    if (themes.life_aspect === 'relationships') {
      suggestions.push('Explore loving-kindness practice');
      suggestions.push('Learn about compassionate communication');
    }

    if (themes.emotional_state === 'anxious') {
      suggestions.push('Try a guided breathing exercise');
      suggestions.push('Explore teachings on impermanence');
    }

    return suggestions;
  }

  /**
   * Generate related topics
   */
  private generateRelatedTopics(themes: any): string[] {
    const topics: string[] = [];

    if (themes.principles.length > 0) {
      topics.push(...themes.principles.map(p => `The Path of ${p}`));
    }

    if (themes.traditions.length > 0) {
      topics.push(...themes.traditions.map(t => `${t} Teachings`));
    }

    topics.push('Daily Spiritual Practice', 'Mindful Living', 'Inner Peace');

    return topics.slice(0, 5);
  }

  /**
   * Utility: Sleep function for retry delays
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Clear response cache
   */
  clearCache(): void {
    this.responseCache.clear();
    console.log('Response cache cleared');
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; entries: number } {
    let size = 0;
    this.responseCache.forEach((entry) => {
      size += JSON.stringify(entry).length;
    });
    
    return {
      size: size,
      entries: this.responseCache.size,
    };
  }
}

// Export singleton instance
export const enhancedDharmicChatService = new EnhancedDharmicChatService();
export default enhancedDharmicChatService;
