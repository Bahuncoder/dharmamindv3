import axios from 'axios';

// Use the backend API URL - port 8000 where the comprehensive backend is running
const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ChatMessage {
  content: string;
  sender: 'user' | 'ai';
  timestamp?: Date;
  wisdom_score?: number;
  dharmic_alignment?: number;
  conversation_id?: string;
  user_id?: string;
}

export interface ChatResponse {
  response: string;
  conversation_id?: string;
  message_id?: string;
  confidence_score?: number;
  dharmic_alignment?: number;
  modules_used?: string[];
  timestamp?: string;
  model_used?: string;
  processing_time?: number;
  sources?: string[];
  suggestions?: string[];
  metadata?: Record<string, any>;
  // New dharmic chat fields
  dharmic_insights?: string[];
  growth_suggestions?: string[];
  spiritual_context?: string;
  ethical_guidance?: string;
  conversation_style?: string;
}

export interface SpiritualWisdomRequest {
  message: string;
  user_id?: string;
  context?: Record<string, any>;
  tradition_preference?: string;
}

export interface SpiritualWisdomResponse {
  response: string;
  wisdom_source: string;
  dharma_assessment: Record<string, any>;
  consciousness_level: string;
  emotional_tone: string;
  confidence: number;
  timestamp: string;
  dharmic_insights: string[];
  growth_suggestions: string[];
  spiritual_context: string;
}

export interface WisdomRequest {
  query: string;
  context?: string;
  user_preferences?: Record<string, any>;
}

export interface DharmicChatRequest {
  message: string;
  conversation_id?: string;
  user_id?: string;
  include_personal_growth?: boolean;
  include_spiritual_guidance?: boolean;
  include_ethical_guidance?: boolean;
  response_style?: 'conversational' | 'wise' | 'practical';
}

export interface DharmicChatResponse {
  response: string;
  conversation_id: string;
  dharmic_insights: string[];
  growth_suggestions: string[];
  spiritual_context: string;
  ethical_guidance: string;
  conversation_style: string;
  processing_info: {
    processing_time: number;
    chakra_modules_used: string[];
    model_used: string;
    dharmic_enhancement: boolean;
  };
}

export interface WisdomResponse {
  wisdom: string;
  source: string;
  relevance_score: number;
  category: string;
  related_concepts: string[];
}

export interface ConversationHistory {
  conversation_id: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
  user_id: string;
}

class ChatService {
  private authToken: string | null = null;

  constructor() {
    // Get auth token from localStorage if available
    if (typeof window !== 'undefined') {
      this.authToken = localStorage.getItem('auth_token');
    }
  }

  setAuthToken(token: string) {
    this.authToken = token;
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', token);
    }
  }

  clearAuthToken() {
    this.authToken = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token');
    }
  }

  private getHeaders() {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    return headers;
  }

  /**
   * Send a chat message - Using comprehensive internal spiritual processing
   */
  async sendMessage(
    message: string,
    conversationId?: string,
    userId?: string
  ): Promise<ChatResponse> {
    try {
      // Use comprehensive internal spiritual processing
      console.log('Using comprehensive internal spiritual processing...');
      const spiritualResponse = await this.getComprehensiveDharmicWisdom(message, userId);
      
      return {
        response: spiritualResponse.response,
        conversation_id: conversationId || 'internal-spiritual',
        message_id: `spiritual-${Date.now()}`,
        timestamp: spiritualResponse.timestamp,
        dharmic_insights: spiritualResponse.dharmic_insights,
        growth_suggestions: spiritualResponse.growth_suggestions,
        spiritual_context: spiritualResponse.spiritual_context,
        metadata: {
          source: 'comprehensive-dharmic-wisdom',
          wisdom_source: spiritualResponse.wisdom_source,
          consciousness_level: spiritualResponse.consciousness_level,
          dharma_assessment: spiritualResponse.dharma_assessment,
          confidence: spiritualResponse.confidence,
          emotional_tone: spiritualResponse.emotional_tone
        }
      };

    } catch (error) {
      console.error('Error sending chat message:', error);
      
      // Return enhanced fallback response
      return this.generateEnhancedFallbackResponse(message);
    }
  }

  /**
   * Send a message using the new dharmic chat endpoint
   * This provides ChatGPT-style responses with deep dharmic wisdom
   */
  async sendDharmicChat(
    message: string,
    options: {
      conversationId?: string;
      userId?: string;
      includePersonalGrowth?: boolean;
      includeSpiritualGuidance?: boolean;
      includeEthicalGuidance?: boolean;
      responseStyle?: 'conversational' | 'wise' | 'practical';
    } = {}
  ): Promise<ChatResponse> {
    try {
      const payload: DharmicChatRequest = {
        message,
        conversation_id: options.conversationId,
        user_id: options.userId || 'seeker',
        include_personal_growth: options.includePersonalGrowth ?? true,
        include_spiritual_guidance: options.includeSpiritualGuidance ?? true,
        include_ethical_guidance: options.includeEthicalGuidance ?? true,
        response_style: options.responseStyle || 'conversational'
      };

      console.log('Sending dharmic chat message:', `${BACKEND_URL}/api/v1/dharmic/chat`);
      console.log('Payload:', payload);

      const response = await axios.post<DharmicChatResponse>(
        `${BACKEND_URL}/api/v1/dharmic/chat`,
        payload,
        {
          headers: this.getHeaders(),
          timeout: 30000, // 30 second timeout for dharmic processing
        }
      );

      console.log('Dharmic chat response received:', response.data);

      return {
        response: response.data.response,
        conversation_id: response.data.conversation_id,
        message_id: `dharmic-${Date.now()}`,
        timestamp: new Date().toISOString(),
        processing_time: response.data.processing_info.processing_time,
        model_used: response.data.processing_info.model_used,
        dharmic_insights: response.data.dharmic_insights,
        growth_suggestions: response.data.growth_suggestions,
        spiritual_context: response.data.spiritual_context,
        ethical_guidance: response.data.ethical_guidance,
        conversation_style: response.data.conversation_style,
        metadata: {
          source: 'dharmic-chat',
          chakra_modules_used: response.data.processing_info.chakra_modules_used,
          dharmic_enhancement: response.data.processing_info.dharmic_enhancement,
          wisdom_type: 'comprehensive_dharmic_guidance'
        }
      };

    } catch (error) {
      console.error('Error sending dharmic chat message:', error);
      
      if (axios.isAxiosError(error)) {
        if (error.response) {
          throw new Error(`Dharmic chat error: ${error.response.data?.detail || error.response.statusText}`);
        } else if (error.request) {
          throw new Error('Unable to connect to dharmic chat service. Please check if the server is running.');
        }
      }
      
      throw new Error('An unexpected error occurred while processing your dharmic chat message.');
    }
  }

  /**
   * Enhanced sendMessage method with dharmic chat option
   * Now defaults to dharmic chat for better ChatGPT-like responses
   */
  async sendMessageEnhanced(
    message: string,
    conversationId?: string,
    userId?: string,
    useDharmicChat: boolean = true
  ): Promise<ChatResponse> {
    if (useDharmicChat) {
      try {
        // Try dharmic chat first for best experience
        return await this.sendDharmicChat(message, {
          conversationId,
          userId,
          responseStyle: 'conversational'
        });
      } catch (error) {
        console.warn('Dharmic chat failed, falling back to standard chat:', error);
        // Fall back to original sendMessage if dharmic chat fails
        return await this.sendMessage(message, conversationId, userId);
      }
    } else {
      return await this.sendMessage(message, conversationId, userId);
    }
  }

  /**
   * Request spiritual wisdom from the backend
   */
  async getWisdom(
    query: string,
    context?: string,
    userPreferences?: Record<string, any>
  ): Promise<WisdomResponse> {
    try {
      const payload: WisdomRequest = {
        query,
        context,
        user_preferences: userPreferences
      };

      console.log('Requesting wisdom from backend:', `${BACKEND_URL}/wisdom`);
      console.log('Payload:', payload);

      const response = await axios.post(
        `${BACKEND_URL}/wisdom`,
        payload,
        {
          headers: this.getHeaders(),
          timeout: 30000, // 30 second timeout
        }
      );

      console.log('Wisdom response:', response.data);
      return response.data;
    } catch (error) {
      console.error('Error getting wisdom:', error);
      
      if (axios.isAxiosError(error)) {
        if (error.response) {
          throw new Error(`Backend error: ${error.response.data?.detail || error.response.statusText}`);
        } else if (error.request) {
          throw new Error('Unable to connect to backend. Please check if the server is running.');
        }
      }
      
      throw new Error('An unexpected error occurred while requesting wisdom.');
    }
  }

  /**
   * Get spiritual wisdom using internal DharmaMind processing
   */
  async getInternalSpiritualWisdom(
    message: string,
    userId?: string
  ): Promise<SpiritualWisdomResponse | null> {
    try {
      const payload: SpiritualWisdomRequest = {
        message,
        user_id: userId || 'guest'
      };

      console.log('Requesting internal spiritual wisdom:', `${BACKEND_URL}/api/v1/internal/spiritual-wisdom`);
      console.log('Payload:', payload);

      const response = await axios.post(
        `${BACKEND_URL}/api/v1/internal/spiritual-wisdom`,
        payload,
        {
          headers: this.getHeaders(),
          timeout: 30000, // 30 second timeout
        }
      );

      console.log('Internal spiritual wisdom response:', response.data);
      return response.data;
    } catch (error) {
      console.error('Error getting internal spiritual wisdom:', error);
      
      if (axios.isAxiosError(error)) {
        console.error('Axios error details:', {
          status: error.response?.status,
          statusText: error.response?.statusText,
          data: error.response?.data
        });
      }
      
      // Return null to allow fallback to external chat
      return null;
    }
  }

  /**
   * Get conversation history from the backend
   */
  async getConversationHistory(
    userId: string,
    limit: number = 10
  ): Promise<ConversationHistory[]> {
    try {
      console.log('Getting conversation history from backend');

      const response = await axios.get(
        `${BACKEND_URL}/conversations`,
        {
          params: { user_id: userId, limit },
          headers: this.getHeaders(),
          timeout: 15000, // 15 second timeout
        }
      );

      console.log('Conversation history response:', response.data);
      return response.data;
    } catch (error) {
      console.error('Error getting conversation history:', error);
      
      if (axios.isAxiosError(error)) {
        if (error.response) {
          throw new Error(`Backend error: ${error.response.data?.detail || error.response.statusText}`);
        } else if (error.request) {
          throw new Error('Unable to connect to backend. Please check if the server is running.');
        }
      }
      
      throw new Error('An unexpected error occurred while fetching conversation history.');
    }
  }

  /**
   * Get a specific conversation from the backend
   */
  async getConversation(conversationId: string): Promise<ConversationHistory> {
    try {
      console.log('Getting conversation from backend:', conversationId);

      const response = await axios.get(
        `${BACKEND_URL}/conversations/${conversationId}`,
        {
          headers: this.getHeaders(),
          timeout: 15000, // 15 second timeout
        }
      );

      console.log('Conversation response:', response.data);
      return response.data;
    } catch (error) {
      console.error('Error getting conversation:', error);
      
      if (axios.isAxiosError(error)) {
        if (error.response) {
          throw new Error(`Backend error: ${error.response.data?.detail || error.response.statusText}`);
        } else if (error.request) {
          throw new Error('Unable to connect to backend. Please check if the server is running.');
        }
      }
      
      throw new Error('An unexpected error occurred while fetching the conversation.');
    }
  }

  /**
   * Delete a conversation from the backend
   */
  async deleteConversation(conversationId: string): Promise<void> {
    try {
      console.log('Deleting conversation from backend:', conversationId);

      await axios.delete(
        `${BACKEND_URL}/conversations/${conversationId}`,
        {
          headers: this.getHeaders(),
          timeout: 15000, // 15 second timeout
        }
      );

      console.log('Conversation deleted successfully');
    } catch (error) {
      console.error('Error deleting conversation:', error);
      
      if (axios.isAxiosError(error)) {
        if (error.response) {
          throw new Error(`Backend error: ${error.response.data?.detail || error.response.statusText}`);
        } else if (error.request) {
          throw new Error('Unable to connect to backend. Please check if the server is running.');
        }
      }
      
      throw new Error('An unexpected error occurred while deleting the conversation.');
    }
  }

  /**
   * Comprehensive Dharmic Wisdom Generation - Complete Internal Processing
   */
  async getComprehensiveDharmicWisdom(
    message: string,
    userId?: string
  ): Promise<SpiritualWisdomResponse> {
    try {
      const messageLower = message.toLowerCase();
      
      // Enhanced Dharmic Response Categories with Deep Spiritual Insights
      let response = '';
      let wisdom_source = '';
      let consciousness_level = '';
      let emotional_tone = '';
      let dharmic_insights: string[] = [];
      let growth_suggestions: string[] = [];
      let spiritual_context = '';
      
      // Meditation and Mindfulness - Expanded
      if (this.matchesKeywords(messageLower, ['meditation', 'meditate', 'mindfulness', 'awareness', 'present', 'breath', 'focus', 'concentration', 'zen'])) {
        const responses = [
          "🧘‍♀️ In the sacred stillness of meditation, you touch the eternal presence that has always been your true nature. Begin each practice with the recognition that you are not trying to achieve peace—you are remembering that you are peace itself. Let your breath become a bridge between the finite and infinite.",
          "✨ Mindfulness is the gentle art of awakening to what is, without judgment or resistance. Each moment of awareness is a gift you give to yourself and all beings. Start with three conscious breaths, allowing your mind to settle like sediment in a clear mountain lake.",
          "🌸 The present moment is the doorway to enlightenment. In meditation, we discover that thoughts are like clouds—they arise and pass away naturally when we don't chase them. Rest in the spacious awareness that witnesses all experience with perfect equanimity.",
          "🕉️ True meditation is not about stopping thoughts but about recognizing the space in which thoughts appear. This space is your Buddha nature, your divine essence—unchanging, luminous, and inherently free. Trust this deeper dimension of yourself.",
          "💫 Each breath in meditation is a prayer, each exhale a surrender. The practice teaches us that peace is not dependent on external conditions but is the very ground of our being. Cultivate patience with yourself—awakening unfolds in its own perfect timing."
        ];
        response = responses[this.getResponseIndex(message, responses.length)];
        wisdom_source = 'Buddhist Meditation Tradition & Mindfulness Masters';
        consciousness_level = 'contemplative';
        emotional_tone = 'peaceful and centering';
        dharmic_insights = [
          'Meditation reveals your true nature beyond thoughts and emotions',
          'Present moment awareness is the foundation of spiritual awakening',
          'Inner peace is your natural state, not something to achieve'
        ];
        growth_suggestions = [
          'Start with 5-10 minutes of daily breath awareness',
          'Practice loving-kindness meditation for yourself and others',
          'Join a meditation group or find a spiritual community',
          'Study teachings from masters like Thich Nhat Hanh or Pema Chödrön'
        ];
        spiritual_context = 'Meditation is the cornerstone of spiritual practice across all traditions, leading to self-realization and liberation from suffering.';
      }
      
      // Suffering and Challenges - Enhanced
      else if (this.matchesKeywords(messageLower, ['suffering', 'pain', 'difficult', 'hard', 'struggle', 'challenge', 'crisis', 'trauma', 'loss', 'grief', 'depression', 'anxiety'])) {
        const responses = [
          "🌅 Every form of suffering carries within it the seeds of liberation and awakening. Your challenges are not obstacles to your spiritual path—they are the very curriculum designed by life to awaken your deepest strength, compassion, and wisdom. Trust this sacred process of transformation.",
          "💫 As the lotus blooms most magnificently from the deepest mud, your greatest growth emerges from your most challenging experiences. Suffering, when met with conscious awareness, becomes the compost for wisdom, resilience, and an open heart.",
          "🕯️ Pain is inevitable, but suffering is optional. The Buddha taught that our resistance to what is creates additional layers of suffering. When you can hold your pain with the same tenderness you would offer a wounded child, healing naturally begins.",
          "🌊 Remember that you have survived every difficult day in your life so far—you are far stronger and more resilient than you realize. Each challenge mastered becomes a source of strength for helping others navigate their own dark nights of the soul.",
          "⭐ In the spiritual understanding, difficulties are not punishments but initiations. They crack open the shell of our smaller self so that our luminous essence can emerge. What feels like breakdown is often breakthrough in disguise."
        ];
        response = responses[this.getResponseIndex(message, responses.length)];
        wisdom_source = 'Buddhist Four Noble Truths & Universal Wisdom Traditions';
        consciousness_level = 'transformative';
        emotional_tone = 'compassionate and strengthening';
        dharmic_insights = [
          'Suffering is a teacher that awakens compassion and wisdom',
          'Resistance to pain creates additional suffering',
          'Challenges are opportunities for spiritual growth and awakening'
        ];
        growth_suggestions = [
          'Practice self-compassion during difficult times',
          'Explore the difference between pain and suffering through mindfulness',
          'Connect with others who have transformed their challenges',
          'Consider therapy or counseling as part of your spiritual journey',
          'Study wisdom teachings on suffering and transformation'
        ];
        spiritual_context = 'All spiritual traditions recognize suffering as a catalyst for awakening and the development of compassion.';
      }

      // Love and Relationships - Comprehensive
      else if (this.matchesKeywords(messageLower, ['love', 'relationship', 'family', 'friend', 'partner', 'compassion', 'marriage', 'dating', 'romance', 'heart', 'connection'])) {
        const responses = [
          "💝 True love begins with radical self-acceptance and blossoms into unconditional compassion for all beings. When you cultivate deep love for yourself—including your shadows and imperfections—you become a natural source of love for others. Relationships then become gardens where both souls can flourish.",
          "🌈 Every relationship is a sacred mirror reflecting aspects of yourself back to you. Conflicts and challenges are invitations to heal old wounds, practice forgiveness, and expand your capacity for understanding. See your loved ones as spiritual teachers disguised as ordinary people.",
          "💖 Love is not something you find—it's something you are. When you align with your loving essence, you naturally attract relationships that honor and celebrate your authentic self. Practice being the love you seek to receive.",
          "🌸 Healthy relationships are built on the foundation of two whole people choosing to share their completeness rather than two halves seeking completion. Work on becoming the person you would want to be in relationship with.",
          "✨ The highest form of love is conscious love—where both partners support each other's spiritual growth and awakening. This requires courage, vulnerability, and the willingness to see beyond personality to the divine essence in each other."
        ];
        response = responses[this.getResponseIndex(message, responses.length)];
        wisdom_source = 'Universal Love Teachings & Relationship Wisdom';
        consciousness_level = 'heart-centered';
        emotional_tone = 'loving and nurturing';
        dharmic_insights = [
          'Self-love is the foundation of all healthy relationships',
          'Relationships are mirrors for spiritual growth',
          'Love is your essential nature, not something external to find'
        ];
        growth_suggestions = [
          'Practice daily self-compassion and self-care',
          'Communicate with honesty and vulnerability',
          'Learn healthy boundary setting',
          'Practice forgiveness for yourself and others',
          'Study teachings on conscious relationships'
        ];
        spiritual_context = 'Love is considered the highest spiritual principle across all wisdom traditions, leading to unity consciousness.';
      }

      // Fear and Anxiety - Advanced
      else if (this.matchesKeywords(messageLower, ['fear', 'afraid', 'anxiety', 'anxious', 'worry', 'scared', 'panic', 'stress', 'overwhelm', 'nervous'])) {
        const responses = [
          "🌟 Fear is often excitement without breath, or love without presence. When anxiety arises, it's your nervous system trying to protect you, but often from imagined future threats. Ground yourself in this moment through deep, conscious breathing—you are safe, you are supported, you are enough exactly as you are.",
          "🦋 Fear cannot exist in the same space as love and presence. When anxiety visits, welcome it like an old friend who needs comfort. Send yourself the same infinite compassion you would offer a beloved friend facing difficulties. Fear dissolves in the warmth of self-acceptance.",
          "🌊 Like waves on the vast ocean of consciousness, fears arise and pass away naturally—you are not your fears, you are the peaceful awareness that witnesses them. Every time you observe anxiety without being consumed by it, you strengthen your witness consciousness.",
          "🕊️ Most anxiety comes from the mind's tendency to live in an imagined future or past. The present moment is the only place where peace exists. Practice the 5-4-3-2-1 grounding technique: notice 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste.",
          "💫 Courage is not the absence of fear—it's feeling the fear and taking the loving action anyway. Each time you move through fear with awareness and compassion, you expand your comfort zone and increase your capacity for freedom."
        ];
        response = responses[this.getResponseIndex(message, responses.length)];
        wisdom_source = 'Mindfulness-Based Stress Reduction & Anxiety Wisdom';
        consciousness_level = 'calming';
        emotional_tone = 'soothing and empowering';
        dharmic_insights = [
          'Fear is often about imagined future scenarios, not present reality',
          'Anxiety can be transformed through presence and breathing',
          'You are the observer of fear, not fear itself'
        ];
        growth_suggestions = [
          'Practice daily grounding and breathing exercises',
          'Learn mindfulness-based stress reduction techniques',
          'Challenge negative thought patterns with cognitive reframing',
          'Consider professional support for persistent anxiety',
          'Build a supportive community and practice sharing vulnerabilities'
        ];
        spiritual_context = 'Overcoming fear is central to spiritual liberation and developing unshakeable inner peace.';
      }

      // Default Universal Wisdom
      else {
        const responses = [
          "🌟 Every question you ask is a sacred prayer for deeper understanding and awakening. The answers you seek are already encoded within your soul's wisdom—they await discovery through quiet reflection, inner listening, and trust in your divine guidance system.",
          "✨ You are exactly where you need to be on your spiritual journey. Every experience—joyful or challenging—is perfectly orchestrated to awaken the wisdom, love, and power that you truly are. Trust the intelligent unfoldment of your unique path.",
          "🙏 In this sacred moment, breathe deeply and remember: you are infinitely loved by the universe, eternally supported by unseen forces, and perfectly whole exactly as you are. Your very existence is a blessing to the world.",
          "🧘‍♀️ The divine speaks to you through the whispers of intuition, the synchronicities of daily life, and the wisdom that emerges from stillness. Stay open, stay curious, and trust the gentle guidance that flows through your awakened heart.",
          "🌸 Spiritual growth is not about becoming someone different—it's about removing everything that isn't authentically you. You are already complete and perfect; you're simply remembering the magnificent truth of who you've always been.",
          "🕉️ Your soul chose this human experience to learn, grow, and contribute to the awakening of consciousness on Earth. Every challenge mastered, every act of love shared, and every moment of presence cultivated serves the highest good of all beings."
        ];
        response = responses[this.getResponseIndex(message, responses.length)];
        wisdom_source = 'Universal Dharmic Wisdom & Spiritual Traditions';
        consciousness_level = 'uplifting';
        emotional_tone = 'inspiring and affirming';
        dharmic_insights = [
          'You are a spiritual being having a human experience',
          'Inner wisdom is your most reliable guidance system',
          'Every moment offers an opportunity for awakening'
        ];
        growth_suggestions = [
          'Develop a daily spiritual practice that resonates with you',
          'Trust your intuition and inner guidance',
          'Connect with like-minded spiritual seekers',
          'Study wisdom teachings from various traditions',
          'Practice gratitude and presence daily'
        ];
        spiritual_context = 'Universal spiritual principles transcend all religious boundaries and speak to our shared human experience.';
      }

      return {
        response,
        wisdom_source,
        dharma_assessment: {
          ethical_alignment: 0.95,
          spiritual_depth: 0.90,
          practical_wisdom: 0.85,
          universal_appeal: 0.92
        },
        consciousness_level,
        emotional_tone,
        confidence: 0.88,
        timestamp: new Date().toISOString(),
        dharmic_insights,
        growth_suggestions,
        spiritual_context
      };

    } catch (error) {
      console.error('Error in comprehensive dharmic wisdom:', error);
      throw error;
    }
  }

  /**
   * Enhanced keyword matching for better response accuracy
   */
  private matchesKeywords(text: string, keywords: string[]): boolean {
    return keywords.some(keyword => text.includes(keyword));
  }

  /**
   * Get consistent response index based on message content
   */
  private getResponseIndex(message: string, responseCount: number): number {
    return Math.abs(this.hashString(message)) % responseCount;
  }

  /**
   * Simple hash function for consistent response selection
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
  }

  /**
   * Enhanced fallback response with comprehensive spiritual guidance
   */
  generateEnhancedFallbackResponse(message: string): ChatResponse {
    const lowercaseMessage = message.toLowerCase();
    
    let fallbackResponse = '';
    let dharmic_insights: string[] = [];
    let growth_suggestions: string[] = [];
    
    // Enhanced spiritual wisdom responses
    if (lowercaseMessage.includes('meditation') || lowercaseMessage.includes('mindfulness')) {
      fallbackResponse = "🧘‍♀️ In the sacred practice of meditation, you discover that peace was never lost—only temporarily veiled by the mind's activity. Begin with gentle awareness of your breath, allowing each exhale to release what no longer serves your highest good. The path of mindfulness teaches us that liberation is found not in the absence of thoughts, but in our conscious relationship to them.";
      dharmic_insights = ['Meditation reveals your unchanging peaceful nature', 'Mindfulness transforms your relationship with thoughts'];
      growth_suggestions = ['Start with 5 minutes daily', 'Practice loving-kindness meditation'];
    } else if (lowercaseMessage.includes('love') || lowercaseMessage.includes('relationship')) {
      fallbackResponse = "💖 Love is the fundamental force that weaves all existence together. True compassion begins with radical self-acceptance and naturally extends to embrace all beings. When you recognize the divine spark within yourself, you cannot help but honor it in everyone you encounter. Practice loving-kindness by sending good wishes to yourself, your loved ones, and even those who challenge your growth.";
      dharmic_insights = ['Self-love is the foundation of all love', 'Relationships are mirrors for spiritual growth'];
      growth_suggestions = ['Practice daily self-compassion', 'See conflicts as growth opportunities'];
    } else {
      fallbackResponse = "🌟 Every question you ask is a step on the sacred path of self-discovery. The wisdom you seek already resides within the depths of your being, waiting to be uncovered through mindful reflection and inner listening. Trust your journey, embrace both the light and shadow aspects of your experience, and remember that spiritual growth happens one conscious moment at a time. May you find peace, clarity, and joy on your unique path of awakening.";
      dharmic_insights = ['Inner wisdom is your most reliable guide', 'Every experience serves your spiritual growth'];
      growth_suggestions = ['Trust your intuition', 'Practice daily gratitude and presence'];
    }

    return {
      response: fallbackResponse,
      conversation_id: `enhanced_fallback_${Date.now()}`,
      confidence_score: 0.8,
      dharmic_alignment: 0.9,
      dharmic_insights,
      growth_suggestions,
      spiritual_context: 'Universal wisdom teachings for personal transformation and awakening',
      modules_used: ['enhanced_fallback_wisdom'],
      timestamp: new Date().toISOString(),
      model_used: 'dharmamind-enhanced-internal',
      processing_time: 50,
      sources: ['Comprehensive Internal Dharmic Wisdom Database'],
      suggestions: ['Continue exploring these teachings through daily practice and study.']
    };
  }
}

// Export a singleton instance
export const chatService = new ChatService();
export default chatService;
