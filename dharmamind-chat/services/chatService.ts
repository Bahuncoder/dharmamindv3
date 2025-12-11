import axios from 'axios';

<<<<<<< HEAD
// API Gateway URL - All requests go through the backend
const API_GATEWAY_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
=======
// Use the backend API URL - port 8000 where the comprehensive backend is running
const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc

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
<<<<<<< HEAD
  rishi_id?: string;
  rishi_name?: string;
  usage?: {
    messages_used: number;
    limit: number;
    plan: string;
  };
}

export interface RishiInfo {
  id: string;
  name: string;
  domain: string;
  description?: string;
  available: boolean;
}

export interface SubscriptionStatus {
  user_id: string;
  plan: string;
  authenticated: boolean;
  usage: {
    messages_used: number;
    limit: number;
    remaining: number;
  };
  features: {
    rishis: boolean;
    history: boolean;
    api_access: boolean;
    priority_support: boolean;
  };
=======
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
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
}

class ChatService {
  private authToken: string | null = null;

  constructor() {
<<<<<<< HEAD
=======
    // Get auth token from localStorage if available
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
<<<<<<< HEAD
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }
=======

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
    return headers;
  }

  /**
<<<<<<< HEAD
   * Send a chat message through the API Gateway
=======
   * Send a chat message to the backend - Now using internal spiritual processing
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
   */
  async sendMessage(
    message: string,
    conversationId?: string,
<<<<<<< HEAD
    rishiId?: string
  ): Promise<ChatResponse> {
    try {
      console.log('Sending message through API Gateway...');
      
      const response = await axios.post(`${API_GATEWAY_URL}/api/v1/chat`, {
        message,
        conversation_id: conversationId,
        rishi_id: rishiId,
        temperature: 0.8,
        max_tokens: 1024
      }, {
        timeout: 30000,
        headers: this.getHeaders()
      });

      if (response.status === 200) {
        const result = response.data;
        console.log('Chat response received successfully');
        
        return {
          response: result.message,
          conversation_id: result.conversation_id,
          message_id: result.message_id,
          timestamp: result.timestamp,
          confidence_score: result.confidence,
          rishi_id: result.rishi_id,
          rishi_name: result.rishi_name,
          usage: result.usage,
          metadata: result.metadata,
          model_used: result.metadata?.model || 'DharmaLLM'
        };
      }
      
      throw new Error(`API returned status ${response.status}`);
      
    } catch (error: any) {
      console.error('Chat error:', error);
      
      // Handle rate limit error
      if (error.response?.status === 429) {
        return {
          response: 'You have reached your message limit. Please upgrade your plan for unlimited access.',
          conversation_id: conversationId,
          message_id: `error_${Date.now()}`,
          metadata: {
            error: 'rate_limit',
            upgrade_url: '/pricing'
          }
        };
      }
      
      // Return fallback response
      return this.getFallbackResponse(message, conversationId);
    }
  }

  /**
   * Get available Rishis/Guides
   */
  async getRishis(): Promise<RishiInfo[]> {
    try {
      const response = await axios.get(`${API_GATEWAY_URL}/api/v1/rishis`, {
        headers: this.getHeaders(),
        timeout: 10000
      });
      
      return response.data.rishis || [];
    } catch (error) {
      console.error('Failed to fetch rishis:', error);
      // Return default list
      return [
        { id: 'vasishtha', name: 'Vasishtha', domain: 'Dharma & Ethics', available: true },
        { id: 'vishwamitra', name: 'Vishwamitra', domain: 'Willpower & Achievement', available: true },
        { id: 'bharadvaja', name: 'Bharadvaja', domain: 'Knowledge & Learning', available: true },
        { id: 'gautama', name: 'Gautama', domain: 'Justice & Logic', available: true },
        { id: 'jamadagni', name: 'Jamadagni', domain: 'Discipline & Focus', available: true },
        { id: 'kashyapa', name: 'Kashyapa', domain: 'Creation & Nurturing', available: true },
        { id: 'atri', name: 'Atri', domain: 'Balance & Harmony', available: true },
        { id: 'agastya', name: 'Agastya', domain: 'Medicine & Healing', available: true },
        { id: 'narada', name: 'Narada', domain: 'Devotion & Communication', available: true },
      ];
    }
  }

  /**
   * Get subscription status
   */
  async getSubscriptionStatus(): Promise<SubscriptionStatus | null> {
    try {
      const response = await axios.get(`${API_GATEWAY_URL}/api/v1/subscription/status`, {
        headers: this.getHeaders(),
        timeout: 10000
      });
      return response.data;
    } catch (error) {
      console.error('Failed to fetch subscription status:', error);
=======
    userId?: string
  ): Promise<ChatResponse> {
    try {
      // First try internal spiritual processing
      console.log('Using internal spiritual processing...');
      const spiritualResponse = await this.getInternalSpiritualWisdom(message, userId);
      
      if (spiritualResponse) {
        return {
          response: spiritualResponse.response,
          conversation_id: conversationId || 'internal-spiritual',
          message_id: `spiritual-${Date.now()}`,
          timestamp: spiritualResponse.timestamp,
          metadata: {
            source: 'internal-chakra-modules',
            wisdom_source: spiritualResponse.wisdom_source,
            consciousness_level: spiritualResponse.consciousness_level,
            dharma_assessment: spiritualResponse.dharma_assessment,
            confidence: spiritualResponse.confidence
          }
        };
      }

      // Fallback to original chat endpoint if internal processing fails
      const payload = {
        message,
        conversation_id: conversationId,
        user_id: userId || 'anonymous',
        timestamp: new Date().toISOString()
      };

      console.log('Sending chat message to backend:', `${BACKEND_URL}/api/v1/chat`);
      console.log('Payload:', payload);

      const response = await axios.post(
        `${BACKEND_URL}/api/v1/chat`,
        payload,
        {
          headers: this.getHeaders(),
          timeout: 45000, // 45 second timeout for LLM processing
        }
      );

      console.log('Backend response:', response.data);
      return response.data;
    } catch (error) {
      console.error('Error sending chat message:', error);
      
      if (axios.isAxiosError(error)) {
        if (error.response) {
          throw new Error(`Backend error: ${error.response.data?.detail || error.response.statusText}`);
        } else if (error.request) {
          throw new Error('Unable to connect to backend. Please check if the server is running.');
        }
      }
      
      throw new Error('An unexpected error occurred while sending the message.');
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
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
      return null;
    }
  }

  /**
<<<<<<< HEAD
   * Get subscription plans
   */
  async getSubscriptionPlans(): Promise<any[]> {
    try {
      const response = await axios.get(`${API_GATEWAY_URL}/api/v1/subscription/plans`, {
        timeout: 10000
      });
      return response.data.plans || [];
    } catch (error) {
      console.error('Failed to fetch plans:', error);
      return [];
    }
  }

  /**
   * Check API health
   */
  async checkHealth(): Promise<{ gateway: boolean; ai: boolean }> {
    try {
      const response = await axios.get(`${API_GATEWAY_URL}/api/v1/health`, {
        timeout: 5000
      });
      return {
        gateway: response.data.status === 'healthy',
        ai: response.data.dharmallm === 'healthy'
      };
    } catch (error) {
      return { gateway: false, ai: false };
    }
  }

  /**
   * Fallback response when API is unavailable
   */
  private getFallbackResponse(message: string, conversationId?: string): ChatResponse {
    const fallbackResponses = [
      "I appreciate your question. While our AI service is temporarily unavailable, I encourage you to reflect on the wisdom within yourself. What does your inner voice tell you about this matter?",
      "Thank you for reaching out. Our service is being updated to serve you better. In the meantime, consider journaling your thoughts - writing often brings clarity.",
      "Your question shows thoughtfulness. While I cannot provide a full response right now, remember that patience and reflection are themselves forms of wisdom.",
    ];
    
    const randomResponse = fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
    
    return {
      response: randomResponse,
      conversation_id: conversationId || `fallback_${Date.now()}`,
      message_id: `fallback_${Date.now()}`,
      confidence_score: 0.5,
      metadata: {
        source: 'fallback',
        reason: 'service_unavailable'
      }
=======
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
   * Health check for the backend
   */
  async healthCheck(): Promise<{ status: string; version?: string }> {
    try {
      const response = await axios.get(`${BACKEND_URL}/health`, {
        timeout: 5000, // 5 second timeout for health check
      });

      return response.data;
    } catch (error) {
      console.error('Backend health check failed:', error);
      throw new Error('Backend is not available');
    }
  }

  /**
   * Generate a fallback response when backend is unavailable
   */
  generateFallbackResponse(message: string): ChatResponse {
    const lowercaseMessage = message.toLowerCase();
    
    let fallbackResponse = '';
    
    // Spiritual wisdom responses based on common topics
    if (lowercaseMessage.includes('meditation') || lowercaseMessage.includes('mindfulness')) {
      fallbackResponse = "In the stillness of meditation, we find the eternal presence that has always been within us. Begin with just a few minutes of focused breathing, allowing your mind to settle like a clear mountain lake. The path of mindfulness teaches us that peace is not found in the absence of thoughts, but in our relationship to them.";
    } else if (lowercaseMessage.includes('suffering') || lowercaseMessage.includes('pain') || lowercaseMessage.includes('difficult')) {
      fallbackResponse = "Suffering, as the Buddha taught, is part of the human experience, but it is not our permanent state. Every challenge carries within it the seeds of wisdom and growth. Like a lotus that blooms from muddy waters, our struggles can become the very foundation of our spiritual awakening. Be gentle with yourself during difficult times.";
    } else if (lowercaseMessage.includes('love') || lowercaseMessage.includes('compassion') || lowercaseMessage.includes('kindness')) {
      fallbackResponse = "Love is the fundamental force that connects all beings. True compassion begins with self-acceptance and extends naturally to others. When we recognize the divine spark in ourselves, we cannot help but see it in everyone we meet. Practice loving-kindness by sending good wishes to yourself, your loved ones, and even those who challenge you.";
    } else if (lowercaseMessage.includes('fear') || lowercaseMessage.includes('anxiety') || lowercaseMessage.includes('worry')) {
      fallbackResponse = "Fear is often the shadow cast by our attachment to outcomes we cannot control. Remember that you are much more than your thoughts and emotions. Like clouds passing through the sky, fears arise and dissolve naturally when we don't resist them. Ground yourself in the present moment through breath and mindful awareness.";
    } else if (lowercaseMessage.includes('purpose') || lowercaseMessage.includes('meaning') || lowercaseMessage.includes('direction')) {
      fallbackResponse = "Your purpose unfolds naturally when you align with your authentic self. Listen deeply to your heart's calling, not the voices of expectation from others. Every experience, whether perceived as success or failure, contributes to your spiritual growth. Trust the journey, even when the path seems unclear.";
    } else {
      fallbackResponse = "Every question you ask is a step on the path of self-discovery. The answers you seek already reside within you, waiting to be uncovered through mindful reflection and inner wisdom. Trust your journey, embrace both the light and shadow aspects of your experience, and remember that spiritual growth happens in the present moment. May you find peace and clarity on your path.";
    }

    return {
      response: fallbackResponse,
      conversation_id: `fallback_${Date.now()}`,
      confidence_score: 0.7,
      dharmic_alignment: 0.8,
      modules_used: ['fallback_wisdom'],
      timestamp: new Date().toISOString(),
      model_used: 'dharmamind-fallback',
      processing_time: 50,
      sources: ['Internal Wisdom Database'],
      suggestions: ['When the backend is available, you will receive more personalized guidance.']
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
    };
  }
}

<<<<<<< HEAD
// Export singleton instance
const chatService = new ChatService();
export default chatService;

// Named export for flexibility
export { ChatService };
=======
// Export a singleton instance
export const chatService = new ChatService();
export default chatService;
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
