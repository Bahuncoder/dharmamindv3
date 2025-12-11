import axios from 'axios';

// API Gateway URL - All requests go through the backend
const API_GATEWAY_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
  rishi_id?: string;
  rishi_name?: string;
  dharmic_insights?: string[];
  growth_suggestions?: string[];
  spiritual_context?: string;
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
}

class ChatService {
  private authToken: string | null = null;

  constructor() {
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
   * Send a chat message through the API Gateway
   */
  async sendMessage(
    message: string,
    conversationId?: string,
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
   * Enhanced message sending with dharmic insights
   */
  async sendMessageEnhanced(
    message: string,
    conversationId?: string,
    role: string = 'user',
    useDharmicChat: boolean = true
  ): Promise<ChatResponse> {
    // Use the standard sendMessage with enhanced metadata
    const response = await this.sendMessage(message, conversationId);

    // Add dharmic insights if using dharmic chat mode
    if (useDharmicChat && response) {
      return {
        ...response,
        dharmic_insights: response.metadata?.dharmic_insights || [],
        growth_suggestions: response.metadata?.growth_suggestions || [],
        spiritual_context: response.metadata?.spiritual_context || '',
        metadata: {
          ...response.metadata,
          dharmic_chat_enabled: true,
          confidence: response.confidence_score || 0.9,
          dharmic_alignment: response.dharmic_alignment || 0.85
        }
      };
    }

    return response;
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
      return null;
    }
  }

  /**
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
    };
  }

  /**
   * Generate an enhanced fallback response with spiritual wisdom
   */
  generateEnhancedFallbackResponse(message: string): ChatResponse {
    const spiritualResponses = [
      {
        response: "🙏 Thank you for your question. While our service is reconnecting, I invite you to sit in stillness for a moment. Often, the answers we seek are already within us, waiting to be discovered through quiet reflection.",
        dharmic_insights: ["Patience is a virtue that reveals hidden wisdom", "Inner peace comes from accepting what is"],
        growth_suggestions: ["Practice 5 minutes of mindful breathing", "Journal your thoughts to gain clarity"],
      },
      {
        response: "🕉️ Your inquiry shows a seeking heart. Though our full service is momentarily unavailable, remember that the greatest teachers have always pointed us back to ourselves. What truth resonates within you regarding your question?",
        dharmic_insights: ["The guru within knows the path", "Self-inquiry is the highest sadhana"],
        growth_suggestions: ["Contemplate your question in meditation", "Trust your intuition"],
      },
      {
        response: "✨ I acknowledge your thoughtful question. While we restore our connection, consider this: every challenge is an opportunity for growth. What wisdom might this moment of patience be offering you?",
        dharmic_insights: ["Every moment holds a teaching", "Obstacles are stepping stones to understanding"],
        growth_suggestions: ["Embrace this pause as sacred time", "Reflect on what you truly need right now"],
      },
    ];

    const selected = spiritualResponses[Math.floor(Math.random() * spiritualResponses.length)];

    return {
      response: selected.response,
      conversation_id: `spiritual_${Date.now()}`,
      message_id: `spiritual_${Date.now()}`,
      confidence_score: 0.7,
      dharmic_insights: selected.dharmic_insights,
      growth_suggestions: selected.growth_suggestions,
      spiritual_context: "This response is generated with spiritual wisdom to guide you during service interruption.",
      metadata: {
        source: 'spiritual_fallback',
        reason: 'service_unavailable',
        original_message: message.substring(0, 100)
      }
    };
  }
}

// Export singleton instance
const chatService = new ChatService();
export default chatService;

// Named export for flexibility
export { ChatService };
