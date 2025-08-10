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
  confidence_score?: number;
  dharmic_alignment?: number;
  modules_used?: string[];
  timestamp?: string;
  model_used?: string;
  processing_time?: number;
  sources?: string[];
  suggestions?: string[];
}

export interface WisdomRequest {
  query: string;
  context?: string;
  user_preferences?: Record<string, any>;
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
   * Send a chat message to the backend
   */
  async sendMessage(
    message: string,
    conversationId?: string,
    userId?: string
  ): Promise<ChatResponse> {
    try {
      const payload = {
        message,
        conversation_id: conversationId,
        user_id: userId || 'anonymous',
        timestamp: new Date().toISOString()
      };

      console.log('Sending chat message to backend:', `${BACKEND_URL}/chat`);
      console.log('Payload:', payload);

      const response = await axios.post(
        `${BACKEND_URL}/chat`,
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
    };
  }
}

// Export a singleton instance
export const chatService = new ChatService();
export default chatService;
