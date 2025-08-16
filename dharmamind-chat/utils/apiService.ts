// API service for DharmaMind frontend integration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8003';

interface ApiResponse<T> {
  data?: T;
  error?: string;
  success: boolean;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ChatRequest {
  message: string;
  conversation_id?: string;
  context?: string;
}

interface ChatResponse {
  message: string;
  conversation_id: string;
  sources?: string[];
  metadata?: any;
}

interface WisdomQuote {
  quote: string;
  source: string;
  category: string;
  wisdom_level: number;
}

interface UserProfile {
  id: string;
  email: string;
  name: string;
  subscription_plan: 'free' | 'professional' | 'enterprise';
  subscription_status: 'active' | 'canceled' | 'past_due';
  spiritual_level: number;
  preferences: {
    wisdom_traditions: string[];
    notification_settings: any;
  };
}

class ApiService {
  private getAuthHeaders(): HeadersInit {
    const token = localStorage.getItem('access_token');
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` })
    };
  }

  private async handleResponse<T>(response: Response): Promise<ApiResponse<T>> {
    try {
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return {
          success: false,
          error: errorData.detail || `HTTP ${response.status}: ${response.statusText}`
        };
      }

      const data = await response.json();
      return {
        success: true,
        data
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  // Chat API integration
  async sendChatMessage(request: ChatRequest): Promise<ApiResponse<ChatResponse>> {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(request)
      });

      return this.handleResponse<ChatResponse>(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to send chat message'
      };
    }
  }

  // Get wisdom quote
  async getWisdomQuote(category?: string): Promise<ApiResponse<WisdomQuote>> {
    try {
      const url = category 
        ? `${API_BASE_URL}/wisdom?category=${encodeURIComponent(category)}`
        : `${API_BASE_URL}/wisdom`;
      
      const response = await fetch(url, {
        headers: this.getAuthHeaders()
      });

      return this.handleResponse<WisdomQuote>(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get wisdom quote'
      };
    }
  }

  // User authentication
  async login(email: string, password: string): Promise<ApiResponse<{ access_token: string; user: UserProfile }>> {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const result = await this.handleResponse<{ access_token: string; user: UserProfile }>(response);
      
      if (result.success && result.data) {
        localStorage.setItem('access_token', result.data.access_token);
        localStorage.setItem('user_profile', JSON.stringify(result.data.user));
      }

      return result;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Login failed'
      };
    }
  }

  async register(userData: {
    email: string;
    password: string;
    name: string;
    spiritual_interests?: string[];
  }): Promise<ApiResponse<{ access_token: string; user: UserProfile }>> {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData)
      });

      const result = await this.handleResponse<{ access_token: string; user: UserProfile }>(response);
      
      if (result.success && result.data) {
        localStorage.setItem('access_token', result.data.access_token);
        localStorage.setItem('user_profile', JSON.stringify(result.data.user));
      }

      return result;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Registration failed'
      };
    }
  }

  // User profile management
  async getUserProfile(): Promise<ApiResponse<UserProfile>> {
    try {
      const response = await fetch(`${API_BASE_URL}/user/profile`, {
        headers: this.getAuthHeaders()
      });

      return this.handleResponse<UserProfile>(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get user profile'
      };
    }
  }

  async updateUserProfile(updates: Partial<UserProfile>): Promise<ApiResponse<UserProfile>> {
    try {
      const response = await fetch(`${API_BASE_URL}/user/profile`, {
        method: 'PUT',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(updates)
      });

      return this.handleResponse<UserProfile>(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update profile'
      };
    }
  }

  // Subscription management
  async createSubscription(data: {
    plan_id: string;
    payment_method_id?: string;
    metadata?: any;
  }): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${API_BASE_URL}/subscription/create`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(data)
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create subscription'
      };
    }
  }

  async cancelSubscription(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${API_BASE_URL}/subscription/cancel`, {
        method: 'POST',
        headers: this.getAuthHeaders()
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to cancel subscription'
      };
    }
  }

  async getSubscriptionStatus(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${API_BASE_URL}/subscription/status`, {
        headers: this.getAuthHeaders()
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get subscription status'
      };
    }
  }

  // Payment methods
  async addPaymentMethod(data: {
    method_type: 'credit_card' | 'bank_account';
    card_details?: {
      number: string;
      exp_month: number;
      exp_year: number;
      cvc: string;
      cardholder_name: string;
    };
    billing_address: any;
    set_as_default?: boolean;
  }): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${API_BASE_URL}/payment/methods`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(data)
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to add payment method'
      };
    }
  }

  async getPaymentMethods(): Promise<ApiResponse<any[]>> {
    try {
      const response = await fetch(`${API_BASE_URL}/payment/methods`, {
        headers: this.getAuthHeaders()
      });

      return this.handleResponse<any[]>(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get payment methods'
      };
    }
  }

  // System health
  async getSystemHealth(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to check system health'
      };
    }
  }

  // Logout
  logout(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_profile');
  }

  // Check if user is authenticated
  isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token');
  }

  // Get current user from localStorage
  getCurrentUser(): UserProfile | null {
    const userStr = localStorage.getItem('user_profile');
    if (userStr) {
      try {
        return JSON.parse(userStr);
      } catch {
        return null;
      }
    }
    return null;
  }
}

// Export singleton instance
export const apiService = new ApiService();

// Export types for use in components
export type {
  ApiResponse,
  ChatMessage,
  ChatRequest,
  ChatResponse,
  WisdomQuote,
  UserProfile
};
