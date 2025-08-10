/**
 * Authentication Service for DharmaMind
 * Handles user authentication, registration, and token management
 */

interface User {
  user_id: string;
  email: string;
  first_name: string;
  last_name: string;
  subscription_plan: 'basic' | 'pro' | 'max' | 'enterprise';
  status: string;
  email_verified: boolean;
  auth_provider: string;
}

interface AuthResponse {
  success: boolean;
  user?: User;
  session_token?: string;
  expires_at?: number;
  message?: string;
  error?: string;
}

interface LoginCredentials {
  email: string;
  password: string;
}

interface RegisterData {
  first_name: string;
  last_name: string;
  email: string;
  password: string;
  accept_terms: boolean;
  accept_privacy: boolean;
  marketing_consent?: boolean;
}

class AuthService {
  private baseUrl: string;
  
  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }

  /**
   * Register a new user
   */
  async register(userData: RegisterData): Promise<AuthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      if (!response.ok) {
        const error = await response.json();
        return { success: false, error: error.detail || 'Registration failed' };
      }

      const data = await response.json();
      
      // Store tokens if registration is successful
      if (data.session_token) {
        this.setTokens(data.session_token);
        this.setUser(data.user);
      }

      return { success: true, ...data };
    } catch (error) {
      console.error('Registration error:', error);
      return { success: false, error: 'Network error. Please try again.' };
    }
  }

  /**
   * Login user with email and password
   */
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        const error = await response.json();
        return { success: false, error: error.detail || 'Login failed' };
      }

      const data = await response.json();
      
      // Store tokens if login is successful
      if (data.session_token) {
        this.setTokens(data.session_token);
        this.setUser(data.user);
      }

      return { success: true, ...data };
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, error: 'Network error. Please try again.' };
    }
  }

  /**
   * Demo login for testing purposes
   */
  async demoLogin(plan: 'basic' | 'pro' | 'max' | 'enterprise' = 'basic'): Promise<AuthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/auth/demo-login?plan=${plan}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const error = await response.json();
        return { success: false, error: error.detail || 'Demo login failed' };
      }

      const data = await response.json();
      
      // Store tokens if demo login is successful
      if (data.session_token) {
        this.setTokens(data.session_token);
        this.setUser(data.user);
      }

      return { success: true, ...data };
    } catch (error) {
      console.error('Demo login error:', error);
      return { success: false, error: 'Network error. Please try again.' };
    }
  }

  /**
   * Logout user
   */
  async logout(): Promise<void> {
    try {
      const token = this.getAccessToken();
      if (token) {
        // Call backend logout endpoint
        await fetch(`${this.baseUrl}/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear local storage regardless of backend response
      this.clearTokens();
      this.clearUser();
    }
  }

  /**
   * Get current user from localStorage
   */
  getCurrentUser(): User | null {
    if (typeof window === 'undefined') return null;
    
    const userStr = localStorage.getItem('dharmamind_user');
    if (!userStr) return null;

    try {
      return JSON.parse(userStr);
    } catch {
      return null;
    }
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!this.getAccessToken() && !!this.getCurrentUser();
  }

  /**
   * Get access token
   */
  getAccessToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem('dharmamind_session_token');
  }

  /**
   * Make authenticated API request
   */
  async authenticatedFetch(url: string, options: RequestInit = {}): Promise<Response> {
    const token = this.getAccessToken();
    
    if (!token) {
      throw new Error('No access token available');
    }

    // Make request with current token
    const response = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        'Authorization': `Bearer ${token}`,
      },
    });

    // If token expired, logout user
    if (response.status === 401) {
      this.logout();
      throw new Error('Authentication expired');
    }

    return response;
  }

  // Private methods
  private setTokens(sessionToken: string): void {
    if (typeof window === 'undefined') return;
    localStorage.setItem('dharmamind_session_token', sessionToken);
  }

  private setUser(user: User): void {
    if (typeof window === 'undefined') return;
    localStorage.setItem('dharmamind_user', JSON.stringify(user));
  }

  private clearTokens(): void {
    if (typeof window === 'undefined') return;
    localStorage.removeItem('dharmamind_session_token');
  }

  private clearUser(): void {
    if (typeof window === 'undefined') return;
    localStorage.removeItem('dharmamind_user');
  }
}

// Export singleton instance
export const authService = new AuthService();

// Export types
export type { User, AuthResponse, LoginCredentials, RegisterData };
