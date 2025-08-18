/**
 * DharmaMind Brand Website Authentication Service
 * Centralized authentication through Chat App (dharmamind.ai)
 */

interface DharmaMindUser {
  user_id: string;
  email: string;
  first_name: string;
  last_name: string;
  subscription_plan: 'basic' | 'pro' | 'max' | 'enterprise';
  status: string;
  email_verified: boolean;
  auth_provider: string;
  avatar_url?: string;
  joined_date?: string;
}

interface AuthResponse {
  success: boolean;
  user?: DharmaMindUser;
  session_token?: string;
  expires_at?: number;
  message?: string;
  error?: string;
}

interface BrandAuthService {
  checkAutoLogin(): Promise<AuthResponse | null>;
  getCurrentUser(): Promise<DharmaMindUser | null>;
  isAuthenticated(): boolean;
  logout(): void;
  redirectToCentralAuth(mode: 'login' | 'signup', returnUrl?: string): void;
}

class DharmaMindBrandAuth implements BrandAuthService {
  private centralAuthUrl: string;
  private apiUrl: string;
  
  constructor() {
    // Central Auth Hub (Chat App)
    this.centralAuthUrl = process.env.NEXT_PUBLIC_CENTRAL_AUTH_URL || 'https://dharmamind.ai';
    // Brand API endpoint
    this.apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }

  /**
   * Check for auto-login from central auth
   */
  async checkAutoLogin(): Promise<AuthResponse | null> {
    // Check for token in URL parameters (SSO from central auth)
    if (typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      const ssoToken = urlParams.get('dharmamind_token');
      const authCode = urlParams.get('auth_code');
      
      if (ssoToken) {
        // Clean URL
        window.history.replaceState({}, document.title, window.location.pathname);
        return await this.loginWithToken(ssoToken);
      }
      
      if (authCode) {
        // Exchange auth code for token
        try {
          const response = await fetch(`${this.apiUrl}/auth/exchange-token`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
              auth_code: authCode,
              platform: 'brand'
            })
          });
          
          if (response.ok) {
            const data = await response.json();
            // Clean URL
            window.history.replaceState({}, document.title, window.location.pathname);
            return await this.loginWithToken(data.token);
          }
        } catch (error) {
          console.error('Token exchange error:', error);
        }
      }
    }

    // Check for existing valid session
    const token = this.getStoredToken();
    if (token) {
      return await this.loginWithToken(token);
    }

    return null;
  }

  /**
   * Login with token from central auth
   */
  async loginWithToken(token: string): Promise<AuthResponse> {
    try {
      // Verify token with central auth
      const response = await fetch(`${this.centralAuthUrl}/api/auth/verify`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept': 'application/json'
        }
      });

      if (!response.ok) {
        return { success: false, error: 'Invalid or expired session' };
      }

      const data = await response.json();
      
      if (data.user) {
        this.setTokens(token);
        this.setUser(data.user);
        
        return { 
          success: true, 
          user: this.mapToBrandUser(data.user),
          session_token: token 
        };
      }

      return { success: false, error: 'User verification failed' };
    } catch (error) {
      console.error('Token verification error:', error);
      return { success: false, error: 'Session verification failed' };
    }
  }

  /**
   * Get current authenticated user
   */
  async getCurrentUser(): Promise<DharmaMindUser | null> {
    const token = this.getStoredToken();
    if (!token) return null;

    try {
      const response = await fetch(`${this.centralAuthUrl}/api/auth/me`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept': 'application/json'
        }
      });

      if (!response.ok) {
        this.logout();
        return null;
      }

      const data = await response.json();
      return this.mapToBrandUser(data.user);
    } catch (error) {
      console.error('Get user error:', error);
      this.logout();
      return null;
    }
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    const token = this.getStoredToken();
    const user = this.getStoredUser();
    return !!(token && user);
  }

  /**
   * Logout user
   */
  logout(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('dharmamind_token');
      localStorage.removeItem('dharmamind_user');
      
      // Optional: Call logout endpoint
      const token = this.getStoredToken();
      if (token) {
        fetch(`${this.centralAuthUrl}/api/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }).catch(err => console.log('Logout cleanup error:', err));
      }
    }
  }

  /**
   * Redirect to central auth hub
   */
  redirectToCentralAuth(mode: 'login' | 'signup' = 'login', returnUrl?: string): void {
    const currentUrl = typeof window !== 'undefined' ? window.location.origin : '';
    const finalReturnUrl = returnUrl || '/';
    const brandReturnUrl = `${currentUrl}/auth?returnUrl=${encodeURIComponent(finalReturnUrl)}`;
    
    // Central auth hub (Chat App) with brand return URL
    const centralAuthUrl = `${this.centralAuthUrl}/auth?mode=${mode}&returnUrl=${encodeURIComponent(brandReturnUrl)}&platform=brand&source=${encodeURIComponent(currentUrl)}`;
    window.location.href = centralAuthUrl;
  }

  /**
   * Get stored token
   */
  getStoredToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem('dharmamind_token');
  }

  /**
   * Store tokens
   */
  private setTokens(token: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('dharmamind_token', token);
    }
  }

  /**
   * Store user data
   */
  private setUser(user: any): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('dharmamind_user', JSON.stringify(user));
    }
  }

  /**
   * Get stored user data
   */
  private getStoredUser(): any {
    if (typeof window === 'undefined') return null;
    const userData = localStorage.getItem('dharmamind_user');
    return userData ? JSON.parse(userData) : null;
  }

  /**
   * Map user data to brand format
   */
  private mapToBrandUser(user: any): DharmaMindUser {
    return {
      user_id: user.user_id || user.id,
      email: user.email,
      first_name: user.first_name || user.name?.split(' ')[0] || '',
      last_name: user.last_name || user.name?.split(' ').slice(1).join(' ') || '',
      subscription_plan: user.subscription_plan || user.plan || 'basic',
      status: user.status || 'active',
      email_verified: user.email_verified || false,
      auth_provider: user.auth_provider || 'dharmamind',
      avatar_url: user.avatar_url || this.generateAvatarUrl(user.email),
      joined_date: user.created_at || user.joined_date || new Date().toISOString()
    };
  }

  /**
   * Generate avatar URL
   */
  private generateAvatarUrl(email: string): string {
    const initial = email.charAt(0).toUpperCase();
    return `data:image/svg+xml;base64,${btoa(`
      <svg width="40" height="40" xmlns="http://www.w3.org/2000/svg">
        <circle cx="20" cy="20" r="20" fill="#10b981"/>
        <text x="20" y="28" text-anchor="middle" fill="white" font-family="Arial" font-size="16" font-weight="bold">
          ${initial}
        </text>
      </svg>
    `)}`;
  }
}

// Export singleton instance
const brandAuth = new DharmaMindBrandAuth();
export default brandAuth;
export type { DharmaMindUser, AuthResponse };
