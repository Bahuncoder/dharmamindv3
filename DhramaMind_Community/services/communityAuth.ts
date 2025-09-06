/**
 * DharmaMind Community Authentication Service
 * Connects with main DharmaMind platform authentication system
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
  community_role?: 'member' | 'contributor' | 'moderator' | 'guide';
}

interface AuthResponse {
  success: boolean;
  user?: DharmaMindUser;
  session_token?: string;
  expires_at?: number;
  message?: string;
  error?: string;
}

interface LoginCredentials {
  email: string;
  password: string;
}

interface CommunityAuthService {
  login(credentials: LoginCredentials): Promise<AuthResponse>;
  loginWithToken(token: string): Promise<AuthResponse>;
  logout(): void;
  getCurrentUser(): Promise<DharmaMindUser | null>;
  isAuthenticated(): boolean;
  getStoredToken(): string | null;
}

class DharmaMindCommunityAuth implements CommunityAuthService {
  private baseUrl: string;
  private communityUrl: string;
  
  constructor() {
    // Central Auth Hub (Chat App)
    this.baseUrl = process.env.NEXT_PUBLIC_CENTRAL_AUTH_URL || 'https://dharmamind.ai';
    // Community API endpoint
    this.communityUrl = process.env.NEXT_PUBLIC_COMMUNITY_API_URL || 'http://localhost:8000';
  }

  /**
   * Login with DharmaMind credentials
   */
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      // Use main DharmaMind authentication endpoint
      const response = await fetch(`${this.baseUrl}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        const error = await response.json();
        return { 
          success: false, 
          error: error.detail || 'Invalid email or password. Please try again.' 
        };
      }

      const data = await response.json();
      
      if (data.session_token && data.user) {
        // Store authentication data
        this.setTokens(data.session_token);
        this.setUser(data.user);
        
        // Sync with community platform
        await this.syncWithCommunity(data.user, data.session_token);
        
        return { 
          success: true, 
          user: this.mapToCommunityUser(data.user),
          session_token: data.session_token 
        };
      }

      return { success: false, error: 'Authentication failed' };
    } catch (error) {
      console.error('Login error:', error);
      return { 
        success: false, 
        error: 'Unable to connect to DharmaMind. Please try again.' 
      };
    }
  }

  /**
   * Login with existing DharmaMind token (SSO)
   */
  async loginWithToken(token: string): Promise<AuthResponse> {
    try {
      // Verify token with main DharmaMind API
      const response = await fetch(`${this.baseUrl}/auth/verify`, {
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
        await this.syncWithCommunity(data.user, token);
        
        return { 
          success: true, 
          user: this.mapToCommunityUser(data.user),
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
      const response = await fetch(`${this.baseUrl}/auth/me`, {
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
      return this.mapToCommunityUser(data.user);
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
   * Logout user from all platforms
   */
  logout(): void {
    // Clear local storage
    if (typeof window !== 'undefined') {
      localStorage.removeItem('dharmamind_token');
      localStorage.removeItem('dharmamind_user');
      localStorage.removeItem('dharmamind_community_profile');
      
      // Optional: Call logout endpoint to invalidate server session
      const token = this.getStoredToken();
      if (token) {
        fetch(`${this.baseUrl}/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }).catch(err => console.log('Logout cleanup error:', err));
      }
    }
  }

  /**
   * Get stored authentication token
   */
  getStoredToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem('dharmamind_token');
  }

  /**
   * Store authentication tokens
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
   * Map main platform user to community user format
   */
  private mapToCommunityUser(user: any): DharmaMindUser {
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
      joined_date: user.created_at || user.joined_date || new Date().toISOString(),
      community_role: user.community_role || 'member'
    };
  }

  /**
   * Sync user with community platform
   */
  private async syncWithCommunity(user: any, token: string): Promise<void> {
    try {
      // Sync user profile with community database
      await fetch(`${this.communityUrl}/community/sync-user`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: user.user_id || user.id,
          email: user.email,
          name: user.first_name || user.name,
          subscription_plan: user.subscription_plan || user.plan,
          last_login: new Date().toISOString()
        })
      });
    } catch (error) {
      console.log('Community sync warning:', error);
      // Non-critical error, don't fail login
    }
  }

  /**
   * Generate avatar URL from email
   */
  private generateAvatarUrl(email: string): string {
    // Simple initial-based avatar or gravatar
    const initial = email.charAt(0).toUpperCase();
    return `data:image/svg+xml;base64,${btoa(`
      <svg width="40" height="40" xmlns="http://www.w3.org/2000/svg">
        <circle cx="20" cy="20" r="20" fill="#F2A300"/>
        <text x="20" y="28" text-anchor="middle" fill="white" font-family="Arial" font-size="16" font-weight="bold">
          ${initial}
        </text>
      </svg>
    `)}`;
  }

  /**
   * Auto-login if coming from central auth (Chat App)
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
          const response = await fetch(`${this.communityUrl}/auth/exchange-token`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
              auth_code: authCode,
              platform: 'community'
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
   * Generate central auth URL (Chat App)
   */
  generateCentralAuthUrl(mode: 'login' | 'signup' = 'login', returnUrl?: string): string {
    const currentUrl = typeof window !== 'undefined' ? window.location.origin : '';
    const finalReturnUrl = returnUrl || '/community';
    const communityReturnUrl = `${currentUrl}/login?returnUrl=${encodeURIComponent(finalReturnUrl)}`;
    
    // Central auth hub (Chat App) with community return URL
    return `${this.baseUrl}/auth?mode=${mode}&returnUrl=${encodeURIComponent(communityReturnUrl)}&platform=community&source=${encodeURIComponent(currentUrl)}`;
  }

  /**
   * Redirect to central auth hub
   */
  redirectToCentralAuth(mode: 'login' | 'signup' = 'login', returnUrl?: string): void {
    const centralAuthUrl = this.generateCentralAuthUrl(mode, returnUrl);
    window.location.href = centralAuthUrl;
  }
}

// Export singleton instance
const communityAuth = new DharmaMindCommunityAuth();
export default communityAuth;
export type { DharmaMindUser, AuthResponse, LoginCredentials };
