/**
 * Authentication Context for DharmaMind
 * Provides authentication state management across the application
 */

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { authService, User, AuthResponse, LoginCredentials, RegisterData } from '../services/authService';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isGuest: boolean;
  isLoading: boolean;
  login: (credentials: LoginCredentials) => Promise<AuthResponse>;
  register: (userData: RegisterData) => Promise<AuthResponse>;
  demoLogin: (plan?: 'basic' | 'pro' | 'max' | 'enterprise') => Promise<AuthResponse>;
  guestLogin: () => void;
  logout: () => Promise<void>;
  refreshUser: () => void;
  updateUserPlan: (plan: 'basic' | 'pro' | 'max' | 'enterprise') => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isGuest, setIsGuest] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Initialize authentication state
  useEffect(() => {
    const initAuth = () => {
      const currentUser = authService.getCurrentUser();
      setUser(currentUser);
      
      // Check if user was in guest mode
      if (typeof window !== 'undefined') {
        const guestMode = sessionStorage.getItem('dharma_guest_mode');
        if (guestMode === 'true' && !currentUser) {
          setIsGuest(true);
        }
      }
      
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = async (credentials: LoginCredentials): Promise<AuthResponse> => {
    try {
      setIsLoading(true);
      const response = await authService.login(credentials);
      
      if (response.success && response.user) {
        setUser(response.user);
      }
      
      return response;
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, error: 'Login failed' };
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (userData: RegisterData): Promise<AuthResponse> => {
    try {
      setIsLoading(true);
      const response = await authService.register(userData);
      
      if (response.success && response.user) {
        setUser(response.user);
      }
      
      return response;
    } catch (error) {
      console.error('Registration error:', error);
      return { success: false, error: 'Registration failed' };
    } finally {
      setIsLoading(false);
    }
  };

  const demoLogin = async (plan: 'basic' | 'pro' | 'max' | 'enterprise' = 'basic'): Promise<AuthResponse> => {
    try {
      setIsLoading(true);
      const response = await authService.demoLogin(plan);
      
      if (response.success && response.user) {
        setUser(response.user);
        setIsGuest(false);
        if (typeof window !== 'undefined') {
          sessionStorage.removeItem('dharma_guest_mode');
        }
      }
      
      return response;
    } catch (error) {
      console.error('Demo login error:', error);
      return { success: false, error: 'Demo login failed' };
    } finally {
      setIsLoading(false);
    }
  };

  // Guest login - no account needed, just start chatting
  const guestLogin = (): void => {
    setIsGuest(true);
    if (typeof window !== 'undefined') {
      sessionStorage.setItem('dharma_guest_mode', 'true');
    }
  };

  const logout = async (): Promise<void> => {
    try {
      setIsLoading(true);
      await authService.logout();
      setUser(null);
      setIsGuest(false);
      if (typeof window !== 'undefined') {
        sessionStorage.removeItem('dharma_guest_mode');
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const refreshUser = (): void => {
    const currentUser = authService.getCurrentUser();
    setUser(currentUser);
  };

  const updateUserPlan = (plan: 'basic' | 'pro' | 'max' | 'enterprise'): void => {
    if (user) {
      const updatedUser = { ...user, subscription_plan: plan };
      setUser(updatedUser);
      // Also update in localStorage/sessionStorage if needed
      if (typeof window !== 'undefined') {
        const userData = localStorage.getItem('dharma_user') || sessionStorage.getItem('dharma_user');
        if (userData) {
          const parsedUser = JSON.parse(userData);
          parsedUser.subscription_plan = plan;
          if (localStorage.getItem('dharma_user')) {
            localStorage.setItem('dharma_user', JSON.stringify(parsedUser));
          } else {
            sessionStorage.setItem('dharma_user', JSON.stringify(parsedUser));
          }
        }
      }
    }
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user && authService.isAuthenticated(),
    isGuest,
    isLoading,
    login,
    register,
    demoLogin,
    guestLogin,
    logout,
    refreshUser,
    updateUserPlan,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use authentication context
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;
