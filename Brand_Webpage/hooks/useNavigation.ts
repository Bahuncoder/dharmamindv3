/**
 * 🧭 DharmaMind Centralized Navigation System
 * 
 * Unified routing and navigation management
 * Replaces scattered router.push() implementations
 */

import { useRouter } from 'next/router';
import { useCallback } from 'react';

// ===============================
// ROUTE CONSTANTS
// ===============================

export const ROUTES = {
  // Public pages
  HOME: '/',
  AUTH: '/auth',
  WELCOME: '/welcome',
  CONTACT: '/contact',
  HELP: '/help',
  ABOUT: '/about',
  PRICING: '/pricing',
  PRIVACY: '/privacy',
  TERMS: '/terms',
  
  // Authentication flows
  LOGIN: '/auth?mode=login',
  SIGNUP: '/auth?mode=signup',
  FORGOT_PASSWORD: '/auth?mode=forgot-password',
  VERIFY: '/auth?mode=verify',
  RESET_PASSWORD: '/reset-password',
  
  // Protected pages
  CHAT: '/chat',
  DASHBOARD: '/dashboard',
  PROFILE: '/profile',
  SETTINGS: '/settings',
  SUBSCRIPTION: '/subscription',
  BILLING: '/billing',
  
  // API routes
  API: {
    AUTH: {
      LOGIN: '/api/auth/login',
      REGISTER: '/api/auth/register',
      LOGOUT: '/api/auth/logout',
      VERIFY: '/api/auth/verify',
      FORGOT_PASSWORD: '/api/auth/forgot-password',
      RESET_PASSWORD: '/api/auth/reset-password'
    },
    USER: {
      PROFILE: '/api/user/profile',
      SETTINGS: '/api/user/settings',
      SUBSCRIPTION: '/api/user/subscription'
    },
    CHAT: {
      MESSAGES: '/api/chat/messages',
      CONVERSATION: '/api/chat/conversation'
    }
  }
} as const;

// ===============================
// NAVIGATION HOOK
// ===============================

export interface NavigationOptions {
  replace?: boolean;
  shallow?: boolean;
  scroll?: boolean;
  query?: Record<string, string>;
}

export interface UseNavigationReturn {
  // Basic navigation
  navigateTo: (path: string, options?: NavigationOptions) => Promise<boolean>;
  goBack: () => void;
  goForward: () => void;
  reload: () => void;
  
  // Route-specific helpers
  goHome: () => Promise<boolean>;
  goToAuth: (mode?: 'login' | 'signup' | 'forgot-password' | 'verify') => Promise<boolean>;
  goToChat: () => Promise<boolean>;
  goToDashboard: () => Promise<boolean>;
  goToProfile: () => Promise<boolean>;
  goToSettings: () => Promise<boolean>;
  goToSubscription: () => Promise<boolean>;
  goToContact: () => Promise<boolean>;
  goToHelp: () => Promise<boolean>;
  
  // Conditional navigation
  redirectAfterAuth: () => Promise<boolean>;
  redirectToLogin: (returnUrl?: string) => Promise<boolean>;
  
  // Utility functions
  isCurrentRoute: (path: string) => boolean;
  buildUrl: (path: string, query?: Record<string, string>) => string;
  getCurrentPath: () => string;
  getQueryParam: (key: string) => string | string[] | undefined;
}

/**
 * Main navigation hook
 */
export const useNavigation = (): UseNavigationReturn => {
  const router = useRouter();

  const navigateTo = useCallback(async (
    path: string, 
    options: NavigationOptions = {}
  ): Promise<boolean> => {
    const { replace = false, shallow = false, scroll = true, query = {} } = options;
    
    const url = buildUrl(path, query);
    
    try {
      if (replace) {
        return await router.replace(url, undefined, { shallow, scroll });
      } else {
        return await router.push(url, undefined, { shallow, scroll });
      }
    } catch (error) {
      console.error('Navigation error:', error);
      return false;
    }
  }, [router]);

  const goBack = useCallback(() => {
    if (window.history.length > 1) {
      router.back();
    } else {
      navigateTo(ROUTES.HOME);
    }
  }, [router, navigateTo]);

  const goForward = useCallback(() => {
    router.forward();
  }, [router]);

  const reload = useCallback(() => {
    router.reload();
  }, [router]);

  // Route-specific helpers
  const goHome = useCallback(() => navigateTo(ROUTES.HOME), [navigateTo]);
  
  const goToAuth = useCallback((mode: 'login' | 'signup' | 'forgot-password' | 'verify' = 'login') => {
    return navigateTo(ROUTES.AUTH, { query: { mode } });
  }, [navigateTo]);
  
  const goToChat = useCallback(() => navigateTo(ROUTES.CHAT), [navigateTo]);
  const goToDashboard = useCallback(() => navigateTo(ROUTES.DASHBOARD), [navigateTo]);
  const goToProfile = useCallback(() => navigateTo(ROUTES.PROFILE), [navigateTo]);
  const goToSettings = useCallback(() => navigateTo(ROUTES.SETTINGS), [navigateTo]);
  const goToSubscription = useCallback(() => navigateTo(ROUTES.SUBSCRIPTION), [navigateTo]);
  const goToContact = useCallback(() => navigateTo(ROUTES.CONTACT), [navigateTo]);
  const goToHelp = useCallback(() => navigateTo(ROUTES.HELP), [navigateTo]);

  // Conditional navigation
  const redirectAfterAuth = useCallback(async (): Promise<boolean> => {
    const returnUrl = router.query.returnUrl as string;
    const defaultRedirect = ROUTES.CHAT;
    
    return navigateTo(returnUrl || defaultRedirect);
  }, [router.query.returnUrl, navigateTo]);

  const redirectToLogin = useCallback(async (returnUrl?: string): Promise<boolean> => {
    const currentUrl = returnUrl || router.asPath;
    return navigateTo(ROUTES.LOGIN, { 
      query: currentUrl !== ROUTES.HOME ? { returnUrl: currentUrl } : {} 
    });
  }, [router.asPath, navigateTo]);

  // Utility functions
  const isCurrentRoute = useCallback((path: string): boolean => {
    return router.pathname === path || router.asPath === path;
  }, [router.pathname, router.asPath]);

  const buildUrl = useCallback((path: string, query: Record<string, string> = {}): string => {
    const url = new URL(path, window.location.origin);
    Object.entries(query).forEach(([key, value]) => {
      url.searchParams.set(key, value);
    });
    return url.pathname + url.search;
  }, []);

  const getCurrentPath = useCallback((): string => {
    return router.asPath;
  }, [router.asPath]);

  const getQueryParam = useCallback((key: string): string | string[] | undefined => {
    return router.query[key];
  }, [router.query]);

  return {
    navigateTo,
    goBack,
    goForward,
    reload,
    goHome,
    goToAuth,
    goToChat,
    goToDashboard,
    goToProfile,
    goToSettings,
    goToSubscription,
    goToContact,
    goToHelp,
    redirectAfterAuth,
    redirectToLogin,
    isCurrentRoute,
    buildUrl,
    getCurrentPath,
    getQueryParam
  };
};

// ===============================
// NAVIGATION GUARDS
// ===============================

export interface NavigationGuardResult {
  canNavigate: boolean;
  redirectTo?: string;
  reason?: string;
}

export type NavigationGuard = (
  to: string,
  from: string,
  context?: any
) => NavigationGuardResult | Promise<NavigationGuardResult>;

/**
 * Hook for protected routes
 */
export const useNavigationGuards = () => {
  const navigation = useNavigation();

  const requireAuth: NavigationGuard = useCallback(async (to: string, from: string) => {
    // Check if user is authenticated
    try {
      const response = await fetch('/api/auth/check');
      if (response.ok) {
        return { canNavigate: true };
      } else {
        return {
          canNavigate: false,
          redirectTo: ROUTES.LOGIN,
          reason: 'Authentication required'
        };
      }
    } catch (error) {
      return {
        canNavigate: false,
        redirectTo: ROUTES.LOGIN,
        reason: 'Unable to verify authentication'
      };
    }
  }, []);

  const requireGuest: NavigationGuard = useCallback(async (to: string, from: string) => {
    // Check if user is NOT authenticated
    try {
      const response = await fetch('/api/auth/check');
      if (!response.ok) {
        return { canNavigate: true };
      } else {
        return {
          canNavigate: false,
          redirectTo: ROUTES.CHAT,
          reason: 'Already authenticated'
        };
      }
    } catch (error) {
      return { canNavigate: true }; // Allow access if can't verify
    }
  }, []);

  const requireSubscription: NavigationGuard = useCallback(async (to: string, from: string) => {
    // Check if user has active subscription
    try {
      const response = await fetch('/api/user/subscription');
      if (response.ok) {
        const data = await response.json();
        if (data.hasActiveSubscription) {
          return { canNavigate: true };
        } else {
          return {
            canNavigate: false,
            redirectTo: ROUTES.SUBSCRIPTION,
            reason: 'Active subscription required'
          };
        }
      } else {
        return {
          canNavigate: false,
          redirectTo: ROUTES.SUBSCRIPTION,
          reason: 'Unable to verify subscription'
        };
      }
    } catch (error) {
      return {
        canNavigate: false,
        redirectTo: ROUTES.SUBSCRIPTION,
        reason: 'Unable to verify subscription'
      };
    }
  }, []);

  const withGuards = useCallback(async (
    path: string,
    guards: NavigationGuard[],
    options?: NavigationOptions
  ): Promise<boolean> => {
    const currentPath = navigation.getCurrentPath();

    for (const guard of guards) {
      const result = await guard(path, currentPath);
      if (!result.canNavigate) {
        if (result.redirectTo) {
          console.warn(`Navigation blocked: ${result.reason}`);
          return navigation.navigateTo(result.redirectTo, options);
        }
        return false;
      }
    }

    return navigation.navigateTo(path, options);
  }, [navigation]);

  return {
    requireAuth,
    requireGuest,
    requireSubscription,
    withGuards
  };
};

// ===============================
// ROUTE PATTERNS
// ===============================

export const ROUTE_PATTERNS = {
  PUBLIC: [
    ROUTES.HOME,
    ROUTES.AUTH,
    ROUTES.WELCOME,
    ROUTES.CONTACT,
    ROUTES.HELP,
    ROUTES.ABOUT,
    ROUTES.PRICING,
    ROUTES.PRIVACY,
    ROUTES.TERMS,
    ROUTES.RESET_PASSWORD
  ],
  PROTECTED: [
    ROUTES.CHAT,
    ROUTES.DASHBOARD,
    ROUTES.PROFILE,
    ROUTES.SETTINGS,
    ROUTES.SUBSCRIPTION,
    ROUTES.BILLING
  ],
  AUTH_ONLY: [
    ROUTES.LOGIN,
    ROUTES.SIGNUP,
    ROUTES.FORGOT_PASSWORD,
    ROUTES.VERIFY
  ]
} as const;

/**
 * Check if a route requires authentication
 */
export const isProtectedRoute = (path: string): boolean => {
  return ROUTE_PATTERNS.PROTECTED.some(route => 
    path === route || path.startsWith(route + '/')
  );
};

/**
 * Check if a route is auth-only (for guests)
 */
export const isAuthOnlyRoute = (path: string): boolean => {
  return ROUTE_PATTERNS.AUTH_ONLY.some(route => 
    path === route || path.startsWith(route + '/')
  );
};

/**
 * Check if a route is public
 */
export const isPublicRoute = (path: string): boolean => {
  return ROUTE_PATTERNS.PUBLIC.some(route => 
    path === route || path.startsWith(route + '/')
  );
};
