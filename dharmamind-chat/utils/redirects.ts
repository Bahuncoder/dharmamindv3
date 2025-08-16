/**
 * Central  // Subscription redirects
  '/subscription': '/chat?subscription=true',
  '/subscription_new': '/chat?subscription=true',
  '/subscribe': '/chat?subscription=true',
  '/billing': '/chat?subscription=true',
  '/upgrade': '/chat?subscription=true',
  '/plans': '/chat?subscription=true',
  
  // Password reset redirects
  '/forgot-password': '/auth?mode=forgot-password',
  '/forgot-password-new': '/auth?mode=forgot-password',
  '/reset-password': '/auth?mode=forgot-password',
  
  // Mobile redirects
  '/mobile-chat': '/chat',direct configuration
 * This approach eliminates the need for multiple redirect pages
 * All redirects are handled in one place for easier maintenance
 */

export const LEGACY_REDIRECTS = {
  // Auth related redirects
  '/login': '/auth?mode=login',
  '/signup': '/auth?mode=signup', 
  '/login-redirect': '/auth?mode=login',
  '/signup-redirect': '/auth?mode=signup',
  '/register': '/auth?mode=signup',
  '/signin': '/auth?mode=login',
  
  // Subscription related redirects
  '/subscription': '/chat?subscription=true',
  '/subscription_new': '/chat?subscription=true',
  '/subscribe': '/chat?subscription=true',
  '/billing': '/chat?subscription=true',
  '/upgrade': '/chat?subscription=true',
  '/plans': '/chat?subscription=true',
  
  // Other potential legacy redirects
  '/welcome': '/',
  '/home': '/',
  '/landing': '/',
  '/dashboard': '/chat',
} as const;

/**
 * Get redirect destination for a given path
 * Returns null if no redirect is needed
 */
export function getRedirectDestination(path: string): string | null {
  return LEGACY_REDIRECTS[path as keyof typeof LEGACY_REDIRECTS] || null;
}

/**
 * Check if a path needs to be redirected
 */
export function shouldRedirect(path: string): boolean {
  return path in LEGACY_REDIRECTS;
}
