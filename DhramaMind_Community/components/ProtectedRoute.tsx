import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import communityAuth from '../services/communityAuth';
import type { DharmaMindUser } from '../services/communityAuth';

interface ProtectedRouteProps {
  children: React.ReactNode;
  redirectTo?: string;
  requireAuth?: boolean;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ 
  children, 
  redirectTo = '/login',
  requireAuth = true 
}) => {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState<DharmaMindUser | null>(null);

  useEffect(() => {
    const checkAuth = async () => {
      // Check for SSO login first
      const authResult = await communityAuth.checkAutoLogin();
      if (authResult?.success && authResult.user) {
        setUser(authResult.user);
        setIsAuthenticated(true);
        setIsLoading(false);
        return;
      }

      // Check existing authentication
      if (communityAuth.isAuthenticated()) {
        const currentUser = await communityAuth.getCurrentUser();
        if (currentUser) {
          setUser(currentUser);
          setIsAuthenticated(true);
        } else {
          setIsAuthenticated(false);
        }
      } else {
        setIsAuthenticated(false);
      }

      setIsLoading(false);
    };

    checkAuth();
  }, []);

  useEffect(() => {
    if (!isLoading && requireAuth && !isAuthenticated) {
      // Redirect to login with current path as return URL
      const currentPath = router.asPath;
      router.push(`${redirectTo}?returnUrl=${encodeURIComponent(currentPath)}`);
    }
  }, [isLoading, isAuthenticated, requireAuth, redirectTo, router]);

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-secondary-bg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-xl font-semibold text-secondary">Loading...</p>
        </div>
      </div>
    );
  }

  // If auth is required but user is not authenticated, don't render children
  // (redirect will happen in useEffect)
  if (requireAuth && !isAuthenticated) {
    return null;
  }

  // Render children with user context
  return (
    <div data-user-authenticated={isAuthenticated} data-user-id={user?.user_id}>
      {children}
    </div>
  );
};

export default ProtectedRoute;
