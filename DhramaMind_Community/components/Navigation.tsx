import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import Logo from './Logo';
import communityAuth from '../services/communityAuth';
import type { DharmaMindUser } from '../services/communityAuth';

const Navigation: React.FC = () => {
  const router = useRouter();
  const [user, setUser] = useState<DharmaMindUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      if (communityAuth.isAuthenticated()) {
        const currentUser = await communityAuth.getCurrentUser();
        setUser(currentUser);
      }
      setIsLoading(false);
    };

    checkAuth();
  }, []);

  const handleLogout = () => {
    communityAuth.logout();
    setUser(null);
    router.push('/');
  };

  const handleLogin = () => {
    // Redirect to central auth hub (Chat App)
    communityAuth.redirectToCentralAuth('login', router.asPath);
  };

  return (
    <nav className="w-full bg-primary-bg/95 backdrop-blur-sm shadow-sm border-b-2 border-border-primary">
      <div className="max-w-6xl mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo - Clickable */}
          <Link 
            href="/"
            className="hover:opacity-80 transition-opacity duration-200"
          >
            <Logo size="md" showText={true} />
          </Link>
          
          {/* Navigation Links */}
          <div className="flex items-center space-x-6">
            <Link 
              href="/community" 
              className="text-secondary hover:text-primary transition-colors duration-200 font-bold"
            >
              Community
            </Link>
            <Link 
              href="/blog" 
              className="text-secondary hover:text-primary transition-colors duration-200 font-bold"
            >
              Blog
            </Link>
            <a 
              href="https://dharmamind.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-secondary hover:text-primary transition-colors duration-200 font-bold"
            >
              Home
            </a>
            <a 
              href="https://dharmamind.ai" 
              target="_blank" 
              rel="noopener noreferrer"
              className="btn-outline px-4 py-2 rounded-lg font-bold hover:bg-primary hover:text-white transition-all duration-200"
            >
              AI Chat
            </a>
            
            {/* Authentication Section */}
            {!isLoading && (
              <>
                {user ? (
                  <div className="flex items-center space-x-3">
                    {/* User Avatar/Name */}
                    <div className="flex items-center space-x-2">
                      <div className="w-8 h-8 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-bold">
                        {user.first_name?.charAt(0) || user.email.charAt(0).toUpperCase()}
                      </div>
                      <span className="text-secondary font-medium">
                        {user.first_name || user.email.split('@')[0]}
                      </span>
                    </div>
                    
                    {/* Logout Button */}
                    <button
                      onClick={handleLogout}
                      className="text-muted hover:text-secondary transition-colors duration-200 text-sm font-medium"
                    >
                      Logout
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={handleLogin}
                    className="btn-primary px-6 py-2 rounded-lg font-bold hover:opacity-90 transition-opacity duration-200"
                  >
                    Login
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
