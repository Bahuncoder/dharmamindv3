import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import Logo from './Logo';
import Button from './Button';
import communityAuth from '../services/communityAuth';
import type { DharmaMindUser } from '../services/communityAuth';

const Navigation: React.FC = () => {
  const router = useRouter();
  const [user, setUser] = useState<DharmaMindUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

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
    setIsMobileMenuOpen(false);
  };

  const handleLogin = () => {
    // Redirect to central auth hub (Chat App)
    communityAuth.redirectToCentralAuth('login', router.asPath);
  };

  const navigationLinks = [
    { href: '/community', label: 'Community', icon: '👥' },
    { href: '/blog', label: 'Blog', icon: '📝' },
    { href: 'https://dharmamind.com', label: 'Home', icon: '🏠', external: true },
  ];

  const isActiveLink = (href: string) => {
    if (href.startsWith('http')) return false;
    return router.pathname === href;
  };

  return (
    <nav className="w-full bg-primary-bg/95 backdrop-blur-md shadow-lg border-b-2 border-border-primary sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link 
            href="/"
            className="hover:opacity-80 transition-all duration-300 hover:scale-105"
          >
            <Logo size="md" showText={true} />
          </Link>
          
          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center space-x-8">
            {navigationLinks.map((link) => (
              link.external ? (
                <a
                  key={link.href}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-2 text-secondary hover:text-primary transition-all duration-300 font-bold group"
                >
                  <span className="group-hover:scale-110 transition-transform">{link.icon}</span>
                  <span>{link.label}</span>
                </a>
              ) : (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center space-x-2 transition-all duration-300 font-bold group relative ${
                    isActiveLink(link.href)
                      ? 'text-primary'
                      : 'text-secondary hover:text-primary'
                  }`}
                >
                  <span className="group-hover:scale-110 transition-transform">{link.icon}</span>
                  <span>{link.label}</span>
                  {isActiveLink(link.href) && (
                    <div className="absolute -bottom-2 left-0 right-0 h-0.5 bg-primary rounded-full"></div>
                  )}
                </Link>
              )
            ))}
            
            {/* AI Chat Button */}
            <a 
              href="https://dharmamind.ai" 
              target="_blank" 
              rel="noopener noreferrer"
              className="btn-outline px-4 py-2 rounded-lg font-bold hover:bg-primary hover:text-white transition-all duration-300 group"
            >
              <span className="mr-2 group-hover:scale-110 transition-transform">🤖</span>
              AI Chat
            </a>
            
            {/* Authentication Section */}
            {!isLoading && (
              <>
                {user ? (
                  <div className="flex items-center space-x-4">
                    {/* User Profile */}
                    <div className="flex items-center space-x-3 p-2 rounded-lg hover:bg-neutral-100 transition-all duration-300 cursor-pointer group">
                      <div className="relative">
                        <div className="w-10 h-10 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-bold group-hover:scale-110 transition-transform shadow-md">
                          {user.first_name?.charAt(0) || user.email.charAt(0).toUpperCase()}
                        </div>
                        <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-success rounded-full border-2 border-white"></div>
                      </div>
                      <div className="hidden xl:block">
                        <div className="text-sm font-bold text-primary">
                          {user.first_name || 'User'}
                        </div>
                        <div className="text-xs text-muted">
                          Online
                        </div>
                      </div>
                    </div>
                    
                    {/* Logout Button */}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleLogout}
                      className="font-bold"
                      icon={<span>🚪</span>}
                      iconPosition="left"
                    >
                      Logout
                    </Button>
                  </div>
                ) : (
                  <Button
                    variant="primary"
                    size="md"
                    onClick={handleLogin}
                    className="font-bold"
                    icon={<span>🔑</span>}
                    iconPosition="left"
                  >
                    Login
                  </Button>
                )}
              </>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="lg:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-2 rounded-lg text-secondary hover:text-primary hover:bg-neutral-100 transition-all duration-300"
              aria-label="Toggle mobile menu"
            >
              <div className="w-6 h-6 flex flex-col justify-center items-center">
                <span className={`block w-6 h-0.5 bg-current transition-all duration-300 ${isMobileMenuOpen ? 'rotate-45 translate-y-1' : '-translate-y-1'}`}></span>
                <span className={`block w-6 h-0.5 bg-current transition-all duration-300 ${isMobileMenuOpen ? 'opacity-0' : 'opacity-100'}`}></span>
                <span className={`block w-6 h-0.5 bg-current transition-all duration-300 ${isMobileMenuOpen ? '-rotate-45 -translate-y-1' : 'translate-y-1'}`}></span>
              </div>
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        <div className={`lg:hidden transition-all duration-300 overflow-hidden ${
          isMobileMenuOpen ? 'max-h-96 pb-6' : 'max-h-0'
        }`}>
          <div className="space-y-4 pt-4 border-t border-border-light">
            {navigationLinks.map((link) => (
              link.external ? (
                <a
                  key={link.href}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-3 p-3 rounded-lg hover:bg-neutral-100 transition-all duration-300 group"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  <span className="text-xl group-hover:scale-110 transition-transform">{link.icon}</span>
                  <span className="font-bold text-secondary group-hover:text-primary">{link.label}</span>
                </a>
              ) : (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center space-x-3 p-3 rounded-lg transition-all duration-300 group ${
                    isActiveLink(link.href)
                      ? 'bg-primary-gradient text-white'
                      : 'hover:bg-neutral-100'
                  }`}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  <span className="text-xl group-hover:scale-110 transition-transform">{link.icon}</span>
                  <span className={`font-bold ${isActiveLink(link.href) ? 'text-white' : 'text-secondary group-hover:text-primary'}`}>
                    {link.label}
                  </span>
                </Link>
              )
            ))}
            
            {/* Mobile AI Chat Button */}
            <a 
              href="https://dharmamind.ai" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center space-x-3 p-3 rounded-lg border-2 border-primary text-primary hover:bg-primary hover:text-white transition-all duration-300 group"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              <span className="text-xl group-hover:scale-110 transition-transform">🤖</span>
              <span className="font-bold">AI Chat</span>
            </a>
            
            {/* Mobile Authentication */}
            {!isLoading && (
              <div className="pt-4 border-t border-border-light">
                {user ? (
                  <div className="space-y-3">
                    <div className="flex items-center space-x-3 p-3 bg-neutral-50 rounded-lg">
                      <div className="w-12 h-12 bg-primary-gradient rounded-full flex items-center justify-center text-white font-bold text-lg shadow-md">
                        {user.first_name?.charAt(0) || user.email.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <div className="font-bold text-primary">
                          {user.first_name || 'User'}
                        </div>
                        <div className="text-sm text-secondary">
                          {user.email}
                        </div>
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="md"
                      onClick={handleLogout}
                      fullWidth
                      icon={<span>🚪</span>}
                      iconPosition="left"
                    >
                      Logout
                    </Button>
                  </div>
                ) : (
                  <Button
                    variant="primary"
                    size="lg"
                    onClick={handleLogin}
                    fullWidth
                    icon={<span>🔑</span>}
                    iconPosition="left"
                  >
                    Login to Continue
                  </Button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
