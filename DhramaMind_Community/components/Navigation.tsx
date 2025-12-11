import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import Logo from './Logo';
import Button from './Button';
import NotificationCenter from './NotificationCenter';
import communityAuth from '../services/communityAuth';
import type { DharmaMindUser } from '../services/communityAuth';

const Navigation: React.FC = () => {
  const router = useRouter();
  const [user, setUser] = useState<DharmaMindUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

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

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleLogout = () => {
    communityAuth.logout();
    setUser(null);
    router.push('/');
    setIsMobileMenuOpen(false);
  };

  const handleLogin = () => {
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
    <nav 
      className={`
        fixed top-0 left-0 right-0 z-50 
        transition-all duration-500 ease-out
        ${isScrolled 
          ? 'bg-white/90 backdrop-blur-xl shadow-lg border-b border-gold-100' 
          : 'bg-white/70 backdrop-blur-md border-b border-transparent'
        }
      `}
    >
      {/* Gradient accent line */}
      <div className={`
        absolute top-0 left-0 right-0 h-0.5 
        bg-gradient-to-r from-gold-400 via-gold-500 to-gold-500
        transition-opacity duration-500
        ${isScrolled ? 'opacity-100' : 'opacity-0'}
      `} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 lg:h-20">
          {/* Logo */}
          <Link
            href="/"
            className="group flex items-center gap-3 hover:opacity-90 transition-all duration-300"
          >
            <div className="relative">
              <Logo size="md" showText={false} />
              {/* Glow effect on hover */}
              <div className="absolute inset-0 bg-gold-500/20 blur-xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            </div>
            <div className="hidden sm:block">
              <span className="text-xl font-bold text-gray-900 group-hover:text-gold-600 transition-colors">
                DharmaMind
              </span>
              <span className="block text-xs text-gray-500 -mt-0.5">Community</span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center gap-1">
            {navigationLinks.map((link, index) => (
              link.external ? (
                <a
                  key={link.href}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="
                    group relative px-4 py-2 rounded-xl
                    text-gray-600 hover:text-gray-900
                    transition-all duration-300
                  "
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <span className="flex items-center gap-2">
                    <span className="text-lg transition-transform group-hover:scale-110 duration-300">
                      {link.icon}
                    </span>
                    <span className="font-medium">{link.label}</span>
                    <svg className="w-3 h-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </span>
                  {/* Hover background */}
                  <span className="absolute inset-0 rounded-xl bg-gray-100 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10" />
                </a>
              ) : (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`
                    group relative px-4 py-2 rounded-xl
                    transition-all duration-300
                    ${isActiveLink(link.href)
                      ? 'text-gold-600'
                      : 'text-gray-600 hover:text-gray-900'
                    }
                  `}
                >
                  <span className="flex items-center gap-2">
                    <span className="text-lg transition-transform group-hover:scale-110 duration-300">
                      {link.icon}
                    </span>
                    <span className="font-medium">{link.label}</span>
                  </span>
                  
                  {/* Active indicator */}
                  {isActiveLink(link.href) && (
                    <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-0.5 bg-gold-500 rounded-full" />
                  )}
                  
                  {/* Hover background */}
                  <span className={`
                    absolute inset-0 rounded-xl transition-opacity duration-300 -z-10
                    ${isActiveLink(link.href) 
                      ? 'bg-gold-50 opacity-100' 
                      : 'bg-gray-100 opacity-0 group-hover:opacity-100'
                    }
                  `} />
                </Link>
              )
            ))}

            {/* AI Chat Button */}
            <a
              href="https://dharmamind.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="
                group relative ml-2 px-5 py-2.5 rounded-xl
                bg-gradient-to-r from-gold-500 to-gold-600
                text-white font-semibold
                shadow-lg shadow-gold-500/25
                hover:shadow-xl hover:shadow-gold-500/30 hover:-translate-y-0.5
                transition-all duration-300
                overflow-hidden
              "
            >
              <span className="relative z-10 flex items-center gap-2">
                <span className="text-lg">🤖</span>
                <span>AI Chat</span>
                <svg className="w-4 h-4 transition-transform group-hover:translate-x-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </span>
              {/* Shimmer effect */}
              <span className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 bg-gradient-to-r from-transparent via-white/20 to-transparent" />
            </a>

            {/* Divider */}
            <div className="w-px h-8 bg-gray-200 mx-3" />

            {/* Authentication Section */}
            {!isLoading && (
              <>
                {user ? (
                  <div className="flex items-center gap-3">
                    {/* Notifications */}
                    <NotificationCenter />

                    {/* User Profile */}
                    <div className="
                      group flex items-center gap-3 px-3 py-2 rounded-xl
                      hover:bg-gray-50 transition-all duration-300 cursor-pointer
                    ">
                      <div className="relative">
                        <div className="
                          w-10 h-10 rounded-xl
                          bg-gradient-to-br from-gold-400 to-gold-600
                          flex items-center justify-center
                          text-white font-bold
                          shadow-md shadow-gold-500/20
                          transition-transform group-hover:scale-105 duration-300
                        ">
                          {user.first_name?.charAt(0) || user.email.charAt(0).toUpperCase()}
                        </div>
                        {/* Online indicator */}
                        <div className="absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 bg-gold-500 rounded-full border-2 border-white" />
                      </div>
                      <div className="hidden xl:block text-left">
                        <div className="text-sm font-semibold text-gray-900">
                          {user.first_name || 'User'}
                        </div>
                        <div className="text-xs text-gold-600 flex items-center gap-1">
                          <span className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-pulse" />
                          Online
                        </div>
                      </div>
                    </div>

                    {/* Logout Button */}
                    <button
                      onClick={handleLogout}
                      className="
                        px-3 py-2 rounded-xl
                        text-gray-500 hover:text-gray-700 hover:bg-gray-100
                        transition-all duration-300
                      "
                      title="Logout"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                      </svg>
                    </button>
                  </div>
                ) : (
                  <Button
                    variant="primary"
                    size="md"
                    onClick={handleLogin}
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
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="
              lg:hidden p-2 rounded-xl
              text-gray-600 hover:text-gray-900 hover:bg-gray-100
              transition-all duration-300
            "
            aria-label="Toggle mobile menu"
          >
            <div className="w-6 h-6 flex flex-col justify-center items-center gap-1.5">
              <span className={`
                block w-6 h-0.5 bg-current rounded-full
                transition-all duration-300
                ${isMobileMenuOpen ? 'rotate-45 translate-y-2' : ''}
              `} />
              <span className={`
                block w-6 h-0.5 bg-current rounded-full
                transition-all duration-300
                ${isMobileMenuOpen ? 'opacity-0 scale-0' : ''}
              `} />
              <span className={`
                block w-6 h-0.5 bg-current rounded-full
                transition-all duration-300
                ${isMobileMenuOpen ? '-rotate-45 -translate-y-2' : ''}
              `} />
            </div>
          </button>
        </div>

        {/* Mobile Navigation Menu */}
        <div className={`
          lg:hidden overflow-hidden
          transition-all duration-500 ease-out
          ${isMobileMenuOpen ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'}
        `}>
          <div className="py-4 space-y-2 border-t border-gray-100">
            {navigationLinks.map((link, index) => (
              link.external ? (
                <a
                  key={link.href}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="
                    flex items-center gap-3 px-4 py-3 rounded-xl
                    text-gray-600 hover:text-gray-900 hover:bg-gray-50
                    transition-all duration-300
                  "
                  style={{ animationDelay: `${index * 0.1}s` }}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  <span className="text-xl">{link.icon}</span>
                  <span className="font-medium">{link.label}</span>
                  <svg className="w-4 h-4 ml-auto opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              ) : (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`
                    flex items-center gap-3 px-4 py-3 rounded-xl
                    transition-all duration-300
                    ${isActiveLink(link.href)
                      ? 'bg-gold-50 text-gold-600'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }
                  `}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  <span className="text-xl">{link.icon}</span>
                  <span className="font-medium">{link.label}</span>
                  {isActiveLink(link.href) && (
                    <span className="ml-auto w-2 h-2 bg-gold-500 rounded-full" />
                  )}
                </Link>
              )
            ))}

            {/* Mobile AI Chat Button */}
            <a
              href="https://dharmamind.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="
                flex items-center justify-center gap-2 px-4 py-3 mx-2 rounded-xl
                bg-gradient-to-r from-gold-500 to-gold-600
                text-white font-semibold
                shadow-lg shadow-gold-500/25
                transition-all duration-300
              "
              onClick={() => setIsMobileMenuOpen(false)}
            >
              <span className="text-xl">🤖</span>
              <span>Open AI Chat</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </a>

            {/* Mobile Authentication */}
            {!isLoading && (
              <div className="pt-4 mt-2 border-t border-gray-100">
                {user ? (
                  <div className="space-y-3 px-2">
                    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-xl">
                      <div className="
                        w-12 h-12 rounded-xl
                        bg-gradient-to-br from-gold-400 to-gold-600
                        flex items-center justify-center
                        text-white font-bold text-lg
                        shadow-md
                      ">
                        {user.first_name?.charAt(0) || user.email.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <div className="font-semibold text-gray-900">
                          {user.first_name || 'User'}
                        </div>
                        <div className="text-sm text-gray-500">
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
                  <div className="px-2">
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
                  </div>
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
