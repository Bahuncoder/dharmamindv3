/**
 * 🕉️ DharmaMind Centralized Footer Component
 * 
 * Unified footer used across all pages to ensure consistency
 */

import React from 'react';
import { useRouter } from 'next/router';
import Logo from './Logo';
import { ContactButton, SupportSection } from './CentralizedSupport';

interface FooterProps {
  variant?: 'simple' | 'detailed';
  className?: string;
}

const Footer: React.FC<FooterProps> = ({ 
  variant = 'simple', 
  className = '' 
}) => {
  const router = useRouter();

  const handleNavigation = (path: string) => {
    router.push(path);
  };

  const handleAnchorLink = (anchor: string) => {
    // For anchor links, scroll to section on current page or navigate to home page with anchor
    if (router.pathname === '/') {
      // Already on home page, just scroll
      const element = document.getElementById(anchor.replace('#', ''));
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      // Navigate to home page with anchor
      router.push(`/${anchor}`);
    }
  };

  if (variant === 'simple') {
    return (
      <footer className={`border-t backdrop-blur-sm ${className}`} style={{
        borderColor: 'var(--color-border-light)',
        backgroundColor: 'var(--color-bg-card)'
      }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <Logo 
              size="xs"
              className="justify-center mb-4"
            />
            <nav className="flex justify-center items-center space-x-8 mb-6">
              <button
                onClick={() => handleAnchorLink('#features')}
                className="text-sm transition-colors inline-block hover:opacity-80"
                style={{ color: 'var(--color-text-secondary)' }}
              >
                Features
              </button>
              <button
                onClick={() => handleAnchorLink('#pricing')}
                className="text-sm transition-colors inline-block hover:opacity-80"
                style={{ color: 'var(--color-text-secondary)' }}
              >
                Pricing
              </button>
              <button 
                onClick={() => handleNavigation('/enterprise')}
                className="text-sm transition-colors inline-block hover:opacity-80"
                style={{ color: 'var(--color-text-secondary)' }}
              >
                Enterprise
              </button>
              <button 
                onClick={() => handleNavigation('/api-docs')}
                className="text-sm text-stone-600 hover:text-stone-800 transition-colors inline-block"
              >
                API
              </button>
              <button 
                onClick={() => window.open('https://dharmamind.com/help', '_blank')}
                className="text-sm text-stone-600 hover:text-stone-800 transition-colors inline-flex items-center gap-1"
              >
                Help
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </button>
              <ContactButton 
                variant="link"
                prefillCategory="general"
                showIcon={false}
                className="!text-sm !text-stone-600 hover:!text-stone-800 !transition-colors !inline-block !no-underline"
              >
                Contact
              </ContactButton>
            </nav>
            <p className="text-sm text-stone-600 mb-4">
              AI with soul powered by dharma • Supreme consciousness through integrated wisdom
            </p>
            <p className="text-xs text-stone-500">
              © 2025 DharmaMind. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    );
  }

  // Detailed footer variant
  return (
    <footer className={`border-t border-gray-200 ${className}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          
          {/* Brand Section */}
          <div className="col-span-1">
            <Logo 
              size="sm"
              showText={true}
              onClick={() => handleNavigation('/')}
              className="mb-4 cursor-pointer"
            />
            <p className="text-sm text-slate-600">
              Professional AI spiritual guidance platform for modern professionals and leaders.
            </p>
          </div>

          {/* Product Section */}
          <div>
            <h3 className="text-sm font-semibold text-slate-900 mb-4">Product</h3>
            <ul className="space-y-2 text-sm text-slate-600">
              <li>
                <button 
                  onClick={() => handleAnchorLink('#features')} 
                  className="hover:text-slate-900 text-left"
                >
                  Features
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleAnchorLink('#pricing')} 
                  className="hover:text-slate-900 text-left"
                >
                  Pricing
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleNavigation('/api-docs')} 
                  className="hover:text-slate-900 text-left"
                >
                  API Documentation
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleNavigation('/enterprise')} 
                  className="hover:text-slate-900 text-left"
                >
                  Enterprise
                </button>
              </li>
            </ul>
          </div>

          {/* Support Section */}
          <div>
            <h3 className="text-sm font-semibold text-slate-900 mb-4">Support</h3>
            <ul className="space-y-2 text-sm text-slate-600">
              <li>
                <button 
                  onClick={() => window.open('https://dharmamind.com/help', '_blank')} 
                  className="hover:text-slate-900 text-left inline-flex items-center gap-1"
                >
                  Help Center
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleNavigation('/docs')} 
                  className="hover:text-slate-900 text-left"
                >
                  Documentation
                </button>
              </li>
              <li>
                <ContactButton 
                  variant="link"
                  prefillCategory="support"
                  showIcon={false}
                  className="!text-sm !text-slate-600 hover:!text-slate-900 !text-left !no-underline"
                >
                  Contact Support
                </ContactButton>
              </li>
              <li>
                <ContactButton 
                  variant="link"
                  prefillCategory="feedback"
                  showIcon={false}
                  className="!text-sm !text-slate-600 hover:!text-slate-900 !text-left !no-underline"
                >
                  Feedback
                </ContactButton>
              </li>
            </ul>
          </div>

          {/* Company Section */}
          <div>
            <h3 className="text-sm font-semibold text-slate-900 mb-4">Company</h3>
            <ul className="space-y-2 text-sm text-slate-600">
              <li>
                <button 
                  onClick={() => handleAnchorLink('#about')} 
                  className="hover:text-slate-900 text-left"
                >
                  About
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleNavigation('/privacy')} 
                  className="hover:text-slate-900 text-left"
                >
                  Privacy Policy
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleNavigation('/terms')} 
                  className="hover:text-slate-900 text-left"
                >
                  Terms of Service
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleNavigation('/security')} 
                  className="hover:text-slate-900 text-left"
                >
                  Security
                </button>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-200 mt-8 pt-8 text-center">
          <p className="text-sm text-slate-600">
            © 2025 DharmaMind. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
