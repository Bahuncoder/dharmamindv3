/**
 * 🕉️ DharmaMind Centralized Footer Component
 * 
 * Professional footer used across all pages to ensure consistency
 */

import React from 'react';
import { useRouter } from 'next/router';
import Logo from './Logo';
import { ContactButton } from './CentralizedSupport';

interface FooterProps {
  variant?: 'simple' | 'professional';
  className?: string;
}

const Footer: React.FC<FooterProps> = ({
  variant = 'professional',
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

  // Simple footer for minimal pages
  if (variant === 'simple') {
    return (
      <footer className={`border-t border-stone-200/50 bg-white/80 backdrop-blur-sm ${className}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <Logo
              size="xs"
              className="justify-center mb-4"
            />
            <nav className="flex justify-center items-center space-x-8 mb-6">
              <button
                onClick={() => handleAnchorLink('#features')}
                className="text-sm text-stone-600 hover:text-stone-800 transition-colors inline-block"
              >
                Features
              </button>
              <button
                onClick={() => handleAnchorLink('#pricing')}
                className="text-sm text-stone-600 hover:text-stone-800 transition-colors inline-block"
              >
                Pricing
              </button>
              <button
                onClick={() => handleNavigation('/enterprise')}
                className="text-sm text-stone-600 hover:text-stone-800 transition-colors inline-block"
              >
                Enterprise
              </button>
              <button
                onClick={() => handleNavigation('/help')}
                className="text-sm text-stone-600 hover:text-stone-800 transition-colors inline-block"
              >
                Help
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

            {/* Social Media Links */}
            <div className="flex justify-center items-center space-x-6 mb-4">
              <a
                href="https://facebook.com/dharmamindai"
                target="_blank"
                rel="noopener noreferrer"
                className="text-stone-400 hover:text-stone-600 transition-colors"
                aria-label="Follow us on Facebook"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" />
                </svg>
              </a>
              <a
                href="https://twitter.com/dharmamindai"
                target="_blank"
                rel="noopener noreferrer"
                className="text-stone-400 hover:text-stone-600 transition-colors"
                aria-label="Follow us on Twitter/X"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z" />
                </svg>
              </a>
              <a
                href="https://linkedin.com/company/dharmamindai"
                target="_blank"
                rel="noopener noreferrer"
                className="text-stone-400 hover:text-stone-600 transition-colors"
                aria-label="Follow us on LinkedIn"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                </svg>
              </a>
            </div>

            <p className="text-sm text-stone-600 mb-4">
              AI with soul powered by dharma • Supreme consciousness through integrated wisdom
            </p>
          </div>
        </div>
      </footer>
    );
  }

  // Professional footer (default) - comprehensive and polished
  return (
    <footer className={`bg-white border-t border-gray-100 ${className}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">

        {/* Main Footer Content */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-8 mb-12">

          {/* Brand Section - Takes 2 columns on large screens */}
          <div className="col-span-2 md:col-span-2">
            <Logo
              size="md"
              showText={true}
              onClick={() => handleNavigation('/')}
              className="mb-6 cursor-pointer"
            />
            <p className="text-secondary mb-6 max-w-sm leading-relaxed">
              Revolutionary AI spiritual guidance platform that combines cutting-edge technology with ancient wisdom for modern professionals and leaders.
            </p>

            {/* Always visible Home link */}
            <button
              onClick={() => handleNavigation('/')}
              className="inline-flex items-center px-4 py-2 bg-primary/10 hover:bg-primary/20 text-primary rounded-lg transition-colors mb-4"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
              Back to Home
            </button>

            {/* Social Media */}
            <div className="flex items-center space-x-4">
              <span className="text-sm font-medium text-primary">Follow us:</span>
              <div className="flex space-x-3">
                <a
                  href="https://facebook.com/dharmamindai"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-8 h-8 bg-section-light rounded-full flex items-center justify-center text-secondary hover:bg-brand-primary hover:text-primary transition-all duration-200"
                  aria-label="Follow us on Facebook"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" />
                  </svg>
                </a>
                <a
                  href="https://twitter.com/dharmamindai"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-8 h-8 bg-section-light rounded-full flex items-center justify-center text-secondary hover:bg-brand-primary hover:text-primary transition-all duration-200"
                  aria-label="Follow us on Twitter/X"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z" />
                  </svg>
                </a>
                <a
                  href="https://linkedin.com/company/dharmamindai"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-8 h-8 bg-section-light rounded-full flex items-center justify-center text-secondary hover:bg-brand-primary hover:text-primary transition-all duration-200"
                  aria-label="Follow us on LinkedIn"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                  </svg>
                </a>
                <a
                  href="https://github.com/dharmamind"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-8 h-8 bg-section-light rounded-full flex items-center justify-center text-secondary hover:bg-brand-primary hover:text-primary transition-all duration-200"
                  aria-label="Follow us on GitHub"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                </a>
              </div>
            </div>
          </div>

          {/* Product Section */}
          <div>
            <h3 className="text-sm font-semibold text-primary mb-4 uppercase tracking-wider">Product</h3>
            <ul className="space-y-3">
              <li>
                <button
                  onClick={() => handleAnchorLink('#features')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  Features
                </button>
              </li>
              <li>
                <button
                  onClick={() => handleAnchorLink('#pricing')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  Pricing
                </button>
              </li>
              <li>
                <button
                  onClick={() => handleNavigation('/enterprise')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  Enterprise
                </button>
              </li>
              <li>
                <button
                  onClick={() => handleNavigation('/api-docs')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  API Documentation
                </button>
              </li>
            </ul>
          </div>

          {/* Support Section */}
          <div>
            <h3 className="text-sm font-semibold text-primary mb-4 uppercase tracking-wider">Support</h3>
            <ul className="space-y-3">
              <li>
                <button
                  onClick={() => handleNavigation('/help')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  Help Center
                </button>
              </li>
              <li>
                <ContactButton
                  variant="link"
                  prefillCategory="support"
                  showIcon={false}
                  className="!text-secondary hover:!text-primary !text-left !block !text-sm !transition-colors !no-underline !justify-start !w-full !font-normal !p-0 !m-0"
                >
                  Contact Support
                </ContactButton>
              </li>
              <li>
                <button
                  onClick={() => handleNavigation('/docs')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  Documentation
                </button>
              </li>
              <li>
                <button
                  onClick={() => handleNavigation('/status')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  System Status
                </button>
              </li>
            </ul>
          </div>

          {/* Company Section */}
          <div>
            <h3 className="text-sm font-semibold text-primary mb-4 uppercase tracking-wider">Company</h3>
            <ul className="space-y-3">
              <li>
                <button
                  onClick={() => handleNavigation('/about')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  About Us
                </button>
              </li>
              <li>
                <a
                  href="https://dharmamind.org"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  Blog
                </a>
              </li>
              <li>
                <button
                  onClick={() => handleNavigation('/careers')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  Careers
                </button>
              </li>
              <li>
                <button
                  onClick={() => handleNavigation('/privacy')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  Privacy Policy
                </button>
              </li>
              <li>
                <button
                  onClick={() => handleNavigation('/terms')}
                  className="text-secondary hover:text-primary text-left block text-sm transition-colors"
                >
                  Terms of Service
                </button>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-brand-accent pt-8">
          <div className="flex flex-col lg:flex-row justify-between items-center space-y-4 lg:space-y-0">

            {/* Legal Links */}
            <div className="flex items-center space-x-4 text-sm text-secondary">
              <button
                onClick={() => handleNavigation('/security')}
                className="hover:text-primary transition-colors"
              >
                Security
              </button>
              <button
                onClick={() => handleNavigation('/cookies')}
                className="hover:text-primary transition-colors"
              >
                Cookie Policy
              </button>
            </div>

            {/* Copyright - Center */}
            <div className="text-sm text-secondary">
              <span>© 2025 DharmaMind. All rights reserved.</span>
            </div>

            {/* Tagline */}
            <div className="text-sm text-secondary font-medium">
              Built with consciousness and compassion 🕉️
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
