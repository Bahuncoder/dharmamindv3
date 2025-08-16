import React from 'react';
import Link from 'next/link';
import Logo from './Logo';

const Navigation: React.FC = () => {
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
              className="btn-primary px-6 py-2 rounded-lg font-bold"
            >
              Chat with DharmaMind AI
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
