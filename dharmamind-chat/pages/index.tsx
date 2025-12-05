import React, { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';

// Spiritual interfaces
interface LoadingStep {
  progress: number;
  text: string;
  delay: number;
  icon?: string;
}

interface LoadingState {
  progress: number;
  currentStep: number;
  isComplete: boolean;
  hasError: boolean;
}

const HomePage: React.FC = () => {
  const router = useRouter();
  const [showContent, setShowContent] = useState(false);
  const [loadingState, setLoadingState] = useState<LoadingState>({
    progress: 0,
    currentStep: 0,
    isComplete: false,
    hasError: false
  });
  const [loadingText, setLoadingText] = useState('Awakening consciousness...');

  // Spiritual loading configuration - INSTANT FEEL
  const loadingSteps: LoadingStep[] = [
    { progress: 25, text: 'Welcome to DharmaMind', delay: 50, icon: '🧘' },
    { progress: 50, text: 'AI with Soul awakening...', delay: 100, icon: '✨' },
    { progress: 75, text: 'Dharmic intelligence ready', delay: 150, icon: '🕉️' },
    { progress: 100, text: 'Entering conscious chat...', delay: 200, icon: '💫' }
  ];

  // Progressive loading with demo chat redirect
  const initializeLoading = useCallback(() => {
    setShowContent(true);
    
    // Single smooth transition instead of steps
    setTimeout(() => {
      setLoadingState(prev => ({
        ...prev,
        progress: 100,
        currentStep: 3,
        isComplete: true
      }));
      setLoadingText('Welcome to DharmaMind');
      
      // Immediate redirect - no waiting
      setTimeout(() => {
        router.replace('/auth?mode=login');
      }, 100);
    }, 200);
  }, [router]);

  useEffect(() => {
    initializeLoading();
  }, [initializeLoading]);

  return (
    <>
      <Head>
        <title>DharmaMind - AI with Soul, Powered by Dharma</title>
        <meta name="description" content="Experience AI with soul - Conscious conversations guided by dharmic wisdom and spiritual intelligence" />
        <meta name="keywords" content="dharmic ai, ai with soul, conscious ai, spiritual wisdom, dharma, meditation, enlightenment" />
        
        {/* Spiritual Metadata */}
        <meta property="og:title" content="DharmaMind - AI with Soul" />
        <meta property="og:description" content="Conscious AI conversations powered by dharmic wisdom" />
        <meta property="og:type" content="website" />
        
        {/* Performance optimization */}
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="theme-color" content="#10b981" />
        
        {/* Spiritual Icons */}
        <link rel="icon" type="image/x-icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen flex items-center justify-center relative overflow-hidden" 
           style={{
             background: 'linear-gradient(135deg, var(--color-neutral-800) 0%, var(--color-neutral-700) 50%, var(--color-neutral-600) 100%)',
           }}
           role="main" 
           aria-label="DharmaMind - AI with Soul Loading Interface">
        
        {/* Spiritual Floating Elements - PEACEFUL */}
        <div className="absolute inset-0 overflow-hidden" aria-hidden="true">
          {/* Gentle static elements */}
          <div className="absolute top-1/4 left-1/4 w-2 h-2 rounded-full opacity-40"
               style={{background: 'var(--color-success)'}}></div>
          <div className="absolute top-3/4 right-1/4 w-2 h-2 rounded-full opacity-30"
               style={{background: 'var(--color-accent)'}}></div>
          <div className="absolute top-1/2 left-3/4 w-3 h-3 rounded-full opacity-20"
               style={{background: 'var(--color-success)'}}></div>
          <div className="absolute bottom-1/4 left-1/2 w-2 h-2 rounded-full opacity-35"
               style={{background: 'var(--color-accent)'}}></div>
          
          {/* Static spiritual symbols */}
          <div className="absolute top-1/6 right-1/3 text-xl opacity-15" style={{color: 'var(--color-logo-emerald)'}}>🕉️</div>
          <div className="absolute bottom-1/3 right-1/6 text-lg opacity-10" style={{color: 'var(--color-accent)'}}>✨</div>
          <div className="absolute top-2/3 left-1/6 text-base opacity-20" style={{color: 'var(--color-logo-emerald)'}}>🧘</div>
          
          {/* Soft static orbs */}
          <div className="absolute -top-40 -right-40 w-80 h-80 rounded-full opacity-5"
               style={{background: 'radial-gradient(circle, var(--color-success), transparent)'}}></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 rounded-full opacity-5"
               style={{background: 'radial-gradient(circle, var(--color-accent), transparent)'}}></div>
        </div>

        {/* Main Content */}
        <div className={`relative z-10 text-center max-w-lg mx-auto px-6 transition-all duration-700 ease-out ${
          showContent ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        }`}>
          
          {/* Spiritual Logo Section - Matches Logo component sm size exactly */}
          <div className="mb-12">
            <div className="w-28 h-28 mx-auto mb-8 rounded-lg flex items-center justify-center relative overflow-hidden"
                 style={{
                   background: 'linear-gradient(135deg, var(--color-accent), var(--color-accent-hover))',
                   border: '3px solid var(--color-logo-emerald)',
                   boxShadow: '0 20px 60px rgba(16, 185, 129, 0.4), 0 8px 30px rgba(249, 115, 22, 0.3)'
                 }}>
              {/* Use exact same structure as Logo component */}
              <img
                src="/logo.jpeg"
                alt="DharmaMind Logo"
                className="w-full h-full object-cover rounded-lg relative z-10"
                style={{
                  objectFit: 'cover',
                  objectPosition: 'center center'
                }}
              />
              {/* Professional overlay gradient exactly like Logo component */}
              <div 
                className="absolute inset-0 rounded-lg"
                style={{
                  background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.1), rgba(16, 185, 129, 0.1))',
                  zIndex: 5
                }}
              />
              {/* Gentle static glow */}
              <div className="absolute inset-0 rounded-lg opacity-30"
                   style={{background: 'linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent)'}}></div>
            </div>
            
            <h1 className="text-5xl md:text-6xl font-bold mb-6 text-white">
              <span className="text-transparent bg-clip-text bg-gradient-to-r" 
                   style={{backgroundImage: 'linear-gradient(45deg, var(--color-accent), var(--color-success))'}}>
                DharmaMind
              </span>
            </h1>
            
            <p className="text-xl text-gray-300 mb-4">
              AI with Soul • Powered by Dharma
            </p>
            
            <div className="flex justify-center items-center space-x-4 text-sm text-gray-400">
              <span className="flex items-center space-x-1">
                <span>🧘</span>
                <span>Conscious</span>
              </span>
              <span>•</span>
              <span className="flex items-center space-x-1">
                <span>✨</span>
                <span>Spiritual</span>
              </span>
              <span>•</span>
              <span className="flex items-center space-x-1">
                <span>🕉️</span>
                <span>Dharmic</span>
              </span>
            </div>
          </div>

          {/* Loading Section */}
          <div className="space-y-8">
            
            {/* Spiritual Loading Animation - COMPLETELY STATIC */}
            <div className="flex justify-center items-center space-x-3">
              <div className="relative">
                {/* Simple static circle with icon */}
                <div className="w-16 h-16 rounded-full flex items-center justify-center"
                     style={{
                       background: 'linear-gradient(45deg, var(--color-success), var(--color-accent))',
                       boxShadow: '0 0 20px rgba(16, 185, 129, 0.3)'
                     }}>
                  <span className="text-2xl">🧘</span>
                </div>
              </div>
            </div>

            {/* Spiritual Loading Text */}
            <div className="space-y-4">
              <p className="text-2xl font-semibold text-white">
                Welcome to DharmaMind
              </p>
              
              <div className="text-gray-400 space-y-2">
                <p className="text-sm">
                  AI with Soul • Powered by Dharma
                </p>
                <p className="text-xs flex items-center justify-center space-x-4">
                  <span>🧘 Conscious</span>
                  <span>•</span>
                  <span>✨ Spiritual</span>
                  <span>•</span>
                  <span>🕉️ Dharmic</span>
                </p>
              </div>
            </div>

            {/* Spiritual Progress Bar - SIMPLE */}
            <div className="w-full max-w-md mx-auto space-y-3">
              <div className="flex justify-between text-sm text-gray-400">
                <span>Entering Conscious Chat</span>
                <span className="font-mono">Ready</span>
              </div>
              
              <div className="w-full h-2 rounded-full overflow-hidden"
                   style={{backgroundColor: 'rgba(255,255,255,0.1)'}}>
                <div className="h-full rounded-full relative overflow-hidden"
                     style={{
                       background: 'linear-gradient(90deg, var(--color-success), var(--color-accent))',
                       width: '100%',
                       boxShadow: '0 0 15px var(--color-success)'
                     }}>
                  {/* Gentle static shimmer */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-20"></div>
                </div>
              </div>
            </div>

            {/* Immediate entry message */}
            <div className="text-center space-y-2">
              <p className="text-sm" style={{color: 'var(--color-success)'}}>
                ✨ Entering conscious chat experience
              </p>
            </div>

            {/* Simple static dots */}
            <div className="flex justify-center space-x-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} 
                     className="w-3 h-3 rounded-full opacity-100"
                     style={{
                       backgroundColor: i % 2 === 0 ? 'var(--color-success)' : 'var(--color-accent)'
                     }}></div>
              ))}
            </div>
          </div>
        </div>

        {/* Ultra-minimal CSS for seamless experience */}
        <style jsx>{`
          /* Gentle logo hover effect only */
          .logo-professional:hover {
            transform: scale(1.01);
            transition: transform 0.2s ease;
          }
          
          /* Responsive design for mobile */
          @media (max-width: 768px) {
            .spiritual-mobile-optimize {
              transform: scale(0.95);
            }
          }
          
          /* Accessibility - completely static */
          @media (prefers-reduced-motion: reduce) {
            * {
              animation: none !important;
              transition: none !important;
            }
          }
        `}</style>
      </div>
    </>
  );
};

export default HomePage;
