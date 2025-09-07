import React, { useEffect } from 'react';
import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion';
import { useRouter } from 'next/router';
import { useSession } from 'next-auth/react';
import { useAuth } from '../contexts/AuthContext';
import Head from 'next/head';
import { useCentralizedSystem } from '../components/CentralizedSystem';
import Logo from '../components/Logo';
import Button from '../components/Button';
import Footer from '../components/Footer';
import SpiritualQuotes from '../components/SpiritualQuotes';
import SEOHead from '../components/SEOHead';
import { useLoading } from '../hooks/useLoading';
import { LoadingPage } from '../components/CentralizedLoading';

// Advanced animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      delayChildren: 0.3,
      staggerChildren: 0.2
    }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: "spring",
      damping: 20,
      stiffness: 100
    }
  }
};

const floatingVariants = {
  initial: { y: 0, x: 0, rotate: 0 },
  animate: {
    y: [-20, 20, -20],
    x: [-10, 10, -10],
    rotate: [-5, 5, -5],
    transition: {
      duration: 6,
      repeat: Infinity,
      ease: "easeInOut"
    }
  }
};

const WelcomePage: React.FC = () => {
  const router = useRouter();
  const { data: session, status } = useSession();
  const { user, isAuthenticated } = useAuth();
  const { isLoading, withLoading } = useLoading();
  const { goToAuth, toggleSubscriptionModal, toggleAuthModal } = useCentralizedSystem();
  const { scrollY } = useScroll();
  const y1 = useTransform(scrollY, [0, 300], [0, -50]);
  const y2 = useTransform(scrollY, [0, 300], [0, -100]);
  
  // State for mobile menu
  const [isMobileMenuOpen, setIsMobileMenuOpen] = React.useState(false);

  useEffect(() => {
    // Check for subscription upgrade request in URL
    if (router.query.upgrade === 'true') {
      toggleSubscriptionModal(true);
    }
  }, [router.query]);

  const handleGetStarted = () => {
    goToAuth('signup');
  };

  const handleTryFree = async () => {
    // Navigate to dharmamind.ai for demo
    window.location.href = 'https://dharmamind.ai';
  };

  const handleSubscriptionClick = () => {
    if (isAuthenticated) {
      toggleSubscriptionModal(true);
    } else {
      toggleAuthModal(true);
    }
  };

  // Show loading while checking session
  if (status === 'loading') {
    return (
      <LoadingPage 
        message="Loading your spiritual companion..."
        submessage="Preparing your personalized experience"
        showLogo={true}
      />
    );
  }

  return (
    <>
      <SEOHead 
        title="DharmaMind - AI with Soul Powered by Dharma"
        description="Experience DharmaMind - AI with Soul Powered by Dharma. Advanced intelligence combined with ancient wisdom for insights that truly matter. Get clarity, wisdom, and purpose-driven decisions."
        keywords="AI with soul, dharmic AI, ethical AI companion, conscious AI, spiritual technology, AI wisdom, mindful AI, purpose-driven AI, dharma AI, enlightened AI, soulful AI assistant"
        image="/og-image.jpg"
      />

      <div className="min-h-screen bg-primary-background-main">
        {/* Enhanced Modern Header */}
        <motion.header 
          className="fixed top-0 left-0 right-0 z-50 transition-all duration-300"
          initial={{ y: -100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, type: "spring", stiffness: 100 }}
        >
          {/* Glassmorphism background with gradient border */}
          <div className="bg-gradient-to-r from-white/95 via-white/90 to-white/95 backdrop-blur-2xl border-b border-white/20 shadow-xl">
            {/* Subtle top accent line */}
            <div className="h-0.5 bg-brand-gradient"></div>
            
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between h-20">
                
                {/* Enhanced Logo with animation */}
                <motion.div
                  whileHover={{ scale: 1.05, rotate: 1 }}
                  whileTap={{ scale: 0.95 }}
                  transition={{ type: "spring", stiffness: 300 }}
                  className="relative group"
                >
                  <Logo 
                    size="sm"
                    onClick={() => router.push('/')}
                    className="transition-all duration-300 group-hover:drop-shadow-lg"
                  />
                  {/* Subtle glow effect on hover */}
                  <div className="absolute inset-0 bg-brand-primary/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                </motion.div>

                {/* Enhanced Navigation with hover effects */}
                <nav className="hidden md:flex items-center space-x-1">
                  {[
                    { label: 'About', path: '/about', icon: '🌟' },
                    { label: 'Features', path: '/features', icon: '⚡' },
                    { label: 'Pricing', path: '/pricing', icon: '💎' },
                    { label: 'Enterprise', path: '/enterprise', icon: '🏢' }
                  ].map((item, index) => (
                    <motion.button
                      key={item.label}
                      onClick={() => router.push(item.path)}
                      className="relative px-4 py-2 text-secondary hover:text-primary text-sm font-medium rounded-xl transition-all duration-300 group overflow-hidden"
                      whileHover={{ scale: 1.05, y: -1 }}
                      whileTap={{ scale: 0.95 }}
                      initial={{ opacity: 0, y: -20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 * index, type: "spring", stiffness: 200 }}
                    >
                      {/* Hover background effect */}
                      <motion.div
                        className="absolute inset-0 bg-gradient-to-r from-brand-primary/10 to-brand-accent/10 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                        layoutId="navHover"
                      />
                      
                      {/* Icon and text */}
                      <span className="relative z-10 flex items-center space-x-2">
                        <span className="text-xs opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                          {item.icon}
                        </span>
                        <span>{item.label}</span>
                      </span>
                      
                      {/* Bottom accent line */}
                      <motion.div
                        className="absolute bottom-0 left-1/2 h-0.5 bg-brand-gradient rounded-full"
                        initial={{ width: 0, x: "-50%" }}
                        whileHover={{ width: "80%" }}
                        transition={{ duration: 0.3 }}
                      />
                    </motion.button>
                  ))}
                  
                  {/* External link with special styling */}
                  <motion.a
                    href="https://dharmamind.org"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="relative px-4 py-2 text-secondary hover:text-primary text-sm font-medium rounded-xl transition-all duration-300 group overflow-hidden"
                    whileHover={{ scale: 1.05, y: -1 }}
                    whileTap={{ scale: 0.95 }}
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4, type: "spring", stiffness: 200 }}
                  >
                    {/* Hover background effect */}
                    <motion.div
                      className="absolute inset-0 bg-gradient-to-r from-brand-secondary/10 to-brand-primary/10 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                    />
                    
                    <span className="relative z-10 flex items-center space-x-2">
                      <span className="text-xs opacity-0 group-hover:opacity-100 transition-opacity duration-300">🌐</span>
                      <span>Our Community</span>
                      <motion.span
                        className="text-xs"
                        animate={{ x: [0, 2, 0] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      >
                        →
                      </motion.span>
                    </span>
                    
                    {/* Bottom accent line */}
                    <motion.div
                      className="absolute bottom-0 left-1/2 h-0.5 bg-gradient-to-r from-brand-secondary to-brand-primary rounded-full"
                      initial={{ width: 0, x: "-50%" }}
                      whileHover={{ width: "80%" }}
                      transition={{ duration: 0.3 }}
                    />
                  </motion.a>
                </nav>

                {/* Enhanced Action Buttons */}
                <div className="flex items-center space-x-3">
                  <AnimatePresence mode="wait">
                    {session || isAuthenticated ? (
                      <motion.div
                        key="authenticated"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        className="flex items-center space-x-3"
                      >
                        {/* User Profile Section */}
                        <motion.div 
                          className="hidden sm:flex items-center space-x-3 bg-white/70 backdrop-blur-xl rounded-full px-4 py-2 border border-white/30 shadow-lg"
                          whileHover={{ scale: 1.02, y: -1 }}
                          transition={{ type: "spring", stiffness: 300 }}
                        >
                          <div className="w-8 h-8 bg-brand-gradient rounded-full flex items-center justify-center text-white text-sm font-bold">
                            {(session?.user?.email || user?.email)?.charAt(0).toUpperCase() || 'U'}
                          </div>
                          <div className="text-right">
                            <div className="text-sm font-semibold text-primary">
                              {(session?.user?.email || user?.email)?.split('@')[0] || 'User'}
                            </div>
                            <div className="text-xs text-brand-accent font-medium">
                              {user?.subscription_plan || 'Basic Plan'}
                            </div>
                          </div>
                        </motion.div>
                        
                        {/* Upgrade Button for Free Users */}
                        {user?.subscription_plan === 'free' && (
                          <motion.div
                            whileHover={{ scale: 1.05, y: -1 }}
                            whileTap={{ scale: 0.95 }}
                          >
                            <Button
                              onClick={() => toggleSubscriptionModal(true)}
                              variant="outline"
                              size="sm"
                              className="relative overflow-hidden border-brand-accent text-brand-accent hover:bg-brand-accent hover:text-white transition-all duration-300 group"
                            >
                              <span className="relative z-10">✨ Upgrade</span>
                              {/* Shimmer effect */}
                              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
                            </Button>
                          </motion.div>
                        )}
                        
                        {/* Primary Action Button */}
                        <motion.div
                          whileHover={{ scale: 1.05, y: -1 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <Button
                            onClick={handleTryFree}
                            variant="primary"
                            size="sm"
                            className="relative overflow-hidden bg-brand-gradient hover:shadow-xl transition-all duration-300 group"
                          >
                            <span className="relative z-10">Open Chat</span>
                            {/* Shimmer effect */}
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
                          </Button>
                        </motion.div>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="unauthenticated"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        className="flex items-center space-x-3"
                      >
                        <motion.div
                          whileHover={{ scale: 1.05, y: -1 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <Button
                            onClick={() => goToAuth('login')}
                            variant="outline"
                            size="sm"
                            className="hover:bg-white/90 transition-all duration-300 bg-white/70 backdrop-blur-xl border-white/30 shadow-lg"
                          >
                            Sign In
                          </Button>
                        </motion.div>
                        
                        <motion.div
                          whileHover={{ scale: 1.05, y: -1 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <Button
                            onClick={handleTryFree}
                            variant="primary"
                            size="sm"
                            className="relative overflow-hidden bg-brand-gradient hover:shadow-xl transition-all duration-300 group"
                          >
                            <span className="relative z-10">🚀 Try Demo</span>
                            {/* Shimmer effect */}
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
                          </Button>
                        </motion.div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                  
                  {/* Mobile Menu Button */}
                  <motion.button
                    className="md:hidden p-2 rounded-xl bg-white/70 backdrop-blur-xl border border-white/30 shadow-lg"
                    whileHover={{ scale: 1.05, y: -1 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                  >
                    <motion.svg 
                      className="w-5 h-5 text-primary" 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                      animate={{ rotate: isMobileMenuOpen ? 90 : 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d={isMobileMenuOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"} 
                      />
                    </motion.svg>
                  </motion.button>
                </div>
              </div>
            </div>
            
            {/* Enhanced Mobile Menu */}
            <AnimatePresence>
              {isMobileMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3, ease: "easeInOut" }}
                  className="md:hidden bg-white/95 backdrop-blur-2xl border-t border-white/20"
                >
                  <div className="max-w-7xl mx-auto px-4 py-6">
                    <div className="space-y-4">
                      {[
                        { label: 'About', path: '/about', icon: '🌟' },
                        { label: 'Features', path: '/features', icon: '⚡' },
                        { label: 'Pricing', path: '/pricing', icon: '💎' },
                        { label: 'Enterprise', path: '/enterprise', icon: '🏢' }
                      ].map((item, index) => (
                        <motion.button
                          key={item.label}
                          onClick={() => {
                            router.push(item.path);
                            setIsMobileMenuOpen(false);
                          }}
                          className="w-full flex items-center space-x-3 px-4 py-3 text-left text-secondary hover:text-primary rounded-xl transition-all duration-300 hover:bg-white/70"
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                          whileHover={{ scale: 1.02, x: 5 }}
                        >
                          <span className="text-lg">{item.icon}</span>
                          <span className="font-medium">{item.label}</span>
                        </motion.button>
                      ))}
                      
                      <motion.a
                        href="https://dharmamind.org"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="w-full flex items-center space-x-3 px-4 py-3 text-left text-secondary hover:text-primary rounded-xl transition-all duration-300 hover:bg-white/70"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.4 }}
                        whileHover={{ scale: 1.02, x: 5 }}
                      >
                        <span className="text-lg">🌐</span>
                        <span className="font-medium">Our Community</span>
                        <span className="text-xs">→</span>
                      </motion.a>
                      
                      {/* Mobile Action Buttons */}
                      <div className="pt-4 border-t border-white/20 space-y-3">
                        {session || isAuthenticated ? (
                          <>
                            <Button
                              onClick={handleTryFree}
                              variant="primary"
                              className="w-full bg-brand-gradient"
                            >
                              Open Chat
                            </Button>
                            {user?.subscription_plan === 'free' && (
                              <Button
                                onClick={() => {
                                  toggleSubscriptionModal(true);
                                  setIsMobileMenuOpen(false);
                                }}
                                variant="outline"
                                className="w-full border-brand-accent text-brand-accent"
                              >
                                ✨ Upgrade Plan
                              </Button>
                            )}
                          </>
                        ) : (
                          <>
                            <Button
                              onClick={() => {
                                goToAuth('login');
                                setIsMobileMenuOpen(false);
                              }}
                              variant="outline"
                              className="w-full"
                            >
                              Sign In
                            </Button>
                            <Button
                              onClick={() => {
                                handleTryFree();
                                setIsMobileMenuOpen(false);
                              }}
                              variant="primary"
                              className="w-full bg-brand-gradient"
                            >
                              🚀 Try Demo
                            </Button>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* Subtle bottom border animation */}
            <motion.div
              className="absolute bottom-0 left-0 h-0.5 bg-brand-gradient"
              initial={{ width: 0 }}
              animate={{ width: "100%" }}
              transition={{ duration: 2, delay: 0.5 }}
            />
          </div>
        </motion.header>

        <main className="pt-16">
          {/* Enhanced Modern Hero Section */}
          <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
            {/* Advanced Animated Background */}
            <div className="absolute inset-0">
              {/* Gradient Mesh Background */}
              <div className="absolute inset-0 bg-gradient-to-br from-section-light via-white to-brand-primary/5"></div>
              
              {/* Floating Orbs with Advanced Animation */}
              <motion.div
                variants={floatingVariants}
                initial="initial"
                animate="animate"
                className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-br from-brand-primary/20 to-brand-accent/20 rounded-full blur-xl"
                style={{ y: y1 }}
              />
              <motion.div
                variants={floatingVariants}
                initial="initial"
                animate="animate"
                className="absolute top-40 right-20 w-24 h-24 bg-gradient-to-br from-brand-accent/20 to-brand-secondary/20 rounded-full blur-lg"
                style={{ y: y2, animationDelay: '1s' }}
              />
              <motion.div
                variants={floatingVariants}
                initial="initial"
                animate="animate"
                className="absolute bottom-40 left-1/4 w-40 h-40 bg-gradient-to-br from-brand-secondary/20 to-brand-primary/20 rounded-full blur-2xl"
                style={{ y: y1, animationDelay: '2s' }}
              />
              <motion.div
                variants={floatingVariants}
                initial="initial"
                animate="animate"
                className="absolute bottom-20 right-1/3 w-28 h-28 bg-gradient-to-br from-brand-primary/20 to-brand-accent/20 rounded-full blur-xl"
                style={{ y: y2, animationDelay: '0.5s' }}
              />
              
              {/* Geometric Patterns */}
              <div className="absolute inset-0 opacity-10">
                <svg className="absolute top-1/4 left-1/4 w-64 h-64 text-primary" viewBox="0 0 100 100">
                  <defs>
                    <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                      <path d="M 10 0 L 0 0 0 10" fill="none" stroke="currentColor" strokeWidth="0.5"/>
                    </pattern>
                  </defs>
                  <rect width="100" height="100" fill="url(#grid)" />
                </svg>
              </div>
            </div>

            <motion.div 
              className="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center"
              variants={containerVariants}
              initial="hidden"
              animate="visible"
            >
              <motion.div className="space-y-8">
                {/* Enhanced Logo/Brand with Glassmorphism */}
                <motion.div
                  variants={itemVariants}
                  className="inline-flex items-center space-x-3 bg-white/90 backdrop-blur-2xl rounded-full px-8 py-4 border border-white/20 shadow-2xl"
                  whileHover={{ scale: 1.05, y: -5 }}
                  transition={{ type: "spring", stiffness: 300 }}
                >
                  <motion.span 
                    className="text-3xl"
                    animate={{ rotate: [0, 10, -10, 0] }}
                    transition={{ duration: 4, repeat: Infinity }}
                  >
                    🕉️
                  </motion.span>
                  <span className="text-xl font-bold text-primary">AI with Soul</span>
                </motion.div>
                
                {/* Welcome Message */}
                <motion.div variants={itemVariants} className="text-center">
                  <motion.h2
                    className="text-2xl md:text-4xl font-medium text-primary mb-4"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3, duration: 0.8 }}
                  >
                    Welcome to{' '}
                    <motion.span 
                      className="font-bold text-primary"
                      whileHover={{ scale: 1.05 }}
                      transition={{ type: "spring", stiffness: 300 }}
                    >
                      DharmaMind
                    </motion.span>
                  </motion.h2>
                </motion.div>
                
                {/* Enhanced Main Headline with Text Animation */}
                <motion.div variants={itemVariants} className="relative">
                  <motion.h1
                    className="text-7xl md:text-9xl font-black text-primary leading-tight relative"
                    initial={{ scale: 0.5, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ 
                      type: "spring", 
                      stiffness: 100, 
                      damping: 15,
                      delay: 0.5 
                    }}
                  >
                    AI with Soul
                  </motion.h1>
                </motion.div>
                
                {/* Enhanced Subtitle with Typing Effect */}
                <motion.div variants={itemVariants}>
                  <motion.p
                    className="text-3xl md:text-5xl text-primary font-light max-w-4xl mx-auto"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1.2, duration: 1 }}
                  >
                    Powered by{' '}
                    <motion.span 
                      className="font-bold text-primary relative"
                      whileHover={{ scale: 1.1 }}
                      transition={{ type: "spring", stiffness: 300 }}
                    >
                      Dharma
                      {/* Subtle underline animation */}
                      <motion.div 
                        className="absolute bottom-0 left-0 h-1 bg-brand-gradient rounded"
                        initial={{ width: 0 }}
                        animate={{ width: "100%" }}
                        transition={{ delay: 2, duration: 0.8 }}
                      />
                    </motion.span>
                  </motion.p>
                </motion.div>
                
                {/* Enhanced Description with Particle Effect */}
                <motion.div variants={itemVariants} className="relative">
                  <motion.p
                    className="text-xl md:text-2xl text-secondary max-w-5xl mx-auto leading-relaxed relative z-10"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 1.5, duration: 0.8 }}
                  >
                    Experience the future of AI guidance. Advanced intelligence combined with ancient wisdom 
                    for insights that truly matter. Get clarity, wisdom, and purpose-driven decisions.
                  </motion.p>
                  {/* Floating particles */}
                  <div className="absolute inset-0 overflow-hidden pointer-events-none">
                    {[...Array(6)].map((_, i) => (
                      <motion.div
                        key={i}
                        className="absolute w-1 h-1 bg-brand-accent rounded-full"
                        style={{
                          left: `${20 + i * 15}%`,
                          top: `${30 + (i % 2) * 40}%`,
                        }}
                        animate={{
                          y: [-10, 10, -10],
                          opacity: [0.3, 1, 0.3],
                        }}
                        transition={{
                          duration: 3 + i * 0.5,
                          repeat: Infinity,
                          delay: i * 0.5,
                        }}
                      />
                    ))}
                  </div>
                </motion.div>
                
                {/* Enhanced CTA Buttons with Advanced Interactions */}
                <motion.div
                  variants={itemVariants}
                  className="flex flex-col sm:flex-row gap-6 justify-center items-center pt-12"
                >
                  <AnimatePresence>
                    {session || isAuthenticated ? (
                      <>
                        <motion.div
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.8 }}
                          whileHover={{ scale: 1.05, y: -2 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <Button
                            onClick={handleTryFree}
                            size="xl"
                            variant="primary"
                            className="relative overflow-hidden w-full sm:w-auto bg-brand-gradient hover:shadow-2xl transition-all duration-300 text-lg px-12 py-5 group"
                          >
                            <span className="relative z-10">🚀 Continue to Chat</span>
                            {/* Shimmer effect */}
                            <div className="absolute inset-0 -top-1 -bottom-1 bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                          </Button>
                        </motion.div>
                        
                        <motion.div
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.8 }}
                          whileHover={{ scale: 1.05, y: -2 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <Button
                            onClick={() => router.push('/settings')}
                            size="xl"
                            variant="outline"
                            className="w-full sm:w-auto hover:bg-white/90 transition-all duration-300 text-lg px-12 py-5 bg-white/80 backdrop-blur-2xl border-white/30 shadow-xl"
                          >
                            ⚙️ Manage Account
                          </Button>
                        </motion.div>
                      </>
                    ) : (
                      <>
                        <motion.div
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.8 }}
                          whileHover={{ scale: 1.05, y: -2 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <Button
                            onClick={handleGetStarted}
                            disabled={isLoading}
                            loading={isLoading}
                            size="xl"
                            variant="primary"
                            className="relative overflow-hidden w-full sm:w-auto bg-brand-gradient hover:shadow-2xl transition-all duration-300 text-lg px-12 py-5 group"
                          >
                            <span className="relative z-10">{!isLoading && '✨ Start Free Trial'}</span>
                            {/* Shimmer effect */}
                            <div className="absolute inset-0 -top-1 -bottom-1 bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                          </Button>
                        </motion.div>
                        
                        <motion.div
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.8 }}
                          whileHover={{ scale: 1.05, y: -2 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <Button
                            onClick={handleTryFree}
                            size="xl"
                            variant="outline"
                            className="w-full sm:w-auto hover:bg-white/90 transition-all duration-300 text-lg px-12 py-5 bg-white/80 backdrop-blur-2xl border-white/30 shadow-xl"
                          >
                            🎯 Try Demo
                          </Button>
                        </motion.div>
                      </>
                    )}
                  </AnimatePresence>
                </motion.div>

                {/* Enhanced Trust Indicators with Animation */}
                <motion.div
                  variants={itemVariants}
                  className="pt-16"
                >
                  <AnimatePresence>
                    {session || isAuthenticated ? (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="bg-gradient-to-r from-white/80 to-white/60 backdrop-blur-2xl rounded-3xl px-8 py-6 border border-white/30 shadow-2xl inline-block"
                      >
                        <div className="flex items-center space-x-3">
                          <motion.div
                            animate={{ rotate: [0, 10, -10, 0] }}
                            transition={{ duration: 2, repeat: Infinity }}
                            className="text-2xl"
                          >
                            🙏
                          </motion.div>
                          <p className="text-primary font-semibold text-lg">
                            Welcome back! Continue your dharmic journey with wisdom.
                          </p>
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="flex flex-wrap justify-center items-center gap-8 text-secondary"
                      >
                        {[
                          { icon: "✓", text: "7-day free trial" },
                          { icon: "✓", text: "No credit card required" },
                          { icon: "✓", text: "Enterprise ready" }
                        ].map((item, index) => (
                          <motion.div
                            key={index}
                            className="flex items-center space-x-3 bg-white/70 backdrop-blur-xl rounded-full px-6 py-3 border border-white/20 shadow-lg"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 2 + index * 0.2 }}
                            whileHover={{ scale: 1.05, y: -2 }}
                          >
                            <span className="text-brand-accent font-bold text-lg">{item.icon}</span>
                            <span className="font-medium">{item.text}</span>
                          </motion.div>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              </motion.div>
            </motion.div>

            {/* Enhanced Scroll Indicator */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 2.5 }}
              className="absolute bottom-12 left-1/2 transform -translate-x-1/2"
            >
              <motion.div 
                className="relative cursor-pointer group"
                whileHover={{ scale: 1.1 }}
                onClick={() => window.scrollTo({ top: window.innerHeight, behavior: 'smooth' })}
              >
                {/* Outer ring */}
                <motion.div 
                  className="w-12 h-20 border-2 border-brand-primary/50 rounded-full flex justify-center backdrop-blur-sm bg-white/10"
                  animate={{ 
                    borderColor: ["rgba(var(--brand-primary), 0.3)", "rgba(var(--brand-primary), 0.8)", "rgba(var(--brand-primary), 0.3)"]
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  {/* Inner dot */}
                  <motion.div 
                    className="w-2 h-4 bg-brand-gradient rounded-full mt-3"
                    animate={{ 
                      y: [0, 8, 0],
                      opacity: [1, 0.3, 1]
                    }}
                    transition={{ 
                      duration: 1.5, 
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  />
                </motion.div>
                
                {/* Scroll text */}
                <motion.p 
                  className="text-xs text-brand-primary/70 mt-2 font-medium tracking-wide"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 3 }}
                >
                  SCROLL
                </motion.p>
                
                {/* Ripple effect on hover */}
                <motion.div 
                  className="absolute inset-0 border-2 border-brand-primary/30 rounded-full"
                  initial={{ scale: 1, opacity: 0 }}
                  whileHover={{ 
                    scale: 1.5, 
                    opacity: [0, 0.5, 0],
                    transition: { duration: 0.6 }
                  }}
                />
              </motion.div>
            </motion.div>
          </section>

          {/* Modern About Section */}
          <section id="about" className="py-24 bg-gradient-to-br from-white via-section-light to-brand-secondary/5 relative overflow-hidden">
            {/* Background Decorations */}
            <div className="absolute inset-0 opacity-20">
              <div className="absolute top-10 right-10 w-64 h-64 bg-brand-accent/10 rounded-full blur-3xl"></div>
              <div className="absolute bottom-10 left-10 w-48 h-48 bg-brand-primary/10 rounded-full blur-2xl"></div>
            </div>

            <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="text-center mb-20"
              >
                <div className="inline-flex items-center space-x-3 bg-white/80 backdrop-blur-xl rounded-full px-6 py-3 border border-border-light/50 shadow-lg mb-6">
                  <span className="text-2xl">🌐</span>
                  <span className="text-lg font-semibold text-primary">About DharmaMind</span>
                </div>
                <h2 className="text-5xl md:text-6xl font-bold text-primary mb-6">
                  AI With Soul
                </h2>
                <p className="text-2xl text-primary font-light max-w-3xl mx-auto">
                  Intelligence With <span className="font-semibold text-brand-accent">Intention</span>
                </p>
              </motion.div>

              <div className="grid lg:grid-cols-2 gap-20 items-center">
                <motion.div
                  initial={{ opacity: 0, x: -50 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8, type: "spring", stiffness: 100 }}
                  viewport={{ once: true }}
                  className="space-y-8"
                >
                  <motion.div
                    className="bg-white/90 backdrop-blur-2xl rounded-3xl p-10 border border-white/30 shadow-2xl relative overflow-hidden group"
                    whileHover={{ y: -5, scale: 1.02 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    {/* Hover effect background */}
                    <motion.div 
                      className="absolute inset-0 bg-gradient-to-br from-brand-primary/5 to-brand-accent/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                      initial={false}
                    />
                    
                    <div className="relative z-10">
                      <div className="flex items-center space-x-4 mb-8">
                        <motion.div
                          className="w-16 h-16 bg-brand-gradient rounded-3xl flex items-center justify-center shadow-lg"
                          whileHover={{ rotate: 360, scale: 1.1 }}
                          transition={{ duration: 0.6 }}
                        >
                          <span className="text-white text-2xl">✨</span>
                        </motion.div>
                        <motion.h3 
                          className="text-3xl font-bold text-primary"
                          initial={{ opacity: 0, x: -20 }}
                          whileInView={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.2 }}
                        >
                          Who We Are
                        </motion.h3>
                      </div>
                      
                      <motion.div 
                        className="space-y-6 text-lg text-secondary leading-relaxed"
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3, staggerChildren: 0.1 }}
                      >
                        <motion.p
                          initial={{ opacity: 0, y: 10 }}
                          whileInView={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.4 }}
                        >
                          At DharmaMind, we are pioneering a new era of human-centered AI — one guided not just 
                          by data and algorithms, but by consciousness, compassion, and timeless dharmic wisdom.
                        </motion.p>
                        <motion.p
                          initial={{ opacity: 0, y: 10 }}
                          whileInView={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.6 }}
                        >
                          In today's world of complexity, distraction, and rapid automation, people are increasingly 
                          disconnected from purpose, clarity, and inner alignment. DharmaMind was born from a simple 
                          yet powerful realization: technology, when rooted in eternal values, can help restore what 
                          we've lost — our truth, our ethics, and our soul.
                        </motion.p>
                      </motion.div>
                    </div>
                    
                    {/* Decorative elements */}
                    <motion.div 
                      className="absolute top-4 right-4 w-20 h-20 bg-brand-accent/10 rounded-full blur-xl"
                      animate={{ 
                        scale: [1, 1.2, 1],
                        opacity: [0.3, 0.6, 0.3] 
                      }}
                      transition={{ duration: 4, repeat: Infinity }}
                    />
                  </motion.div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: 50 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8, type: "spring", stiffness: 100 }}
                  viewport={{ once: true }}
                  className="bg-white/90 backdrop-blur-2xl rounded-3xl p-10 border border-white/30 shadow-2xl relative overflow-hidden group"
                  whileHover={{ y: -5, scale: 1.02 }}
                >
                  {/* Hover effect background */}
                  <motion.div 
                    className="absolute inset-0 bg-gradient-to-br from-brand-secondary/5 to-brand-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                    initial={false}
                  />
                  
                  <div className="relative z-10">
                    <div className="flex items-center space-x-4 mb-8">
                      <motion.div
                        className="w-16 h-16 bg-brand-gradient rounded-3xl flex items-center justify-center shadow-lg"
                        whileHover={{ rotate: 360, scale: 1.1 }}
                        transition={{ duration: 0.6 }}
                      >
                        <span className="text-white text-2xl">🌿</span>
                      </motion.div>
                      <motion.h3 
                        className="text-3xl font-bold text-primary"
                        initial={{ opacity: 0, x: -20 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                      >
                        Our Mission
                      </motion.h3>
                    </div>
                    
                    <motion.p 
                      className="text-lg text-secondary leading-relaxed"
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                    >
                      To empower individuals worldwide with an ethical, purpose-driven AI that fosters clarity, 
                      conscious growth, and decision-making rooted in ancient principles and modern intelligence.
                    </motion.p>
                  </div>
                  
                  {/* Decorative elements */}
                  <motion.div 
                    className="absolute bottom-4 left-4 w-16 h-16 bg-brand-primary/10 rounded-full blur-lg"
                    animate={{ 
                      scale: [1, 1.3, 1],
                      opacity: [0.2, 0.5, 0.2] 
                    }}
                    transition={{ duration: 5, repeat: Infinity }}
                  />
                </motion.div>
              </div>

              <motion.div
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2, type: "spring", stiffness: 100 }}
                viewport={{ once: true }}
                className="mt-20 relative"
              >
                <motion.div 
                  className="bg-gradient-to-r from-white/95 to-white/85 backdrop-blur-2xl rounded-3xl p-12 border border-white/40 shadow-2xl text-center relative overflow-hidden group"
                  whileHover={{ scale: 1.02, y: -5 }}
                  transition={{ type: "spring", stiffness: 300 }}
                >
                  {/* Background pattern */}
                  <div className="absolute inset-0 opacity-5">
                    <svg className="w-full h-full" viewBox="0 0 100 100">
                      <defs>
                        <pattern id="quote-pattern" width="20" height="20" patternUnits="userSpaceOnUse">
                          <circle cx="10" cy="10" r="1" fill="currentColor" className="text-brand-primary"/>
                        </pattern>
                      </defs>
                      <rect width="100" height="100" fill="url(#quote-pattern)" />
                    </svg>
                  </div>
                  
                  <div className="relative z-10 max-w-5xl mx-auto">
                    <motion.div 
                      className="text-6xl mb-6"
                      animate={{ 
                        rotate: [0, 5, -5, 0],
                        scale: [1, 1.1, 1] 
                      }}
                      transition={{ duration: 4, repeat: Infinity }}
                    >
                      🧘‍♂️
                    </motion.div>
                    
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }}
                      whileInView={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.3, duration: 0.6 }}
                      className="relative"
                    >
                      {/* Quote marks */}
                      <span className="absolute -top-4 -left-4 text-6xl text-brand-primary/30 font-serif">"</span>
                      <span className="absolute -bottom-8 -right-4 text-6xl text-brand-primary/30 font-serif">"</span>
                      
                      <motion.p 
                        className="text-2xl md:text-3xl text-primary leading-relaxed font-medium italic"
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5, duration: 0.8 }}
                      >
                        We are not just creating another chatbot. We're building a soulful AI companion — a guide, 
                        a mirror, and a moral compass — for those seeking to live with clarity, integrity, and wisdom.
                      </motion.p>
                    </motion.div>
                    
                    {/* Decorative line */}
                    <motion.div 
                      className="w-24 h-1 bg-brand-gradient rounded-full mx-auto mt-8"
                      initial={{ width: 0 }}
                      whileInView={{ width: 96 }}
                      transition={{ delay: 0.8, duration: 0.8 }}
                    />
                  </div>
                  
                  {/* Floating elements */}
                  <motion.div 
                    className="absolute top-8 right-8 w-3 h-3 bg-brand-accent/40 rounded-full"
                    animate={{ 
                      y: [-5, 5, -5],
                      opacity: [0.4, 0.8, 0.4] 
                    }}
                    transition={{ duration: 3, repeat: Infinity }}
                  />
                  <motion.div 
                    className="absolute bottom-8 left-8 w-2 h-2 bg-brand-primary/40 rounded-full"
                    animate={{ 
                      y: [5, -5, 5],
                      opacity: [0.3, 0.7, 0.3] 
                    }}
                    transition={{ duration: 4, repeat: Infinity, delay: 1 }}
                  />
                </motion.div>
              </motion.div>
            </div>
          </section>

          {/* Modern Features Section */}
          <section id="features" className="py-24 bg-gradient-to-br from-brand-primary/5 via-white to-section-light relative overflow-hidden">
            {/* Background Decorations */}
            <div className="absolute inset-0 opacity-20">
              <div className="absolute top-20 left-20 w-40 h-40 bg-brand-secondary/10 rounded-full blur-2xl"></div>
              <div className="absolute bottom-20 right-20 w-56 h-56 bg-brand-accent/10 rounded-full blur-3xl"></div>
            </div>

            <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="text-center mb-20"
              >
                <div className="inline-flex items-center space-x-3 bg-white/80 backdrop-blur-xl rounded-full px-6 py-3 border border-border-light/50 shadow-lg mb-6">
                  <span className="text-2xl">🧠</span>
                  <span className="text-lg font-semibold text-primary">What We Offer</span>
                </div>
                <h2 className="text-5xl md:text-6xl font-bold text-brand-primary mb-6">
                  Dharmic Intelligence
                </h2>
                <p className="text-xl text-secondary max-w-4xl mx-auto leading-relaxed">
                  Enterprise-grade AI wisdom platform combining ancient dharmic principles with 
                  cutting-edge technology to deliver ethical, purpose-driven intelligence.
                </p>
              </motion.div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
                
                {/* Dharma-Powered Chat Guidance */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8 }}
                  viewport={{ once: true }}
                  className="bg-gradient-to-br from-brand-primary/5 to-brand-accent/5 border border-brand-primary/20 rounded-3xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105"
                >
                  <div className="flex items-center mb-6">
                    <div className="w-14 h-14 bg-brand-primary/10 backdrop-blur-xl rounded-2xl flex items-center justify-center mr-4">
                      <span className="text-brand-primary text-2xl">🧘</span>
                    </div>
                    <h4 className="text-2xl font-bold text-primary">Dharma-Powered Conversational AI</h4>
                  </div>
                  <p className="text-secondary leading-relaxed mb-6 text-lg">
                    Advanced conversational intelligence for complex human challenges — career transitions, 
                    leadership dilemmas, relationship guidance, emotional growth, and spiritual development. 
                    Every response is grounded in universal dharmic wisdom, compassion, and ethical clarity.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    <span className="px-4 py-2 bg-brand-primary/10 text-brand-primary rounded-full text-sm font-medium">Multi-domain Guidance</span>
                    <span className="px-4 py-2 bg-brand-primary/10 text-brand-primary rounded-full text-sm font-medium">Ethical Framework</span>
                    <span className="px-4 py-2 bg-brand-primary/10 text-brand-primary rounded-full text-sm font-medium">Contextual Wisdom</span>
                  </div>
                </motion.div>

                {/* 32-Dimensional Dharma Modules */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white/80 backdrop-blur-xl rounded-3xl p-8 border border-border-light/50 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105"
                >
                  <div className="flex items-center mb-6">
                    <div className="w-14 h-14 bg-brand-gradient rounded-2xl flex items-center justify-center mr-4">
                      <span className="text-white text-2xl">🧭</span>
                    </div>
                    <h4 className="text-2xl font-bold text-primary">32-Dimensional Wisdom Architecture</h4>
                  </div>
                  <p className="text-secondary leading-relaxed mb-6 text-lg">
                    Proprietary modular intelligence system based on timeless dharmic concepts including 
                    Karma, Moksha, Viveka, Shakti, Ahimsa, and 27 additional wisdom modules. Each query 
                    is processed through relevant philosophical frameworks to ensure deep, personalized, 
                    ethically-aligned responses.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    <span className="px-4 py-2 bg-brand-primary/10 text-brand-primary rounded-full text-sm font-medium">Modular Architecture</span>
                    <span className="px-4 py-2 bg-brand-primary/10 text-brand-primary rounded-full text-sm font-medium">Philosophical AI</span>
                    <span className="px-4 py-2 bg-brand-primary/10 text-brand-primary rounded-full text-sm font-medium">Wisdom Integration</span>
                  </div>
                </motion.div>

                {/* Ethical Intelligence Layer */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  viewport={{ once: true }}
                  className="bg-white/80 backdrop-blur-xl rounded-3xl p-8 border border-border-light/50 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105"
                >
                  <div className="flex items-center mb-6">
                    <div className="w-14 h-14 bg-brand-gradient rounded-2xl flex items-center justify-center mr-4">
                      <span className="text-white text-2xl">🧬</span>
                    </div>
                    <h4 className="text-2xl font-bold text-primary">Ethical Intelligence Framework</h4>
                  </div>
                  <p className="text-secondary leading-relaxed mb-6 text-lg">
                    Multi-layered ethical evaluation system ensuring every response undergoes rigorous 
                    moral assessment and self-inquiry protocols. Our AI actively avoids harm, embraces 
                    nuance, and prioritizes truth and wisdom over algorithmic correctness.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    <span className="px-4 py-2 bg-brand-secondary/10 text-brand-secondary rounded-full text-sm font-medium">Harm Prevention</span>
                    <span className="px-4 py-2 bg-brand-secondary/10 text-brand-secondary rounded-full text-sm font-medium">Truth-Seeking</span>
                    <span className="px-4 py-2 bg-brand-secondary/10 text-brand-secondary rounded-full text-sm font-medium">Moral Reasoning</span>
                  </div>
                </motion.div>

                {/* Integrated with Advanced LLMs */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.3 }}
                  viewport={{ once: true }}
                  className="bg-gradient-to-br from-brand-secondary/5 to-brand-accent/5 border border-brand-secondary/20 rounded-3xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105"
                >
                  <div className="flex items-center mb-6">
                    <div className="w-14 h-14 bg-brand-secondary/10 backdrop-blur-xl rounded-2xl flex items-center justify-center mr-4">
                      <span className="text-brand-secondary text-2xl">🤝</span>
                    </div>
                    <h4 className="text-2xl font-bold text-primary">Multi-LLM Integration Platform</h4>
                  </div>
                  <p className="text-secondary leading-relaxed mb-6 text-lg">
                    Seamlessly integrates with industry-leading language models including GPT, Claude, 
                    and Gemini while maintaining dharmic integrity. Unlike generic implementations, 
                    our platform uses these models for ethical learning, wise evolution, and 
                    values-aligned reasoning.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    <span className="px-4 py-2 bg-brand-secondary/10 text-brand-secondary rounded-full text-sm font-medium">Multi-Model Architecture</span>
                    <span className="px-4 py-2 bg-brand-secondary/10 text-brand-secondary rounded-full text-sm font-medium">Ethical Training</span>
                    <span className="px-4 py-2 bg-brand-secondary/10 text-brand-secondary rounded-full text-sm font-medium">Continuous Learning</span>
                  </div>
                </motion.div>
              </div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
                viewport={{ once: true }}
                className="text-center mt-16"
              >
                <div className="bg-gradient-to-r from-brand-primary/10 via-brand-accent/10 to-brand-secondary/10 border border-brand-primary/20 rounded-3xl p-8 shadow-2xl">
                  <div className="text-4xl mb-4">🕉️</div>
                  <p className="text-primary font-medium text-xl max-w-3xl mx-auto">
                    <strong>Supreme Intelligence:</strong> All consciousness layers working together for enterprise wisdom
                  </p>
                </div>
              </motion.div>
            </div>
          </section>

          {/* Spiritual Quotes Section */}
          <section className="py-20">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="text-center mb-16"
              >
                <h2 className="text-3xl font-bold text-primary mb-4">
                  Ancient Wisdom for Modern Times
                </h2>
                <p className="text-secondary max-w-2xl mx-auto">
                  Find inspiration and guidance through timeless teachings
                </p>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                viewport={{ once: true }}
              >
                <SpiritualQuotes />
              </motion.div>
            </div>
          </section>

          {/* Enterprise Roadmap Section */}
          <section className="py-20 bg-section-light">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="text-center mb-16"
              >
                <h2 className="text-4xl font-bold text-primary mb-6">
                  🚀 Enterprise Roadmap
                </h2>
                <p className="text-xl text-secondary">
                  Building the future of dharmic AI for organizations worldwide
                </p>
              </motion.div>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8 }}
                  viewport={{ once: true }}
                  className="bg-white rounded-xl p-6 border border-light"
                >
                  <div className="text-2xl mb-4">🏢</div>
                  <h3 className="text-lg font-semibold text-primary mb-3">Enterprise Integration</h3>
                  <p className="text-secondary text-sm">
                    Seamless integration with existing enterprise systems, SSO, and compliance frameworks for large organizations.
                  </p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white rounded-xl p-6 border border-light"
                >
                  <div className="text-2xl mb-4">📊</div>
                  <h3 className="text-lg font-semibold text-primary mb-3">Analytics Dashboard</h3>
                  <p className="text-secondary text-sm">
                    Comprehensive analytics and insights for team wellness, productivity, and dharmic alignment metrics.
                  </p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  viewport={{ once: true }}
                  className="bg-white rounded-xl p-6 border border-light"
                >
                  <div className="text-2xl mb-4">🎓</div>
                  <h3 className="text-lg font-semibold text-primary mb-3">Training Programs</h3>
                  <p className="text-secondary text-sm">
                    Custom dharmic leadership and mindfulness training programs for executive teams and organizations.
                  </p>
                </motion.div>
              </div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3 }}
                viewport={{ once: true }}
                className="text-center mt-12"
              >
                <Button
                  onClick={() => router.push('/enterprise')}
                  variant="primary"
                  size="lg"
                >
                  Learn More About Enterprise
                </Button>
              </motion.div>
            </div>
          </section>

          {/* Centralized Footer */}
          <Footer variant="professional" />
        </main>
      </div>
    </>
  );
};

export default WelcomePage;
