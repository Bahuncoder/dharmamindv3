import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import Logo from './Logo';
import Button from './Button';
import ThemeToggle from './ThemeToggle';

interface NavigationHeaderProps {
    title?: string;
    className?: string;
    variant?: 'default' | 'transparent';
    showBackButton?: boolean;
    showHomeButton?: boolean;
    customBackAction?: () => void;
}

const NavigationHeader: React.FC<NavigationHeaderProps> = ({
    title,
    showBackButton = true,
    showHomeButton = false,
    customBackAction,
    className = ""
}) => {
    const router = useRouter();
    const [canGoBack, setCanGoBack] = useState(false);

    useEffect(() => {
        // Check if we have history to go back to
        // We'll use a more intelligent approach by checking sessionStorage
        const hasHistory = typeof window !== 'undefined' &&
            (window.history.length > 1 || sessionStorage.getItem('dharma_nav_history') !== null);
        setCanGoBack(hasHistory);

        // Track navigation history
        if (typeof window !== 'undefined') {
            const currentPath = router.asPath;
            const history = JSON.parse(sessionStorage.getItem('dharma_nav_history') || '[]');

            // Only add to history if it's different from the last entry
            if (history.length === 0 || history[history.length - 1] !== currentPath) {
                history.push(currentPath);
                // Keep only last 10 entries
                if (history.length > 10) {
                    history.shift();
                }
                sessionStorage.setItem('dharma_nav_history', JSON.stringify(history));
            }
        }
    }, [router.asPath]);

    const handleBack = () => {
        if (customBackAction) {
            customBackAction();
            return;
        }

        // Smart back navigation
        if (typeof window !== 'undefined') {
            const history = JSON.parse(sessionStorage.getItem('dharma_nav_history') || '[]');

            if (history.length > 1) {
                // Remove current page and go to previous
                history.pop();
                const previousPage = history[history.length - 1];

                // Update history
                sessionStorage.setItem('dharma_nav_history', JSON.stringify(history));

                // Navigate to previous page
                if (previousPage && previousPage !== router.asPath) {
                    router.push(previousPage);
                } else {
                    // Fallback to browser back
                    router.back();
                }
            } else if (canGoBack) {
                // Use browser back as fallback
                router.back();
            } else {
                // Ultimate fallback - go home
                router.push('/');
            }
        } else {
            // Server-side fallback
            router.push('/');
        }
    };

    const handleHome = () => {
        // Clear history when going home
        if (typeof window !== 'undefined') {
            sessionStorage.setItem('dharma_nav_history', JSON.stringify(['/']));
        }
        router.push('/');
    };

    return (
        <motion.header
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className={`bg-neutral-100/90 dark:bg-neutral-900/90 backdrop-blur-sm border-b border-neutral-300/50 dark:border-neutral-700/50 sticky top-0 z-50 ${className}`}
        >
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <div className="flex items-center space-x-6">
                        {/* Smart Back Button - Clean arrow only */}
                        {showBackButton && (
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={handleBack}
                                className="flex items-center justify-center w-10 h-10 rounded-lg text-neutral-600 dark:text-neutral-400 hover:text-gold-600 dark:hover:text-gold-400 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-all duration-200 group"
                                title={canGoBack ? "Go back to previous page" : "Go to homepage"}
                            >
                                <svg
                                    className="w-5 h-5 transition-transform group-hover:-translate-x-0.5"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                                </svg>
                            </motion.button>
                        )}

                        {/* Logo - Always clickable to home */}
                        <Logo
                            size="sm"
                            onClick={handleHome}
                            className="cursor-pointer hover:opacity-80 transition-opacity"
                        />

                        {/* Page Title */}
                        {title && (
                            <>
                                <div className="h-6 w-px bg-primary-background dark:bg-neutral-700 ml-2"></div>
                                <h1 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 hidden md:block ml-2">
                                    {title}
                                </h1>
                            </>
                        )}
                    </div>

                    {/* Right Side Navigation */}
                    <nav className="hidden md:flex items-center space-x-6">
                        <button
                            onClick={() => router.push('/features')}
                            className="text-neutral-600 dark:text-neutral-400 hover:text-gold-600 dark:hover:text-gold-400 text-sm font-medium transition-colors"
                        >
                            Features
                        </button>
                        <button
                            onClick={() => router.push('/pricing')}
                            className="text-neutral-600 dark:text-neutral-400 hover:text-gold-600 dark:hover:text-gold-400 text-sm font-medium transition-colors"
                        >
                            Pricing
                        </button>
                        <button
                            onClick={() => router.push('/about')}
                            className="text-neutral-600 dark:text-neutral-400 hover:text-gold-600 dark:hover:text-gold-400 text-sm font-medium transition-colors"
                        >
                            About
                        </button>

                        {/* Theme Toggle */}
                        <ThemeToggle />

                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => router.push('/contact')}
                        >
                            Contact
                        </Button>
                    </nav>

                    {/* Mobile Menu Button */}
                    <div className="md:hidden flex items-center space-x-2">
                        {/* Theme Toggle for Mobile */}
                        <ThemeToggle />

                        <button
                            onClick={() => {
                                // You can implement mobile menu here
                                console.log('Mobile menu clicked');
                            }}
                            className="p-2 rounded-lg text-neutral-600 dark:text-neutral-400 hover:text-gold-600 dark:hover:text-gold-400 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                        >
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </motion.header>
    );
};

export default NavigationHeader;
