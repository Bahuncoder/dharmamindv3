import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { motion, AnimatePresence } from 'framer-motion';
import Logo from './Logo';
import Button from './Button';

interface BreadcrumbItem {
    label: string;
    href: string;
}

interface BrandHeaderProps {
    breadcrumbs?: BreadcrumbItem[];
    isEnterprise?: boolean;
    className?: string;
}

const BrandHeader: React.FC<BrandHeaderProps> = ({
    breadcrumbs = [],
    isEnterprise = false,
    className = ""
}) => {
    const router = useRouter();
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    const handleNavigation = (path: string) => {
        router.push(path);
        setIsMobileMenuOpen(false);
    };

    const enterpriseNavItems = [
        { label: 'Overview', href: '/enterprise' },
        { label: 'Solutions', href: '/enterprise/solutions' },
        { label: 'Security', href: '/enterprise/security' },
        { label: 'Support', href: '/enterprise/support' },
        { label: 'Pricing', href: '/enterprise/pricing' }
    ];

    const regularNavItems = [
        { label: 'Features', href: '/features' },
        { label: 'Pricing', href: '/pricing' },
        { label: 'About', href: '/about' },
        { label: 'Help', href: '/help' }
    ];

    const navItems = isEnterprise ? enterpriseNavItems : regularNavItems;

    return (
        <>
            {/* Main Header */}
            <motion.header
                initial={{ y: -20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className={`bg-white/95 backdrop-blur-sm border-b border-brand-accent/50 sticky top-0 z-50 ${className}`}
            >
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16">
                        {/* Left: Logo */}
                        <div className="flex items-center">
                            <Logo
                                size="sm"
                                onClick={() => handleNavigation('/')}
                                className="cursor-pointer hover:opacity-80 transition-opacity"
                            />
                            {isEnterprise && (
                                <div className="ml-4 px-3 py-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-xs font-semibold rounded-full">
                                    ENTERPRISE
                                </div>
                            )}
                        </div>

                        {/* Center: Navigation Menu */}
                        <nav className="hidden lg:flex items-center space-x-8">
                            {navItems.map((item) => (
                                <motion.button
                                    key={item.href}
                                    whileHover={{ y: -1 }}
                                    onClick={() => handleNavigation(item.href)}
                                    className={`text-sm font-medium transition-colors px-3 py-2 rounded-lg ${router.pathname === item.href
                                            ? 'text-primary bg-primary/10'
                                            : 'text-primary hover:text-primary hover:bg-section-light'
                                        }`}
                                >
                                    {item.label}
                                </motion.button>
                            ))}
                        </nav>

                        {/* Right: CTA Buttons */}
                        <div className="flex items-center space-x-4">
                            {!isEnterprise && (
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleNavigation('/enterprise')}
                                    className="hidden md:flex"
                                >
                                    Enterprise
                                </Button>
                            )}
                            <Button
                                variant="primary"
                                size="sm"
                                onClick={() => handleNavigation('/contact')}
                            >
                                {isEnterprise ? 'Contact Sales' : 'Get Started'}
                            </Button>

                            {/* Mobile Menu Button */}
                            <button
                                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                                className="lg:hidden p-2 rounded-lg text-secondary hover:bg-section-light"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    {isMobileMenuOpen ? (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    ) : (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                                    )}
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Mobile Menu */}
                <AnimatePresence>
                    {isMobileMenuOpen && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="lg:hidden border-t border-brand-accent/50 bg-white"
                        >
                            <div className="max-w-7xl mx-auto px-4 py-4 space-y-2">
                                {navItems.map((item) => (
                                    <button
                                        key={item.href}
                                        onClick={() => handleNavigation(item.href)}
                                        className={`block w-full text-left px-3 py-2 rounded-lg text-sm font-medium ${router.pathname === item.href
                                                ? 'text-primary bg-primary/10'
                                                : 'text-primary hover:text-primary hover:bg-section-light'
                                            }`}
                                    >
                                        {item.label}
                                    </button>
                                ))}
                                {!isEnterprise && (
                                    <button
                                        onClick={() => handleNavigation('/enterprise')}
                                        className="block w-full text-left px-3 py-2 rounded-lg text-sm font-medium text-primary hover:text-primary hover:bg-section-light"
                                    >
                                        Enterprise
                                    </button>
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.header>

            {/* Breadcrumb Navigation */}
            {breadcrumbs.length > 0 && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="bg-section-light/80 border-b border-brand-accent/50"
                >
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
                        <nav className="flex items-center space-x-2 text-sm">
                            {breadcrumbs.map((crumb, index) => (
                                <React.Fragment key={crumb.href}>
                                    <motion.button
                                        whileHover={{ scale: 1.02 }}
                                        onClick={() => handleNavigation(crumb.href)}
                                        className={`font-medium transition-colors ${index === breadcrumbs.length - 1
                                                ? 'text-primary cursor-default'
                                                : 'text-primary hover:text-primary-dark cursor-pointer'
                                            }`}
                                    >
                                        {crumb.label}
                                    </motion.button>
                                    {index < breadcrumbs.length - 1 && (
                                        <svg
                                            className="w-4 h-4 text-secondary"
                                            fill="none"
                                            stroke="currentColor"
                                            viewBox="0 0 24 24"
                                        >
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                        </svg>
                                    )}
                                </React.Fragment>
                            ))}
                        </nav>
                    </div>
                </motion.div>
            )}
        </>
    );
};

export default BrandHeader;
