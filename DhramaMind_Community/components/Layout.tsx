/**
 * DharmaMind Community - Layout Components
 * Shared header, footer, and layout wrapper
 */

import React from 'react';
import Link from 'next/link';
import Head from 'next/head';
import { useRouter } from 'next/router';
import ThemeToggle from './ThemeToggle';

// ===========================================
// HEADER COMPONENT
// ===========================================
interface HeaderProps {
    transparent?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ transparent = false }) => {
    const router = useRouter();
    const currentPath = router.pathname;

    const navItems = [
        { name: 'Discussions', href: '/discussions' },
        { name: 'Members', href: '/members' },
        { name: 'Events', href: '/events' },
    ];

    const isActive = (href: string) => currentPath === href || currentPath.startsWith(href + '/');

    return (
        <header className={`${transparent ? 'bg-transparent' : 'bg-white dark:bg-neutral-900'} border-b border-neutral-200 dark:border-neutral-700`}>
            <div className="max-w-6xl mx-auto px-6">
                <div className="flex items-center justify-between h-16">
                    {/* Logo */}
                    <Link href="/" className="flex items-center gap-3">
                        <img src="/logo.jpeg" alt="DharmaMind" className="w-8 h-8 rounded-lg object-cover" />
                        <span className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">Community</span>
                    </Link>

                    {/* Desktop Navigation */}
                    <nav className="hidden md:flex items-center gap-8">
                        {navItems.map((item) => (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`text-base transition-colors ${isActive(item.href)
                                    ? 'font-medium text-gold-600 dark:text-gold-400'
                                    : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
                                    }`}
                            >
                                {item.name}
                            </Link>
                        ))}
                    </nav>

                    {/* Right Side: Theme Toggle + Auth Buttons */}
                    <div className="flex items-center gap-4">
                        <ThemeToggle />
                        <Link
                            href="/login"
                            className="text-base text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
                        >
                            Sign in
                        </Link>
                        <Link
                            href="/signup"
                            className="px-5 py-2 bg-gold-600 text-white text-base font-medium rounded-lg hover:bg-gold-700 transition-colors"
                        >
                            Join
                        </Link>
                    </div>
                </div>
            </div>
        </header>
    );
};

// ===========================================
// FOOTER COMPONENT
// ===========================================
export const Footer: React.FC = () => {
    return (
        <footer className="bg-white dark:bg-neutral-900 border-t border-neutral-200 dark:border-neutral-700 py-10 px-6 mt-auto">
            <div className="max-w-6xl mx-auto">
                <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                    <div className="flex items-center gap-3">
                        <img src="/logo.jpeg" alt="DharmaMind" className="w-6 h-6 rounded object-cover" />
                        <span className="text-base text-neutral-500 dark:text-neutral-400">© 2024 DharmaMind Community</span>
                    </div>
                    <div className="flex items-center gap-8">
                        <Link href="/privacy" className="text-base text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors">
                            Privacy
                        </Link>
                        <Link href="/terms" className="text-base text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors">
                            Terms
                        </Link>
                        <a href="https://dharmamind.com" className="text-base text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors">
                            DharmaMind
                        </a>
                        <a href="https://dharmamind.ai" className="text-base text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors">
                            AI Chat
                        </a>
                    </div>
                </div>
            </div>
        </footer>
    );
};

// ===========================================
// LAYOUT COMPONENT
// ===========================================
interface LayoutProps {
    children: React.ReactNode;
    title?: string;
    description?: string;
    transparentHeader?: boolean;
}

export const Layout: React.FC<LayoutProps> = ({
    children,
    title,
    description,
    transparentHeader = false,
}) => {
    const pageTitle = title ? `${title} - DharmaMind Community` : 'DharmaMind Community';
    const pageDescription = description || 'Connect with thoughtful professionals. Share insights, learn, grow together.';

    return (
        <>
            <Head>
                <title>{pageTitle}</title>
                <meta name="description" content={pageDescription} />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900 flex flex-col">
                <Header transparent={transparentHeader} />
                <main className="flex-1">
                    {children}
                </main>
                <Footer />
            </div>
        </>
    );
};

export default Layout;
