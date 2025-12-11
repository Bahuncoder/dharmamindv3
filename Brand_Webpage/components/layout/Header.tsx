import Link from 'next/link';
import Image from 'next/image';
import { useRouter } from 'next/router';
import { siteConfig } from '../../config/site.config';
import { useState } from 'react';

export default function Header() {
    const router = useRouter();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    return (
        <header className="fixed top-0 left-0 right-0 bg-neutral-100/90 backdrop-blur-md z-50 border-b border-neutral-300">
            <nav className="max-w-7xl mx-auto px-6 py-4">
                <div className="flex items-center justify-between">
                    {/* Logo */}
                    <Link href="/" className="flex items-center space-x-3">
                        <Image
                            src="/logo.jpeg"
                            alt={`${siteConfig.company.name} Logo`}
                            width={40}
                            height={40}
                            className="rounded-lg"
                        />
                        <span className="text-xl font-semibold text-neutral-900">{siteConfig.company.name}</span>
                    </Link>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center space-x-8">
                        {siteConfig.navigation.main.map((item) => (
                            <Link
                                key={item.name}
                                href={item.href}
                                className={`text-sm transition-colors ${router.pathname === item.href
                                    ? 'text-neutral-900 font-medium'
                                    : 'text-neutral-600 hover:text-neutral-900'
                                    }`}
                            >
                                {item.name}
                            </Link>
                        ))}
                    </div>

                    {/* CTA Buttons */}
                    <div className="hidden md:flex items-center space-x-4">
                        <Link
                            href="/enterprise"
                            className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors"
                        >
                            Enterprise
                        </Link>
                        <Link
                            href="/team"
                            className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors"
                        >
                            Team
                        </Link>
                        <Link
                            href="/careers"
                            className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors"
                        >
                            Careers
                        </Link>
                        <Link
                            href="/blog"
                            className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors"
                        >
                            Blog
                        </Link>
                        <a
                            href="https://chat.dharmamind.ai"
                            className="bg-gold-600 text-white px-4 py-2 rounded-full text-sm font-medium hover:bg-gold-700 transition-colors"
                        >
                            Try DharmaMind
                        </a>
                    </div>

                    {/* Mobile Menu Button */}
                    <button
                        className="md:hidden p-2"
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                        aria-label="Toggle menu"
                    >
                        <svg
                            className="w-6 h-6 text-neutral-900"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            {mobileMenuOpen ? (
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            ) : (
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                            )}
                        </svg>
                    </button>
                </div>

                {/* Mobile Menu */}
                {mobileMenuOpen && (
                    <div className="md:hidden mt-4 pb-4 border-t border-neutral-200 pt-4">
                        <div className="flex flex-col space-y-4">
                            {siteConfig.navigation.main.map((item) => (
                                <Link
                                    key={item.name}
                                    href={item.href}
                                    className={`text-sm ${router.pathname === item.href
                                        ? 'text-neutral-900 font-medium'
                                        : 'text-neutral-600'
                                        }`}
                                    onClick={() => setMobileMenuOpen(false)}
                                >
                                    {item.name}
                                </Link>
                            ))}
                            <Link
                                href="/enterprise"
                                className="text-sm text-neutral-600"
                                onClick={() => setMobileMenuOpen(false)}
                            >
                                Enterprise
                            </Link>
                            <Link
                                href="/team"
                                className="text-sm text-neutral-600"
                                onClick={() => setMobileMenuOpen(false)}
                            >
                                Team
                            </Link>
                            <Link
                                href="/careers"
                                className="text-sm text-neutral-600"
                                onClick={() => setMobileMenuOpen(false)}
                            >
                                Careers
                            </Link>
                            <Link
                                href="/blog"
                                className="text-sm text-neutral-600"
                                onClick={() => setMobileMenuOpen(false)}
                            >
                                Blog
                            </Link>
                            <a
                                href="https://chat.dharmamind.ai"
                                className="bg-gold-600 text-white px-4 py-2 rounded-full text-sm font-medium text-center hover:bg-gold-700 transition-colors"
                            >
                                Try DharmaMind
                            </a>
                        </div>
                    </div>
                )}
            </nav>
        </header>
    );
}
