'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

export default function CookieConsent() {
    const [showBanner, setShowBanner] = useState(false);

    useEffect(() => {
        // Check if user has already consented
        const consent = localStorage.getItem('cookie-consent');
        if (!consent) {
            // Delay showing banner for better UX
            const timer = setTimeout(() => setShowBanner(true), 1500);
            return () => clearTimeout(timer);
        }
    }, []);

    const acceptAll = () => {
        localStorage.setItem('cookie-consent', JSON.stringify({
            necessary: true,
            analytics: true,
            marketing: true,
            timestamp: new Date().toISOString(),
        }));
        setShowBanner(false);
    };

    const acceptNecessary = () => {
        localStorage.setItem('cookie-consent', JSON.stringify({
            necessary: true,
            analytics: false,
            marketing: false,
            timestamp: new Date().toISOString(),
        }));
        setShowBanner(false);
    };

    if (!showBanner) return null;

    return (
        <div className="fixed bottom-0 left-0 right-0 z-50 p-4 md:p-6">
            <div className="max-w-4xl mx-auto bg-neutral-100 rounded-2xl shadow-2xl border border-neutral-200 p-6 md:p-8">
                <div className="flex flex-col md:flex-row md:items-start gap-6">
                    {/* Content */}
                    <div className="flex-grow">
                        <div className="flex items-center gap-2 mb-3">
                            <span className="text-2xl">🍪</span>
                            <h3 className="text-lg font-semibold text-neutral-900">We value your privacy</h3>
                        </div>
                        <p className="text-neutral-600 text-sm leading-relaxed mb-4">
                            We use cookies to enhance your browsing experience, analyze site traffic,
                            and personalize content. By clicking "Accept All", you consent to our use of cookies.
                            Read our{' '}
                            <Link href="/privacy" className="text-neutral-900 hover:underline">
                                Privacy Policy
                            </Link>{' '}
                            and{' '}
                            <Link href="/terms" className="text-neutral-900 hover:underline">
                                Cookie Policy
                            </Link>{' '}
                            for more information.
                        </p>
                    </div>

                    {/* Buttons */}
                    <div className="flex flex-col sm:flex-row gap-3 flex-shrink-0">
                        <button
                            onClick={acceptNecessary}
                            className="px-5 py-2.5 text-sm font-medium text-neutral-700 bg-neutral-100 hover:bg-neutral-200 rounded-full transition-colors"
                        >
                            Necessary Only
                        </button>
                        <button
                            onClick={acceptAll}
                            className="px-5 py-2.5 text-sm font-medium text-white bg-gold-600 hover:bg-gold-700 rounded-full transition-colors"
                        >
                            Accept All
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
