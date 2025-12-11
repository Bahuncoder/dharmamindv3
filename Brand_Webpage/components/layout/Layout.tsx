import Head from 'next/head';
import Header from './Header';
import Footer from './Footer';
import CookieConsent from '../ui/CookieConsent';
import { siteConfig } from '../../config/site.config';

interface LayoutProps {
    children: React.ReactNode;
    title?: string;
    description?: string;
}

export default function Layout({ children, title, description }: LayoutProps) {
    const pageTitle = title
        ? `${title} | ${siteConfig.company.name}`
        : `${siteConfig.company.name} - ${siteConfig.company.tagline}`;

    const pageDescription = description || siteConfig.company.description;

    return (
        <>
            <Head>
                <title>{pageTitle}</title>
                <meta name="description" content={pageDescription} />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <link rel="icon" href="/favicon.ico" />

                {/* Open Graph */}
                <meta property="og:title" content={pageTitle} />
                <meta property="og:description" content={pageDescription} />
                <meta property="og:type" content="website" />
                <meta property="og:image" content="/og-image.png" />

                {/* Twitter Card */}
                <meta name="twitter:card" content="summary_large_image" />
                <meta name="twitter:title" content={pageTitle} />
                <meta name="twitter:description" content={pageDescription} />
            </Head>

            <div className="min-h-screen flex flex-col">
                <Header />
                <main className="flex-grow pt-16">
                    {children}
                </main>
                <Footer />
                <CookieConsent />
            </div>
        </>
    );
}
