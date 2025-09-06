import Head from 'next/head';
import { useRouter } from 'next/router';

interface SEOProps {
  title?: string;
  description?: string;
  keywords?: string;
  image?: string;
  url?: string;
  type?: 'website' | 'article' | 'profile';
  author?: string;
  publishedTime?: string;
  modifiedTime?: string;
  noIndex?: boolean;
  canonical?: string;
}

const defaultSEO = {
  title: 'DharmaMind - AI Life Coach for Finding Purpose & Conscious Living',
  description: 'Discover your life purpose with DharmaMind AI - an ethical AI companion for personal growth, mindfulness, and conscious living. Get guidance for meaningful decisions and inner peace.',
  keywords: 'DharmaMind, finding life purpose, AI life coach, conscious living guide, personal growth AI, mindfulness and purpose, ethical AI companion, wisdom AI chatbot, inner peace guidance, stress management AI, AI for self-reflection, meaningful life AI',
  image: '/og-image.jpg',
  type: 'website' as const,
  author: 'DharmaMind Team'
};

export default function SEOHead({
  title,
  description = defaultSEO.description,
  keywords = defaultSEO.keywords,
  image = defaultSEO.image,
  url,
  type = defaultSEO.type,
  author = defaultSEO.author,
  publishedTime,
  modifiedTime,
  noIndex = false,
  canonical
}: SEOProps) {
  const router = useRouter();
  
  // Construct full URL
  const baseUrl = process.env.NEXT_PUBLIC_SITE_URL || 'https://dharmamind.vercel.app';
  const fullUrl = url || `${baseUrl}${router.asPath}`;
  const fullImageUrl = image.startsWith('http') ? image : `${baseUrl}${image}`;
  
  // Construct title
  const fullTitle = title 
    ? (title.includes('DharmaMind') ? title : `${title} | DharmaMind`)
    : defaultSEO.title;

  return (
    <Head>
      {/* Basic Meta Tags */}
      <title>{fullTitle}</title>
      <meta name="description" content={description} />
      <meta name="keywords" content={keywords} />
      <meta name="author" content={author} />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <meta httpEquiv="Content-Language" content="en" />
      
      {/* Canonical URL */}
      <link rel="canonical" href={canonical || fullUrl} />
      
      {/* Robots */}
      {noIndex ? (
        <meta name="robots" content="noindex, nofollow" />
      ) : (
        <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1" />
      )}
      
      {/* Open Graph / Facebook */}
      <meta property="og:type" content={type} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:url" content={fullUrl} />
      <meta property="og:image" content={fullImageUrl} />
      <meta property="og:image:alt" content={title || defaultSEO.title} />
      <meta property="og:site_name" content="DharmaMind" />
      <meta property="og:locale" content="en_US" />
      
      {/* Article specific */}
      {type === 'article' && (
        <>
          {publishedTime && <meta property="article:published_time" content={publishedTime} />}
          {modifiedTime && <meta property="article:modified_time" content={modifiedTime} />}
          <meta property="article:author" content={author} />
          <meta property="article:section" content="Spirituality" />
          <meta property="article:tag" content={keywords} />
        </>
      )}
      
      {/* Twitter Card */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={fullTitle} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={fullImageUrl} />
      <meta name="twitter:image:alt" content={title || defaultSEO.title} />
      <meta name="twitter:creator" content="@dharmamind" />
      <meta name="twitter:site" content="@dharmamind" />
      
      {/* Additional Meta Tags */}
      <meta name="theme-color" content="#8B5CF6" />
      <meta name="application-name" content="DharmaMind" />
      <meta name="apple-mobile-web-app-title" content="DharmaMind" />
      <meta name="apple-mobile-web-app-capable" content="yes" />
      <meta name="apple-mobile-web-app-status-bar-style" content="default" />
      <meta name="mobile-web-app-capable" content="yes" />
      
      {/* Structured Data for Organization */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "Organization",
            "name": "DharmaMind",
            "description": description,
            "url": baseUrl,
            "logo": `${baseUrl}/logo.jpeg`,
            "sameAs": [
              "https://dharmamind.org"
            ],
            "contactPoint": {
              "@type": "ContactPoint",
              "contactType": "customer service",
              "availableLanguage": "English"
            }
          })
        }}
      />
      
      {/* Structured Data for WebSite */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": "DharmaMind",
            "description": description,
            "url": baseUrl,
            "potentialAction": {
              "@type": "SearchAction",
              "target": `${baseUrl}/search?q={search_term_string}`,
              "query-input": "required name=search_term_string"
            }
          })
        }}
      />
      
      {/* Favicon and Icons */}
      <link rel="icon" type="image/x-icon" href="/favicon.ico" />
      <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
      <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
      <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
      <link rel="manifest" href="/site.webmanifest" />
    </Head>
  );
}
