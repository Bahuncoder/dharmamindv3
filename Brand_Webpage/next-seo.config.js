// SEO Configuration for Next.js

/** @type {import('next').NextConfig} */
const nextConfig = {
  // SEO and Meta Configuration
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
    NEXT_PUBLIC_SITE_URL: process.env.NEXT_PUBLIC_SITE_URL || 'https://dharmamind.vercel.app',
  },

  // Image optimization
  images: {
    domains: ['dharmamind.vercel.app', 'via.placeholder.com'],
    formats: ['image/avif', 'image/webp'],
  },

  // Compression
  compress: true,

  // Headers for SEO
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
        ],
      },
    ];
  },

  // Redirects for SEO
  async redirects() {
    return [
      {
        source: '/home',
        destination: '/',
        permanent: true,
      },
      {
        source: '/live-app',
        destination: 'https://dharmamind.org',
        permanent: true,
      },
    ];
  },

  // Rewrites for API
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/:path*`,
      },
      {
        source: '/api/chat/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/chat/:path*`,
      },
    ];
  },

  // PWA and performance
  experimental: {
    optimizeCss: true,
  },

  // Generate static pages for better SEO
  trailingSlash: false,
  
  // Enable SWC minification for better performance
  swcMinify: true,
};

module.exports = nextConfig;
