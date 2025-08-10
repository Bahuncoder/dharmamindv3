/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
    NEXT_PUBLIC_SITE_URL: process.env.NEXT_PUBLIC_SITE_URL || 'https://dharmamind.vercel.app',
  },

  // Image optimization for SEO
  images: {
    domains: ['dharmamind.vercel.app', 'via.placeholder.com'],
    formats: ['image/avif', 'image/webp'],
  },

  // Compression for performance
  compress: true,

  // Security headers for SEO
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

  // SEO-friendly redirects
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

  async rewrites() {
    return [
      // Only proxy specific API routes to backend, not auth routes
      {
        source: '/api/v1/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/:path*`,
      },
      {
        source: '/api/chat/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/chat/:path*`,
      },
      // Keep NextAuth routes local (don't proxy /api/auth/*)
    ]
  },

  // Performance optimizations
  swcMinify: true,
  trailingSlash: false,
}

module.exports = nextConfig
