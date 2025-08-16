/** @type {import('next').NextConfig} */
const nextConfig = {
  // Force Pages Router structure - App Router is disabled by default in Next.js 14
  pageExtensions: ['ts', 'tsx', 'js', 'jsx'],
  
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
  },
  async redirects() {
    return [
      // Centralized redirects - replaces multiple redirect pages
      {
        source: '/login',
        destination: '/auth?mode=login',
        permanent: true,
      },
      {
        source: '/signup',
        destination: '/auth?mode=signup',
        permanent: true,
      },
      {
        source: '/signin',
        destination: '/auth?mode=login',
        permanent: true,
      },
      {
        source: '/register',
        destination: '/auth?mode=signup',
        permanent: true,
      },
      {
        source: '/login-redirect',
        destination: '/auth?mode=login',
        permanent: true,
      },
      {
        source: '/signup-redirect',
        destination: '/auth?mode=signup',
        permanent: true,
      },
      // Subscription redirects - centralized subscription management
      {
        source: '/subscription',
        destination: '/chat?subscription=true',
        permanent: true,
      },
      {
        source: '/subscription_new',
        destination: '/chat?subscription=true',
        permanent: true,
      },
      {
        source: '/subscribe',
        destination: '/chat?subscription=true',
        permanent: true,
      },
      {
        source: '/billing',
        destination: '/chat?subscription=true',
        permanent: true,
      },
      {
        source: '/upgrade',
        destination: '/chat?subscription=true',
        permanent: true,
      },
      {
        source: '/plans',
        destination: '/chat?subscription=true',
        permanent: true,
      },
      // Password reset redirects
      {
        source: '/forgot-password',
        destination: '/auth?mode=forgot-password',
        permanent: true,
      },
      {
        source: '/forgot-password-new',
        destination: '/auth?mode=forgot-password',
        permanent: true,
      },
      {
        source: '/reset-password',
        destination: '/auth?mode=forgot-password',
        permanent: true,
      },
      // Mobile redirects
      {
        source: '/mobile-chat',
        destination: '/chat',
        permanent: true,
      },
      // Other legacy redirects
      {
        source: '/welcome',
        destination: '/',
        permanent: true,
      },
      {
        source: '/home',
        destination: '/',
        permanent: true,
      },
      {
        source: '/landing',
        destination: '/',
        permanent: true,
      },
      {
        source: '/dashboard',
        destination: '/chat',
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
}

module.exports = nextConfig
