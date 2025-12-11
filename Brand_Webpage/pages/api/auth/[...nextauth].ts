<<<<<<< HEAD
import NextAuth, { NextAuthOptions } from 'next-auth'
import GoogleProvider from 'next-auth/providers/google'
import CredentialsProvider from 'next-auth/providers/credentials'

// Central Auth Configuration for DharmaMind Platform
// This is the main auth server - Chat and Community redirect here

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID || '',
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || '',
    }),
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        // TODO: Replace with real database lookup
        // For now, using test accounts for development
        const testAccounts = [
          {
            id: 'user-1',
            email: 'test@dharmamind.com',
            password: 'test1234',
            name: 'Test User',
            plan: 'free',
            image: null
          },
          {
            id: 'user-2',
            email: 'pro@dharmamind.com',
            password: 'pro12345',
            name: 'Pro User',
            plan: 'pro',
            image: null
          },
          {
            id: 'user-3',
            email: 'enterprise@dharmamind.com',
            password: 'ent12345',
            name: 'Enterprise User',
            plan: 'enterprise',
            image: null
          },
        ];

        const user = testAccounts.find(
          acc => acc.email === credentials?.email && acc.password === credentials?.password
        );

        if (user) {
          return {
            id: user.id,
            email: user.email,
            name: user.name,
            plan: user.plan,
            image: user.image
          };
        }

        return null;
      }
    })
  ],
  callbacks: {
    async jwt({ token, account, user }) {
=======
import NextAuth from 'next-auth'
import GoogleProvider from 'next-auth/providers/google'

export default NextAuth({
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    })
  ],
  callbacks: {
    async jwt({ token, account, profile }) {
      // Persist the OAuth access_token and or the user id to the token right after signin
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
      if (account) {
        token.accessToken = account.access_token
        token.provider = account.provider
      }
<<<<<<< HEAD
      if (user) {
        token.id = user.id
        token.plan = (user as any).plan || 'free'
      }
      return token
    },
    async session({ session, token }: { session: any; token: any }) {
      if (session.user) {
        session.user.id = token.id
        session.user.plan = token.plan
        session.accessToken = token.accessToken
        session.provider = token.provider
      }
      return session
    },
    async redirect({ url, baseUrl }) {
      // Handle cross-origin redirects for Chat and Community
      const allowedOrigins = [
        process.env.NEXTAUTH_URL || baseUrl,
        process.env.CHAT_URL || 'http://localhost:3000',
        process.env.COMMUNITY_URL || 'http://localhost:3002',
      ];

      // Check if redirect URL is from allowed origins
      try {
        const urlOrigin = new URL(url).origin;
        if (allowedOrigins.some(origin => urlOrigin === origin)) {
          return url;
        }
      } catch {
        // Invalid URL, use default
      }

      if (url.startsWith("/")) return `${baseUrl}${url}`;
      return baseUrl;
    }
  },
  pages: {
    signIn: '/auth/login',
    signOut: '/auth/logout',
    error: '/auth/error',
    newUser: '/auth/welcome',
  },
  session: {
    strategy: 'jwt',
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },
  secret: process.env.NEXTAUTH_SECRET,
}

export default NextAuth(authOptions)
=======
      return token
    },
    async session({ session, token }) {
      // Send properties to the client, like an access_token and user id from a provider.
      session.accessToken = token.accessToken as string
      session.provider = token.provider as string
      return session
    },
    async redirect({ url, baseUrl }) {
      // Allows relative callback URLs
      if (url.startsWith("/")) return `${baseUrl}${url}`
      // Allows callback URLs on the same origin
      else if (new URL(url).origin === baseUrl) return url;
      return baseUrl + "/chat";
    }
  },
  pages: {
    signIn: '/login',
    error: '/login', // Error code passed in query string as ?error=
  },
  session: {
    strategy: 'jwt',
  },
  secret: process.env.NEXTAUTH_SECRET,
})
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
