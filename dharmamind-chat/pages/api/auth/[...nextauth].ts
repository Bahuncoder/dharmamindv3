import NextAuth from 'next-auth'
import GoogleProvider from 'next-auth/providers/google'
import CredentialsProvider from 'next-auth/providers/credentials'

export default NextAuth({
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        // Hardcoded test accounts for development
        const testAccounts = [
          { 
            id: 'test-basic',
            email: 'test@dharmamind.com', 
            password: 'test1234',
            name: 'Test User',
            plan: 'basic'
          },
          { 
            id: 'test-pro',
            email: 'pro@dharmamind.com', 
            password: 'pro12345',
            name: 'Pro User',
            plan: 'pro'
          },
          { 
            id: 'test-max',
            email: 'max@dharmamind.com', 
            password: 'max12345',
            name: 'Max User',
            plan: 'max'
          },
          { 
            id: 'bahuncoder',
            email: 'bahuncoder@gmail.com', 
            password: 'bahun1234',
            name: 'Bahun Coder',
            plan: 'max'
          }
        ];

        const user = testAccounts.find(
          acc => acc.email === credentials?.email && acc.password === credentials?.password
        );

        if (user) {
          return {
            id: user.id,
            email: user.email,
            name: user.name,
            plan: user.plan
          };
        }
        
        return null;
      }
    })
  ],
  callbacks: {
    async jwt({ token, account, profile, user }) {
      // Persist the OAuth access_token and or the user id to the token right after signin
      if (account) {
        token.accessToken = account.access_token
        token.provider = account.provider
      }
      // Add plan from credentials provider
      if (user?.plan) {
        token.plan = user.plan
      }
      return token
    },
    async session({ session, token }) {
      // Send properties to the client, like an access_token and user id from a provider.
      session.accessToken = token.accessToken as string
      session.provider = token.provider as string
      session.user.plan = token.plan as string
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
