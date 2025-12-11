import { DefaultSession, DefaultUser } from "next-auth"

declare module "next-auth" {
  interface User extends DefaultUser {
    plan?: string
  }

  interface Session {
    accessToken?: string
    provider?: string
    user: {
      id?: string
      plan?: string
    } & DefaultSession["user"]
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken?: string
    provider?: string
    plan?: string
  }
}
