import type { AppProps } from 'next/app'
import { SessionProvider } from 'next-auth/react'
import { AuthProvider } from '../contexts/AuthContext'
import { ColorProvider } from '../contexts/ColorContext'
import { SubscriptionProvider } from '../contexts/SubscriptionContext'
import { ThemeProvider } from '../contexts/ThemeContext'
import '../styles/globals.css'
import '../styles/chat-interface-enhanced.css'
import '../styles/advanced-spiritual-design.css'

export default function App({ 
  Component, 
  pageProps: { session, ...pageProps } 
}: AppProps) {
  return (
    <SessionProvider session={session}>
      <AuthProvider>
        <SubscriptionProvider>
          <ColorProvider>
            <ThemeProvider>
              <Component {...pageProps} />
            </ThemeProvider>
          </ColorProvider>
        </SubscriptionProvider>
      </AuthProvider>
    </SessionProvider>
  )
}
