import type { AppProps } from 'next/app'
import { SessionProvider } from 'next-auth/react'
import { AuthProvider } from '../contexts/AuthContext'
import { ColorProvider } from '../contexts/ColorContext'
import { SubscriptionProvider } from '../contexts/SubscriptionContext'
import { ThemeProvider } from '../contexts/ThemeContext'
import { RishiChatProvider } from '../contexts/RishiChatContext'
import '../styles/globals.css'
import '../styles/colors.css'
import '../styles/chat.css'
import '../styles/chat-enhanced.css'
import '../styles/chat-interface-enhanced.css'
import '../styles/enhanced-chat-ui.css'
import '../styles/enhanced-message-bubble.css'
import '../styles/enhanced-input.css'
import '../styles/advanced-spiritual-design.css'
import '../styles/enhanced-layout.css'

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
              <RishiChatProvider>
                <Component {...pageProps} />
              </RishiChatProvider>
            </ThemeProvider>
          </ColorProvider>
        </SubscriptionProvider>
      </AuthProvider>
    </SessionProvider>
  )
}
