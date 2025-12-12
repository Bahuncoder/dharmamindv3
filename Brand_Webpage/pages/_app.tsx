import type { AppProps } from 'next/app'
import { SessionProvider } from 'next-auth/react'
import { AuthProvider } from '../contexts/AuthContext'
import { ColorProvider } from '../contexts/ColorContext'
import { ThemeProvider } from '../contexts/ThemeContext'
import { CentralizedSystemProvider } from '../components/CentralizedSystem'
import '../styles/globals.css'

export default function App({ 
  Component, 
  pageProps: { session, ...pageProps } 
}: AppProps) {
  return (
    <SessionProvider session={session}>
      <ThemeProvider defaultTheme="system">
        <AuthProvider>
          <ColorProvider>
            <CentralizedSystemProvider>
              <Component {...pageProps} />
            </CentralizedSystemProvider>
          </ColorProvider>
        </AuthProvider>
      </ThemeProvider>
    </SessionProvider>
  )
}
