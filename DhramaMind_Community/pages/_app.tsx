import type { AppProps } from 'next/app'
import { ColorProvider } from '../contexts/ColorContext'
import '../styles/globals.css'
import '../styles/colors.css'

export default function App({ 
  Component, 
  pageProps 
}: AppProps) {
  return (
    <ColorProvider>
      <Component {...pageProps} />
    </ColorProvider>
  )
}
