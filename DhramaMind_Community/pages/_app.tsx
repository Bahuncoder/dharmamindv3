import type { AppProps } from 'next/app';
import { ColorProvider } from '../contexts/ColorContext';
import { ToastProvider } from '../contexts/ToastContext';
import ErrorBoundary from '../components/ErrorBoundary';
import { useSessionCleanup } from '../utils/secureStorage';
import '../styles/globals.css';
import '../styles/colors.css';

// Session cleanup component
const SessionManager: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  useSessionCleanup();
  return <>{children}</>;
};

export default function App({ 
  Component, 
  pageProps 
}: AppProps) {
  return (
    <ErrorBoundary>
      <ColorProvider>
        <ToastProvider>
          <SessionManager>
            <Component {...pageProps} />
          </SessionManager>
        </ToastProvider>
      </ColorProvider>
    </ErrorBoundary>
  );
}
