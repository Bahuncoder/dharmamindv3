import type { AppProps } from 'next/app';
import { ColorProvider } from '../contexts/ColorContext';
import { ToastProvider } from '../contexts/ToastContext';
import { NotificationProvider } from '../contexts/NotificationContext';
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
          <NotificationProvider>
            <SessionManager>
              <Component {...pageProps} />
            </SessionManager>
          </NotificationProvider>
        </ToastProvider>
      </ColorProvider>
    </ErrorBoundary>
  );
}
