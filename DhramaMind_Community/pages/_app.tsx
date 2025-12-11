import type { AppProps } from 'next/app';
import { ColorProvider } from '../contexts/ColorContext';
import { ToastProvider } from '../contexts/ToastContext';
<<<<<<< HEAD
import { NotificationProvider } from '../contexts/NotificationContext';
=======
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
import ErrorBoundary from '../components/ErrorBoundary';
import { useSessionCleanup } from '../utils/secureStorage';
import '../styles/globals.css';
import '../styles/colors.css';

// Session cleanup component
const SessionManager: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  useSessionCleanup();
  return <>{children}</>;
};

<<<<<<< HEAD
export default function App({
  Component,
  pageProps
=======
export default function App({ 
  Component, 
  pageProps 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
}: AppProps) {
  return (
    <ErrorBoundary>
      <ColorProvider>
        <ToastProvider>
<<<<<<< HEAD
          <NotificationProvider>
            <SessionManager>
              <Component {...pageProps} />
            </SessionManager>
          </NotificationProvider>
=======
          <SessionManager>
            <Component {...pageProps} />
          </SessionManager>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
        </ToastProvider>
      </ColorProvider>
    </ErrorBoundary>
  );
}
