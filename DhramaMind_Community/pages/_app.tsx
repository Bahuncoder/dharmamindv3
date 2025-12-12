import type { AppProps } from 'next/app';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { ColorProvider } from '../contexts/ColorContext';
import { ThemeProvider } from '../contexts/ThemeContext';
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

// Top loading bar component
const TopLoadingBar = ({ isLoading }: { isLoading: boolean }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (isLoading) {
      setProgress(0);
      const timer1 = setTimeout(() => setProgress(30), 100);
      const timer2 = setTimeout(() => setProgress(60), 300);
      const timer3 = setTimeout(() => setProgress(80), 600);
      return () => {
        clearTimeout(timer1);
        clearTimeout(timer2);
        clearTimeout(timer3);
      };
    } else {
      setProgress(100);
      const timer = setTimeout(() => setProgress(0), 200);
      return () => clearTimeout(timer);
    }
  }, [isLoading]);

  if (progress === 0) return null;

  return (
    <div className="fixed top-0 left-0 right-0 z-[9999] h-1 bg-transparent">
      <div
        className="h-full bg-gradient-to-r from-gold-500 via-gold-600 to-gold-500 transition-all duration-300 ease-out"
        style={{ 
          width: `${progress}%`,
          boxShadow: '0 0 10px rgba(212, 168, 84, 0.5)'
        }}
      />
    </div>
  );
};

export default function App({
  Component,
  pageProps
}: AppProps) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const handleStart = () => setIsLoading(true);
    const handleComplete = () => setIsLoading(false);

    router.events.on('routeChangeStart', handleStart);
    router.events.on('routeChangeComplete', handleComplete);
    router.events.on('routeChangeError', handleComplete);

    return () => {
      router.events.off('routeChangeStart', handleStart);
      router.events.off('routeChangeComplete', handleComplete);
      router.events.off('routeChangeError', handleComplete);
    };
  }, [router]);

  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="system">
        <ColorProvider>
          <ToastProvider>
            <NotificationProvider>
              <SessionManager>
                <TopLoadingBar isLoading={isLoading} />
                <Component {...pageProps} />
              </SessionManager>
            </NotificationProvider>
          </ToastProvider>
        </ColorProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}
