import { useEffect } from 'react';
import { useRouter } from 'next/router';

export default function ForgotPassword() {
  const router = useRouter();
  
  useEffect(() => {
    router.replace('/auth?mode=forgot-password');
  }, [router]);

  // Show loading while redirecting
  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-100 via-neutral-50 to-neutral-100 flex items-center justify-center">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-neutral-300 border-t-stone-800 rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-neutral-600">Redirecting to password reset...</p>
      </div>
    </div>
  );
}
