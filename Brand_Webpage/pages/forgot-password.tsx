import { useEffect } from 'react';
import { useRouter } from 'next/router';

export default function ForgotPassword() {
  const router = useRouter();
  
  useEffect(() => {
    router.replace('/auth?mode=forgot-password');
  }, [router]);

  // Show loading while redirecting
  return (
    <div className="min-h-screen bg-gradient-to-br from-stone-100 via-amber-50 to-emerald-50 flex items-center justify-center">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-stone-300 border-t-stone-800 rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-stone-600">Redirecting to password reset...</p>
      </div>
    </div>
  );
}
