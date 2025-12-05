import { useEffect, useCallback } from 'react';

// Secure storage with encryption for sensitive data
class SecureStorage {
  private static encrypt(data: string, key: string): string {
    // Simple XOR encryption (use proper encryption in production)
    let result = '';
    for (let i = 0; i < data.length; i++) {
      result += String.fromCharCode(data.charCodeAt(i) ^ key.charCodeAt(i % key.length));
    }
    return btoa(result);
  }

  private static decrypt(encryptedData: string, key: string): string {
    try {
      const data = atob(encryptedData);
      let result = '';
      for (let i = 0; i < data.length; i++) {
        result += String.fromCharCode(data.charCodeAt(i) ^ key.charCodeAt(i % key.length));
      }
      return result;
    } catch {
      return '';
    }
  }

  private static getKey(): string {
    return 'dharmamind-secure-key-2025';
  }

  static setSecureItem(key: string, value: string): void {
    try {
      const encrypted = this.encrypt(value, this.getKey());
      localStorage.setItem(`secure_${key}`, encrypted);
    } catch (error) {
      console.warn('Failed to store secure item:', error);
    }
  }

  static getSecureItem(key: string): string | null {
    try {
      const encrypted = localStorage.getItem(`secure_${key}`);
      if (!encrypted) return null;
      return this.decrypt(encrypted, this.getKey());
    } catch (error) {
      console.warn('Failed to retrieve secure item:', error);
      return null;
    }
  }

  static removeSecureItem(key: string): void {
    try {
      localStorage.removeItem(`secure_${key}`);
    } catch (error) {
      console.warn('Failed to remove secure item:', error);
    }
  }

  static clearSecureStorage(): void {
    try {
      const keys = Object.keys(localStorage).filter(key => key.startsWith('secure_'));
      keys.forEach(key => localStorage.removeItem(key));
    } catch (error) {
      console.warn('Failed to clear secure storage:', error);
    }
  }
}

// Enhanced storage hook with security
export const useSecureStorage = (key: string, defaultValue: string = '') => {
  const setStoredValue = useCallback((value: string) => {
    SecureStorage.setSecureItem(key, value);
  }, [key]);

  const getStoredValue = useCallback((): string => {
    return SecureStorage.getSecureItem(key) || defaultValue;
  }, [key, defaultValue]);

  const removeStoredValue = useCallback(() => {
    SecureStorage.removeSecureItem(key);
  }, [key]);

  return {
    getStoredValue,
    setStoredValue,
    removeStoredValue
  };
};

// Session management with security
export class SessionManager {
  private static readonly SESSION_KEY = 'dharma_session';
  private static readonly ADMIN_KEY = 'dharma_admin';
  private static readonly TOKEN_KEY = 'dharma_token';

  static setUserSession(userData: any): void {
    SecureStorage.setSecureItem(this.SESSION_KEY, JSON.stringify({
      ...userData,
      timestamp: Date.now(),
      expiresAt: Date.now() + (24 * 60 * 60 * 1000) // 24 hours
    }));
  }

  static getUserSession(): any | null {
    try {
      const sessionData = SecureStorage.getSecureItem(this.SESSION_KEY);
      if (!sessionData) return null;

      const parsed = JSON.parse(sessionData);
      
      // Check if session is expired
      if (parsed.expiresAt && Date.now() > parsed.expiresAt) {
        this.clearUserSession();
        return null;
      }

      return parsed;
    } catch {
      return null;
    }
  }

  static setAdminSession(adminData: any): void {
    SecureStorage.setSecureItem(this.ADMIN_KEY, JSON.stringify({
      ...adminData,
      timestamp: Date.now(),
      expiresAt: Date.now() + (8 * 60 * 60 * 1000) // 8 hours for admin
    }));
  }

  static getAdminSession(): any | null {
    try {
      const adminData = SecureStorage.getSecureItem(this.ADMIN_KEY);
      if (!adminData) return null;

      const parsed = JSON.parse(adminData);
      
      // Check if session is expired
      if (parsed.expiresAt && Date.now() > parsed.expiresAt) {
        this.clearAdminSession();
        return null;
      }

      return parsed;
    } catch {
      return null;
    }
  }

  static setAuthToken(token: string): void {
    SecureStorage.setSecureItem(this.TOKEN_KEY, token);
  }

  static getAuthToken(): string | null {
    return SecureStorage.getSecureItem(this.TOKEN_KEY);
  }

  static clearUserSession(): void {
    SecureStorage.removeSecureItem(this.SESSION_KEY);
    SecureStorage.removeSecureItem(this.TOKEN_KEY);
  }

  static clearAdminSession(): void {
    SecureStorage.removeSecureItem(this.ADMIN_KEY);
  }

  static clearAllSessions(): void {
    this.clearUserSession();
    this.clearAdminSession();
  }

  static isUserAuthenticated(): boolean {
    return this.getUserSession() !== null;
  }

  static isAdminAuthenticated(): boolean {
    return this.getAdminSession() !== null;
  }
}

// Auto-cleanup expired sessions
export const useSessionCleanup = () => {
  useEffect(() => {
    const cleanup = () => {
      SessionManager.getUserSession(); // This will auto-remove if expired
      SessionManager.getAdminSession(); // This will auto-remove if expired
    };

    // Run cleanup on mount
    cleanup();

    // Run cleanup every 5 minutes
    const interval = setInterval(cleanup, 5 * 60 * 1000);

    return () => clearInterval(interval);
  }, []);
};

export default SecureStorage;
