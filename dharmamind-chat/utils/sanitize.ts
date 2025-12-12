/**
 * 🔒 Secure HTML Sanitization Utility
 * =====================================
 * 
 * Provides XSS protection using DOMPurify for safely rendering
 * user-generated or external HTML content.
 * 
 * Usage:
 *   import { sanitizeHtml, SafeHtml } from '@/utils/sanitize';
 *   
 *   // Direct sanitization
 *   const clean = sanitizeHtml(dirtyHtml);
 *   
 *   // React component
 *   <SafeHtml html={content} />
 */

import DOMPurify from 'dompurify';

/**
 * DOMPurify configuration for maximum security
 */
const DOMPURIFY_CONFIG: DOMPurify.Config = {
    // Allowed HTML tags
    ALLOWED_TAGS: [
        'p', 'br', 'b', 'i', 'em', 'strong', 'u', 's', 'strike',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li',
        'blockquote', 'pre', 'code',
        'a', 'img',
        'table', 'thead', 'tbody', 'tr', 'th', 'td',
        'div', 'span',
        'hr',
    ],

    // Allowed attributes
    ALLOWED_ATTR: [
        'href', 'src', 'alt', 'title', 'class', 'id',
        'target', 'rel',
        'width', 'height',
    ],

    // Force all links to open in new tab with security attributes
    ADD_ATTR: ['target', 'rel'],

    // Disallow potentially dangerous protocols
    ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto|tel):|[^a-z]|[a-z+.-]+(?:[^a-z+.\-:]|$))/i,

    // Additional security settings
    FORBID_TAGS: ['script', 'style', 'iframe', 'form', 'input', 'button'],
    FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover', 'onfocus', 'onblur'],

    // Data attributes (disabled for security)
    ALLOW_DATA_ATTR: false,

    // Return string (not DOM node)
    RETURN_DOM: false,
    RETURN_DOM_FRAGMENT: false,
};

/**
 * Sanitize HTML string using DOMPurify
 * 
 * @param dirty - Potentially unsafe HTML string
 * @param config - Optional custom DOMPurify config
 * @returns Sanitized HTML string
 */
export function sanitizeHtml(
    dirty: string,
    config?: DOMPurify.Config
): string {
    if (!dirty) return '';

    // Merge custom config with defaults
    const finalConfig = config ? { ...DOMPURIFY_CONFIG, ...config } : DOMPURIFY_CONFIG;

    // Add security hooks
    DOMPurify.addHook('afterSanitizeAttributes', (node) => {
        // Set target="_blank" and rel="noopener noreferrer" for all links
        if (node.tagName === 'A') {
            node.setAttribute('target', '_blank');
            node.setAttribute('rel', 'noopener noreferrer');
        }

        // Ensure images have alt attribute
        if (node.tagName === 'IMG' && !node.getAttribute('alt')) {
            node.setAttribute('alt', 'Image');
        }
    });

    const clean = DOMPurify.sanitize(dirty, finalConfig);

    // Remove hook after use
    DOMPurify.removeHook('afterSanitizeAttributes');

    return clean as string;
}

/**
 * Sanitize plain text (escape HTML entities)
 * 
 * @param text - Plain text that might contain HTML
 * @returns Escaped text safe for display
 */
export function escapeHtml(text: string): string {
    if (!text) return '';

    const escapeMap: Record<string, string> = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;',
        '/': '&#x2F;',
        '`': '&#x60;',
        '=': '&#x3D;',
    };

    return text.replace(/[&<>"'`=/]/g, (char) => escapeMap[char]);
}

/**
 * Sanitize URL to prevent javascript: and data: protocols
 * 
 * @param url - URL string to sanitize
 * @returns Sanitized URL or empty string if dangerous
 */
export function sanitizeUrl(url: string): string {
    if (!url) return '';

    const trimmed = url.trim().toLowerCase();

    // Block dangerous protocols
    const dangerousProtocols = [
        'javascript:',
        'data:',
        'vbscript:',
        'file:',
    ];

    for (const protocol of dangerousProtocols) {
        if (trimmed.startsWith(protocol)) {
            console.warn(`[Security] Blocked dangerous URL: ${url.substring(0, 50)}`);
            return '';
        }
    }

    // Allow safe protocols
    const safeProtocols = ['http://', 'https://', 'mailto:', 'tel:', '/'];
    const isSafe = safeProtocols.some(p => trimmed.startsWith(p));

    if (!isSafe && !trimmed.startsWith('#')) {
        // Relative URLs are okay, but log unknown patterns
        if (trimmed.includes(':')) {
            console.warn(`[Security] Blocked unknown protocol: ${url.substring(0, 50)}`);
            return '';
        }
    }

    return url;
}

/**
 * React component for safely rendering HTML content
 * 
 * @example
 * <SafeHtml html={userContent} className="prose" />
 */
interface SafeHtmlProps {
    html: string;
    className?: string;
    as?: keyof JSX.IntrinsicElements;
}

export function SafeHtml({
    html,
    className = '',
    as: Component = 'div'
}: SafeHtmlProps): JSX.Element {
    const cleanHtml = sanitizeHtml(html);

    return (
        <Component
      className= { className }
    dangerouslySetInnerHTML = {{ __html: cleanHtml }
}
    />
  );
}

/**
 * Hook for sanitized HTML with memoization
 * 
 * @example
 * const cleanHtml = useSanitizedHtml(dirtyHtml);
 */
import { useMemo } from 'react';

export function useSanitizedHtml(dirty: string): string {
    return useMemo(() => sanitizeHtml(dirty), [dirty]);
}

// Default export for convenience
export default {
    sanitizeHtml,
    escapeHtml,
    sanitizeUrl,
    SafeHtml,
    useSanitizedHtml,
};
