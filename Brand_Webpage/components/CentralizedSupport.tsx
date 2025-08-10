/**
 * 🕉️ DharmaMind Centralized Support Components
 * 
 * Unified components for contact, help, and support functionality
 * Used across multiple pages to ensure consistency
 */

import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { useColors } from '../contexts/ColorContext';

// ===============================
// TYPES & INTERFACES
// ===============================

export interface ContactFormData {
  name: string;
  email: string;
  subject: string;
  category: 'general' | 'support' | 'billing' | 'feedback' | 'partnership';
  message: string;
  priority: 'low' | 'medium' | 'high';
}

export interface SupportLinkProps {
  variant?: 'button' | 'link' | 'card';
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  className?: string;
  children?: React.ReactNode;
}

export interface ContactButtonProps extends SupportLinkProps {
  type?: 'contact' | 'help' | 'support';
  prefillCategory?: ContactFormData['category'];
}

// ===============================
// CENTRALIZED CONTACT BUTTON
// ===============================

export const ContactButton: React.FC<ContactButtonProps> = ({
  type = 'contact',
  variant = 'button',
  size = 'md',
  showIcon = true,
  prefillCategory,
  className = '',
  children
}) => {
  const router = useRouter();

  const handleClick = () => {
    const params = new URLSearchParams();
    if (prefillCategory) {
      params.set('category', prefillCategory);
    }
    
    const url = type === 'help' 
      ? '/help' 
      : `/contact${params.toString() ? '?' + params.toString() : ''}`;
    
    router.push(url);
  };

  const getVariantClasses = () => {
    const sizeClasses = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2',
      lg: 'px-6 py-3 text-lg'
    };

    switch (variant) {
      case 'link':
        return `${sizeClasses[size]} text-primary hover:text-primary-dark underline transition-colors`;
      case 'card':
        return `${sizeClasses[size]} card-background border border-card-border rounded-lg hover:shadow-md transition-shadow`;
      default:
        return `${sizeClasses[size]} btn-primary transition-colors`;
    }
  };

  const getIcon = () => {
    if (!showIcon) return null;
    
    const iconClass = "w-4 h-4 mr-2";
    
    switch (type) {
      case 'help':
        return (
          <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'support':
        return (
          <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192L5.636 18.364M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-5 0a4 4 0 11-8 0 4 4 0 018 0z" />
          </svg>
        );
      default:
        return (
          <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        );
    }
  };

  const getDefaultText = () => {
    switch (type) {
      case 'help': return 'Help Center';
      case 'support': return 'Contact Support';
      default: return 'Contact';
    }
  };

  return (
    <button
      onClick={handleClick}
      className={`flex items-center justify-center font-medium transition-all ${getVariantClasses()} ${className}`}
    >
      {getIcon()}
      {children || getDefaultText()}
    </button>
  );
};

// ===============================
// CENTRALIZED SUPPORT SECTION
// ===============================

export const SupportSection: React.FC<{
  title?: string;
  subtitle?: string;
  showLinks?: boolean;
  compact?: boolean;
  className?: string;
}> = ({
  title = "Support",
  subtitle = "Get help when you need it",
  showLinks = true,
  compact = false,
  className = ''
}) => {
  const router = useRouter();

  if (compact) {
    return (
      <div className={`space-y-2 ${className}`}>
        <h3 className="text-sm font-semibold text-primary mb-2">{title}</h3>
        <ul className="space-y-1 text-sm text-secondary">
          <li>
            <button 
              onClick={() => router.push('/help')} 
              className="hover:text-primary text-left"
            >
              Help Center
            </button>
          </li>
          <li>
            <button 
              onClick={() => router.push('/contact')} 
              className="hover:text-primary text-left"
            >
              Contact
            </button>
          </li>
        </ul>
      </div>
    );
  }

  return (
    <div className={`bg-page-background rounded-lg border border-card-border p-6 ${className}`}>
      <h3 className="text-lg font-semibold text-primary mb-2">{title}</h3>
      <p className="text-secondary mb-4">{subtitle}</p>
      
      {showLinks && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <ContactButton 
            type="help" 
            variant="card" 
            className="text-left"
          >
            <div>
              <div className="font-medium text-primary">Help Center</div>
              <div className="text-sm text-secondary">Browse articles and guides</div>
            </div>
          </ContactButton>
          
          <ContactButton 
            type="contact" 
            variant="card" 
            className="text-left"
          >
            <div>
              <div className="font-medium text-primary">Contact Us</div>
              <div className="text-sm text-secondary">Send us a message</div>
            </div>
          </ContactButton>
        </div>
      )}
    </div>
  );
};

// ===============================
// CENTRALIZED CONTACT FORM
// ===============================

export const ContactForm: React.FC<{
  initialCategory?: ContactFormData['category'];
  onSubmit?: (data: ContactFormData) => void;
  compact?: boolean;
}> = ({ 
  initialCategory = 'general',
  onSubmit,
  compact = false 
}) => {
  const [formData, setFormData] = useState<ContactFormData>({
    name: '',
    email: '',
    subject: '',
    category: initialCategory,
    message: '',
    priority: 'medium'
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      // Here you would normally send to your API
      console.log('Contact form submitted:', formData);
      
      if (onSubmit) {
        await onSubmit(formData);
      }
      
      setSuccess(true);
      setFormData({
        name: '',
        email: '',
        subject: '',
        category: 'general',
        message: '',
        priority: 'medium'
      });
    } catch (error) {
      console.error('Failed to submit contact form:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleChange = (field: keyof ContactFormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  if (success) {
    return (
      <div className="text-center py-8">
        <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-primary mb-2">Message Sent! 🙏</h3>
        <p className="text-secondary">We'll get back to you within 24 hours.</p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className={`grid gap-4 ${compact ? 'grid-cols-1' : 'grid-cols-2'}`}>
        <div>
          <label className="block text-sm font-medium text-primary mb-2">
            Name *
          </label>
          <input
            type="text"
            required
            value={formData.name}
            onChange={(e) => handleChange('name', e.target.value)}
            className="input-primary"
            placeholder="Your name"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-primary mb-2">
            Email *
          </label>
          <input
            type="email"
            required
            value={formData.email}
            onChange={(e) => handleChange('email', e.target.value)}
            className="input-primary"
            placeholder="your@email.com"
          />
        </div>
      </div>

      <div className={`grid gap-4 ${compact ? 'grid-cols-1' : 'grid-cols-2'}`}>
        <div>
          <label className="block text-sm font-medium text-primary mb-2">
            Category
          </label>
          <select
            value={formData.category}
            onChange={(e) => handleChange('category', e.target.value as ContactFormData['category'])}
            className="input-primary"
          >
            <option value="general">General Inquiry</option>
            <option value="support">Technical Support</option>
            <option value="billing">Billing Question</option>
            <option value="feedback">Product Feedback</option>
            <option value="partnership">Partnership</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-primary mb-2">
            Priority
          </label>
          <select
            value={formData.priority}
            onChange={(e) => handleChange('priority', e.target.value as ContactFormData['priority'])}
            className="input-primary"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-primary mb-2">
          Subject *
        </label>
        <input
          type="text"
          required
          value={formData.subject}
          onChange={(e) => handleChange('subject', e.target.value)}
          className="input-primary"
          placeholder="Brief description of your inquiry"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-primary mb-2">
          Message *
        </label>
        <textarea
          required
          rows={compact ? 4 : 6}
          value={formData.message}
          onChange={(e) => handleChange('message', e.target.value)}
          className="input-primary resize-none"
          placeholder="Please provide as much detail as possible..."
        />
      </div>

      <button
        type="submit"
        disabled={isSubmitting}
        className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isSubmitting ? 'Sending...' : 'Send Message'}
      </button>
    </form>
  );
};

// ===============================
// CENTRALIZED HELP LINK
// ===============================

export const HelpLink: React.FC<SupportLinkProps> = ({
  variant = 'link',
  size = 'md',
  showIcon = true,
  className = '',
  children
}) => {
  return (
    <ContactButton
      type="help"
      variant={variant}
      size={size}
      showIcon={showIcon}
      className={className}
    >
      {children}
    </ContactButton>
  );
};

// ===============================
// CENTRALIZED SUPPORT CONTACT
// ===============================

export const SupportContact: React.FC<{
  title?: string;
  description?: string;
  category?: ContactFormData['category'];
} & SupportLinkProps> = ({
  title = "Still need help?",
  description = "Our support team is here to help.",
  category = 'support',
  variant = 'button',
  size = 'md',
  className = ''
}) => {
  return (
    <div className={`text-center p-6 card-background rounded-lg ${className}`}>
      <h3 className="text-lg font-semibold text-primary mb-2">{title}</h3>
      <p className="text-secondary mb-4">{description}</p>
      <ContactButton
        type="support"
        variant={variant}
        size={size}
        prefillCategory={category}
      />
    </div>
  );
};

export default {
  ContactButton,
  SupportSection,
  ContactForm,
  HelpLink,
  SupportContact
};
