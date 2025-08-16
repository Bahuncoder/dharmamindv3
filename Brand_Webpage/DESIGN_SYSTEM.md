# 🎨 DharmaMind Design System Documentation

## Overview

The DharmaMind Design System provides a comprehensive set of colors, components, and guidelines to ensure consistent visual design across all applications. This system emphasizes **emerald green accents** with **light gray foundations** for a professional, calming aesthetic.

---

## 🎯 Color Palette

### Primary Colors

| Color | Hex Code | Usage | Variable |
|-------|----------|-------|----------|
| **Dark Charcoal** | `#2C2C2C` | Primary text, headings | `--text-primary` |
| **Medium Gray** | `#6E6E6E` | Secondary text, descriptions | `--text-secondary` |
| **Light Gray** | `#6b7280` | UI elements, buttons | `--primary` |
| **Emerald Green** | `#10b981` | Accents, highlights, CTAs | `--accent` |

### Background Colors

| Color | Hex Code | Usage | Variable |
|-------|----------|-------|----------|
| **Pure White** | `#ffffff` | Main content areas | `--bg-primary` |
| **Light Gray** | `#f9fafb` | Page backgrounds | `--bg-secondary` |
| **Soft Gray** | `#f3f4f6` | Card backgrounds | `--bg-tertiary` |

### Border & State Colors

| Color | Hex Code | Usage | Variable |
|-------|----------|-------|----------|
| **Light Border** | `#e5e7eb` | Subtle borders | `--border-light` |
| **Medium Border** | `#d1d5db` | Standard borders | `--border-medium` |
| **Success Green** | `#10b981` | Success states | `--success` |
| **Warning Amber** | `#f59e0b` | Warning states | `--warning` |
| **Error Red** | `#ef4444` | Error states | `--error` |

---

## 🔘 Button System

### Primary Button
The main call-to-action button with emerald accent.

```css
.btn-primary {
  background-color: #6b7280;      /* Light gray background */
  color: #2C2C2C;                 /* Dark charcoal text */
  border: 2px solid #10b981;      /* Emerald border */
  border-radius: 0.5rem;
  padding: 0.75rem 1.5rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.btn-primary:hover {
  background-color: #4b5563;      /* Darker gray */
  transform: translateY(-1px);    /* Subtle lift */
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}
```

**Usage:**
- Main CTAs (Sign Up, Submit, Save)
- Primary navigation actions
- Form submissions

### Secondary Button
Outline style for secondary actions.

```css
.btn-secondary {
  background-color: transparent;
  color: #2C2C2C;
  border: 2px solid #10b981;
  border-radius: 0.5rem;
  padding: 0.75rem 1.5rem;
}

.btn-secondary:hover {
  background-color: #f3f4f6;
  transform: translateY(-1px);
}
```

**Usage:**
- Secondary actions (Cancel, Back)
- Alternative options
- Less important CTAs

### Button States

| State | Background | Text | Border |
|-------|------------|------|--------|
| **Default** | `#6b7280` | `#2C2C2C` | `#10b981` |
| **Hover** | `#4b5563` | `#2C2C2C` | `#10b981` |
| **Active** | `#374151` | `#2C2C2C` | `#059669` |
| **Disabled** | `#d1d5db` | `#9ca3af` | `#d1d5db` |

---

## 💬 Chat Interface

### Message Bubbles

#### User Messages
```css
.message-user {
  background-color: #10b981;      /* Emerald background */
  color: #ffffff;                 /* White text */
  border-radius: 1rem 1rem 0.25rem 1rem;
  padding: 0.75rem 1rem;
  margin-left: 20%;               /* Right-aligned */
  max-width: 80%;
}
```

#### AI Messages
```css
.message-ai {
  background-color: #f3f4f6;      /* Light gray background */
  color: #2C2C2C;                 /* Dark charcoal text */
  border-radius: 1rem 1rem 1rem 0.25rem;
  padding: 0.75rem 1rem;
  margin-right: 20%;              /* Left-aligned */
  border-left: 4px solid #10b981; /* Emerald accent */
  max-width: 80%;
}
```

### Input Field
```css
.chat-input {
  background-color: #ffffff;
  color: #2C2C2C;
  border: 2px solid #e5e7eb;
  border-radius: 1rem;
  padding: 0.75rem 1rem;
  font-size: 1rem;
  width: 100%;
}

.chat-input:focus {
  outline: none;
  border-color: #10b981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
}
```

---

## 📦 Component Library

### Content Cards

#### Standard Card
```css
.content-card {
  background-color: #ffffff;
  color: #2C2C2C;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 2rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;
}

.content-card:hover {
  box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}
```

#### Feature Box
For highlighted content with emerald accent.

```css
.feature-box {
  background-color: #6b7280;
  color: #ffffff;
  border: 2px solid #10b981;
  border-radius: 0.75rem;
  padding: 2rem;
  text-align: center;
}
```

#### Quote Box
For testimonials and highlighted quotes.

```css
.quote-box {
  background-color: #f9fafb;
  color: #2C2C2C;
  border-left: 4px solid #10b981;
  border-radius: 0.5rem;
  padding: 1.5rem;
  font-size: 1.125rem;
  line-height: 1.6;
}
```

### Forms

#### Input Fields
```css
.form-input {
  background-color: #ffffff;
  color: #2C2C2C;
  border: 2px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 0.75rem 1rem;
  font-size: 1rem;
  width: 100%;
}

.form-input:focus {
  outline: none;
  border-color: #10b981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
}

.form-input:invalid {
  border-color: #ef4444;
}
```

#### Form Labels
```css
.form-label {
  color: #2C2C2C;
  font-weight: 500;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
  display: block;
}
```

---

## 🎨 Logo & Branding

### Logo Container
```css
.logo-container {
  width: 40px;
  height: 40px;
  border-radius: 0.5rem;
  background-color: #ffffff;
  border: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.logo-container::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 4px;
  background-color: #10b981;
}
```

### Logo Sizes

| Size | Dimensions | Usage |
|------|------------|-------|
| **xs** | 24x24px | Small icons, nav items |
| **sm** | 32x32px | Standard navigation |
| **md** | 40x40px | Default size |
| **lg** | 48x48px | Headers, prominent areas |
| **xl** | 64x64px | Landing pages, heroes |

---

## 📱 Layout Guidelines

### Page Structure
```css
.page-container {
  background-color: #f9fafb;      /* Light gray page background */
  min-height: 100vh;
  padding: 1rem;
}

.content-section {
  background-color: #ffffff;      /* White content areas */
  border-radius: 0.75rem;
  border: 1px solid #e5e7eb;
  margin: 1rem 0;
  padding: 2rem;
}
```

### Grid System
```css
.grid-container {
  display: grid;
  gap: 2rem;
  padding: 2rem;
}

.grid-2-col {
  grid-template-columns: repeat(2, 1fr);
}

.grid-3-col {
  grid-template-columns: repeat(3, 1fr);
}

@media (max-width: 768px) {
  .grid-2-col,
  .grid-3-col {
    grid-template-columns: 1fr;
  }
}
```

### Spacing Scale

| Token | Value | Usage |
|-------|-------|-------|
| `xs` | 0.5rem (8px) | Small gaps |
| `sm` | 1rem (16px) | Standard spacing |
| `md` | 1.5rem (24px) | Section spacing |
| `lg` | 2rem (32px) | Large sections |
| `xl` | 3rem (48px) | Page sections |

---

## 📐 Typography

### Text Hierarchy

```css
.heading-primary {
  color: #2C2C2C;
  font-size: 2.5rem;
  font-weight: 700;
  line-height: 1.2;
  margin-bottom: 1rem;
}

.heading-secondary {
  color: #2C2C2C;
  font-size: 2rem;
  font-weight: 600;
  line-height: 1.3;
  margin-bottom: 0.75rem;
}

.heading-tertiary {
  color: #2C2C2C;
  font-size: 1.5rem;
  font-weight: 600;
  line-height: 1.4;
  margin-bottom: 0.5rem;
}

.body-text {
  color: #2C2C2C;
  font-size: 1rem;
  line-height: 1.6;
  margin-bottom: 1rem;
}

.body-secondary {
  color: #6E6E6E;
  font-size: 0.875rem;
  line-height: 1.5;
}

.caption {
  color: #9ca3af;
  font-size: 0.75rem;
  line-height: 1.4;
}
```

### Font Weights

| Weight | Value | Usage |
|--------|-------|-------|
| **Light** | 300 | Large headings |
| **Regular** | 400 | Body text |
| **Medium** | 500 | Labels, buttons |
| **Semibold** | 600 | Subheadings |
| **Bold** | 700 | Main headings |

---

## 🎯 State Management

### Loading States
```css
.loading-spinner {
  border: 2px solid #e5e7eb;
  border-top: 2px solid #10b981;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
```

### Success States
```css
.success-message {
  background-color: #d1fae5;
  color: #065f46;
  border: 1px solid #10b981;
  border-radius: 0.5rem;
  padding: 1rem;
}
```

### Error States
```css
.error-message {
  background-color: #fee2e2;
  color: #991b1b;
  border: 1px solid #ef4444;
  border-radius: 0.5rem;
  padding: 1rem;
}
```

---

## 🔧 Implementation Guide

### CSS Variables Setup
```css
:root {
  /* Text Colors */
  --text-primary: #2C2C2C;
  --text-secondary: #6E6E6E;
  --text-muted: #9ca3af;
  --text-inverse: #ffffff;
  
  /* UI Colors */
  --primary: #6b7280;
  --primary-hover: #4b5563;
  --accent: #10b981;
  --accent-hover: #059669;
  
  /* Backgrounds */
  --bg-primary: #ffffff;
  --bg-secondary: #f9fafb;
  --bg-tertiary: #f3f4f6;
  
  /* Borders */
  --border-light: #e5e7eb;
  --border-medium: #d1d5db;
  
  /* States */
  --success: #10b981;
  --warning: #f59e0b;
  --error: #ef4444;
  --info: #3b82f6;
  
  /* Shadows */
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
}
```

### Usage Examples

#### React Component
```jsx
import React from 'react';
import './Button.css';

const Button = ({ variant = 'primary', children, ...props }) => {
  return (
    <button className={`btn btn-${variant}`} {...props}>
      {children}
    </button>
  );
};

export default Button;
```

#### CSS Module
```css
.btn {
  border-radius: 0.5rem;
  padding: 0.75rem 1.5rem;
  font-weight: 500;
  transition: all 0.2s ease;
  cursor: pointer;
  border: 2px solid;
}

.btn-primary {
  background-color: var(--primary);
  color: var(--text-primary);
  border-color: var(--accent);
}

.btn-primary:hover {
  background-color: var(--primary-hover);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}
```

---

## 📱 Responsive Design

### Breakpoints

| Breakpoint | Min Width | Usage |
|------------|-----------|-------|
| **Mobile** | 320px | Small screens |
| **Tablet** | 768px | Medium screens |
| **Desktop** | 1024px | Large screens |
| **Wide** | 1440px | Extra large screens |

### Mobile-First Approach
```css
/* Mobile first */
.container {
  padding: 1rem;
}

/* Tablet and up */
@media (min-width: 768px) {
  .container {
    padding: 2rem;
    max-width: 768px;
    margin: 0 auto;
  }
}

/* Desktop and up */
@media (min-width: 1024px) {
  .container {
    max-width: 1024px;
    padding: 3rem;
  }
}
```

---

## ✅ Best Practices

### Do's ✅
- Use emerald green (`#10b981`) for all accent colors
- Maintain consistent spacing using the defined scale
- Use dark charcoal (`#2C2C2C`) for primary text
- Apply subtle shadows and hover effects
- Keep borders light and minimal
- Use white backgrounds for content areas

### Don'ts ❌
- Don't use orange or amber colors (replaced with emerald)
- Don't use bright or neon colors
- Don't mix different accent colors
- Don't use heavy shadows or effects
- Don't ignore hover states
- Don't use pure black for text

### Accessibility
- Maintain minimum 4.5:1 contrast ratio for text
- Use emerald focus rings for keyboard navigation
- Provide adequate touch targets (44px minimum)
- Test with screen readers
- Support reduced motion preferences

---

## 🚀 Quick Start Checklist

1. **Setup CSS Variables** - Copy the root variables to your main CSS file
2. **Implement Button System** - Use the defined button classes
3. **Apply Typography** - Use the text hierarchy classes
4. **Add Component Styles** - Implement card and form components
5. **Test Responsiveness** - Ensure mobile-first approach
6. **Validate Accessibility** - Check contrast and focus states

---

## 📞 Support

For questions about the design system or implementation help, please refer to:
- **Design Guidelines**: This documentation
- **Code Examples**: Check component files in `/components`
- **Color Reference**: See `/styles/colors.css`

---

*Last updated: August 8, 2025*
*Version: 2.0.0*
