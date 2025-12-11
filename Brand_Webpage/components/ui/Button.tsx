import Link from 'next/link';

interface ButtonProps {
    children: React.ReactNode;
    href?: string;
    variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
    size?: 'sm' | 'md' | 'lg';
    className?: string;
    onClick?: () => void;
    external?: boolean;
}

export default function Button({
    children,
    href,
    variant = 'primary',
    size = 'md',
    className = '',
    onClick,
    external = false,
}: ButtonProps) {
    const baseStyles = 'inline-flex items-center justify-center font-medium transition-all rounded-full';

    const variantStyles = {
        primary: 'bg-gold-600 text-white hover:bg-gold-700',
        secondary: 'bg-neutral-200 text-neutral-900 hover:bg-neutral-300',
        outline: 'border-2 border-gold-600 text-gold-700 hover:bg-gold-600 hover:text-white',
        ghost: 'text-neutral-600 hover:text-gold-600 hover:bg-neutral-100',
    };

    const sizeStyles = {
        sm: 'px-4 py-2 text-sm',
        md: 'px-6 py-3 text-sm',
        lg: 'px-8 py-4 text-base',
    };

    const styles = `${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${className}`;

    if (href) {
        if (external) {
            return (
                <a href={href} className={styles} target="_blank" rel="noopener noreferrer">
                    {children}
                </a>
            );
        }
        return (
            <Link href={href} className={styles}>
                {children}
            </Link>
        );
    }

    return (
        <button onClick={onClick} className={styles}>
            {children}
        </button>
    );
}
