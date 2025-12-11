interface CardProps {
    children: React.ReactNode;
    className?: string;
    hover?: boolean;
    padding?: 'none' | 'sm' | 'md' | 'lg';
}

export default function Card({
    children,
    className = '',
    hover = false,
    padding = 'md'
}: CardProps) {
    const paddingStyles = {
        none: '',
        sm: 'p-4',
        md: 'p-6',
        lg: 'p-8',
    };

    const hoverStyles = hover ? 'hover:shadow-lg hover:-translate-y-1 transition-all' : '';

    return (
        <div className={`bg-neutral-100 rounded-2xl border border-neutral-200 ${paddingStyles[padding]} ${hoverStyles} ${className}`}>
            {children}
        </div>
    );
}
