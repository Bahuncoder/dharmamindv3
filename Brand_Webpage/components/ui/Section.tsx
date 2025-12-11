interface SectionProps {
    children: React.ReactNode;
    className?: string;
    background?: 'white' | 'light' | 'dark';
    padding?: 'sm' | 'md' | 'lg' | 'xl';
    id?: string;
}

export default function Section({
    children,
    className = '',
    background = 'white',
    padding = 'lg',
    id
}: SectionProps) {
    const backgroundStyles = {
        white: 'bg-neutral-50',           // Medium light gray instead of white
        light: 'bg-neutral-100',          // Medium gray
        dark: 'bg-neutral-200 text-neutral-900',  // Darker medium gray
    };

    const paddingStyles = {
        sm: 'py-12',
        md: 'py-16',
        lg: 'py-24',
        xl: 'py-32',
    };

    return (
        <section
            id={id}
            className={`${backgroundStyles[background]} ${paddingStyles[padding]} ${className}`}
        >
            <div className="max-w-7xl mx-auto px-6">
                {children}
            </div>
        </section>
    );
}
