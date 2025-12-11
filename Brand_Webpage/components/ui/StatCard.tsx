interface StatCardProps {
    value: string;
    label: string;
    description?: string;
    dark?: boolean;
}

export default function StatCard({ value, label, description, dark = false }: StatCardProps) {
    return (
        <div className="text-center">
            <div className={`text-4xl md:text-5xl font-bold mb-2 ${dark ? 'text-gold-600' : 'text-neutral-900'}`}>
                {value}
            </div>
            <div className={`font-medium ${dark ? 'text-white' : 'text-neutral-900'}`}>
                {label}
            </div>
            {description && (
                <div className={`text-sm mt-1 ${dark ? 'text-white/60' : 'text-neutral-500'}`}>
                    {description}
                </div>
            )}
        </div>
    );
}
