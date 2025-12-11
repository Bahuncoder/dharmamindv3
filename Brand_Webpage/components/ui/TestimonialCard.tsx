import { Testimonial } from '../../config/site.config';

interface TestimonialCardProps {
    testimonial: Testimonial;
}

export default function TestimonialCard({ testimonial }: TestimonialCardProps) {
    return (
        <div className="bg-neutral-100 rounded-2xl border border-neutral-200 p-6 h-full flex flex-col">
            {/* Stars */}
            <div className="flex mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                    <span key={i} className="text-gold-600 text-lg">★</span>
                ))}
            </div>

            {/* Quote */}
            <blockquote className="text-neutral-700 flex-grow mb-6">
                &quot;{testimonial.quote}&quot;
            </blockquote>

            {/* Author */}
            <div className="flex items-center">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gold-200/20 to-gold-300/30 flex items-center justify-center mr-3">
                    <span className="text-neutral-900 text-sm font-medium">
                        {testimonial.author.charAt(0)}
                    </span>
                </div>
                <div>
                    <div className="font-medium text-neutral-900">{testimonial.author}</div>
                    <div className="text-sm text-neutral-500">
                        {testimonial.role} · {testimonial.location}
                    </div>
                </div>
            </div>
        </div>
    );
}
