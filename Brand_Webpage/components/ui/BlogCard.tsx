import Link from 'next/link';
import { BlogPost } from '../../config/site.config';

interface BlogCardProps {
    post: BlogPost | (BlogPost & { image?: string });
    featured?: boolean;
}

export default function BlogCard({ post, featured = false }: BlogCardProps) {
    const formattedDate = new Date(post.date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
    });

    if (featured) {
        return (
            <Link href={`/blog/${post.slug}`} className="block group">
                <article className="grid md:grid-cols-2 gap-8 items-center">
                    {/* Image */}
                    <div className="aspect-video bg-gradient-to-br from-neutral-100 to-neutral-200 rounded-2xl overflow-hidden">
                        <div className="w-full h-full flex items-center justify-center">
                            <span className="text-6xl opacity-30">📝</span>
                        </div>
                    </div>

                    {/* Content */}
                    <div>
                        <span className="inline-block px-3 py-1 bg-gold-600 text-white text-xs rounded-full mb-4">
                            {post.category}
                        </span>
                        <h2 className="text-2xl md:text-3xl font-bold text-neutral-900 mb-3 group-hover:text-neutral-700 transition-colors">
                            {post.title}
                        </h2>
                        <p className="text-neutral-600 mb-4 line-clamp-2">
                            {post.excerpt}
                        </p>
                        <div className="flex items-center space-x-4 text-sm text-neutral-500">
                            <span>{post.author}</span>
                            <span>·</span>
                            <span>{formattedDate}</span>
                            <span>·</span>
                            <span>{post.readTime}</span>
                        </div>
                    </div>
                </article>
            </Link>
        );
    }

    return (
        <Link href={`/blog/${post.slug}`} className="block group">
            <article>
                {/* Image */}
                <div className="aspect-video bg-gradient-to-br from-neutral-100 to-neutral-200 rounded-xl overflow-hidden mb-4">
                    <div className="w-full h-full flex items-center justify-center">
                        <span className="text-4xl opacity-30">📝</span>
                    </div>
                </div>

                {/* Content */}
                <span className="inline-block px-2 py-1 bg-neutral-100 text-neutral-600 text-xs rounded-full mb-2">
                    {post.category}
                </span>
                <h3 className="text-lg font-semibold text-neutral-900 mb-2 group-hover:text-neutral-700 transition-colors line-clamp-2">
                    {post.title}
                </h3>
                <p className="text-neutral-600 text-sm mb-3 line-clamp-2">
                    {post.excerpt}
                </p>
                <div className="flex items-center space-x-3 text-xs text-neutral-500">
                    <span>{post.author}</span>
                    <span>·</span>
                    <span>{formattedDate}</span>
                </div>
            </article>
        </Link>
    );
}
