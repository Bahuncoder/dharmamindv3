import { Layout, Section, BlogCard, Button } from '../components';
import { siteConfig } from '../config/site.config';

export default function Blog() {
    const allPosts = [...siteConfig.blog.featured, ...siteConfig.blog.posts];
    const featuredPost = siteConfig.blog.featured[0];
    const recentPosts = [...siteConfig.blog.featured.slice(1), ...siteConfig.blog.posts];

    return (
        <Layout
            title="Blog"
            description="Insights, research, and updates from the DharmaMind team."
        >
            {/* Hero Section */}
            <Section background="light" padding="xl">
                <div className="max-w-3xl">
                    <h1 className="text-4xl md:text-6xl font-bold text-neutral-900 mb-6">
                        Blog
                    </h1>
                    <p className="text-xl text-neutral-600 leading-relaxed">
                        Insights on AI, spirituality, research, and building technology that
                        serves humanity's highest aspirations.
                    </p>
                </div>
            </Section>

            {/* Categories */}
            <Section padding="sm">
                <div className="flex flex-wrap gap-3">
                    <button className="px-4 py-2 bg-gold-600 text-white text-sm rounded-full">
                        All Posts
                    </button>
                    {siteConfig.blog.categories.map((category) => (
                        <button
                            key={category}
                            className="px-4 py-2 bg-neutral-100 text-neutral-700 text-sm rounded-full hover:bg-neutral-200 transition-colors"
                        >
                            {category}
                        </button>
                    ))}
                </div>
            </Section>

            {/* Featured Post */}
            <Section>
                <div className="mb-8">
                    <span className="text-sm font-medium text-neutral-500 uppercase tracking-wider">Featured</span>
                </div>
                <BlogCard post={featuredPost} featured />
            </Section>

            {/* Recent Posts */}
            <Section background="light">
                <div className="mb-12">
                    <h2 className="text-2xl font-bold text-neutral-900">Recent Posts</h2>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {recentPosts.map((post) => (
                        <BlogCard key={post.slug} post={post} />
                    ))}
                </div>

                {/* Load More */}
                <div className="text-center mt-12">
                    <Button variant="outline">
                        Load More Posts
                    </Button>
                </div>
            </Section>

            {/* Newsletter Signup */}
            <Section>
                <div className="bg-neutral-200 border border-neutral-300 rounded-3xl p-8 md:p-12 text-center">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                        Stay Updated
                    </h2>
                    <p className="text-lg text-neutral-600 mb-8 max-w-2xl mx-auto">
                        Subscribe to our newsletter for the latest research, product updates,
                        and insights on AI and spirituality.
                    </p>
                    <form className="max-w-md mx-auto flex flex-col sm:flex-row gap-3">
                        <input
                            type="email"
                            placeholder="Enter your email"
                            className="flex-grow px-4 py-3 rounded-full bg-neutral-100 border border-neutral-300 text-neutral-900 placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-gold-400"
                        />
                        <button
                            type="submit"
                            className="px-6 py-3 bg-gold-600 text-white rounded-full font-medium hover:bg-gold-700 transition-colors"
                        >
                            Subscribe
                        </button>
                    </form>
                    <p className="text-sm text-neutral-500 mt-4">
                        No spam, unsubscribe anytime.
                    </p>
                </div>
            </Section>

            {/* Topics Section */}
            <Section background="light">
                <div className="text-center mb-12">
                    <h2 className="text-2xl font-bold text-neutral-900 mb-4">Explore by Topic</h2>
                    <p className="text-neutral-600">Deep dives into what we're working on.</p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {[
                        {
                            category: "Engineering",
                            description: "Technical deep-dives into our infrastructure, models, and tools.",
                            icon: "⚙️",
                            count: allPosts.filter(p => p.category === "Engineering").length
                        },
                        {
                            category: "Research",
                            description: "Our latest findings in AI, NLP, and spiritual computing.",
                            icon: "🔬",
                            count: allPosts.filter(p => p.category === "Research").length
                        },
                        {
                            category: "Philosophy",
                            description: "Reflections on ethics, spirituality, and the future of AI.",
                            icon: "🕉️",
                            count: allPosts.filter(p => p.category === "Philosophy").length
                        },
                        {
                            category: "Company",
                            description: "Culture, values, and life at DharmaMind.",
                            icon: "🏢",
                            count: allPosts.filter(p => p.category === "Company").length
                        },
                        {
                            category: "Product Updates",
                            description: "New features, releases, and improvements.",
                            icon: "🚀",
                            count: allPosts.filter(p => p.category === "Product Updates").length
                        },
                    ].map((topic) => (
                        <div
                            key={topic.category}
                            className="bg-neutral-100 rounded-2xl border border-neutral-200 p-6 hover:shadow-md hover:border-neutral-300 transition-all cursor-pointer"
                        >
                            <span className="text-3xl mb-4 block">{topic.icon}</span>
                            <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                                {topic.category}
                            </h3>
                            <p className="text-neutral-600 text-sm mb-4">
                                {topic.description}
                            </p>
                            <span className="text-sm text-neutral-500">
                                {topic.count} {topic.count === 1 ? 'post' : 'posts'}
                            </span>
                        </div>
                    ))}
                </div>
            </Section>
        </Layout>
    );
}
