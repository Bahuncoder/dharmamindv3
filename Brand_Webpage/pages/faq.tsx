import { Layout, Section, Button } from '../components';
import { siteConfig } from '../config/site.config';
import { useState } from 'react';

interface FAQItem {
    question: string;
    answer: string;
    category: string;
}

const faqs: FAQItem[] = [
    // General
    {
        category: "General",
        question: "What is DharmaMind?",
        answer: "DharmaMind is an AI-powered platform designed to provide spiritual guidance and support. We combine advanced language models with deep knowledge of wisdom traditions from around the world to help users on their spiritual journey."
    },
    {
        category: "General",
        question: "How is DharmaMind different from other AI chatbots?",
        answer: "Unlike general-purpose AI chatbots, DharmaMind is specifically trained on authentic spiritual texts, teachings, and wisdom traditions. Our AI understands the nuances of spiritual guidance while maintaining respect for all traditions. We also prioritize ethical AI practices and user privacy."
    },
    {
        category: "General",
        question: "What spiritual traditions does DharmaMind support?",
        answer: "DharmaMind has been trained on teachings from Buddhism, Hinduism, Yoga, Vedanta, Taoism, Zen, Christian mysticism, Sufism, and many other wisdom traditions. We continuously expand our knowledge base while ensuring accuracy and respect for each tradition."
    },
    // Products
    {
        category: "Products",
        question: "What is DharmaMind Chat?",
        answer: "DharmaMind Chat is our flagship conversational AI that provides personalized spiritual guidance, meditation support, and answers to questions about various wisdom traditions. It's designed to be a thoughtful companion on your spiritual journey."
    },
    {
        category: "Products",
        question: "What is DharmaMind API?",
        answer: "DharmaMind API allows developers and organizations to integrate our spiritual AI capabilities into their own applications. It's perfect for wellness apps, meditation platforms, educational tools, and more."
    },
    {
        category: "Products",
        question: "Is there a mobile app?",
        answer: "We're currently developing native mobile apps for iOS and Android. In the meantime, our web application is fully responsive and works great on mobile devices."
    },
    // Pricing
    {
        category: "Pricing",
        question: "Is DharmaMind free to use?",
        answer: "Yes! We offer a free tier that includes basic conversations and access to our core features. For unlimited access and advanced features, we offer Pro and Enterprise plans."
    },
    {
        category: "Pricing",
        question: "What's included in the Pro plan?",
        answer: "The Pro plan ($19/month) includes unlimited conversations, advanced spiritual guidance, priority support, personalized practice recommendations, and access to premium meditation content."
    },
    {
        category: "Pricing",
        question: "Do you offer enterprise solutions?",
        answer: "Yes, we offer custom enterprise solutions for organizations including private deployments, custom integrations, dedicated support, and volume licensing. Contact us for a custom quote."
    },
    // Privacy & Safety
    {
        category: "Privacy & Safety",
        question: "How do you protect my privacy?",
        answer: "We take privacy extremely seriously. Your conversations are encrypted, never sold to third parties, and you can delete your data at any time. We collect only what's necessary to provide the service and improve your experience."
    },
    {
        category: "Privacy & Safety",
        question: "Is my conversation data used to train the AI?",
        answer: "By default, your conversations are not used for training. You can opt-in to help improve our AI, but this is entirely optional and your data would be anonymized."
    },
    {
        category: "Privacy & Safety",
        question: "What safety measures are in place?",
        answer: "DharmaMind includes comprehensive safety systems to prevent harmful outputs. Our AI is designed to recognize when users may need professional help and will recommend appropriate resources. We also have strict content policies aligned with ethical spiritual guidance."
    },
    // Technical
    {
        category: "Technical",
        question: "What AI model powers DharmaMind?",
        answer: "DharmaMind is powered by DharmaLLM, our proprietary language model specifically fine-tuned for spiritual understanding. It's built on advanced transformer architecture and trained on carefully curated spiritual texts and teachings."
    },
    {
        category: "Technical",
        question: "Can DharmaMind remember our previous conversations?",
        answer: "Yes, DharmaMind can maintain context across conversations to provide more personalized guidance. You can also choose to start fresh at any time by clearing your conversation history."
    },
    {
        category: "Technical",
        question: "Is the API rate-limited?",
        answer: "Yes, API usage has rate limits that vary by plan. Free tier has more restrictive limits, while paid plans offer higher throughput. Enterprise customers can get custom rate limits based on their needs."
    },
];

function FAQAccordion({ item, isOpen, onClick }: { item: FAQItem; isOpen: boolean; onClick: () => void }) {
    return (
        <div className="border-b border-neutral-200">
            <button
                className="w-full py-6 flex items-center justify-between text-left hover:text-gold-600 transition-colors"
                onClick={onClick}
            >
                <span className="text-lg font-medium text-neutral-900 pr-8">{item.question}</span>
                <span className={`flex-shrink-0 w-6 h-6 rounded-full border-2 border-neutral-300 flex items-center justify-center transition-all ${isOpen ? 'bg-gold-600 border-gold-600 rotate-180' : ''}`}>
                    <svg
                        className={`w-3 h-3 transition-colors ${isOpen ? 'text-white' : 'text-neutral-400'}`}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </span>
            </button>
            <div className={`overflow-hidden transition-all duration-300 ${isOpen ? 'max-h-96 pb-6' : 'max-h-0'}`}>
                <p className="text-neutral-600 leading-relaxed pr-12">{item.answer}</p>
            </div>
        </div>
    );
}

export default function FAQ() {
    const [openIndex, setOpenIndex] = useState<number | null>(0);
    const [activeCategory, setActiveCategory] = useState<string>("All");

    const categories = ["All", ...Array.from(new Set(faqs.map(f => f.category)))];
    const filteredFaqs = activeCategory === "All" ? faqs : faqs.filter(f => f.category === activeCategory);

    return (
        <Layout
            title="FAQ"
            description="Frequently asked questions about DharmaMind, our products, pricing, and more."
        >
            {/* Hero Section */}
            <Section background="light" padding="xl">
                <div className="max-w-3xl mx-auto text-center">
                    <h1 className="text-4xl md:text-6xl font-bold text-neutral-900 mb-6">
                        Frequently Asked Questions
                    </h1>
                    <p className="text-xl text-neutral-600 leading-relaxed">
                        Find answers to common questions about {siteConfig.company.name},
                        our products, and how we can help you on your spiritual journey.
                    </p>
                </div>
            </Section>

            {/* Category Filters */}
            <Section padding="sm">
                <div className="flex flex-wrap justify-center gap-3">
                    {categories.map((category) => (
                        <button
                            key={category}
                            onClick={() => {
                                setActiveCategory(category);
                                setOpenIndex(0);
                            }}
                            className={`px-4 py-2 text-sm rounded-full transition-colors ${activeCategory === category
                                ? 'bg-gold-600 text-white'
                                : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
                                }`}
                        >
                            {category}
                        </button>
                    ))}
                </div>
            </Section>

            {/* FAQ List */}
            <Section>
                <div className="max-w-3xl mx-auto">
                    {filteredFaqs.map((faq, index) => (
                        <FAQAccordion
                            key={index}
                            item={faq}
                            isOpen={openIndex === index}
                            onClick={() => setOpenIndex(openIndex === index ? null : index)}
                        />
                    ))}
                </div>
            </Section>

            {/* Still Have Questions */}
            <Section background="light">
                <div className="max-w-2xl mx-auto text-center">
                    <h2 className="text-2xl font-bold text-neutral-900 mb-4">
                        Still have questions?
                    </h2>
                    <p className="text-neutral-600 mb-6">
                        Can't find the answer you're looking for? Our team is here to help.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button href="/contact" variant="primary">
                            Contact Support
                        </Button>
                        <Button href="/docs" variant="outline">
                            View Documentation
                        </Button>
                    </div>
                </div>
            </Section>

            {/* Quick Links */}
            <Section>
                <div className="grid md:grid-cols-3 gap-6">
                    <div className="p-6 rounded-xl border border-neutral-200 text-center">
                        <span className="text-3xl mb-4 block">📚</span>
                        <h3 className="font-semibold text-neutral-900 mb-2">Documentation</h3>
                        <p className="text-neutral-600 text-sm mb-4">
                            Detailed guides and API references
                        </p>
                        <Button href="/docs" variant="ghost" size="sm">
                            Browse Docs →
                        </Button>
                    </div>
                    <div className="p-6 rounded-xl border border-neutral-200 text-center">
                        <span className="text-3xl mb-4 block">💬</span>
                        <h3 className="font-semibold text-neutral-900 mb-2">Community</h3>
                        <p className="text-neutral-600 text-sm mb-4">
                            Join our Discord community
                        </p>
                        <Button href={siteConfig.social.discord} variant="ghost" size="sm" external>
                            Join Discord →
                        </Button>
                    </div>
                    <div className="p-6 rounded-xl border border-neutral-200 text-center">
                        <span className="text-3xl mb-4 block">📧</span>
                        <h3 className="font-semibold text-neutral-900 mb-2">Email Support</h3>
                        <p className="text-neutral-600 text-sm mb-4">
                            Get help from our team
                        </p>
                        <Button href={`mailto:${siteConfig.company.supportEmail}`} variant="ghost" size="sm">
                            Email Us →
                        </Button>
                    </div>
                </div>
            </Section>
        </Layout>
    );
}
