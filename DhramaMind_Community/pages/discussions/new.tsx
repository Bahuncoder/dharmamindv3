/**
 * DharmaMind Community - New Discussion Page
 */

import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { Layout } from '../../components/Layout';
import Link from 'next/link';

const categories = [
  'Mindfulness',
  'Leadership',
  'Philosophy',
  'Technology',
  'Wellness',
  'Career',
  'Community',
  'Other',
];

const NewDiscussionPage: React.FC = () => {
  const router = useRouter();
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [category, setCategory] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // In production, this would call an API
    console.log('New discussion:', { title, content, category });
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Redirect to discussions page
    router.push('/discussions');
  };

  const isValid = title.trim() && content.trim() && category;

  return (
    <Layout title="Start a Discussion" description="Create a new discussion in DharmaMind Community">
      <div className="max-w-3xl mx-auto px-6 py-12">
        {/* Breadcrumb */}
        <nav className="mb-8">
          <Link href="/discussions" className="text-gold-600 hover:text-gold-700 transition-colors">
            ← Back to Discussions
          </Link>
        </nav>

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-neutral-900 mb-2">Start a New Discussion</h1>
          <p className="text-neutral-600">
            Share your thoughts, ask questions, or start a meaningful conversation with the community.
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-neutral-200 p-8">
          {/* Category Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-neutral-700 mb-3">
              Category
            </label>
            <div className="flex flex-wrap gap-2">
              {categories.map((cat) => (
                <button
                  key={cat}
                  type="button"
                  onClick={() => setCategory(cat)}
                  className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                    category === cat
                      ? 'bg-gold-600 text-white'
                      : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>

          {/* Title */}
          <div className="mb-6">
            <label htmlFor="title" className="block text-sm font-medium text-neutral-700 mb-2">
              Title
            </label>
            <input
              id="title"
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="What's your discussion about?"
              className="w-full px-4 py-3 border border-neutral-200 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
              maxLength={200}
            />
            <p className="text-sm text-neutral-400 mt-1">{title.length}/200 characters</p>
          </div>

          {/* Content */}
          <div className="mb-8">
            <label htmlFor="content" className="block text-sm font-medium text-neutral-700 mb-2">
              Content
            </label>
            <textarea
              id="content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Share your thoughts, experiences, or questions..."
              className="w-full px-4 py-3 border border-neutral-200 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent resize-none"
              rows={10}
            />
            <p className="text-sm text-neutral-400 mt-1">
              Tip: Use Markdown for formatting. You can add **bold**, *italic*, and lists.
            </p>
          </div>

          {/* Guidelines */}
          <div className="bg-gold-50 rounded-lg p-4 mb-8">
            <h3 className="font-medium text-gold-800 mb-2">Community Guidelines</h3>
            <ul className="text-sm text-gold-700 space-y-1">
              <li>• Be respectful and kind to all community members</li>
              <li>• Keep discussions relevant and constructive</li>
              <li>• No spam, self-promotion, or offensive content</li>
              <li>• Use clear, descriptive titles for your discussions</li>
            </ul>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={() => router.back()}
              className="px-6 py-3 text-neutral-600 hover:text-neutral-900 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!isValid || isSubmitting}
              className="px-8 py-3 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? 'Publishing...' : 'Publish Discussion'}
            </button>
          </div>
        </form>
      </div>
    </Layout>
  );
};

export default NewDiscussionPage;
