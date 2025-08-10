import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../../components/Logo';

interface BlogPost {
  id?: string;
  title: string;
  content: string;
  excerpt: string;
  category: string;
  tags: string[];
  status: 'draft' | 'published' | 'scheduled';
  featuredImage: string;
  author: string;
  publishDate: string;
  seoTitle: string;
  seoDescription: string;
}

const AdminEditor: React.FC = () => {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('content');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'saved' | 'saving' | 'error' | null>(null);
  
  const [post, setPost] = useState<BlogPost>({
    title: '',
    content: '',
    excerpt: '',
    category: 'meditation',
    tags: [],
    status: 'draft',
    featuredImage: '',
    author: 'Admin',
    publishDate: new Date().toISOString().split('T')[0],
    seoTitle: '',
    seoDescription: ''
  });

  const [newTag, setNewTag] = useState('');

  const categories = [
    { value: 'meditation', label: 'Meditation' },
    { value: 'philosophy', label: 'Philosophy' },
    { value: 'community', label: 'Community' },
    { value: 'practices', label: 'Practices' },
    { value: 'ai-wisdom', label: 'AI Wisdom' },
    { value: 'dharma', label: 'Dharma' }
  ];

  useEffect(() => {
    // Check authentication
    const isAuthenticated = localStorage.getItem('dharma_admin');
    if (!isAuthenticated) {
      router.push('/admin/login');
      return;
    }
  }, [router]);

  const handleInputChange = (field: keyof BlogPost, value: any) => {
    setPost(prev => ({
      ...prev,
      [field]: value
    }));
    setSaveStatus(null);
  };

  const addTag = () => {
    if (newTag.trim() && !post.tags.includes(newTag.trim())) {
      setPost(prev => ({
        ...prev,
        tags: [...prev.tags, newTag.trim()]
      }));
      setNewTag('');
    }
  };

  const removeTag = (tagToRemove: string) => {
    setPost(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }));
  };

  const handleSave = async (status: 'draft' | 'published') => {
    setIsSaving(true);
    setSaveStatus('saving');
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const updatedPost = {
        ...post,
        status,
        id: post.id || Date.now().toString(),
        publishDate: status === 'published' ? new Date().toISOString().split('T')[0] : post.publishDate
      };
      
      // In a real app, this would be an API call
      console.log('Saving post:', updatedPost);
      
      setPost(updatedPost);
      setSaveStatus('saved');
      
      // Auto-clear save status after 3 seconds
      setTimeout(() => setSaveStatus(null), 3000);
      
    } catch (error) {
      setSaveStatus('error');
    } finally {
      setIsSaving(false);
    }
  };

  const getSaveStatusMessage = () => {
    switch (saveStatus) {
      case 'saving': return { text: 'Saving...', color: 'text-warning' };
      case 'saved': return { text: 'Saved successfully!', color: 'text-success' };
      case 'error': return { text: 'Save failed. Please try again.', color: 'text-error' };
      default: return null;
    }
  };

  return (
    <>
      <Head>
        <title>Content Editor - DharmaMind Admin</title>
        <meta name="description" content="DharmaMind Content Editor - Create and manage blog posts" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/logo.jpeg" />
      </Head>
      
      <div className="min-h-screen bg-secondary-bg">
        {/* Top Navigation Bar */}
        <nav className="bg-primary-bg shadow-lg border-b-2 border-border-primary">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center">
                <Logo size="sm" showText={true} />
                <div className="ml-4 hidden md:block">
                  <span className="text-lg font-black text-primary">Content Editor</span>
                </div>
              </div>
              
              <div className="flex items-center space-x-4">
                {getSaveStatusMessage() && (
                  <span className={`text-sm font-semibold ${getSaveStatusMessage()!.color}`}>
                    {getSaveStatusMessage()!.text}
                  </span>
                )}
                <button 
                  onClick={() => router.push('/admin/dashboard')}
                  className="btn-outline px-4 py-2 rounded-lg font-bold text-sm hover:bg-primary hover:text-white transition-all"
                >
                  ← Back to Dashboard
                </button>
              </div>
            </div>
          </div>
        </nav>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-black text-primary mb-2 tracking-tight">
              {post.id ? 'Edit Post' : 'Create New Post'}
            </h1>
            <p className="text-xl text-secondary font-semibold">
              Write and publish engaging content for your community
            </p>
          </div>

          {/* Action Buttons */}
          <div className="mb-8 flex flex-wrap gap-4">
            <button 
              onClick={() => handleSave('draft')}
              disabled={isSaving}
              className="btn-outline px-6 py-3 rounded-lg font-bold hover:bg-primary hover:text-white transition-all disabled:opacity-50"
            >
              {isSaving && post.status === 'draft' ? 'Saving...' : 'Save as Draft'}
            </button>
            <button 
              onClick={() => handleSave('published')}
              disabled={isSaving || !post.title || !post.content}
              className="btn-primary px-6 py-3 rounded-lg font-bold hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              {isSaving && post.status === 'published' ? 'Publishing...' : 'Publish Post'}
            </button>
            <button className="btn-outline px-6 py-3 rounded-lg font-bold hover:bg-secondary hover:text-white transition-all">
              Preview
            </button>
          </div>

          {/* Tab Navigation */}
          <div className="mb-8">
            <div className="border-b border-medium">
              <nav className="-mb-px flex space-x-8">
                {[
                  { id: 'content', label: 'Content', icon: '📝' },
                  { id: 'settings', label: 'Post Settings', icon: '⚙️' },
                  { id: 'seo', label: 'SEO', icon: '🔍' }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`py-2 px-1 border-b-2 font-bold text-sm transition-colors ${
                      activeTab === tab.id
                        ? 'border-primary text-primary'
                        : 'border-transparent text-secondary hover:text-primary hover:border-medium'
                    }`}
                  >
                    <span className="mr-2">{tab.icon}</span>
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Editor Content */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main Content Area */}
            <div className="lg:col-span-2">
              {activeTab === 'content' && (
                <div className="space-y-6">
                  {/* Title */}
                  <div className="card-primary p-6">
                    <label className="block text-sm font-bold text-primary mb-2">
                      Post Title *
                    </label>
                    <input
                      type="text"
                      value={post.title}
                      onChange={(e) => handleInputChange('title', e.target.value)}
                      placeholder="Enter your post title..."
                      className="w-full px-4 py-3 text-xl font-bold border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors"
                    />
                  </div>

                  {/* Content Editor */}
                  <div className="card-primary p-6">
                    <label className="block text-sm font-bold text-primary mb-2">
                      Post Content *
                    </label>
                    <textarea
                      value={post.content}
                      onChange={(e) => handleInputChange('content', e.target.value)}
                      placeholder="Write your post content here... You can use Markdown formatting."
                      rows={20}
                      className="w-full px-4 py-3 border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors resize-none font-medium"
                    />
                    <div className="mt-2 text-xs text-muted font-semibold">
                      Tip: You can use Markdown formatting for rich text content
                    </div>
                  </div>

                  {/* Excerpt */}
                  <div className="card-primary p-6">
                    <label className="block text-sm font-bold text-primary mb-2">
                      Post Excerpt
                    </label>
                    <textarea
                      value={post.excerpt}
                      onChange={(e) => handleInputChange('excerpt', e.target.value)}
                      placeholder="Brief description of your post (will be used in post previews)..."
                      rows={3}
                      className="w-full px-4 py-3 border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors resize-none font-medium"
                    />
                  </div>
                </div>
              )}

              {activeTab === 'settings' && (
                <div className="space-y-6">
                  {/* Category and Tags */}
                  <div className="card-primary p-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-bold text-primary mb-2">
                          Category
                        </label>
                        <select
                          value={post.category}
                          onChange={(e) => handleInputChange('category', e.target.value)}
                          className="w-full px-4 py-3 border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors font-medium"
                        >
                          {categories.map(cat => (
                            <option key={cat.value} value={cat.value}>{cat.label}</option>
                          ))}
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-bold text-primary mb-2">
                          Author
                        </label>
                        <input
                          type="text"
                          value={post.author}
                          onChange={(e) => handleInputChange('author', e.target.value)}
                          className="w-full px-4 py-3 border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors font-medium"
                        />
                      </div>
                    </div>

                    <div className="mt-6">
                      <label className="block text-sm font-bold text-primary mb-2">
                        Tags
                      </label>
                      <div className="flex flex-wrap gap-2 mb-3">
                        {post.tags.map(tag => (
                          <span 
                            key={tag}
                            className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold bg-primary-gradient-light text-primary"
                          >
                            {tag}
                            <button
                              onClick={() => removeTag(tag)}
                              className="ml-2 text-primary hover:text-primary-hover"
                            >
                              ×
                            </button>
                          </span>
                        ))}
                      </div>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={newTag}
                          onChange={(e) => setNewTag(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && addTag()}
                          placeholder="Add a tag..."
                          className="flex-1 px-4 py-2 border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors font-medium"
                        />
                        <button 
                          onClick={addTag}
                          className="btn-primary px-4 py-2 rounded-lg font-bold"
                        >
                          Add
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Publishing Options */}
                  <div className="card-primary p-6">
                    <h4 className="text-lg font-black text-primary mb-4">Publishing Options</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-bold text-primary mb-2">
                          Status
                        </label>
                        <select
                          value={post.status}
                          onChange={(e) => handleInputChange('status', e.target.value as 'draft' | 'published' | 'scheduled')}
                          className="w-full px-4 py-3 border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors font-medium"
                        >
                          <option value="draft">Draft</option>
                          <option value="published">Published</option>
                          <option value="scheduled">Scheduled</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-bold text-primary mb-2">
                          Publish Date
                        </label>
                        <input
                          type="date"
                          value={post.publishDate}
                          onChange={(e) => handleInputChange('publishDate', e.target.value)}
                          className="w-full px-4 py-3 border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors font-medium"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'seo' && (
                <div className="space-y-6">
                  <div className="card-primary p-6">
                    <h4 className="text-lg font-black text-primary mb-4">SEO Settings</h4>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-bold text-primary mb-2">
                          SEO Title
                        </label>
                        <input
                          type="text"
                          value={post.seoTitle}
                          onChange={(e) => handleInputChange('seoTitle', e.target.value)}
                          placeholder="SEO optimized title (leave empty to use post title)"
                          className="w-full px-4 py-3 border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors font-medium"
                        />
                        <div className="mt-1 text-xs text-muted font-semibold">
                          Recommended: 50-60 characters
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-bold text-primary mb-2">
                          Meta Description
                        </label>
                        <textarea
                          value={post.seoDescription}
                          onChange={(e) => handleInputChange('seoDescription', e.target.value)}
                          placeholder="Brief description for search engines..."
                          rows={3}
                          className="w-full px-4 py-3 border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors resize-none font-medium"
                        />
                        <div className="mt-1 text-xs text-muted font-semibold">
                          Recommended: 150-160 characters
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Sidebar */}
            <div className="lg:col-span-1">
              <div className="space-y-6">
                {/* Post Status */}
                <div className="card-primary p-6">
                  <h4 className="text-lg font-black text-primary mb-4">Post Status</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm font-semibold text-secondary">Status:</span>
                      <span className={`text-sm font-bold px-2 py-1 rounded ${
                        post.status === 'published' ? 'bg-success-light text-success' :
                        post.status === 'draft' ? 'bg-warning-light text-warning' :
                        'bg-info-light text-info'
                      }`}>
                        {post.status.charAt(0).toUpperCase() + post.status.slice(1)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-semibold text-secondary">Word Count:</span>
                      <span className="text-sm font-bold text-primary">
                        {post.content.split(' ').filter(word => word.length > 0).length}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-semibold text-secondary">Reading Time:</span>
                      <span className="text-sm font-bold text-primary">
                        {Math.ceil(post.content.split(' ').filter(word => word.length > 0).length / 200)} min
                      </span>
                    </div>
                  </div>
                </div>

                {/* Featured Image */}
                <div className="card-primary p-6">
                  <h4 className="text-lg font-black text-primary mb-4">Featured Image</h4>
                  <div className="space-y-3">
                    <input
                      type="url"
                      value={post.featuredImage}
                      onChange={(e) => handleInputChange('featuredImage', e.target.value)}
                      placeholder="Enter image URL..."
                      className="w-full px-4 py-3 border-2 border-light focus:border-primary focus:outline-none rounded-lg transition-colors font-medium"
                    />
                    {post.featuredImage && (
                      <div className="border-2 border-light rounded-lg overflow-hidden">
                        <img 
                          src={post.featuredImage} 
                          alt="Featured" 
                          className="w-full h-32 object-cover"
                          onError={(e) => {
                            (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjEyMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjRjdGN0Y3Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OTk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=';
                          }}
                        />
                      </div>
                    )}
                  </div>
                </div>

                {/* Quick Tips */}
                <div className="card-primary p-6">
                  <h4 className="text-lg font-black text-primary mb-4">Writing Tips</h4>
                  <div className="space-y-2 text-sm text-secondary font-medium">
                    <div className="flex items-start space-x-2">
                      <span>💡</span>
                      <span>Use clear, engaging headings to structure your content</span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <span>🎯</span>
                      <span>Keep paragraphs short and easy to read</span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <span>🔍</span>
                      <span>Include relevant keywords naturally</span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <span>📝</span>
                      <span>End with a clear call-to-action</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default AdminEditor;
