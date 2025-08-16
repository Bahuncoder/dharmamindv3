import { GetServerSideProps } from 'next';

function generateSiteMap(pages: string[]) {
  const baseUrl = process.env.NEXT_PUBLIC_SITE_URL || 'https://dharmamind.vercel.app';
  
  return `<?xml version="1.0" encoding="UTF-8"?>
   <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
     ${pages
       .map((page) => {
         const route = page === 'index' ? '' : page;
         const lastMod = new Date().toISOString();
         
         // Priority based on page importance
         let priority = '0.5';
         if (page === 'index') priority = '1.0';
         else if (['about', 'pricing', 'features'].includes(page)) priority = '0.8';
         else if (['contact', 'help', 'enterprise'].includes(page)) priority = '0.7';
         
         // Change frequency based on page type
         let changefreq = 'monthly';
         if (page === 'index') changefreq = 'weekly';
         else if (['pricing', 'features'].includes(page)) changefreq = 'monthly';
         else if (['about', 'contact'].includes(page)) changefreq = 'yearly';
         
         return `
           <url>
               <loc>${baseUrl}/${route}</loc>
               <lastmod>${lastMod}</lastmod>
               <changefreq>${changefreq}</changefreq>
               <priority>${priority}</priority>
           </url>
         `;
       })
       .join('')}
   </urlset>
 `;
}

function SiteMap() {
  // getServerSideProps will do the heavy lifting
}

export const getServerSideProps: GetServerSideProps = async ({ res }) => {
  // Static pages to include in sitemap with strategic keywords
  const staticPages = [
    'index', // Primary: finding life purpose, AI life coach
    'about', // Focus: ethical AI companion, personal transformation  
    'pricing', // Focus: personal growth AI, conscious living
    'features', // Focus: AI for self-reflection, meaningful life AI
    'contact', // Focus: DharmaMind community
    'help', // Focus: inner peace guidance, stress management AI
    'enterprise', // Focus: workplace well-being, employee engagement
    'auth', // Focus: AI life coach signup
    'signin', // Focus: conscious living guide access
    'docs', // Focus: wisdom AI chatbot documentation
    'api-docs', // Technical: conversational philosophy AI
    'technical-specs', // Focus: ethical AI features
    'security', // Focus: ethical AI companion safety
    'privacy', // Focus: AI for personal ethics
    'terms', // Legal: purpose-driven AI guide
    'status', // System: DharmaMind AI status
    'support', // Focus: AI for personal growth support
    'demo-request', // Focus: workplace purpose alignment demo
    'feature-requests' // Focus: meaningful life AI features
  ];

  // Generate the XML sitemap
  const sitemap = generateSiteMap(staticPages);

  res.setHeader('Content-Type', 'text/xml');
  // Cache for 24 hours
  res.setHeader('Cache-Control', 'public, s-maxage=86400, stale-while-revalidate');
  res.write(sitemap);
  res.end();

  return {
    props: {},
  };
};

export default SiteMap;
