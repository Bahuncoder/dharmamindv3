#!/usr/bin/env python3
"""
WISDOM LIBRARY CHAPTER-BY-CHAPTER DOWNLOADER
=============================================

The book pages are just landing pages.
We need to download actual CHAPTERS from each book!

Target priority:
1. Mahabharata - ~100,000 verses
2. Ramayana - ~24,000 verses
3. Bhagavata Purana - ~18,000 verses
4. All other Puranas

Total potential: 200,000+ verses!
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path
from typing import List, Dict

class WisdomLibraryChapterDownloader:
    def __init__(self):
        self.base_url = "https://www.wisdomlib.org"
        self.output_dir = Path("data/authentic_sources/wisdom_library_chapters")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        
        # Priority books to download
        self.priority_books = [
            {
                'slug': 'mahabharata-sanskrit',
                'name': 'Mahabharata',
                'estimated_verses': 100000,
                'url_pattern': '/hinduism/book/mahabharata-sanskrit'
            },
            {
                'slug': 'the-ramayana-sanskrit',
                'name': 'Ramayana',
                'estimated_verses': 24000,
                'url_pattern': '/hinduism/book/the-ramayana-sanskrit'
            },
            {
                'slug': 'bhagavata-purana-sanskrit',
                'name': 'Bhagavata Purana',
                'estimated_verses': 18000,
                'url_pattern': '/hinduism/book/bhagavata-purana-sanskrit'
            },
        ]
    
    def download_page(self, url: str) -> str:
        """Download a page"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            return None
        except Exception as e:
            print(f"    Error downloading {url}: {e}")
            return None
    
    def find_chapter_links(self, html: str, book_url: str) -> List[Dict]:
        """Find all chapter/section links in book page"""
        soup = BeautifulSoup(html, 'html.parser')
        chapters = []
        
        # Method 1: Look for links with chapter/section patterns
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text().strip()
            
            # Look for chapter patterns
            if any(pattern in href.lower() for pattern in ['chapter', 'adhyaya', 'kanda', 'parva']):
                full_url = self.base_url + href if href.startswith('/') else href
                chapters.append({
                    'title': text,
                    'url': full_url,
                    'type': 'chapter'
                })
        
        # Method 2: Look for numbered links (1, 2, 3, etc.)
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text().strip()
            
            # Check if text is a number (chapter number)
            if text.isdigit() and book_url in href:
                full_url = self.base_url + href if href.startswith('/') else href
                chapters.append({
                    'title': f'Chapter {text}',
                    'url': full_url,
                    'type': 'numbered'
                })
        
        # Method 3: Table of contents patterns
        for section in soup.find_all(['div', 'ul'], class_=re.compile(r'toc|contents|chapters', re.I)):
            for link in section.find_all('a', href=True):
                href = link['href']
                text = link.get_text().strip()
                full_url = self.base_url + href if href.startswith('/') else href
                if full_url not in [c['url'] for c in chapters]:
                    chapters.append({
                        'title': text,
                        'url': full_url,
                        'type': 'toc'
                    })
        
        # Remove duplicates
        seen_urls = set()
        unique_chapters = []
        for chapter in chapters:
            if chapter['url'] not in seen_urls:
                seen_urls.add(chapter['url'])
                unique_chapters.append(chapter)
        
        return unique_chapters
    
    def extract_sanskrit_from_chapter(self, html: str) -> List[str]:
        """Extract Sanskrit verses from a chapter page"""
        soup = BeautifulSoup(html, 'html.parser')
        verses = []
        
        # Remove unwanted elements
        for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # Method 1: Find verse containers
        for tag in soup.find_all(['div', 'p', 'span', 'blockquote'], class_=re.compile(r'verse|shloka|sanskrit|devanagari', re.I)):
            text = tag.get_text().strip()
            if re.search(r'[\u0900-\u097F]', text):
                sanskrit = ' '.join(re.findall(r'[\u0900-\u097F\s।॥०-९]+', text))
                if len(sanskrit) > 50:
                    verses.append(sanskrit.strip())
        
        # Method 2: Any text with Devanagari
        for tag in soup.find_all(['div', 'p', 'pre', 'article']):
            text = tag.get_text().strip()
            if re.search(r'[\u0900-\u097F]', text):
                # Extract only Devanagari portions
                sanskrit_parts = re.findall(r'[\u0900-\u097F\s।॥०-९]+', text)
                for part in sanskrit_parts:
                    if len(part.strip()) > 50:
                        verses.append(part.strip())
        
        # Remove duplicates
        unique_verses = []
        seen = set()
        for v in verses:
            if v not in seen and len(v) > 50:
                seen.add(v)
                unique_verses.append(v)
        
        return unique_verses
    
    def download_book(self, book_info: Dict) -> Dict:
        """Download all chapters from a book"""
        print(f"\n{'='*70}")
        print(f"📚 {book_info['name']}")
        print(f"   Estimated: {book_info['estimated_verses']:,} verses")
        print(f"{'='*70}\n")
        
        book_url = self.base_url + book_info['url_pattern']
        print(f"🔍 Exploring: {book_url}")
        
        # Download book main page
        html = self.download_page(book_url)
        if not html:
            print(f"   ✗ Failed to load book page")
            return None
        
        # Find all chapters
        print(f"📖 Finding chapters...")
        chapters = self.find_chapter_links(html, book_info['url_pattern'])
        print(f"   Found {len(chapters)} potential chapters/sections")
        
        if not chapters:
            print(f"   ⚠ No chapters found! Book might have different structure.")
            # Try to extract from main page
            verses = self.extract_sanskrit_from_chapter(html)
            if verses:
                print(f"   ✓ Extracted {len(verses)} verses from main page")
                return {
                    'book': book_info['slug'],
                    'name': book_info['name'],
                    'chapters': [{
                        'title': 'Main Page',
                        'url': book_url,
                        'verses': verses,
                        'verse_count': len(verses)
                    }],
                    'total_verses': len(verses)
                }
            return None
        
        # Download each chapter
        book_data = {
            'book': book_info['slug'],
            'name': book_info['name'],
            'chapters': [],
            'total_verses': 0
        }
        
        print(f"\n⬇️  Downloading chapters...")
        for i, chapter in enumerate(chapters[:100], 1):  # Limit to first 100 chapters for testing
            print(f"   [{i}/{min(len(chapters), 100)}] {chapter['title'][:50]}...")
            
            chapter_html = self.download_page(chapter['url'])
            if not chapter_html:
                print(f"       ✗ Failed")
                continue
            
            verses = self.extract_sanskrit_from_chapter(chapter_html)
            
            if verses:
                book_data['chapters'].append({
                    'title': chapter['title'],
                    'url': chapter['url'],
                    'verses': verses,
                    'verse_count': len(verses)
                })
                book_data['total_verses'] += len(verses)
                print(f"       ✓ {len(verses)} verses")
            else:
                print(f"       ⚠ No Sanskrit found")
            
            # Be respectful
            time.sleep(2)
        
        return book_data
    
    def download_all_priority_books(self):
        """Download all priority books"""
        print("="*70)
        print("🕉️  WISDOM LIBRARY CHAPTER-BY-CHAPTER DOWNLOADER")
        print("="*70)
        print(f"\nTarget: {sum(b['estimated_verses'] for b in self.priority_books):,} verses")
        print(f"Books: {len(self.priority_books)}")
        print("="*70)
        
        all_books = []
        total_verses = 0
        
        for book_info in self.priority_books:
            book_data = self.download_book(book_info)
            
            if book_data:
                all_books.append(book_data)
                total_verses += book_data['total_verses']
                
                # Save individual book
                book_file = self.output_dir / f"{book_info['slug']}.json"
                with open(book_file, 'w', encoding='utf-8') as f:
                    json.dump(book_data, f, ensure_ascii=False, indent=2)
                
                print(f"\n✅ {book_info['name']} saved: {book_file}")
                print(f"   Collected: {book_data['total_verses']:,} verses")
                print(f"   Progress: {book_data['total_verses']/book_info['estimated_verses']*100:.1f}% of estimate")
            
            print("\n" + "="*70 + "\n")
        
        # Save combined
        combined_file = self.output_dir / "all_chapters.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_books, f, ensure_ascii=False, indent=2)
        
        # Summary
        print(f"\n{'='*70}")
        print(f"✅ DOWNLOAD COMPLETE")
        print(f"{'='*70}")
        print(f"\nBooks downloaded: {len(all_books)}")
        print(f"Total verses: {total_verses:,}")
        print(f"Saved to: {combined_file}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    downloader = WisdomLibraryChapterDownloader()
    downloader.download_all_priority_books()
