#!/usr/bin/env python3
"""
SOURCE 4: Wisdom Library - FIND CORRECT URLs
Explore site structure and find Sanskrit texts
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path

class WisdomLibraryFixer:
    def __init__(self):
        self.base_url = "https://www.wisdomlib.org"
        self.output_dir = Path("data/authentic_sources/wisdom_library_fixed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })
        self.texts_collected = []
    
    def download_page(self, url: str) -> str:
        """Download page"""
        try:
            print(f"  Trying: {url}")
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                print(f"    ✓ Status 200")
                return response.text
            else:
                print(f"    ✗ Status {response.status_code}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
        return None
    
    def extract_sanskrit(self, text: str) -> str:
        """Extract Devanagari text"""
        devanagari = re.findall(r'[\u0900-\u097F\s।॥०-९]+', text)
        return ' '.join(devanagari).strip()
    
    def explore_main_page(self):
        """Explore Wisdom Library main page"""
        print("=" * 70)
        print("SOURCE 4: WISDOM LIBRARY - FINDING CORRECT URLs")
        print("=" * 70)
        
        # Try different URL patterns
        url_patterns = [
            "/hinduism",
            "/hinduism/texts",
            "/hinduism/scripture",
            "/hinduism/books",
            "/definition/bhagavad-gita",
            "/definition/upanishad",
            "/definition/purana",
        ]
        
        for pattern in url_patterns:
            url = f"{self.base_url}{pattern}"
            html = self.download_page(url)
            
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Look for book/text links
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    text = a.get_text().strip()
                    
                    # Look for Sanskrit-related links
                    if any(keyword in text.lower() for keyword in ['gita', 'upanishad', 'purana', 'veda', 'sanskrit']):
                        full_url = href if href.startswith('http') else f"{self.base_url}{href}"
                        print(f"    Found: {text} -> {full_url}")
                
                time.sleep(2)
    
    def try_direct_texts(self):
        """Try direct text URLs"""
        print("\n📖 Trying direct text URLs...")
        
        direct_urls = [
            "/hinduism/book/the-bhagavad-gita",
            "/hinduism/book/gita",
            "/hinduism/book/upanishads",
            "/definition/bhagavad-gita",
            "/definition/upanishads",
            "/hinduism/essay/bhagavad-gita",
        ]
        
        for path in direct_urls:
            url = f"{self.base_url}{path}"
            html = self.download_page(url)
            
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract Sanskrit content
                for div in soup.find_all(['div', 'p', 'article']):
                    text = div.get_text().strip()
                    sanskrit = self.extract_sanskrit(text)
                    
                    if sanskrit and len(sanskrit) > 100:
                        self.texts_collected.append({
                            'source': 'wisdom_library',
                            'url': url,
                            'text': sanskrit
                        })
                
                time.sleep(2)
    
    def search_site(self):
        """Use site search"""
        print("\n🔍 Using site search...")
        
        search_terms = [
            "bhagavad+gita+sanskrit",
            "upanishads+sanskrit",
            "puranas+sanskrit",
        ]
        
        for term in search_terms:
            search_url = f"{self.base_url}/search/{term}"
            html = self.download_page(search_url)
            
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find result links
                for a in soup.find_all('a', href=True):
                    if 'sanskrit' in a.get_text().lower():
                        full_url = a['href'] if a['href'].startswith('http') else f"{self.base_url}{a['href']}"
                        print(f"    Result: {full_url}")
                
                time.sleep(2)
    
    def run(self):
        """Main execution"""
        # Explore main page
        self.explore_main_page()
        
        # Try direct URLs
        self.try_direct_texts()
        
        # Search site
        self.search_site()
        
        # Save results
        output_file = self.output_dir / "found_texts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.texts_collected, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 70)
        print(f"✅ SOURCE 4 COMPLETE: {len(self.texts_collected)} texts collected")
        print(f"   Saved to: {output_file}")
        print("=" * 70)
        
        return self.texts_collected

if __name__ == "__main__":
    fixer = WisdomLibraryFixer()
    fixer.run()
