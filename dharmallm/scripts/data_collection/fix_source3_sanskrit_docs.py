#!/usr/bin/env python3
"""
SOURCE 3: Sanskrit Documents - FIX URL MAPPING
Explore actual directory structure and download correctly
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path

class SanskritDocumentsFixer:
    def __init__(self):
        self.base_url = "https://sanskritdocuments.org"
        self.output_dir = Path("data/authentic_sources/sanskrit_docs_fixed")
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
                print(f"    ✓ SUCCESS")
                return response.text
            else:
                print(f"    ✗ Status {response.status_code}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
        return None
    
    def extract_links_from_index(self, url: str):
        """Extract all document links from index page"""
        html = self.download_page(url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Look for .htm, .html, .itx files
            if any(ext in href.lower() for ext in ['.htm', '.html', '.itx', '.txt']):
                full_url = href if href.startswith('http') else f"{url}/{href}"
                links.append(full_url)
        
        return links
    
    def extract_sanskrit(self, text: str) -> str:
        """Extract Devanagari text"""
        devanagari = re.findall(r'[\u0900-\u097F\s।॥०-९]+', text)
        return ' '.join(devanagari).strip()
    
    def explore_category(self, category_name: str, category_path: str):
        """Explore a category and download all texts"""
        print(f"\n📖 Exploring: {category_name}")
        print(f"   Path: {category_path}")
        
        # Try both with and without trailing slash
        for path_variant in [category_path, category_path.rstrip('/')]:
            url = f"{self.base_url}/{path_variant}"
            html = self.download_page(url)
            
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find all document links
                doc_links = []
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if any(ext in href.lower() for ext in ['.htm', '.html', '.itx']):
                        if not href.startswith('http'):
                            full_url = f"{url}/{href}" if not url.endswith('/') else f"{url}{href}"
                        else:
                            full_url = href
                        doc_links.append((href, full_url))
                
                print(f"   Found {len(doc_links)} document links")
                
                # Download each document
                for filename, doc_url in doc_links[:20]:  # Limit to 20 per category
                    doc_html = self.download_page(doc_url)
                    if doc_html:
                        # Extract Sanskrit
                        doc_soup = BeautifulSoup(doc_html, 'html.parser')
                        text = doc_soup.get_text()
                        sanskrit = self.extract_sanskrit(text)
                        
                        if sanskrit and len(sanskrit) > 200:
                            self.texts_collected.append({
                                'source': 'sanskrit_documents',
                                'category': category_name,
                                'filename': filename,
                                'text': sanskrit,
                                'url': doc_url
                            })
                            print(f"     ✓ {filename}: {len(sanskrit)} chars")
                        
                        time.sleep(1)
                
                break  # Found working URL
    
    def run(self):
        """Main execution"""
        print("=" * 70)
        print("SOURCE 3: SANSKRIT DOCUMENTS - FIXED")
        print("=" * 70)
        
        # Explore main categories
        categories = [
            ("Bhagavad Gita", "doc_giitaa"),
            ("Upanishads", "doc_upanishhat"),
            ("Vedas", "doc_veda"),
            ("Stotras", "doc_z_misc_major"),
            ("Puranas", "doc_purana"),
        ]
        
        for cat_name, cat_path in categories:
            self.explore_category(cat_name, cat_path)
            time.sleep(2)
        
        # Save results
        output_file = self.output_dir / "fixed_texts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.texts_collected, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 70)
        print(f"✅ SOURCE 3 COMPLETE: {len(self.texts_collected)} texts collected")
        print(f"   Saved to: {output_file}")
        print("=" * 70)
        
        return self.texts_collected

if __name__ == "__main__":
    fixer = SanskritDocumentsFixer()
    fixer.run()
