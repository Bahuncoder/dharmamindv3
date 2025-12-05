#!/usr/bin/env python3
"""
SOURCE 2: Gita Supersite IIT Kanpur - FIX HTML PARSING
Get all 700 Bhagavad Gita verses in pure Sanskrit
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path

class GitaSupersiteFixer:
    def __init__(self):
        self.base_url = "https://www.gitasupersite.iitk.ac.in/srimad"
        self.output_dir = Path("data/authentic_sources/gita_supersite_fixed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        self.verses_collected = []
        
        # Known verse counts per chapter
        self.verse_counts = {
            1: 47, 2: 72, 3: 43, 4: 42, 5: 29, 6: 47,
            7: 30, 8: 28, 9: 34, 10: 42, 11: 55, 12: 20,
            13: 35, 14: 27, 15: 20, 16: 24, 17: 28, 18: 78
        }
    
    def download_page(self, url: str) -> str:
        """Download page"""
        try:
            print(f"  Downloading: {url}")
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"    Error: {e}")
        return None
    
    def extract_sanskrit_verse(self, html: str) -> str:
        """Extract Sanskrit verse using multiple methods"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Method 1: Look for Devanagari in ANY div/p/span
        for tag in soup.find_all(['div', 'p', 'span', 'td', 'pre']):
            text = tag.get_text().strip()
            # Check if contains Devanagari
            if re.search(r'[\u0900-\u097F]', text):
                # Extract only Devanagari
                devanagari = re.findall(r'[\u0900-\u097F\s।॥०-९]+', text)
                if devanagari:
                    verse = ' '.join(devanagari).strip()
                    # Must have at least 20 chars and some Sanskrit chars
                    if len(verse) > 20 and re.search(r'[\u0915-\u0939]', verse):
                        return verse
        
        # Method 2: Look in table rows
        for tr in soup.find_all('tr'):
            text = tr.get_text().strip()
            if re.search(r'[\u0900-\u097F]', text):
                devanagari = re.findall(r'[\u0900-\u097F\s।॥०-९]+', text)
                if devanagari:
                    verse = ' '.join(devanagari).strip()
                    if len(verse) > 20:
                        return verse
        
        # Method 3: Just extract all Devanagari from entire HTML
        all_text = soup.get_text()
        devanagari = re.findall(r'[\u0900-\u097F\s।॥०-९]+', all_text)
        if devanagari:
            verse = ' '.join(devanagari).strip()
            if len(verse) > 50:  # Has substantial content
                return verse
        
        return None
    
    def download_chapter(self, chapter: int):
        """Download all verses from a chapter"""
        print(f"\n📖 Chapter {chapter}...")
        verse_count = self.verse_counts.get(chapter, 50)
        
        for verse_num in range(1, verse_count + 1):
            url = f"{self.base_url}?language=dv&field_chapter_value={chapter}&field_nsutra_value={verse_num}"
            html = self.download_page(url)
            
            if html:
                sanskrit_verse = self.extract_sanskrit_verse(html)
                
                if sanskrit_verse:
                    self.verses_collected.append({
                        'source': 'gita_supersite_iitk',
                        'chapter': chapter,
                        'verse': verse_num,
                        'text': sanskrit_verse,
                        'url': url
                    })
                    print(f"  ✓ {chapter}.{verse_num} - {len(sanskrit_verse)} chars")
                else:
                    print(f"  ✗ {chapter}.{verse_num} - No Sanskrit found")
            
            time.sleep(1)  # Be respectful
        
        print(f"  Chapter {chapter}: {sum(1 for v in self.verses_collected if v['chapter']==chapter)} verses")
    
    def download_all_chapters(self):
        """Download all 18 chapters"""
        print("=" * 70)
        print("SOURCE 2: GITA SUPERSITE IIT KANPUR - FIXED")
        print("=" * 70)
        
        for chapter in range(1, 19):
            self.download_chapter(chapter)
        
        # Save results
        output_file = self.output_dir / "complete_gita.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.verses_collected, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 70)
        print(f"✅ SOURCE 2 COMPLETE: {len(self.verses_collected)} verses collected")
        print(f"   Total expected: ~700 verses")
        print(f"   Success rate: {len(self.verses_collected)/700*100:.1f}%")
        print(f"   Saved to: {output_file}")
        print("=" * 70)
        
        return self.verses_collected

if __name__ == "__main__":
    fixer = GitaSupersiteFixer()
    fixer.download_all_chapters()
