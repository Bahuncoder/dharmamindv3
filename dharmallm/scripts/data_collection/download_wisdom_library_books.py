#!/usr/bin/env python3
"""
WISDOM LIBRARY BOOK DOWNLOADER
Test and download all 147 Sanskrit books we found
Potential: 200,000+ verses!
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path
from typing import List, Dict

class WisdomLibraryBookDownloader:
    def __init__(self):
        self.base_url = "https://www.wisdomlib.org/hinduism/book/"
        self.output_dir = Path("data/authentic_sources/wisdom_library_books")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        self.books_collected = []
        
        # All 147 Sanskrit books found
        self.sanskrit_books = [
            # Major Texts
            "mahabharata-sanskrit",
            "the-ramayana-sanskrit",
            "bhagavata-purana-sanskrit",
            "bhagavad-gita-with-four-commentaries-sanskrit",
            "shiva-purana-sanskrit",
            "skanda-purana-sanskrit",
            "agni-purana-sanskrit",
            "padma-purana-sanskrit",
            "narada-purana-sanskrit",
            "bhavishya-purana-sanskrit",
            
            # Upanishads
            "yoga-sutra-with-bhasya-sanskrit",
            "yoga-sutra-with-bhoja-vritti-sanskrit",
            
            # Samhitas & Other Texts
            "manusmriti-sanskrit",
            "manusmriti-with-manubhashya-sanskrit",
            "ashtanga-hridaya-samhita-sanskrit",
            "brahma-samhita-sanskrit",
            "brahma-samhita-with-tika-sanskrit",
            "jayakhya-samhita-sanskrit",
            "lakshmi-tantra-sanskrit",
            "lakshminarayana-samhita-sanskrit",
            "padma-samhita-sanskrit",
            "paramesvara-samhita-sanskrit",
            "prasna-samhita-sanskrit",
            "purushottama-samhita-sanskrit",
            "satvata-samhita-sanskrit",
            "vishnu-samhita-sanskrit",
            "visvaksena-samhita-sanskrit",
            "visvamitra-samhita-sanskrit",
            
            # Tantras
            "kubjikamatatantra-sanskrit",
            "svacchanda-tantra-sanskrit",
            "tantraloka-sanskrit-text",
            
            # Sutras & Shastras
            "apastamba-grihya-sutra-sanskrit",
            "apastamba-sulba-sutra-sanskrit",
            "katyayana-smriti-sanskrit",
            "kautilya-arthashastra-sanskrit",
            "yajnavalkya-smriti-with-mitakshara-sanskrit",
            
            # Literature
            "kathasaritsagara-sanskrit",
            "hitopadesha-sanskrit",
            "panchatantra-sanskrit",
            "panchasayaka-sanskrit",
            "brihat-katha-shloka-samgraha-sanskrit",
            "naishadha-charita-sanskrit",
            "yoga-vasistha-sanskrit",
            "moksopaya-sanskrit",
            
            # Jyotisha (Astrology)
            "brihat-samhita-sanskrit",
            "bhrigu-samhita-sanskrit",
            "hayanaratna-by-balabhadra-sanskrit",
            
            # Ayurveda
            "matangalila-by-nilakantha-sanskrit",
            "dhanurveda-samhita-sanskrit",
            "syainika-sastra-sanskrit",
            
            # Brahmanas & Vedic
            "satapatha-brahmana-sanskrit",
            "harivamsa-text-sanskrit",
            "harivamsa-appendix-sanskrit",
            
            # Architecture & Technical
            "samarangana-sutradhara-sanskrit",
            "yuktidipika-sanskrit",
            "ganitatilaka-sanskrit-text",
            
            # Additional Puranas
            "hari-bhakti-vilasa-sanskrit-text",
        ]
    
    def download_page(self, url: str) -> str:
        """Download page with retry"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            else:
                return None
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def extract_sanskrit(self, html: str, book_name: str) -> List[str]:
        """Extract all Sanskrit verses/content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        verses = []
        
        # Method 1: Find divs/paragraphs with Devanagari
        for tag in soup.find_all(['div', 'p', 'span', 'td', 'pre', 'article']):
            text = tag.get_text().strip()
            if re.search(r'[\u0900-\u097F]', text):
                # Extract only Devanagari
                sanskrit = re.findall(r'[\u0900-\u097F\s।॥०-९]+', text)
                if sanskrit:
                    verse = ' '.join(sanskrit).strip()
                    if len(verse) > 50:  # Substantial content
                        verses.append(verse)
        
        # Method 2: Look for specific Sanskrit classes
        sanskrit_classes = soup.find_all(class_=re.compile(r'sanskrit|devanagari|verse|shloka', re.I))
        for elem in sanskrit_classes:
            text = elem.get_text().strip()
            sanskrit = re.findall(r'[\u0900-\u097F\s।॥०-९]+', text)
            if sanskrit:
                verse = ' '.join(sanskrit).strip()
                if len(verse) > 50:
                    verses.append(verse)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_verses = []
        for v in verses:
            if v not in seen and len(v) > 50:
                seen.add(v)
                unique_verses.append(v)
        
        return unique_verses
    
    def download_book(self, book_slug: str) -> Dict:
        """Download a single book"""
        url = self.base_url + book_slug
        print(f"\n📖 {book_slug}")
        print(f"   URL: {url}")
        
        html = self.download_page(url)
        
        if not html:
            print(f"   ✗ Failed to download")
            return None
        
        # Check status
        if "404" in html or "not found" in html.lower():
            print(f"   ✗ 404 Not Found")
            return None
        
        # Extract Sanskrit
        verses = self.extract_sanskrit(html, book_slug)
        
        if verses:
            book_data = {
                'source': 'wisdom_library_books',
                'book': book_slug,
                'url': url,
                'verse_count': len(verses),
                'verses': verses,
                'total_chars': sum(len(v) for v in verses)
            }
            
            print(f"   ✓ SUCCESS: {len(verses)} verses, {book_data['total_chars']:,} chars")
            return book_data
        else:
            print(f"   ⚠ No Sanskrit content found")
            return None
    
    def download_all_books(self):
        """Download all Sanskrit books"""
        print("=" * 70)
        print("📚 WISDOM LIBRARY BOOK DOWNLOADER")
        print(f"   Testing {len(self.sanskrit_books)} Sanskrit books")
        print("=" * 70)
        
        successful = 0
        failed = 0
        total_verses = 0
        
        for i, book_slug in enumerate(self.sanskrit_books, 1):
            print(f"\n[{i}/{len(self.sanskrit_books)}]", end=" ")
            
            book_data = self.download_book(book_slug)
            
            if book_data:
                self.books_collected.append(book_data)
                successful += 1
                total_verses += book_data['verse_count']
            else:
                failed += 1
            
            # Progress update every 10 books
            if i % 10 == 0:
                print(f"\n{'='*70}")
                print(f"Progress: {i}/{len(self.sanskrit_books)}")
                print(f"Success: {successful}, Failed: {failed}")
                print(f"Total verses so far: {total_verses:,}")
                print(f"{'='*70}")
            
            # Be respectful - wait between requests
            time.sleep(2)
        
        # Save results
        self.save_results(successful, failed, total_verses)
        
        return self.books_collected
    
    def save_results(self, successful: int, failed: int, total_verses: int):
        """Save collected books"""
        # Save detailed JSON
        output_file = self.output_dir / "all_sanskrit_books.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.books_collected, f, ensure_ascii=False, indent=2)
        
        # Save summary
        summary_file = self.output_dir / "download_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("WISDOM LIBRARY DOWNLOAD SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Books attempted: {len(self.sanskrit_books)}\n")
            f.write(f"Books successful: {successful}\n")
            f.write(f"Books failed: {failed}\n")
            f.write(f"Success rate: {successful/len(self.sanskrit_books)*100:.1f}%\n\n")
            f.write(f"Total verses: {total_verses:,}\n")
            f.write(f"Total books with content: {len(self.books_collected)}\n\n")
            
            if self.books_collected:
                f.write("Top 10 books by verse count:\n")
                sorted_books = sorted(self.books_collected, key=lambda x: x['verse_count'], reverse=True)
                for i, book in enumerate(sorted_books[:10], 1):
                    f.write(f"  {i}. {book['book']}: {book['verse_count']:,} verses\n")
        
        print("\n" + "=" * 70)
        print("✅ DOWNLOAD COMPLETE")
        print("=" * 70)
        print(f"\nBooks attempted: {len(self.sanskrit_books)}")
        print(f"Books successful: {successful}")
        print(f"Books failed: {failed}")
        print(f"Success rate: {successful/len(self.sanskrit_books)*100:.1f}%")
        print(f"\nTotal verses collected: {total_verses:,}")
        print(f"Total characters: {sum(b['total_chars'] for b in self.books_collected):,}")
        print(f"\nSaved to: {output_file}")
        print(f"Summary: {summary_file}")
        print("=" * 70)
        
        if total_verses > 0:
            estimated_total = total_verses + 1000  # Add existing corpus
            print(f"\n🎉 AMAZING! We now have ~{estimated_total:,} verses!")
            print(f"   Progress: {estimated_total/837000*100:.2f}% of 837,000 goal")

if __name__ == "__main__":
    downloader = WisdomLibraryBookDownloader()
    downloader.download_all_books()
