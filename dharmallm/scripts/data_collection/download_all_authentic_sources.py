#!/usr/bin/env python3
"""
Complete Authentic Hindu Text Downloader
Downloads from all 4 priority sources in order:
1. Sanskrit Documents (sanskritdocuments.org)
2. Wisdom Library (wisdomlib.org)
3. Gita Supersite IIT Kanpur (gitasupersite.iitk.ac.in)
4. Vedic Heritage Portal (vedicheritage.gov.in)
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AuthenticSourceDownloader:
    def __init__(self, output_dir: str = "data/authentic_sources"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        
        # Statistics
        self.stats = {
            'sanskrit_documents': 0,
            'wisdom_library': 0,
            'gita_supersite': 0,
            'vedic_heritage': 0,
            'total_texts': 0,
            'total_verses': 0
        }

    def download_page(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Download page with retries"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading: {url}")
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    return response.text
                else:
                    logger.warning(f"Status {response.status_code} for {url}")
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed for {url}: {e}")
                time.sleep(2)
        return None

    def extract_sanskrit_text(self, text: str) -> str:
        """Extract only Sanskrit/Devanagari text"""
        # Keep only Devanagari characters, digits, punctuation
        devanagari_pattern = r'[\u0900-\u097F\s।॥०-९]+|[0-9\.\s]+'
        matches = re.findall(devanagari_pattern, text)
        return ' '.join(matches).strip()

    # ========================================================================
    # SOURCE 1: SANSKRIT DOCUMENTS (sanskritdocuments.org)
    # ========================================================================
    
    def download_sanskrit_documents(self):
        """
        Source 1: Sanskrit Documents Collection
        Direct downloads of .htm and .itx files
        """
        logger.info("=" * 70)
        logger.info("SOURCE 1: SANSKRIT DOCUMENTS (sanskritdocuments.org)")
        logger.info("=" * 70)
        
        output_dir = self.output_dir / "sanskrit_documents"
        output_dir.mkdir(exist_ok=True)
        
        # Categories to download
        categories = {
            'gita': {
                'url': 'https://sanskritdocuments.org/doc_giitaa/',
                'files': [
                    'gItA.htm',
                    'gItA.itx',
                    'gItAdhyAna.htm',
                ]
            },
            'upanishads': {
                'url': 'https://sanskritdocuments.org/doc_upanishhat/',
                'files': [
                    'iishaa.htm', 'iishaa.itx',
                    'kena.htm', 'kena.itx',
                    'kaTha.htm', 'kaTha.itx',
                    'prashna.htm', 'prashna.itx',
                    'muNDaka.htm', 'muNDaka.itx',
                    'mANDukya.htm', 'mANDukya.itx',
                    'taittiriiya.htm', 'taittiriiya.itx',
                    'aitareya.htm', 'aitareya.itx',
                    'chhaandogya.htm', 'chhaandogya.itx',
                    'bRhad.htm', 'bRhad.itx',
                    'shvetaashvataara.htm', 'shvetaashvataara.itx',
                    'kaushiitaki.htm', 'kaushiitaki.itx',
                    'maitraayaNi.htm', 'maitraayaNi.itx',
                ]
            },
            'vedas': {
                'url': 'https://sanskritdocuments.org/doc_veda/',
                'files': [
                    'rigveda.htm',
                    'yajurveda.htm',
                    'samaveda.htm',
                    'atharvaveda.htm',
                ]
            },
            'stotras': {
                'url': 'https://sanskritdocuments.org/doc_z_misc_major/',
                'files': [
                    'vishNusahasranAma.htm', 'vishNusahasranAma.itx',
                    'lalitAsahasranAma.htm', 'lalitAsahasranAma.itx',
                    'shivamahimnaH.htm', 'shivamahimnaH.itx',
                    'soundaryalahari.htm', 'soundaryalahari.itx',
                    'shriirudranamakaM.htm',
                ]
            },
            'yoga': {
                'url': 'https://sanskritdocuments.org/doc_z_misc_major/',
                'files': [
                    'yogasuutra.htm', 'yogasuutra.itx',
                    'brahmasuutra.htm',
                ]
            }
        }
        
        texts_collected = []
        
        for category, info in categories.items():
            logger.info(f"\n📖 Downloading {category.upper()}...")
            category_dir = output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            for filename in info['files']:
                url = info['url'] + filename
                content = self.download_page(url)
                
                if content:
                    # Save original file
                    file_path = category_dir / filename
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Extract Sanskrit text
                    if filename.endswith('.htm'):
                        soup = BeautifulSoup(content, 'html.parser')
                        text = soup.get_text()
                        sanskrit_text = self.extract_sanskrit_text(text)
                        
                        if sanskrit_text and len(sanskrit_text) > 100:
                            texts_collected.append({
                                'source': 'sanskrit_documents',
                                'category': category,
                                'filename': filename,
                                'text': sanskrit_text,
                                'url': url
                            })
                            logger.info(f"  ✓ {filename} ({len(sanskrit_text)} chars)")
                            self.stats['sanskrit_documents'] += 1
                    
                    time.sleep(1)  # Be respectful
        
        # Save collected texts
        output_file = output_dir / "all_texts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(texts_collected, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ Sanskrit Documents: {len(texts_collected)} texts collected")
        return texts_collected

    # ========================================================================
    # SOURCE 2: WISDOM LIBRARY (wisdomlib.org)
    # ========================================================================
    
    def download_wisdom_library(self):
        """
        Source 2: Wisdom Library
        Complete books: Puranas, Mahabharata, Ramayana, Upanishads
        """
        logger.info("=" * 70)
        logger.info("SOURCE 2: WISDOM LIBRARY (wisdomlib.org)")
        logger.info("=" * 70)
        
        output_dir = self.output_dir / "wisdom_library"
        output_dir.mkdir(exist_ok=True)
        
        # Books to download
        books = {
            'bhagavad_gita': {
                'url': 'https://www.wisdomlib.org/hinduism/book/bhagavad-gita',
                'chapters': 18
            },
            'upanishads': {
                'url': 'https://www.wisdomlib.org/hinduism/compilation/upanishads',
                'sections': True
            },
            'mahabharata': {
                'url': 'https://www.wisdomlib.org/hinduism/book/mahabharata',
                'sections': True
            },
            'ramayana': {
                'url': 'https://www.wisdomlib.org/hinduism/book/ramayana',
                'sections': True
            },
            'vishnu_purana': {
                'url': 'https://www.wisdomlib.org/hinduism/book/vishnu-purana',
                'sections': True
            },
            'bhagavata_purana': {
                'url': 'https://www.wisdomlib.org/hinduism/book/bhagavata-purana',
                'sections': True
            }
        }
        
        texts_collected = []
        
        for book_name, info in books.items():
            logger.info(f"\n📚 Downloading {book_name.replace('_', ' ').title()}...")
            book_dir = output_dir / book_name
            book_dir.mkdir(exist_ok=True)
            
            # Download main page
            html = self.download_page(info['url'])
            if not html:
                logger.warning(f"  ✗ Could not download {book_name}")
                continue
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all chapter/section links
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/book/' in href or '/chapter/' in href or '/verse/' in href:
                    full_url = href if href.startswith('http') else f"https://www.wisdomlib.org{href}"
                    links.append(full_url)
            
            # Remove duplicates
            links = list(set(links))
            logger.info(f"  Found {len(links)} sections/chapters")
            
            # Download each section (limit to first 50 for now)
            for i, link in enumerate(links[:50], 1):
                section_html = self.download_page(link)
                if section_html:
                    soup = BeautifulSoup(section_html, 'html.parser')
                    
                    # Extract Sanskrit verses
                    sanskrit_divs = soup.find_all(['div', 'p', 'span'], class_=re.compile('sanskrit|verse|devanagari', re.I))
                    
                    for div in sanskrit_divs:
                        text = div.get_text().strip()
                        sanskrit_text = self.extract_sanskrit_text(text)
                        
                        if sanskrit_text and len(sanskrit_text) > 50:
                            texts_collected.append({
                                'source': 'wisdom_library',
                                'book': book_name,
                                'section': i,
                                'text': sanskrit_text,
                                'url': link
                            })
                            self.stats['wisdom_library'] += 1
                    
                    if i % 10 == 0:
                        logger.info(f"  Processed {i}/{min(50, len(links))} sections...")
                    
                    time.sleep(2)  # Be respectful
        
        # Save collected texts
        output_file = output_dir / "all_texts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(texts_collected, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ Wisdom Library: {len(texts_collected)} texts collected")
        return texts_collected

    # ========================================================================
    # SOURCE 3: GITA SUPERSITE IIT KANPUR
    # ========================================================================
    
    def download_gita_supersite(self):
        """
        Source 3: Gita Supersite (IIT Kanpur)
        Most authoritative Bhagavad Gita source
        """
        logger.info("=" * 70)
        logger.info("SOURCE 3: GITA SUPERSITE IIT KANPUR")
        logger.info("=" * 70)
        
        output_dir = self.output_dir / "gita_supersite"
        output_dir.mkdir(exist_ok=True)
        
        base_url = "https://www.gitasupersite.iitk.ac.in/srimad"
        texts_collected = []
        
        # Download all 18 chapters
        for chapter in range(1, 19):
            logger.info(f"\n📖 Chapter {chapter}...")
            
            # Chapter page
            chapter_url = f"{base_url}?language=dv&field_chapter_value={chapter}"
            html = self.download_page(chapter_url)
            
            if not html:
                logger.warning(f"  ✗ Could not download chapter {chapter}")
                continue
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find verse count from page
            verse_links = soup.find_all('a', href=re.compile(f'field_chapter_value={chapter}&field_nsutra_value='))
            max_verse = 0
            for link in verse_links:
                match = re.search(r'field_nsutra_value=(\d+)', link['href'])
                if match:
                    max_verse = max(max_verse, int(match.group(1)))
            
            if max_verse == 0:
                # Try to guess based on chapter (approximate verse counts)
                verse_counts = [47, 72, 43, 42, 29, 47, 30, 28, 34, 42, 55, 20, 35, 27, 20, 24, 28, 78]
                max_verse = verse_counts[chapter - 1] if chapter <= 18 else 50
            
            logger.info(f"  Found ~{max_verse} verses")
            
            # Download each verse
            for verse in range(1, max_verse + 1):
                verse_url = f"{base_url}?language=dv&field_chapter_value={chapter}&field_nsutra_value={verse}"
                verse_html = self.download_page(verse_url)
                
                if verse_html:
                    soup = BeautifulSoup(verse_html, 'html.parser')
                    
                    # Extract Sanskrit verse
                    verse_divs = soup.find_all(['div', 'p', 'span'], class_=re.compile('verse|sanskrit|shloka', re.I))
                    
                    for div in verse_divs:
                        text = div.get_text().strip()
                        sanskrit_text = self.extract_sanskrit_text(text)
                        
                        if sanskrit_text and len(sanskrit_text) > 20:
                            texts_collected.append({
                                'source': 'gita_supersite',
                                'chapter': chapter,
                                'verse': verse,
                                'text': sanskrit_text,
                                'url': verse_url
                            })
                            self.stats['gita_supersite'] += 1
                            break  # One verse per URL
                    
                    time.sleep(1)  # Be respectful
            
            logger.info(f"  ✓ Chapter {chapter} complete")
        
        # Save collected texts
        output_file = output_dir / "all_texts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(texts_collected, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ Gita Supersite: {len(texts_collected)} verses collected")
        return texts_collected

    # ========================================================================
    # SOURCE 4: VEDIC HERITAGE PORTAL
    # ========================================================================
    
    def download_vedic_heritage(self):
        """
        Source 4: Vedic Heritage Portal (Government of India)
        Complete 4 Vedas - most authentic source
        """
        logger.info("=" * 70)
        logger.info("SOURCE 4: VEDIC HERITAGE PORTAL (vedicheritage.gov.in)")
        logger.info("=" * 70)
        
        output_dir = self.output_dir / "vedic_heritage"
        output_dir.mkdir(exist_ok=True)
        
        base_url = "https://vedicheritage.gov.in"
        texts_collected = []
        
        # Try to access main page
        html = self.download_page(base_url)
        if not html:
            logger.warning("Could not access Vedic Heritage Portal")
            logger.info("This portal may require registration or special access")
            logger.info("Attempting alternative access methods...")
            
            # Try direct Veda URLs (if they exist)
            veda_urls = [
                f"{base_url}/rigveda",
                f"{base_url}/yajurveda",
                f"{base_url}/samaveda",
                f"{base_url}/atharvaveda",
                f"{base_url}/vedas/rigveda",
                f"{base_url}/digital-library",
            ]
            
            for url in veda_urls:
                html = self.download_page(url)
                if html:
                    logger.info(f"  ✓ Found: {url}")
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract any Sanskrit content
                    for div in soup.find_all(['div', 'p', 'article']):
                        text = div.get_text().strip()
                        sanskrit_text = self.extract_sanskrit_text(text)
                        
                        if sanskrit_text and len(sanskrit_text) > 100:
                            texts_collected.append({
                                'source': 'vedic_heritage',
                                'url': url,
                                'text': sanskrit_text
                            })
                            self.stats['vedic_heritage'] += 1
                    
                    time.sleep(2)
        else:
            soup = BeautifulSoup(html, 'html.parser')
            logger.info("Portal accessible - exploring structure...")
            
            # Find all relevant links
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if any(word in href.lower() for word in ['veda', 'rigveda', 'yajur', 'sama', 'atharva', 'upanishad']):
                    full_url = href if href.startswith('http') else f"{base_url}{href}"
                    links.append(full_url)
            
            links = list(set(links))
            logger.info(f"Found {len(links)} relevant links")
            
            # Download each link
            for link in links[:20]:  # Limit to first 20
                page_html = self.download_page(link)
                if page_html:
                    soup = BeautifulSoup(page_html, 'html.parser')
                    
                    # Extract Sanskrit content
                    for div in soup.find_all(['div', 'p', 'article']):
                        text = div.get_text().strip()
                        sanskrit_text = self.extract_sanskrit_text(text)
                        
                        if sanskrit_text and len(sanskrit_text) > 100:
                            texts_collected.append({
                                'source': 'vedic_heritage',
                                'url': link,
                                'text': sanskrit_text
                            })
                            self.stats['vedic_heritage'] += 1
                    
                    time.sleep(2)
        
        # Save collected texts
        if texts_collected:
            output_file = output_dir / "all_texts.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(texts_collected, f, ensure_ascii=False, indent=2)
            
            logger.info(f"\n✅ Vedic Heritage: {len(texts_collected)} texts collected")
        else:
            logger.warning("\n⚠️  Vedic Heritage: Portal may require registration")
            logger.info("    Consider manual download from: https://vedicheritage.gov.in")
        
        return texts_collected

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def download_all_sources(self):
        """Download from all 4 sources in order"""
        logger.info("\n" + "=" * 70)
        logger.info("🕉️  COMPLETE AUTHENTIC HINDU TEXT DOWNLOADER")
        logger.info("   Processing all 4 priority sources")
        logger.info("=" * 70 + "\n")
        
        all_texts = []
        
        # Source 1: Sanskrit Documents
        try:
            texts = self.download_sanskrit_documents()
            all_texts.extend(texts)
        except Exception as e:
            logger.error(f"Error downloading Sanskrit Documents: {e}")
        
        # Source 2: Wisdom Library
        try:
            texts = self.download_wisdom_library()
            all_texts.extend(texts)
        except Exception as e:
            logger.error(f"Error downloading Wisdom Library: {e}")
        
        # Source 3: Gita Supersite
        try:
            texts = self.download_gita_supersite()
            all_texts.extend(texts)
        except Exception as e:
            logger.error(f"Error downloading Gita Supersite: {e}")
        
        # Source 4: Vedic Heritage
        try:
            texts = self.download_vedic_heritage()
            all_texts.extend(texts)
        except Exception as e:
            logger.error(f"Error downloading Vedic Heritage: {e}")
        
        # Save combined corpus
        self.save_combined_corpus(all_texts)
        
        # Print final statistics
        self.print_final_stats()
        
        return all_texts

    def save_combined_corpus(self, texts: List[Dict]):
        """Save all texts into combined corpus"""
        output_file = self.output_dir / "COMPLETE_AUTHENTIC_CORPUS.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n💾 Saved combined corpus: {output_file}")
        logger.info(f"   Total texts: {len(texts)}")
        
        # Calculate total characters
        total_chars = sum(len(t.get('text', '')) for t in texts)
        logger.info(f"   Total characters: {total_chars:,}")
        
        # Estimate verses (average verse ~80 chars)
        estimated_verses = total_chars // 80
        self.stats['total_texts'] = len(texts)
        self.stats['total_verses'] = estimated_verses

    def print_final_stats(self):
        """Print final statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 FINAL STATISTICS")
        logger.info("=" * 70)
        logger.info(f"\nSource 1 - Sanskrit Documents : {self.stats['sanskrit_documents']:,} texts")
        logger.info(f"Source 2 - Wisdom Library      : {self.stats['wisdom_library']:,} texts")
        logger.info(f"Source 3 - Gita Supersite      : {self.stats['gita_supersite']:,} texts")
        logger.info(f"Source 4 - Vedic Heritage      : {self.stats['vedic_heritage']:,} texts")
        logger.info(f"\n{'=' * 70}")
        logger.info(f"TOTAL TEXTS COLLECTED          : {self.stats['total_texts']:,}")
        logger.info(f"ESTIMATED VERSES               : {self.stats['total_verses']:,}")
        logger.info(f"{'=' * 70}\n")
        
        # Progress toward goal
        target = 837000
        progress = (self.stats['total_verses'] / target) * 100
        logger.info(f"Progress toward 837,000 verses: {progress:.2f}%")
        logger.info(f"Still needed: {target - self.stats['total_verses']:,} verses\n")

def main():
    """Main execution"""
    downloader = AuthenticSourceDownloader()
    downloader.download_all_sources()
    
    print("\n" + "=" * 70)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review downloaded texts in: data/authentic_sources/")
    print("2. Train model with new corpus")
    print("3. Continue expanding collection")
    print("\n")

if __name__ == "__main__":
    main()
