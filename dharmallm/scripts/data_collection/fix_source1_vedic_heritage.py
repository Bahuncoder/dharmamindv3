#!/usr/bin/env python3
"""
SOURCE 1: Vedic Heritage Portal - DEEP DIVE
Expand collection from 310 texts to thousands
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path
from typing import List, Dict

class VedicHeritageExpander:
    def __init__(self):
        self.base_url = "https://vedicheritage.gov.in"
        self.output_dir = Path("data/authentic_sources/vedic_heritage_expanded")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        self.texts_collected = []
    
    def download_page(self, url: str) -> str:
        """Download page with retry"""
        for attempt in range(3):
            try:
                print(f"  Downloading: {url}")
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    return response.text
                time.sleep(2)
            except Exception as e:
                print(f"    Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return None
    
    def extract_sanskrit(self, text: str) -> str:
        """Extract only Devanagari text"""
        devanagari_pattern = r'[\u0900-\u097F\s।॥०-९]+'
        matches = re.findall(devanagari_pattern, text)
        return ' '.join(matches).strip()
    
    def explore_main_sections(self):
        """Explore main sections of Vedic Heritage Portal"""
        print("=" * 70)
        print("SOURCE 1: VEDIC HERITAGE PORTAL - DEEP DIVE")
        print("=" * 70)
        
        # Main sections to explore
        sections = [
            "/samhitas/",
            "/upanishads/",
            "/vedangas/",
            "/brahmanas/",
            "/aranyakas/",
            "/sutras/",
        ]
        
        all_links = []
        
        for section in sections:
            print(f"\n📖 Exploring: {section}")
            url = self.base_url + section
            html = self.download_page(url)
            
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                links = []
                
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if href.startswith('/') and not href.startswith('//'):
                        full_url = self.base_url + href
                        if section[1:-1] in href.lower():
                            links.append(full_url)
                
                links = list(set(links))
                print(f"  Found {len(links)} links in {section}")
                all_links.extend(links)
                time.sleep(1)
        
        return list(set(all_links))
    
    def download_all_veda_pages(self):
        """Download all Veda-related pages"""
        print("\n" + "=" * 70)
        print("DOWNLOADING ALL VEDA PAGES")
        print("=" * 70)
        
        # Specific Veda URLs
        veda_urls = [
            # Rigveda
            "/samhitas/rigveda/",
            "/samhitas/rigveda/ashvalayana-samhita/",
            "/samhitas/rigveda/kaushitaki-samhita/",
            "/samhitas/rigveda/shankhayana-samhita/",
            
            # Yajurveda
            "/samhitas/yajurveda/",
            "/samhitas/yajurveda/krishna-yajurveda/taittiriya-samhita/",
            "/samhitas/yajurveda/vajasneyi-madhyandina-samhita/",
            "/samhitas/yajurveda/vajasaneyi-kanva-samhita/",
            
            # Samaveda
            "/samhitas/samaveda-samhitas/",
            "/samhitas/samaveda-samhitas/kauthuma-samhita/",
            "/samhitas/samaveda-samhitas/jaiminiya-samhita-2/",
            "/samhitas/samaveda-samhitas/ranayaniya-samhita/",
            
            # Atharvaveda
            "/samhitas/atharvaveda-samhitas/",
            "/samhitas/atharvaveda-samhitas/shaunaka-samhita/",
            "/samhitas/atharvaveda-samhitas/atharvaveda-shaunaka-samhita/",
            "/samhitas/atharvaveda-samhitas/paippalada-samhita/",
        ]
        
        for url in veda_urls:
            full_url = self.base_url + url if not url.startswith('http') else url
            html = self.download_page(full_url)
            
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract all text content
                for div in soup.find_all(['div', 'p', 'article', 'section']):
                    text = div.get_text().strip()
                    sanskrit = self.extract_sanskrit(text)
                    
                    if sanskrit and len(sanskrit) > 100:
                        self.texts_collected.append({
                            'source': 'vedic_heritage_expanded',
                            'category': 'veda',
                            'url': full_url,
                            'text': sanskrit
                        })
                
                # Find and follow all sub-links
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if '/samhitas/' in href or '/veda/' in href.lower():
                        if href.startswith('/'):
                            sub_url = self.base_url + href
                            if sub_url not in [full_url] + veda_urls:
                                print(f"    Found sub-page: {sub_url}")
                                sub_html = self.download_page(sub_url)
                                
                                if sub_html:
                                    sub_soup = BeautifulSoup(sub_html, 'html.parser')
                                    for div in sub_soup.find_all(['div', 'p']):
                                        text = div.get_text().strip()
                                        sanskrit = self.extract_sanskrit(text)
                                        
                                        if sanskrit and len(sanskrit) > 100:
                                            self.texts_collected.append({
                                                'source': 'vedic_heritage_expanded',
                                                'category': 'veda',
                                                'url': sub_url,
                                                'text': sanskrit
                                            })
                                    
                                    time.sleep(2)
                
                time.sleep(2)
    
    def download_all_upanishads(self):
        """Download all Upanishads"""
        print("\n" + "=" * 70)
        print("DOWNLOADING ALL UPANISHADS")
        print("=" * 70)
        
        # Major Upanishads
        upanishad_urls = [
            "/upanishads/",
            "/upanishads/isha-upanishad/",
            "/upanishads/kenopanisad/",
            "/upanishads/kathopanishad/",
            "/upanishads/prashnopanishad/",
            "/upanishads/mundakopanishad/",
            "/upanishads/mandukya-upanishad/",
            "/upanishads/taittiriya-upanishads/",
            "/upanishads/aitareya-upanishad/",
            "/upanishads/chandogyopanishad/",
            "/upanishads/brihadaranyaka-upanishad/",
            "/upanishads/shvetashvatara-upanishad/",
            "/upanishads/kaushitaki-upanishad/",
            "/upanishads/maitrayaniya-upanishad/",
            "/upanishads/maitrayani-upanishad/",
        ]
        
        for url in upanishad_urls:
            full_url = self.base_url + url
            html = self.download_page(full_url)
            
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                for div in soup.find_all(['div', 'p', 'article']):
                    text = div.get_text().strip()
                    sanskrit = self.extract_sanskrit(text)
                    
                    if sanskrit and len(sanskrit) > 100:
                        self.texts_collected.append({
                            'source': 'vedic_heritage_expanded',
                            'category': 'upanishad',
                            'url': full_url,
                            'text': sanskrit
                        })
                
                time.sleep(2)
    
    def run(self):
        """Main execution"""
        print("\n🕉️  Starting Vedic Heritage Portal Deep Dive...")
        
        # Download Vedas
        self.download_all_veda_pages()
        print(f"\n  ✓ Vedas: {len([t for t in self.texts_collected if t['category']=='veda'])} texts")
        
        # Download Upanishads
        self.download_all_upanishads()
        print(f"  ✓ Upanishads: {len([t for t in self.texts_collected if t['category']=='upanishad'])} texts")
        
        # Save results
        output_file = self.output_dir / "expanded_texts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.texts_collected, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 70)
        print(f"✅ SOURCE 1 COMPLETE: {len(self.texts_collected)} texts collected")
        print(f"   Saved to: {output_file}")
        print("=" * 70)
        
        return self.texts_collected

if __name__ == "__main__":
    expander = VedicHeritageExpander()
    expander.run()
