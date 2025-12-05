#!/usr/bin/env python3
"""
Sample Web Scraper for Sacred-Texts.com
Uses BeautifulSoup to extract complete texts
"""

import requests
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path

def scrape_sacred_texts_page(url: str) -> str:
    """Scrape a single page from sacred-texts.com"""
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Sacred-texts typically has content in <p> tags
        paragraphs = soup.find_all('p')
        text = '\n\n'.join([p.get_text() for p in paragraphs])
        
        return text
    
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def scrape_bhagavad_gita_complete():
    """Scrape all 18 chapters of Bhagavad Gita"""
    
    base_url = "https://www.sacred-texts.com/hin/gita"
    chapters = []
    
    # Gita has chapters gitaXX.htm (01-18)
    for i in range(1, 19):
        chapter_url = f"{base_url}/gita{i:02d}.htm"
        print(f"Downloading Chapter {i}...")
        
        text = scrape_sacred_texts_page(chapter_url)
        
        chapters.append({
            'chapter': i,
            'text': text,
            'url': chapter_url
        })
        
        time.sleep(1)  # Be polite to server
    
    # Save complete Gita
    output_path = Path('complete_scriptures/bhagavad_gita/complete_gita.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chapters, f, indent=2, ensure_ascii=False)
    
    print(f"Complete Bhagavad Gita saved to {output_path}")

if __name__ == "__main__":
    print("Sacred-Texts.com Scraper")
    print("=" * 50)
    scrape_bhagavad_gita_complete()
