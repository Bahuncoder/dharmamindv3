#!/usr/bin/env python3
"""
Combine all collected Sanskrit sources into one master training corpus
"""

import json
import os
from pathlib import Path
from collections import Counter

def load_json_safely(filepath):
    """Load JSON file safely"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return []

def combine_all_sources():
    """Combine all sources into master corpus"""
    
    print("=" * 70)
    print("🕉️  COMBINING ALL SANSKRIT SOURCES")
    print("=" * 70)
    
    all_texts = []
    
    # Source 1: Existing pure Sanskrit corpus (828 texts)
    print("\n1. Loading existing pure Sanskrit corpus...")
    existing_file = "data/pure_sanskrit_corpus/COMPLETE_PURE_SANSKRIT_CORPUS.json"
    if os.path.exists(existing_file):
        existing_data = load_json_safely(existing_file)
        if isinstance(existing_data, dict) and 'texts' in existing_data:
            existing_texts = existing_data['texts']
        else:
            existing_texts = existing_data if isinstance(existing_data, list) else []
        print(f"   ✓ Loaded {len(existing_texts)} texts from existing corpus")
        # Convert to uniform format
        for text in existing_texts:
            if isinstance(text, str):
                all_texts.append({'source': 'existing_corpus', 'text': text})
            else:
                all_texts.append(text)
    
    # Source 2: New authentic sources (314 texts)
    print("\n2. Loading new authentic sources...")
    authentic_file = "data/authentic_sources/COMPLETE_AUTHENTIC_CORPUS.json"
    if os.path.exists(authentic_file):
        authentic_texts = load_json_safely(authentic_file)
        print(f"   ✓ Loaded {len(authentic_texts)} texts from authentic sources")
        all_texts.extend(authentic_texts)
    
    # Calculate statistics
    total_texts = len(all_texts)
    total_chars = sum(len(t.get('text', '')) for t in all_texts)
    estimated_verses = total_chars // 80  # Average verse ~80 chars
    
    # Source breakdown
    if all_texts:
        sources = Counter([t.get('source', 'unknown') for t in all_texts])
        
        print("\n" + "=" * 70)
        print("📊 CORPUS STATISTICS")
        print("=" * 70)
        print(f"\nTotal texts: {total_texts:,}")
        print(f"Total characters: {total_chars:,}")
        print(f"Estimated verses: {estimated_verses:,}")
        
        print("\nBreakdown by source:")
        for source, count in sources.most_common():
            print(f"  • {source}: {count:,} texts")
        
        # Progress
        target = 837000
        progress = (estimated_verses / target) * 100
        print(f"\n{'=' * 70}")
        print(f"Progress toward 837,000 verses: {progress:.2f}%")
        print(f"Still needed: {target - estimated_verses:,} verses")
        print("=" * 70)
    
    # Save master corpus
    output_dir = Path("data/master_corpus")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "MASTER_SANSKRIT_CORPUS.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Master corpus saved: {output_file}")
    print(f"   Ready for training!")
    
    # Create training-ready format
    training_file = output_dir / "training_texts.txt"
    with open(training_file, 'w', encoding='utf-8') as f:
        for i, text in enumerate(all_texts, 1):
            content = text.get('text', '')
            if content.strip():
                f.write(content.strip() + '\n\n')
    
    print(f"\n✅ Training file saved: {training_file}")
    print(f"   {total_texts:,} texts ready for training")
    
    return all_texts

if __name__ == "__main__":
    combine_all_sources()
