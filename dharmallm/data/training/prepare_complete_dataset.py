#!/usr/bin/env python3
"""
Prepare Complete Training Dataset
==================================
Combines ALL extracted passages into unified training dataset:
- Previous 680 Rishi samples
- Previous 1,047 extracted passages  
- New 2,631 complete canon passages
= 4,358+ total samples (6.4x original dataset!)

Creates train/val split with proper categorization
"""

import json
from pathlib import Path
from collections import Counter
import random

# Paths
BASE_DIR = Path(__file__).parent.parent
EXTRACTED_PASSAGES = BASE_DIR / 'extracted_passages'
OUTPUT_DIR = BASE_DIR / 'training' / 'foundation_complete'

# Input files
RISHI_TRAIN = BASE_DIR / 'training' / 'foundation' / 'train.jsonl'
RISHI_VAL = BASE_DIR / 'training' / 'foundation' / 'val.jsonl'
EXTRACTED_1 = EXTRACTED_PASSAGES / 'all_extracted_passages.jsonl'
EXTRACTED_2 = EXTRACTED_PASSAGES / 'complete_canon_passages.jsonl'

# Output files
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = OUTPUT_DIR / 'train.jsonl'
VAL_FILE = OUTPUT_DIR / 'val.jsonl'

# Set random seed for reproducibility
random.seed(42)


def load_jsonl(file_path):
    """Load JSONL file"""
    samples = []
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return samples
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                samples.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return samples


def normalize_sample(sample, source='unknown'):
    """Normalize sample to common format"""
    # Extract text
    if 'text' in sample:
        text = sample['text']
    elif 'content' in sample:
        text = sample['content']
    else:
        return None
    
    # Clean text
    text = text.strip()
    if len(text) < 50:
        return None
    
    # Determine category and subcategory
    if 'category' in sample:
        category = sample['category']
        subcategory = sample.get('subcategory', 'general')
    elif 'scripture_type' in sample:
        scripture_type = sample['scripture_type']
        if scripture_type in ['veda', 'rigveda', 'samaveda', 'yajurveda', 'atharvaveda']:
            category = 'shruti_veda'
            subcategory = scripture_type
        elif scripture_type == 'upanishad':
            category = 'shruti_upanishad'
            subcategory = 'upanishad'
        elif scripture_type == 'bhagavad_gita':
            category = 'smriti_gita'
            subcategory = 'bhagavad_gita'
        elif scripture_type == 'itihasa':
            category = 'itihasa'
            subcategory = 'epic'
        elif scripture_type == 'purana':
            category = 'purana'
            subcategory = 'purana'
        elif scripture_type == 'dharma_shastra':
            category = 'smriti_dharma'
            subcategory = 'dharma'
        else:
            category = 'other'
            subcategory = scripture_type
    else:
        # From original Rishi dataset
        category = sample.get('category', 'general')
        subcategory = sample.get('source_type', 'rishi')
    
    # Get Rishi assignment if available
    rishi = sample.get('rishi', 'vashishta')
    
    # Create context prefix based on category
    if category.startswith('shruti_veda'):
        prefix = f"[Sacred Veda - {subcategory.title()}]"
    elif category.startswith('shruti_upanishad'):
        prefix = f"[Upanishad - Philosophical Teaching]"
    elif category.startswith('smriti_gita'):
        prefix = f"[Bhagavad Gita - Divine Discourse]"
    elif category.startswith('smriti_dharma'):
        prefix = f"[Dharma Shastra - Righteous Conduct]"
    elif 'itihasa' in category:
        prefix = f"[Itihasa - Epic Narrative]"
    elif 'purana' in category:
        prefix = f"[Purana - Ancient Chronicles]"
    elif 'darshana' in category:
        prefix = f"[Darshana - Philosophical System]"
    elif 'agama' in category or 'tantra' in category:
        prefix = f"[Agama/Tantra - Esoteric Teaching]"
    elif 'yoga' in category:
        prefix = f"[Yoga - Spiritual Practice]"
    else:
        prefix = f"[Dharmic Teaching]"
    
    # Combine prefix with text
    full_text = f"{prefix}\n{text}"
    
    return {
        'text': full_text,
        'category': category,
        'subcategory': subcategory,
        'rishi': rishi,
        'source': source,
        'length': len(full_text)
    }


def prepare_complete_dataset():
    """Prepare complete training dataset"""
    print("🕉️  PREPARE COMPLETE TRAINING DATASET")
    print("=" * 60)
    
    all_samples = []
    
    # Load original Rishi training data
    print("\n📚 Loading original Rishi dataset...")
    rishi_train = load_jsonl(RISHI_TRAIN)
    rishi_val = load_jsonl(RISHI_VAL)
    rishi_samples = rishi_train + rishi_val
    
    for sample in rishi_samples:
        normalized = normalize_sample(sample, source='rishi_original')
        if normalized:
            all_samples.append(normalized)
    
    print(f"  ✅ Loaded {len(all_samples)} Rishi samples")
    
    # Load first extraction (1,047 passages)
    print("\n📚 Loading first extraction (1,047 passages)...")
    extracted_1 = load_jsonl(EXTRACTED_1)
    count_before = len(all_samples)
    
    for sample in extracted_1:
        normalized = normalize_sample(sample, source='extraction_1')
        if normalized:
            all_samples.append(normalized)
    
    print(f"  ✅ Loaded {len(all_samples) - count_before} passages from first extraction")
    
    # Load complete canon extraction (2,631 passages)
    print("\n📚 Loading complete canon extraction (2,631 passages)...")
    extracted_2 = load_jsonl(EXTRACTED_2)
    count_before = len(all_samples)
    
    for sample in extracted_2:
        normalized = normalize_sample(sample, source='complete_canon')
        if normalized:
            all_samples.append(normalized)
    
    print(f"  ✅ Loaded {len(all_samples) - count_before} passages from complete canon")
    
    # Deduplicate by text content
    print("\n🔄 Deduplicating samples...")
    seen_texts = set()
    deduplicated = []
    
    for sample in all_samples:
        text_key = sample['text'].lower()[:200]  # Use first 200 chars as key
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            deduplicated.append(sample)
    
    print(f"  ✅ Removed {len(all_samples) - len(deduplicated)} duplicates")
    all_samples = deduplicated
    
    # Filter by quality
    print("\n🔍 Filtering by quality...")
    filtered = []
    
    for sample in all_samples:
        text = sample['text']
        
        # Length check
        if len(text) < 50 or len(text) > 1500:
            continue
        
        # Word count check
        words = text.split()
        if len(words) < 5 or len(words) > 300:
            continue
        
        # Alpha ratio check (at least 60% letters/spaces)
        alpha_count = sum(c.isalpha() or c.isspace() for c in text)
        if alpha_count / len(text) < 0.6:
            continue
        
        filtered.append(sample)
    
    print(f"  ✅ Kept {len(filtered)} quality samples")
    all_samples = filtered
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Split train/val (90/10)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Calculate statistics
    print("\n📊 Dataset Statistics:")
    print("=" * 60)
    print(f"Total samples:       {len(all_samples):6d}")
    print(f"Training samples:    {len(train_samples):6d} (90%)")
    print(f"Validation samples:  {len(val_samples):6d} (10%)")
    print()
    
    # Category statistics
    print("By Category:")
    categories = Counter(s['category'] for s in all_samples)
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(all_samples)) * 100
        print(f"  {cat:30s}: {count:6d} ({pct:5.1f}%)")
    
    print()
    
    # Source statistics
    print("By Source:")
    sources = Counter(s['source'] for s in all_samples)
    for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(all_samples)) * 100
        print(f"  {src:30s}: {count:6d} ({pct:5.1f}%)")
    
    # Save train set
    print("\n💾 Saving training set...")
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  ✅ Saved to: {TRAIN_FILE}")
    
    # Save val set
    print("\n💾 Saving validation set...")
    with open(VAL_FILE, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  ✅ Saved to: {VAL_FILE}")
    
    print("\n✅ Complete dataset prepared!")
    print(f"\n🎯 Dataset expanded from 680 → {len(all_samples)} samples")
    print(f"   That's {len(all_samples) / 680:.1f}x the original size!")
    print()
    print("🎯 Next Step:")
    print("   python3 training/train_foundation_complete.py")
    print()


if __name__ == '__main__':
    prepare_complete_dataset()
