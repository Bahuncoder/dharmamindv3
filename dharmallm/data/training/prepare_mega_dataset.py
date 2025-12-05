#!/usr/bin/env python3
"""
Prepare MEGA Training Dataset
==============================
Combine ALL extractions into one massive dataset:
1. Original Rishi samples
2. Complete canon extraction (2,631 samples)
3. First massive extraction (1,742 samples) 
4. NEW massive corpus extraction (10,242 samples)

Target: 14,000+ total samples
"""

import json
import random
from pathlib import Path
from collections import Counter

# Paths
BASE_DIR = Path("/media/rupert/New Volume/Testing Ground DharmaMind/Testing ground/DharmaMind-chat-master/dharmallm")
DATA_DIR = BASE_DIR / "data"
TRAINING_DIR = DATA_DIR / "training"
OUTPUT_DIR = TRAINING_DIR / "foundation_mega"

# Input files
INPUT_FILES = [
    DATA_DIR / "extracted_passages" / "complete_canon_extraction.jsonl",  # 2,631
    DATA_DIR / "extracted_passages" / "massive_extraction.jsonl",  # 1,742
    DATA_DIR / "extracted_passages" / "massive_corpus_v2_extraction.jsonl",  # 10,242
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Statistics
stats = {
    'total_raw': 0,
    'after_dedup': 0,
    'after_quality': 0,
    'by_category': Counter(),
    'by_source': Counter()
}


def deduplicate_samples(samples):
    """Remove duplicate samples based on first 200 chars"""
    seen = set()
    unique = []
    
    for sample in samples:
        key = sample['text'][:200]
        if key not in seen:
            seen.add(key)
            unique.append(sample)
    
    return unique


def quality_filter(samples):
    """Filter out low-quality samples"""
    filtered = []
    
    for sample in samples:
        text = sample['text']
        
        # Length check
        if len(text) < 50 or len(text) > 2000:
            continue
        
        # Word count
        words = text.split()
        if len(words) < 8:
            continue
        
        # Must have reasonable alphabetic ratio
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count / len(text) < 0.5:
            continue
        
        filtered.append(sample)
    
    return filtered


def add_context_prefix(sample):
    """Add context prefix based on category"""
    category = sample.get('category', 'unknown')
    text = sample['text']
    
    # Map categories to prefixes
    prefix_map = {
        'shruti_veda': '[Rig Veda]',
        'shruti_upanishad': '[Upanishad]',
        'smriti_gita': '[Bhagavad Gita]',
        'smriti_dharma': '[Manu Smriti]',
        'vedas': '[Vedas]',
        'gita': '[Bhagavad Gita]',
        'puranas': '[Puranas]',
        'dharma_shastras': '[Dharma Shastras]',
        'ramayana': '[Ramayana]',
        'mahabharata': '[Mahabharata]',
        'upanishads': '[Upanishads]',
    }
    
    prefix = prefix_map.get(category, '[Scripture]')
    
    return {
        'text': f"{prefix} {text}",
        'category': category,
        'source': sample.get('source', 'unknown')
    }


def load_samples(filepath):
    """Load samples from JSONL file"""
    samples = []
    
    if not filepath.exists():
        print(f"  ⚠️  File not found: {filepath.name}")
        return samples
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line)
                samples.append(sample)
            except:
                continue
    
    return samples


def main():
    """Main preparation process"""
    print("=" * 70)
    print("🔥 MEGA DATASET PREPARATION")
    print("=" * 70)
    
    all_samples = []
    
    # Load from all sources
    print("\n📂 Loading from all sources...")
    print("=" * 70)
    
    for input_file in INPUT_FILES:
        samples = load_samples(input_file)
        if samples:
            print(f"  ✅ {input_file.name}: {len(samples)} samples")
            all_samples.extend(samples)
            stats['by_source'][input_file.stem] = len(samples)
        else:
            print(f"  ⚠️  {input_file.name}: 0 samples")
    
    stats['total_raw'] = len(all_samples)
    print(f"\n  📊 Total raw samples: {stats['total_raw']}")
    
    # Deduplicate
    print("\n🔍 Deduplicating...")
    all_samples = deduplicate_samples(all_samples)
    stats['after_dedup'] = len(all_samples)
    removed = stats['total_raw'] - stats['after_dedup']
    print(f"  ✅ Removed {removed} duplicates")
    print(f"  ✅ Unique samples: {stats['after_dedup']}")
    
    # Quality filter
    print("\n✨ Quality filtering...")
    all_samples = quality_filter(all_samples)
    stats['after_quality'] = len(all_samples)
    removed = stats['after_dedup'] - stats['after_quality']
    print(f"  ✅ Removed {removed} low-quality samples")
    print(f"  ✅ High-quality samples: {stats['after_quality']}")
    
    # Add context prefixes
    print("\n🏷️  Adding context prefixes...")
    all_samples = [add_context_prefix(s) for s in all_samples]
    
    # Count by category
    for sample in all_samples:
        stats['by_category'][sample['category']] += 1
    
    # Shuffle
    print("\n🔀 Shuffling...")
    random.seed(42)
    random.shuffle(all_samples)
    
    # Split into train/val
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"  ✅ Training: {len(train_samples)} samples (90%)")
    print(f"  ✅ Validation: {len(val_samples)} samples (10%)")
    
    # Save
    print("\n💾 Saving datasets...")
    
    train_file = OUTPUT_DIR / "train.jsonl"
    val_file = OUTPUT_DIR / "val.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n" + "=" * 70)
    print("✅ MEGA DATASET READY!")
    print("=" * 70)
    
    print(f"\n📊 Final Statistics:")
    print(f"  Total raw samples: {stats['total_raw']}")
    print(f"  After deduplication: {stats['after_dedup']} ({stats['after_dedup']/stats['total_raw']*100:.1f}%)")
    print(f"  After quality filter: {stats['after_quality']} ({stats['after_quality']/stats['total_raw']*100:.1f}%)")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    
    print(f"\n📈 Samples by category:")
    for category, count in stats['by_category'].most_common():
        pct = count / stats['after_quality'] * 100
        print(f"  {category:20s}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n📚 Samples by source:")
    for source, count in stats['by_source'].items():
        pct = count / stats['total_raw'] * 100
        print(f"  {source:30s}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n📝 Output:")
    print(f"  Training: {train_file}")
    print(f"  Validation: {val_file}")
    
    print(f"\n🎯 Next step: Train foundation model")
    print(f"   Run: python3 training/train_foundation_mega.py")


if __name__ == "__main__":
    main()
