#!/usr/bin/env python3
"""
Prepare MASSIVE Training Dataset
=================================
Combine ALL available data sources:
1. Original Rishi samples (756)
2. First extraction (1,047)  
3. Complete canon extraction (2,631)
4. NEW: Massive full-text extraction (1,742)

Total raw: ~6,176 samples → Filter to ~4,500-5,000+ quality samples
"""

import json
from pathlib import Path
from collections import defaultdict
import random

# Paths
BASE_DIR = Path("/media/rupert/New Volume/Testing Ground DharmaMind/Testing ground/DharmaMind-chat-master/dharmallm")
RISHI_DATA = BASE_DIR / "data" / "training" / "rishi_original"
EXTRACTION_1 = BASE_DIR / "data" / "extracted_passages" / "authentic_passages.jsonl"
COMPLETE_CANON = BASE_DIR / "data" / "extracted_passages" / "complete_canon_passages.jsonl"
MASSIVE_EXTRACTION = BASE_DIR / "data" / "extracted_passages" / "massive_extraction.jsonl"

OUTPUT_DIR = BASE_DIR / "data" / "training" / "foundation_massive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"

print("🕉️  PREPARING MASSIVE TRAINING DATASET")
print("=" * 70)
print("\n📊 Loading data from all sources...")

# Load all data sources
all_samples = []

# 1. Load original Rishi samples
print("\n📚 Loading original Rishi dataset...")
try:
    rishi_files = list((RISHI_DATA / "train").glob("*.txt")) if (RISHI_DATA / "train").exists() else []
    if not rishi_files and (BASE_DIR / "data" / "training" / "rishi_original.jsonl").exists():
        with open(BASE_DIR / "data" / "training" / "rishi_original.jsonl", 'r') as f:
            for line in f:
                data = json.loads(line)
                all_samples.append({
                    'text': data.get('text', ''),
                    'category': data.get('category', 'rishi'),
                    'subcategory': data.get('subcategory', 'original'),
                    'source': 'rishi_original'
                })
    print(f"  ✅ Loaded {len([s for s in all_samples if s['source'] == 'rishi_original'])} Rishi samples")
except Exception as e:
    print(f"  ⚠️  Could not load Rishi data: {e}")

# 2. Load first extraction
print("\n📚 Loading first extraction...")
if EXTRACTION_1.exists():
    try:
        with open(EXTRACTION_1, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_samples.append({
                    'text': data.get('text', ''),
                    'category': data.get('category', 'unknown'),
                    'subcategory': data.get('subcategory', ''),
                    'source': 'extraction_1'
                })
        print(f"  ✅ Loaded {len([s for s in all_samples if s['source'] == 'extraction_1'])} passages from first extraction")
    except Exception as e:
        print(f"  ⚠️  Could not load first extraction: {e}")

# 3. Load complete canon extraction
print("\n📚 Loading complete canon extraction...")
if COMPLETE_CANON.exists():
    try:
        with open(COMPLETE_CANON, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_samples.append({
                    'text': data.get('text', ''),
                    'category': data.get('category', 'unknown'),
                    'subcategory': data.get('subcategory', ''),
                    'source': 'complete_canon'
                })
        print(f"  ✅ Loaded {len([s for s in all_samples if s['source'] == 'complete_canon'])} passages from complete canon")
    except Exception as e:
        print(f"  ⚠️  Could not load complete canon: {e}")

# 4. Load NEW massive extraction
print("\n📚 Loading NEW massive full-text extraction...")
if MASSIVE_EXTRACTION.exists():
    try:
        with open(MASSIVE_EXTRACTION, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_samples.append({
                    'text': data.get('text', ''),
                    'category': data.get('category', 'unknown'),
                    'subcategory': data.get('subcategory', ''),
                    'source': 'massive_extraction'
                })
        print(f"  ✅ Loaded {len([s for s in all_samples if s['source'] == 'massive_extraction'])} passages from massive extraction")
    except Exception as e:
        print(f"  ⚠️  Could not load massive extraction: {e}")

print(f"\n✅ Total raw samples loaded: {len(all_samples)}")

# Deduplicate
print("\n🔄 Deduplicating samples...")
seen = set()
deduped_samples = []

for sample in all_samples:
    # Use first 200 chars as key
    key = sample['text'][:200].lower().strip()
    if key and key not in seen and len(key) > 50:
        seen.add(key)
        deduped_samples.append(sample)

print(f"  ✅ Removed {len(all_samples) - len(deduped_samples)} duplicates")
print(f"  ✅ Unique samples: {len(deduped_samples)}")

# Filter by quality
print("\n🔍 Filtering by quality...")
quality_samples = []

for sample in deduped_samples:
    text = sample['text']
    
    # Quality checks
    if not text or len(text) < 50:
        continue
    if len(text) > 2000:
        continue
    
    words = text.split()
    if len(words) < 8:
        continue
    
    # Check alpha ratio
    alpha_chars = sum(c.isalpha() or c.isspace() for c in text)
    alpha_ratio = alpha_chars / len(text) if len(text) > 0 else 0
    if alpha_ratio < 0.5:
        continue
    
    quality_samples.append(sample)

print(f"  ✅ Kept {len(quality_samples)} quality samples")
print(f"  ❌ Removed {len(deduped_samples) - len(quality_samples)} low-quality samples")

# Statistics
print("\n📊 Dataset statistics:")
by_category = defaultdict(int)
by_source = defaultdict(int)

for sample in quality_samples:
    by_category[sample['category']] += 1
    by_source[sample['source']] += 1

print(f"\nTotal samples: {len(quality_samples)}")
print(f"\nBy category:")
for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(quality_samples) * 100)
    print(f"  {cat:25s}: {count:5d} ({percentage:5.1f}%)")

print(f"\nBy source:")
for src, count in sorted(by_source.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(quality_samples) * 100)
    print(f"  {src:25s}: {count:5d} ({percentage:5.1f}%)")

# Shuffle and split
print("\n🔀 Shuffling and splitting dataset...")
random.seed(42)
random.shuffle(quality_samples)

train_size = int(len(quality_samples) * 0.9)
train_samples = quality_samples[:train_size]
val_samples = quality_samples[train_size:]

# Add context prefixes based on category
def add_context_prefix(sample):
    """Add context prefix to help model understand source"""
    category = sample['category']
    subcategory = sample['subcategory']
    text = sample['text']
    
    if category == 'vedas' or 'veda' in category.lower():
        if 'rigveda' in subcategory.lower():
            prefix = f"[Rig Veda] "
        elif 'samaveda' in subcategory.lower():
            prefix = f"[Sama Veda] "
        elif 'yajurveda' in subcategory.lower():
            prefix = f"[Yajur Veda] "
        elif 'atharvaveda' in subcategory.lower():
            prefix = f"[Atharva Veda] "
        else:
            prefix = f"[Sacred Veda] "
    elif 'gita' in category.lower():
        prefix = f"[Bhagavad Gita] "
    elif 'purana' in category.lower():
        if 'vishnu' in subcategory.lower():
            prefix = f"[Vishnu Purana] "
        elif 'bhagavata' in subcategory.lower():
            prefix = f"[Bhagavata Purana] "
        elif 'garuda' in subcategory.lower():
            prefix = f"[Garuda Purana] "
        else:
            prefix = f"[Purana] "
    elif 'dharma' in category.lower():
        prefix = f"[Dharma Shastra] "
    elif 'manu' in subcategory.lower():
        prefix = f"[Manu Smriti] "
    elif 'upanishad' in category.lower():
        prefix = f"[Upanishad] "
    elif 'mahabharata' in category.lower():
        prefix = f"[Mahabharata] "
    elif 'ramayana' in category.lower():
        prefix = f"[Ramayana] "
    else:
        prefix = f"[{category.title()}] "
    
    return prefix + text

# Write datasets
print(f"\n💾 Writing training dataset ({len(train_samples)} samples)...")
with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
    for sample in train_samples:
        output = {
            'text': add_context_prefix(sample),
            'category': sample['category'],
            'subcategory': sample['subcategory'],
            'source': sample['source']
        }
        f.write(json.dumps(output, ensure_ascii=False) + '\n')

print(f"💾 Writing validation dataset ({len(val_samples)} samples)...")
with open(VAL_FILE, 'w', encoding='utf-8') as f:
    for sample in val_samples:
        output = {
            'text': add_context_prefix(sample),
            'category': sample['category'],
            'subcategory': sample['subcategory'],
            'source': sample['source']
        }
        f.write(json.dumps(output, ensure_ascii=False) + '\n')

print("\n" + "=" * 70)
print("✅ MASSIVE DATASET PREPARATION COMPLETE!")
print("=" * 70)
print(f"\n📊 Final statistics:")
print(f"  Training samples: {len(train_samples)} (90%)")
print(f"  Validation samples: {len(val_samples)} (10%)")
print(f"  Total: {len(quality_samples)} samples")
print(f"\n📂 Output files:")
print(f"  Train: {TRAIN_FILE}")
print(f"  Val: {VAL_FILE}")
print(f"\n🎯 Next step: Train foundation model on massive dataset!")
print(f"   This is {len(quality_samples) / 2889 * 100:.1f}% more data than before!")
print(f"   Run: cp training/train_foundation_complete.py training/train_foundation_massive.py")
print(f"   Then modify paths and train!")
print()
