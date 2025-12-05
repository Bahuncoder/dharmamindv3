#!/usr/bin/env python3
"""
Prepare expanded training data from extracted passages
Combines existing + new extracted passages
"""

import json
from pathlib import Path
import random

print("🕉️ PREPARING EXPANDED TRAINING DATA")
print("="*70)

# Paths
existing_train = Path("data/training/foundation/train.jsonl")
existing_val = Path("data/training/foundation/val.jsonl")
extracted_passages = Path("data/extracted_passages/all_extracted_passages.jsonl")
output_dir = Path("data/training/foundation_expanded")
output_dir.mkdir(parents=True, exist_ok=True)

# Load existing data
print("\n📚 Loading existing training data...")
existing_samples = []

if existing_train.exists():
    with open(existing_train, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                existing_samples.append(json.loads(line))
    print(f"  Loaded {len(existing_samples)} existing training samples")

# Load new extracted passages
print("\n📖 Loading newly extracted passages...")
new_passages = []

with open(extracted_passages, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            text = data.get('text', '')
            
            # Filter quality
            if len(text) < 50 or len(text.split()) < 5:
                continue
            
            # Format as training sample
            source_type = data.get('source_type', 'other')
            formatted_text = f"[Dharmic Text from {source_type}]\n{text}"
            
            new_passages.append({
                'text': formatted_text,
                'source': data.get('source', 'unknown'),
                'source_type': source_type,
                'category': data.get('category', 'other')
            })

print(f"  Loaded {len(new_passages)} new passages")

# Combine all samples
print("\n🔄 Combining datasets...")
all_samples = existing_samples + new_passages
random.seed(42)
random.shuffle(all_samples)

print(f"  Total samples: {len(all_samples)}")

# Split 90/10 train/val
split_idx = int(0.9 * len(all_samples))
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

print(f"  Train: {len(train_samples)} samples")
print(f"  Val: {len(val_samples)} samples")

# Save expanded dataset
print("\n💾 Saving expanded dataset...")

train_file = output_dir / "train.jsonl"
with open(train_file, 'w', encoding='utf-8') as f:
    for sample in train_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

val_file = output_dir / "val.jsonl"
with open(val_file, 'w', encoding='utf-8') as f:
    for sample in val_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"  ✅ Train saved: {train_file}")
print(f"  ✅ Val saved: {val_file}")

# Statistics
print("\n" + "="*70)
print("📊 DATASET STATISTICS")
print("="*70)

# Count by source type
source_type_counts = {}
for sample in all_samples:
    stype = sample.get('source_type', 'unknown')
    source_type_counts[stype] = source_type_counts.get(stype, 0) + 1

print("\nBy Source Type:")
for stype, count in sorted(source_type_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {stype:20s}: {count:4d} samples")

print("\n" + "="*70)
print("🎯 READY FOR TRAINING!")
print("="*70)
print(f"\nExpanded dataset is {len(all_samples) / len(existing_samples):.1f}x larger!")
print(f"  Before: {len(existing_samples)} samples")
print(f"  After:  {len(all_samples)} samples")
print(f"\n💡 Next step: Train foundation model with expanded data!")
print("   python3 training/train_foundation_cpu.py --data foundation_expanded")
print("\n🕉️ Preparation complete!")
