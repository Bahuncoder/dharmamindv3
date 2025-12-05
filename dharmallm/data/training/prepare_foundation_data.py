#!/usr/bin/env python3
"""
Prepare Foundation Training Data
=================================
Convert extracted Rishi passages to base training format.
No Rishi-specific labels yet - just get the LLM working first.

Author: DharmaMind Team
Date: November 3, 2025
"""

import json
from pathlib import Path
import random

def prepare_training_data():
    """Convert Rishi passages to base training data."""
    
    rishi_dir = Path("data/rishi_training")
    output_dir = Path("data/training/foundation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_passages = []
    stats = {'by_rishi': {}, 'by_source': {}}
    
    print("🕉️ Preparing Foundation Training Data")
    print("="*70)
    print("Loading passages from all Rishis...")
    
    # Load all Rishi passages
    for rishi_folder in rishi_dir.iterdir():
        if not rishi_folder.is_dir():
            continue
        
        rishi_name = rishi_folder.name
        stats['by_rishi'][rishi_name] = 0
        
        for jsonl_file in rishi_folder.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        
                        # Extract text
                        text = data.get('text', '')
                        source = data.get('source', 'unknown')
                        source_type = data.get('source_type', 'unknown')
                        rishi = data.get('rishi', rishi_name)
                        
                        # Filter: minimum length
                        if len(text) < 50:
                            continue
                        
                        # Filter: must have some Sanskrit or meaningful content
                        if text.count(' ') < 5:  # Too short
                            continue
                        
                        # Convert to training format
                        training_sample = {
                            'text': text,
                            'rishi': rishi,
                            'source': source,
                            'source_type': source_type,
                            'category': 'dharmic_text'
                        }
                        all_passages.append(training_sample)
                        
                        # Statistics
                        stats['by_rishi'][rishi_name] += 1
                        stats['by_source'][source_type] = stats['by_source'].get(source_type, 0) + 1
                    
                    except json.JSONDecodeError:
                        continue
    
    print(f"\n✅ Loaded {len(all_passages)} passages")
    print("\nBy Rishi:")
    for rishi, count in sorted(stats['by_rishi'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {rishi:15s}: {count:4d} passages")
    
    print("\nBy Source Type:")
    for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {source:20s}: {count:4d} passages")
    
    # Split train/val (90/10)
    random.seed(42)  # Reproducible split
    random.shuffle(all_passages)
    split_idx = int(len(all_passages) * 0.9)
    
    train_data = all_passages[:split_idx]
    val_data = all_passages[split_idx:]
    
    # Save
    train_file = output_dir / 'train.jsonl'
    val_file = output_dir / 'val.jsonl'
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print("\n" + "="*70)
    print("✅ Training data prepared!")
    print("="*70)
    print(f"  Train: {len(train_data):4d} samples → {train_file}")
    print(f"  Val:   {len(val_data):4d} samples → {val_file}")
    print(f"  Total: {len(all_passages):4d} samples")
    print("="*70)
    print("\n🚀 Next step: Train the foundation model")
    print("   python3 training/train_foundation_llm.py")


if __name__ == "__main__":
    prepare_training_data()
