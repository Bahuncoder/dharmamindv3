#!/usr/bin/env python3
"""
Test the trained Sanskrit master corpus model
Generate sample text and evaluate quality
"""

import torch
import json
from pathlib import Path
import sys


class SanskritTransformerModel(torch.nn.Module):
    """Same architecture as training"""
    
    def __init__(self, vocab_size, hidden_size=768, num_layers=8, nhead=12):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = torch.nn.Parameter(
            torch.randn(1, 1024, hidden_size)
        )
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=3072,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output = torch.nn.Linear(hidden_size, vocab_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.transformer(x)
        return self.output(x)


class ModelTester:
    """Test the trained model"""
    
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        self.model = None
        self.vocab = None
        self.char_to_idx = None
        self.idx_to_char = None
        
    def load_model(self, checkpoint_path):
        """Load trained model"""
        print(f"\n{'='*70}")
        print("🔄 LOADING MODEL")
        print(f"{'='*70}")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"📦 Loading checkpoint: {checkpoint_path}")
        print(f"   Size: {checkpoint_path.stat().st_size / 1024**2:.1f} MB")
        
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device
            )
            
            # Load vocabulary
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_char = checkpoint['idx_to_char']
            self.vocab = list(self.char_to_idx.keys())
            
            vocab_size = checkpoint.get('vocab_size', len(self.vocab))
            print(f"\n✓ Vocabulary loaded: {vocab_size:,} characters")
            
            # Create model
            self.model = SanskritTransformerModel(
                vocab_size=vocab_size,
                hidden_size=768,
                num_layers=8,
                nhead=12
            ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✓ Model loaded successfully")
            print(f"   Architecture: 8-layer Transformer")
            print(f"   Hidden dim: 768")
            print(f"   Attention heads: 12")
            print(f"   Parameters: ~50-60M")
            
            # Print training info
            if 'epoch' in checkpoint:
                print(f"\n📊 Training Info:")
                print(f"   Epoch: {checkpoint['epoch']}")
                print(f"   Loss: {checkpoint.get('loss', 'N/A'):.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_text(self, seed_text, length=200, temperature=0.8):
        """Generate Sanskrit text"""
        if self.model is None:
            print("❌ Model not loaded")
            return None
        
        print(f"\n{'='*70}")
        print("✨ GENERATING TEXT")
        print(f"{'='*70}")
        print(f"Seed: {seed_text}")
        print(f"Length: {length} characters")
        print(f"Temperature: {temperature}")
        print()
        
        # Encode seed
        encoded = [
            self.char_to_idx.get(c, self.char_to_idx.get('<UNK>', 0))
            for c in seed_text
        ]
        
        generated = encoded.copy()
        
        with torch.no_grad():
            for i in range(length):
                # Take last 512 characters (model's context window)
                context = generated[-512:]
                x = torch.tensor([context]).to(self.device)
                
                # Get prediction
                output = self.model(x)
                logits = output[0, -1, :] / temperature
                
                # Sample from distribution
                probs = torch.softmax(logits, dim=0)
                next_idx = torch.multinomial(probs, 1).item()
                
                generated.append(next_idx)
                
                # Show progress
                if (i + 1) % 50 == 0:
                    print(f"Generated {i + 1}/{length} characters...",
                          end='\r')
        
        # Decode
        generated_text = ''.join([
            self.idx_to_char.get(idx, '')
            for idx in generated
        ])
        
        print(f"\n\n{'='*70}")
        print("📝 GENERATED TEXT:")
        print(f"{'='*70}\n")
        print(generated_text)
        print(f"\n{'='*70}\n")
        
        return generated_text
    
    def test_multiple_seeds(self):
        """Test with multiple seed texts"""
        seeds = [
            "ॐ",
            "श्री",
            "धर्म",
            "योग",
            "ब्रह्म",
            "सत्यम्",
            "शान्ति",
            "ज्ञानम्"
        ]
        
        print(f"\n{'='*70}")
        print("🧪 TESTING MULTIPLE SEEDS")
        print(f"{'='*70}\n")
        
        results = []
        for seed in seeds:
            print(f"\n{'─'*70}")
            print(f"Seed: {seed}")
            print(f"{'─'*70}")
            
            generated = self.generate_text(
                seed_text=seed,
                length=100,
                temperature=0.8
            )
            
            results.append({
                'seed': seed,
                'generated': generated
            })
        
        return results
    
    def analyze_generation(self, text):
        """Analyze generated text quality"""
        print(f"\n{'='*70}")
        print("📊 QUALITY ANALYSIS")
        print(f"{'='*70}\n")
        
        # Count Sanskrit characters
        sanskrit_chars = sum(
            1 for c in text if '\u0900' <= c <= '\u097F'
        )
        total_chars = len(text)
        sanskrit_ratio = sanskrit_chars / total_chars if total_chars > 0 else 0
        
        # Count words (approximate)
        words = text.split()
        word_count = len(words)
        
        # Check for repeated patterns
        has_repetition = any(
            text.count(text[i:i+10]) > 3
            for i in range(0, len(text)-10, 5)
        )
        
        print(f"Total characters: {total_chars:,}")
        print(f"Sanskrit (Devanagari) characters: {sanskrit_chars:,}")
        print(f"Sanskrit ratio: {sanskrit_ratio:.1%}")
        print(f"Approximate words: {word_count:,}")
        print(f"Repetition detected: {'Yes' if has_repetition else 'No'}")
        
        # Scoring
        score = 0
        if sanskrit_ratio > 0.8:
            score += 30
            print("✓ Good Sanskrit content (80%+)")
        elif sanskrit_ratio > 0.5:
            score += 20
            print("⚠ Moderate Sanskrit content (50-80%)")
        else:
            print("✗ Low Sanskrit content (<50%)")
        
        if word_count > 10:
            score += 20
            print("✓ Good word count")
        
        if not has_repetition:
            score += 30
            print("✓ No excessive repetition")
        else:
            print("⚠ Some repetition detected")
        
        # Diversity
        unique_chars = len(set(text))
        char_diversity = unique_chars / len(self.vocab) if self.vocab else 0
        if char_diversity > 0.3:
            score += 20
            print(f"✓ Good character diversity ({char_diversity:.1%})")
        
        print(f"\n{'='*70}")
        print(f"OVERALL SCORE: {score}/100")
        print(f"{'='*70}\n")
        
        if score >= 80:
            print("🌟 EXCELLENT - Model generating high-quality Sanskrit!")
        elif score >= 60:
            print("👍 GOOD - Model performing well, minor improvements possible")
        elif score >= 40:
            print("⚠️  FAIR - Model working but needs improvement")
        else:
            print("❌ POOR - Model needs retraining or debugging")
        
        return score


def main():
    print("="*70)
    print("🕉️  SANSKRIT MODEL TESTING")
    print("="*70)
    
    tester = ModelTester()
    
    # Try both checkpoints
    checkpoints = [
        Path("model/checkpoints/best_model_epoch1.pt"),
        Path("model/checkpoints/final_model.pt")
    ]
    
    loaded = False
    for checkpoint in checkpoints:
        if checkpoint.exists():
            print(f"\nFound checkpoint: {checkpoint}")
            if tester.load_model(checkpoint):
                loaded = True
                break
    
    if not loaded:
        print("\n❌ No valid checkpoint found!")
        print("Expected locations:")
        for cp in checkpoints:
            print(f"  - {cp}")
        sys.exit(1)
    
    # Test 1: Generate from Om
    print("\n" + "="*70)
    print("TEST 1: Generate from ॐ (Om)")
    print("="*70)
    
    text1 = tester.generate_text(
        seed_text="ॐ ",
        length=300,
        temperature=0.8
    )
    
    if text1:
        score1 = tester.analyze_generation(text1)
    
    # Test 2: Generate from Dharma
    print("\n" + "="*70)
    print("TEST 2: Generate from धर्म (Dharma)")
    print("="*70)
    
    text2 = tester.generate_text(
        seed_text="धर्म",
        length=300,
        temperature=0.7
    )
    
    if text2:
        score2 = tester.analyze_generation(text2)
    
    # Test 3: Generate from Yoga
    print("\n" + "="*70)
    print("TEST 3: Generate from योग (Yoga)")
    print("="*70)
    
    text3 = tester.generate_text(
        seed_text="योग",
        length=300,
        temperature=0.9
    )
    
    if text3:
        score3 = tester.analyze_generation(text3)
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    
    if text1 and text2 and text3:
        avg_score = (score1 + score2 + score3) / 3
        print(f"\nAverage Score: {avg_score:.1f}/100")
        print(f"\nTest 1 (Om): {score1}/100")
        print(f"Test 2 (Dharma): {score2}/100")
        print(f"Test 3 (Yoga): {score3}/100")
        
        if avg_score >= 80:
            print("\n🎉 MODEL IS EXCELLENT!")
            print("Ready for production deployment")
        elif avg_score >= 60:
            print("\n✅ MODEL IS GOOD!")
            print("Can be used with monitoring")
        elif avg_score >= 40:
            print("\n⚠️  MODEL IS FAIR")
            print("Consider fine-tuning or more training")
        else:
            print("\n❌ MODEL NEEDS IMPROVEMENT")
            print("Consider retraining with adjusted parameters")
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
