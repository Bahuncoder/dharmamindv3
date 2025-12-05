#!/usr/bin/env python3
"""
🕉️ DharmaLLM MASTER CORPUS Training
====================================

Training on 29,099 verses from:
- Vedic Heritage: 1,004 texts (4 Vedas + 113 Upanishads)
- Gita Supersite: 701 verses (Complete Bhagavad Gita)
- Wisdom Library: 22 texts

All authentic original Sanskrit in Devanagari!
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import time

# Color codes
class C:
    G = '\033[92m'  # Green
    Y = '\033[93m'  # Yellow
    B = '\033[94m'  # Blue
    R = '\033[91m'  # Red
    M = '\033[95m'  # Magenta
    C = '\033[96m'  # Cyan
    E = '\033[0m'   # End


class SanskritTransformerModel(nn.Module):
    """Transformer model optimized for Sanskrit Devanagari"""
    
    def __init__(self, vocab_size=10000, hidden_size=768, num_layers=8, nhead=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, hidden_size))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=3072,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.transformer(x)
        return self.output(x)


class MasterCorpusTrainer:
    """Train model on complete master corpus"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.training_start = None
        
    def load_master_corpus(self):
        """Load the master Sanskrit corpus"""
        print(f"\n{C.M}{'='*70}{C.E}")
        print(f"{C.M}🕉️  LOADING MASTER SANSKRIT CORPUS{C.E}")
        print(f"{C.M}{'='*70}{C.E}\n")
        
        corpus_path = Path("data/master_corpus/MASTER_SANSKRIT_CORPUS.json")
        
        if not corpus_path.exists():
            print(f"{C.R}❌ Master corpus not found!{C.E}")
            print(f"   Expected: {corpus_path}")
            print(f"   Run: python3 combine_all_new_downloads.py first")
            return []
        
        print(f"{C.Y}Loading from: {corpus_path}{C.E}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        source_counts = {}
        
        # Extract texts and count by source
        for item in data:
            text = item.get('text', '')
            if text and len(text) > 20:  # Minimum length check
                texts.append(text)
                source = item.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
        
        # Stats
        total_chars = sum(len(t) for t in texts)
        avg_length = total_chars // len(texts) if texts else 0
        
        print(f"\n{C.G}✅ CORPUS LOADED SUCCESSFULLY{C.E}")
        print(f"{C.C}{'─'*70}{C.E}")
        print(f"{C.Y}📊 Statistics:{C.E}")
        print(f"   Total texts: {len(texts):,}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average text length: {avg_length} chars")
        print(f"   Estimated verses: ~{total_chars // 80:,}")
        
        print(f"\n{C.Y}📚 Sources:{C.E}")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {source}: {count:,} texts")
        
        print(f"{C.C}{'─'*70}{C.E}")
        print(f"{C.B}💻 Device: {self.device}{C.E}")
        if torch.cuda.is_available():
            print(f"{C.B}🎮 GPU: {torch.cuda.get_device_name(0)}{C.E}")
            print(f"{C.B}📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB{C.E}")
        
        return texts
    
    def build_vocabulary(self, texts: List[str]):
        """Build character-level vocabulary for Sanskrit"""
        print(f"\n{C.Y}🔤 Building Sanskrit Vocabulary...{C.E}")
        
        # Collect all unique characters
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>', '<MASK>']
        
        # Build vocab
        vocab = special_tokens + sorted(list(chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(vocab)
        
        # Count Devanagari characters (Unicode range U+0900 to U+097F)
        devanagari = [c for c in chars if '\u0900' <= c <= '\u097F']
        
        print(f"{C.G}✓ Vocabulary built successfully{C.E}")
        print(f"   Total characters: {self.vocab_size}")
        print(f"   • Devanagari: {len(devanagari)} chars")
        print(f"   • Special tokens: {len(special_tokens)}")
        print(f"   • Other: {self.vocab_size - len(devanagari) - len(special_tokens)}")
        
        # Show sample Devanagari
        if devanagari:
            sample = ''.join(sorted(devanagari)[:30])
            print(f"   Sample: {sample}...")
    
    def encode_text(self, text: str, max_len: int = 256) -> torch.Tensor:
        """Encode Sanskrit text to character indices"""
        indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) 
                   for char in text[:max_len]]
        
        # Pad if needed
        if len(indices) < max_len:
            indices += [self.char_to_idx['<PAD>']] * (max_len - len(indices))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def create_batches(self, texts: List[str], batch_size: int = 16):
        """Create training batches"""
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch = torch.stack([self.encode_text(text) for text in batch_texts])
            batches.append(batch)
        return batches
    
    def save_checkpoint(self, model, optimizer, epoch, loss, path):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'vocab_size': self.vocab_size,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, path)
    
    def train_model(self, texts: List[str], epochs: int = 20, batch_size: int = 16, 
                    learning_rate: float = 0.0003):
        """Train the Sanskrit model"""
        self.training_start = time.time()
        
        print(f"\n{C.M}{'='*70}{C.E}")
        print(f"{C.M}🚀 STARTING TRAINING{C.E}")
        print(f"{C.M}{'='*70}{C.E}\n")
        
        # Build vocabulary
        self.build_vocabulary(texts)
        
        # Initialize model
        print(f"\n{C.Y}🏗️  Initializing Model...{C.E}")
        model = SanskritTransformerModel(
            vocab_size=self.vocab_size,
            hidden_size=768,
            num_layers=8,
            nhead=12
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / 1024 / 1024
        
        print(f"{C.G}✅ Model Initialized{C.E}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{model_size_mb:.1f} MB")
        print(f"   Architecture: 8-layer Transformer (768-dim, 12-head)")
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss(ignore_index=self.char_to_idx['<PAD>'])
        
        # Create batches
        print(f"\n{C.Y}📦 Creating Training Batches...{C.E}")
        batches = self.create_batches(texts, batch_size)
        print(f"{C.G}✓ Created {len(batches)} batches (batch_size={batch_size}){C.E}")
        
        # Create output directory
        output_dir = Path("model/checkpoints")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        print(f"\n{C.M}{'='*70}{C.E}")
        print(f"{C.M}⚡ TRAINING IN PROGRESS{C.E}")
        print(f"{C.M}{'='*70}{C.E}\n")
        
        print(f"{C.C}Configuration:{C.E}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Total batches per epoch: {len(batches)}")
        print(f"   Total training steps: {epochs * len(batches)}")
        print(f"\n{C.C}{'─'*70}{C.E}\n")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            
            print(f"{C.Y}Epoch {epoch+1}/{epochs}{C.E}")
            
            for batch_idx, batch in enumerate(batches):
                batch = batch.to(self.device)
                
                # Create input and target (shifted by 1)
                input_seq = batch[:, :-1]
                target_seq = batch[:, 1:]
                
                # Forward pass
                optimizer.zero_grad()
                output = model(input_seq)
                
                # Calculate loss
                loss = criterion(
                    output.reshape(-1, self.vocab_size),
                    target_seq.reshape(-1)
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Progress update every 50 batches
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    elapsed = time.time() - self.training_start
                    print(f"   Batch {batch_idx+1}/{len(batches)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Avg Loss: {avg_loss:.4f} | "
                          f"Time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
            
            # Epoch stats
            scheduler.step()
            avg_loss = total_loss / len(batches)
            epoch_time = time.time() - epoch_start
            total_time = time.time() - self.training_start
            
            print(f"\n{C.G}✅ Epoch {epoch+1} Complete{C.E}")
            print(f"   Average Loss: {avg_loss:.4f}")
            print(f"   Epoch Time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
            print(f"   Total Time: {total_time//60:.0f}m {total_time%60:.0f}s")
            print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = output_dir / "best_model.pt"  # Single best file
                self.save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
                print(f"   {C.M}💾 Saved NEW BEST checkpoint (epoch {epoch+1}): {checkpoint_path}{C.E}")
                print(f"   {C.G}🌟 Best loss so far: {best_loss:.4f}{C.E}")
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
                print(f"   {C.B}💾 Saved checkpoint: {checkpoint_path}{C.E}")
            
            print(f"{C.C}{'─'*70}{C.E}\n")
        
        # Final save
        final_path = output_dir / "final_model.pt"
        self.save_checkpoint(model, optimizer, epochs, avg_loss, final_path)
        
        print(f"\n{C.M}{'='*70}{C.E}")
        print(f"{C.M}🎉 TRAINING COMPLETE!{C.E}")
        print(f"{C.M}{'='*70}{C.E}\n")
        
        total_time = time.time() - self.training_start
        print(f"{C.G}Final Statistics:{C.E}")
        print(f"   Best Loss: {best_loss:.4f}")
        print(f"   Final Loss: {avg_loss:.4f}")
        print(f"   Improvement: {(1 - avg_loss/best_loss)*100:.1f}%")
        print(f"   Total Time: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m")
        print(f"   Model saved: {final_path}")
        print(f"\n{C.C}{'─'*70}{C.E}\n")


def main():
    """Main training function"""
    trainer = MasterCorpusTrainer()
    
    # Load corpus
    texts = trainer.load_master_corpus()
    
    if not texts:
        print(f"\n{C.R}❌ No texts to train on!{C.E}")
        return
    
    # Train with improved hyperparameters
    print(f"\n{C.Y}IMPROVED TRAINING CONFIGURATION:{C.E}")
    print(f"   Epochs: 100 (was 20)")
    print(f"   Batch Size: 8 (was 16, reduced for GPU)")
    print(f"   Learning Rate: 0.0001 (was 0.0003, more stable)")
    print(f"   Expected time: 3-6 hours\n")
    
    trainer.train_model(
        texts=texts,
        epochs=100,  # More epochs for better learning
        batch_size=8,  # Smaller batch to avoid OOM
        learning_rate=0.0001  # Lower LR for stability
    )


if __name__ == "__main__":
    main()
