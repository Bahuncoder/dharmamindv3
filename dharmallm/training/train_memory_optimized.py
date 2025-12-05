#!/usr/bin/env python3
"""
Memory-Optimized Training for Complete 31-Module System

Optimizations for GTX 1650 (4GB VRAM):
- Batch size 1 with gradient accumulation
- Mixed precision (FP16) training
- Gradient checkpointing
- Efficient data loading
- Memory monitoring

Model: 225.1M params (61% spiritual)
Target GPU: GTX 1650 with 4GB VRAM
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import json
import time
from pathlib import Path
from datetime import datetime
import gc
import os

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.integrated_dharma_llm import IntegratedDharmaLLM, IntegratedDharmaLLMConfig
from transformers import AutoTokenizer


class OptimizedDharmicDataset(Dataset):
    """Memory-efficient dataset - loads samples on-demand"""
    
    def __init__(self, data_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load only indices, not full data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✅ Dataset loaded: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Load and tokenize on-demand to save memory"""
        item = self.data[idx]
        
        # Combine instruction and input
        text = item.get('instruction', '')
        if 'input' in item and item['input']:
            text += f"\n\n{item['input']}"
        
        # Add response
        response = item.get('output', item.get('response', ''))
        full_text = f"{text}\n\nResponse: {response}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def print_memory_stats():
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def train_memory_optimized(
    data_file='data/complete_hindu_database.json',
    output_dir='models/checkpoints',
    
    # Memory optimization settings
    batch_size=1,  # Small batch for 4GB GPU
    gradient_accumulation_steps=16,  # Effective batch of 16
    max_seq_length=128,  # Shorter sequences
    use_mixed_precision=True,  # FP16 training
    use_gradient_checkpointing=True,  # Trade compute for memory
    
    # Training hyperparameters
    learning_rate=5e-5,
    num_epochs=3,
    warmup_steps=100,
    max_grad_norm=1.0,
    
    # Saving
    save_steps=500,
    logging_steps=10,
    
    # Device
    device=None
):
    """
    Memory-optimized training for 31-module spiritual AI system
    
    Designed to fit 225.1M params in 4GB VRAM
    """
    
    print("=" * 80)
    print("🕉️  MEMORY-OPTIMIZED TRAINING - 31 SPIRITUAL MODULES")
    print("=" * 80)
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.2f}GB")
    
    # Initialize tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"\n📖 Loading dataset: {data_file}")
    dataset = OptimizedDharmicDataset(data_file, tokenizer, max_seq_length)
    
    # Create dataloader with memory-efficient settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing to save memory
        pin_memory=False  # Don't pin memory on small GPUs
    )
    
    # Initialize model
    print("\n🧠 Initializing model...")
    config = IntegratedDharmaLLMConfig()
    model = IntegratedDharmaLLM(config)
    
    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing and device.type == 'cuda':
        print("   ✓ Gradient checkpointing enabled")
        # Enable for base model
        if hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()
    
    model = model.to(device)
    
    # Print model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spiritual_params = sum(p.numel() for p in model.spiritual_modules.parameters())
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Spiritual Parameters: {spiritual_params:,}")
    print(f"   Spiritual Ratio: {spiritual_params/total_params*100:.2f}%")
    print_memory_stats()
    
    # Setup optimizer with memory-efficient settings
    print(f"\n⚙️  Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Setup mixed precision training
    scaler = GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    if scaler:
        print("   ✓ Mixed precision (FP16) enabled")
    
    # Setup learning rate scheduler
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, (total_steps - step) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training configuration
    print(f"\n🎯 Training Configuration:")
    print(f"   Batch Size: {batch_size}")
    print(f"   Gradient Accumulation: {gradient_accumulation_steps}")
    print(f"   Effective Batch Size: {batch_size * gradient_accumulation_steps}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Max Sequence Length: {max_seq_length}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Total Steps: {total_steps}")
    print(f"   Warmup Steps: {warmup_steps}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    
    model.train()
    global_step = 0
    total_loss = 0
    best_loss = float('inf')
    start_time = time.time()
    
    # Clear cache before training
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            epoch_loss = 0
            optimizer.zero_grad()
            
            for step, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass with mixed precision if enabled
                if scaler:
                    with autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        loss = outputs['loss'] / gradient_accumulation_steps
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    loss = outputs['loss'] / gradient_accumulation_steps
                    loss.backward()
                
                # Accumulate loss
                epoch_loss += loss.item() * gradient_accumulation_steps
                total_loss += loss.item() * gradient_accumulation_steps
                
                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Optimizer step
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Logging
                    if global_step % logging_steps == 0:
                        avg_loss = total_loss / (logging_steps * gradient_accumulation_steps)
                        lr = scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        
                        print(f"Step {global_step}/{total_steps} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LM Loss: {outputs['lm_loss'].item():.4f} | "
                              f"Spiritual Loss: {outputs['spiritual_loss'].item():.4f} | "
                              f"LR: {lr:.2e} | "
                              f"Time: {elapsed/60:.1f}min")
                        
                        print_memory_stats()
                        total_loss = 0
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        checkpoint_path = output_dir / f"checkpoint_step_{global_step}"
                        checkpoint_path.mkdir(parents=True, exist_ok=True)
                        
                        print(f"\n💾 Saving checkpoint: {checkpoint_path}")
                        torch.save({
                            'step': global_step,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': avg_loss,
                            'config': config.__dict__
                        }, checkpoint_path / 'checkpoint.pt')
                        
                        # Save best model
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            best_path = output_dir / 'best_model'
                            best_path.mkdir(parents=True, exist_ok=True)
                            torch.save(model.state_dict(), best_path / 'model.pt')
                            print(f"   ✓ Best model updated (loss: {best_loss:.4f})")
                
                # Memory cleanup every 100 steps
                if step % 100 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"\n✅ Epoch {epoch + 1} Complete")
            print(f"   Average Loss: {avg_epoch_loss:.4f}")
            print_memory_stats()
            
            # Save epoch checkpoint
            epoch_path = output_dir / f"epoch_{epoch + 1}"
            epoch_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, epoch_path / 'checkpoint.pt')
        
        # Training complete
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 80)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total Training Time: {total_time/3600:.2f} hours")
        print(f"📊 Best Loss: {best_loss:.4f}")
        print(f"💾 Models saved to: {output_dir}")
        
        # Save final model
        final_path = output_dir / 'final_model'
        final_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_path / 'model.pt')
        print(f"\n✅ Final model saved: {final_path}")
        
        return model, best_loss
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        
        interrupt_path = output_dir / 'interrupted_checkpoint'
        interrupt_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step': global_step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / max(1, logging_steps),
        }, interrupt_path / 'checkpoint.pt')
        
        print(f"✅ Checkpoint saved: {interrupt_path}")
        return model, None
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory-Optimized Training for 31-Module System')
    parser.add_argument('--data', type=str, default='data/complete_hindu_database.json',
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='models/checkpoints_31modules',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (keep at 1 for 4GB GPU)')
    parser.add_argument('--grad-accum', type=int, default=16,
                       help='Gradient accumulation steps')
    parser.add_argument('--max-length', type=int, default=128,
                       help='Max sequence length')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--no-gradient-checkpointing', action='store_true',
                       help='Disable gradient checkpointing')
    parser.add_argument('--cpu', action='store_true',
                       help='Train on CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Determine device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n🕉️  DharmaMind - Memory-Optimized Training")
    print(f"   31 Spiritual Modules | 225.1M Parameters | 61% Spiritual")
    print(f"   Optimized for: GTX 1650 (4GB VRAM)\n")
    
    # Run training
    train_memory_optimized(
        data_file=args.data,
        output_dir=args.output,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_length,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        use_mixed_precision=not args.no_mixed_precision,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        device=device
    )
