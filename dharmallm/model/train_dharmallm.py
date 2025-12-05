"""
🕉️ DharmaLLM Training Pipeline
==============================

Train the custom DharmaLLM from scratch on authentic dharmic texts.
No GPT-2, no HuggingFace - pure PyTorch training.

Usage:
    python train_dharmallm.py --size small --epochs 10
    python train_dharmallm.py --size medium --epochs 20
    python train_dharmallm.py --size large --epochs 30

May this training manifest the wisdom of the Rishis! 🙏
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.custom_dharmallm import (
    DharmaLLM, DharmaLLMConfig, DharmicTokenizer, DharmicDataset,
    create_model_small, create_model_medium, create_model_large
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_training_data(data_dir: Path) -> List[str]:
    """Load all training texts from data directory."""
    texts = []
    
    logger.info(f"Loading training data from {data_dir}...")
    
    # Load text files
    text_patterns = ['**/*.txt']
    for pattern in text_patterns:
        for file_path in data_dir.glob(pattern):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if len(content.strip()) > 100:
                        # Split into paragraphs
                        paragraphs = content.split('\n\n')
                        for para in paragraphs:
                            para = para.strip()
                            if len(para) > 50:
                                texts.append(para)
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
    
    # Load JSON files (structured data)
    for file_path in data_dir.glob('**/*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
                texts.extend(extract_texts_from_json(data))
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
    
    # Load JSONL files
    for file_path in data_dir.glob('**/*.jsonl'):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'text' in item:
                            texts.append(item['text'])
                        elif 'content' in item:
                            texts.append(item['content'])
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
    
    # Deduplicate
    seen = set()
    unique_texts = []
    for text in texts:
        text_hash = hash(text[:100])
        if text_hash not in seen:
            seen.add(text_hash)
            unique_texts.append(text)
    
    logger.info(f"Loaded {len(unique_texts)} unique text segments")
    return unique_texts


def extract_texts_from_json(data, depth: int = 0) -> List[str]:
    """Recursively extract text fields from JSON data."""
    texts = []
    
    if depth > 10:  # Prevent deep recursion
        return texts
    
    if isinstance(data, str) and len(data) > 50:
        texts.append(data)
    elif isinstance(data, list):
        for item in data:
            texts.extend(extract_texts_from_json(item, depth + 1))
    elif isinstance(data, dict):
        # Priority text fields
        text_fields = ['text', 'content', 'translation', 'meaning', 'explanation',
                       'verse', 'shloka', 'wisdom', 'teaching', 'commentary',
                       'sanskrit', 'english', 'description']
        
        for field in text_fields:
            if field in data and isinstance(data[field], str) and len(data[field]) > 50:
                texts.append(data[field])
        
        # Recurse into other fields
        for key, value in data.items():
            if key not in text_fields and isinstance(value, (list, dict)):
                texts.extend(extract_texts_from_json(value, depth + 1))
    
    return texts


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DharmaLLMTrainer:
    """Trainer for custom DharmaLLM."""
    
    def __init__(
        self,
        model: DharmaLLM,
        tokenizer: DharmicTokenizer,
        train_dataset: DharmicDataset,
        val_dataset: Optional[DharmicDataset],
        output_dir: Path,
        learning_rate: float = 3e-4,
        batch_size: int = 8,
        num_epochs: int = 10,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = 'auto'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        else:
            self.val_loader = None
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_log = []
    
    def train(self):
        """Run training loop."""
        logger.info("=" * 60)
        logger.info("🕉️ Starting DharmaLLM Training")
        logger.info("=" * 60)
        logger.info(f"Model parameters: {self.model.num_parameters/1e6:.1f}M")
        logger.info(f"Training examples: {len(self.train_dataset)}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Learning rate: {self.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            train_loss = self._train_epoch()
            
            # Validation
            val_loss = None
            if self.val_loader:
                val_loss = self._validate()
            
            # Log
            log_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'time': time.time() - start_time
            }
            self.training_log.append(log_entry)
            
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - "
                       f"Train Loss: {train_loss:.4f}" +
                       (f" - Val Loss: {val_loss:.4f}" if val_loss else ""))
            
            # Save checkpoint
            if val_loss is None or val_loss < self.best_val_loss:
                if val_loss is not None:
                    self.best_val_loss = val_loss
                self._save_checkpoint('best')
            
            self._save_checkpoint('latest')
        
        # Final save
        self._save_final()
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"✅ Training completed in {total_time/3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Model saved to: {self.output_dir}")
        logger.info("=" * 60)
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss'] / self.gradient_accumulation_steps
            loss.backward()
            
            total_loss += outputs['loss'].item()
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log progress
                if self.global_step % 50 == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"  Step {self.global_step} - Loss: {avg_loss:.4f} - LR: {lr:.2e}")
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs['loss'].item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f'checkpoint_{name}'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(checkpoint_path))
        
        # Save tokenizer
        self.tokenizer.save(str(checkpoint_path / 'tokenizer.json'))
        
        # Save training state
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'training_log': self.training_log
        }
        torch.save(state, str(checkpoint_path / 'training_state.pt'))
    
    def _save_final(self):
        """Save final trained model."""
        final_path = self.output_dir / 'dharma_llm_final'
        final_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(final_path))
        
        # Save tokenizer
        self.tokenizer.save(str(final_path / 'tokenizer.json'))
        
        # Save training summary
        summary = {
            'model_type': 'DharmaLLM',
            'model_config': self.model.config.to_dict(),
            'num_parameters': self.model.num_parameters,
            'training_epochs': self.num_epochs,
            'final_train_loss': self.training_log[-1]['train_loss'] if self.training_log else None,
            'best_val_loss': self.best_val_loss,
            'training_time_seconds': self.training_log[-1]['time'] if self.training_log else 0,
            'completed_at': datetime.now().isoformat()
        }
        
        with open(final_path / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Final model saved to {final_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train Custom DharmaLLM')
    parser.add_argument('--size', choices=['small', 'medium', 'large'], default='small',
                        help='Model size (small: ~25M, medium: ~85M, large: ~200M)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--vocab_size', type=int, default=8000, help='Vocabulary size')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='trained_models/dharmallm',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    logger.info("🕉️ DharmaLLM Training Pipeline")
    logger.info("=" * 60)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    output_dir = base_dir / args.output_dir
    
    # Load training data
    texts = load_training_data(data_dir)
    
    if len(texts) < 100:
        logger.error("Not enough training data! Need at least 100 text segments.")
        return
    
    logger.info(f"Total text segments: {len(texts)}")
    logger.info(f"Total characters: {sum(len(t) for t in texts):,}")
    
    # Train tokenizer
    logger.info("\n📝 Training tokenizer...")
    tokenizer = DharmicTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(texts, min_frequency=2)
    
    # Create model
    logger.info(f"\n🧠 Creating {args.size} model...")
    if args.size == 'small':
        config, model = create_model_small()
    elif args.size == 'medium':
        config, model = create_model_medium()
    else:
        config, model = create_model_large()
    
    # Update vocab size
    if config.vocab_size != len(tokenizer):
        logger.info(f"Adjusting vocab size from {config.vocab_size} to {len(tokenizer)}")
        config.vocab_size = len(tokenizer)
        if args.size == 'small':
            _, model = create_model_small()
        elif args.size == 'medium':
            _, model = create_model_medium()
        else:
            _, model = create_model_large()
        model.config.vocab_size = len(tokenizer)
    
    # Create dataset
    logger.info("\n📊 Creating training dataset...")
    full_dataset = DharmicDataset(texts, tokenizer, max_length=args.max_length)
    
    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Create trainer
    trainer = DharmaLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device
    )
    
    # Train!
    trainer.train()
    
    logger.info("\n🙏 Training complete! Namaste!")


if __name__ == '__main__':
    main()

