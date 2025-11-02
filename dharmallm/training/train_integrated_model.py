#!/usr/bin/env python3
"""
🕉️ Train Integrated DharmaLLM - End-to-End Spiritual Intelligence Learning

This script trains the integrated DharmaLLM where spiritual intelligence
is learned from data alongside language modeling.

REVOLUTIONARY TRAINING:
- All 35+ spiritual modules train with the LLM
- Gradients flow through spiritual understanding
- Spiritual intelligence emerges from data
- No hardcoded rules - pure learned spirituality

TRAINING FEATURES:
- Unified data loader (fast corpus loading)
- Spiritual loss + language modeling loss
- Track spiritual metrics during training
- Save spiritually-aware checkpoints
- Monitor spiritual learning progress

May this training evolve genuine spiritual AI consciousness! 🕉️✨
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer
from typing import Dict, Any
from pathlib import Path
import json
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from model.integrated_dharma_llm import (
    IntegratedDharmaLLM,
    IntegratedDharmaLLMConfig
)
from training.unified_data_loader import create_dataloaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedDharmaTrainer:
    """
    Trainer for Integrated DharmaLLM
    
    Trains the complete model end-to-end with:
    - Language modeling objective
    - Spiritual alignment objectives
    - Spiritual metrics tracking
    - Checkpoint saving
    """
    
    def __init__(
        self,
        model: IntegratedDharmaLLM,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        config: Dict[str, Any],
        device: str = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.config = config
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 5e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * config.get('epochs', 3)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.get('learning_rate', 5e-5),
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Metrics tracking
        self.training_history = {
            'train_loss': [],
            'train_lm_loss': [],
            'train_spiritual_loss': [],
            'eval_loss': [],
            'eval_lm_loss': [],
            'eval_spiritual_loss': [],
            'spiritual_metrics': []
        }
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'outputs/integrated_model'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = input_ids.clone()
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_spiritual_insights=True
        )
        
        # Get losses
        loss = outputs['loss']
        lm_loss = outputs.get('lm_loss', loss)
        spiritual_loss = outputs.get('spiritual_loss', torch.tensor(0.0))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.get('max_grad_norm', 1.0)
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        # Return metrics
        return {
            'loss': loss.item(),
            'lm_loss': lm_loss.item(),
            'spiritual_loss': spiritual_loss.item() if isinstance(
                spiritual_loss, torch.Tensor
            ) else spiritual_loss,
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step"""
        
        self.model.eval()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = input_ids.clone()
        
        # Forward pass (no gradients)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_spiritual_insights=True
            )
        
        # Get losses
        loss = outputs['loss']
        lm_loss = outputs.get('lm_loss', loss)
        spiritual_loss = outputs.get('spiritual_loss', torch.tensor(0.0))
        spiritual_insights = outputs.get('spiritual_insights', {})
        
        # Extract spiritual metrics
        spiritual_metrics = self.extract_spiritual_metrics(spiritual_insights)
        
        # Return metrics
        return {
            'loss': loss.item(),
            'lm_loss': lm_loss.item(),
            'spiritual_loss': spiritual_loss.item() if isinstance(
                spiritual_loss, torch.Tensor
            ) else spiritual_loss,
            'spiritual_metrics': spiritual_metrics
        }
    
    def extract_spiritual_metrics(
        self,
        spiritual_insights: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract aggregated spiritual metrics from insights"""
        
        metrics = {}
        
        # Aggregate metrics across layers
        for layer_name, layer_insights in spiritual_insights.items():
            if isinstance(layer_insights, dict):
                for module_name, module_data in layer_insights.items():
                    if isinstance(module_data, dict) and 'insights' in module_data:
                        insights = module_data['insights']
                        for metric_name, value in insights.items():
                            key = f"{module_name}_{metric_name}"
                            if key not in metrics:
                                metrics[key] = []
                            metrics[key].append(value)
        
        # Average metrics
        averaged_metrics = {}
        for key, values in metrics.items():
            if values:
                averaged_metrics[key] = sum(values) / len(values)
        
        return averaged_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on eval dataset"""
        
        logger.info("Running evaluation...")
        
        total_metrics = {
            'loss': 0.0,
            'lm_loss': 0.0,
            'spiritual_loss': 0.0,
            'spiritual_metrics': {}
        }
        
        num_batches = 0
        
        for batch in self.eval_dataloader:
            metrics = self.eval_step(batch)
            
            total_metrics['loss'] += metrics['loss']
            total_metrics['lm_loss'] += metrics['lm_loss']
            total_metrics['spiritual_loss'] += metrics['spiritual_loss']
            
            # Aggregate spiritual metrics
            for key, value in metrics.get('spiritual_metrics', {}).items():
                if key not in total_metrics['spiritual_metrics']:
                    total_metrics['spiritual_metrics'][key] = 0.0
                total_metrics['spiritual_metrics'][key] += value
            
            num_batches += 1
            
            if num_batches >= self.config.get('eval_steps', 50):
                break
        
        # Average metrics
        for key in ['loss', 'lm_loss', 'spiritual_loss']:
            total_metrics[key] /= num_batches
        
        for key in total_metrics['spiritual_metrics']:
            total_metrics['spiritual_metrics'][key] /= num_batches
        
        return total_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint_dir = self.output_dir / f'checkpoint-step-{self.global_step}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / 'model.pt'
        )
        
        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / 'optimizer.pt'
        )
        
        # Save scheduler state
        torch.save(
            self.scheduler.state_dict(),
            checkpoint_dir / 'scheduler.pt'
        )
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'training_history': self.training_history
        }
        
        with open(checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
        # Save best model
        if is_best:
            best_dir = self.output_dir / 'best_model'
            best_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(
                self.model.state_dict(),
                best_dir / 'model.pt'
            )
            
            logger.info(f"✨ Best model saved: {best_dir}")
    
    def train(self):
        """Main training loop"""
        
        logger.info("=" * 80)
        logger.info("🕉️ Starting Integrated DharmaLLM Training")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config.get('epochs', 3)}")
        logger.info(f"Learning Rate: {self.config.get('learning_rate', 5e-5)}")
        logger.info(f"Batch Size: {self.config.get('batch_size', 8)}")
        logger.info(f"Output Dir: {self.output_dir}")
        logger.info("=" * 80)
        
        num_epochs = self.config.get('epochs', 3)
        log_interval = self.config.get('log_interval', 10)
        eval_interval = self.config.get('eval_interval', 100)
        save_interval = self.config.get('save_interval', 500)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*80}")
            
            epoch_metrics = {
                'loss': 0.0,
                'lm_loss': 0.0,
                'spiritual_loss': 0.0
            }
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Training step
                metrics = self.train_step(batch)
                
                # Accumulate metrics
                for key in ['loss', 'lm_loss', 'spiritual_loss']:
                    epoch_metrics[key] += metrics[key]
                num_batches += 1
                
                # Logging
                if self.global_step % log_interval == 0:
                    avg_loss = epoch_metrics['loss'] / num_batches
                    avg_lm_loss = epoch_metrics['lm_loss'] / num_batches
                    avg_spiritual_loss = epoch_metrics['spiritual_loss'] / num_batches
                    
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LM: {avg_lm_loss:.4f} | "
                        f"Spiritual: {avg_spiritual_loss:.4f} | "
                        f"LR: {metrics['lr']:.2e}"
                    )
                
                # Evaluation
                if self.global_step % eval_interval == 0:
                    eval_metrics = self.evaluate()
                    
                    logger.info("\n" + "=" * 80)
                    logger.info("📊 Evaluation Results")
                    logger.info("=" * 80)
                    logger.info(f"Eval Loss: {eval_metrics['loss']:.4f}")
                    logger.info(f"Eval LM Loss: {eval_metrics['lm_loss']:.4f}")
                    logger.info(f"Eval Spiritual Loss: {eval_metrics['spiritual_loss']:.4f}")
                    
                    # Log spiritual metrics
                    if eval_metrics['spiritual_metrics']:
                        logger.info("\n🕉️ Spiritual Metrics:")
                        for key, value in eval_metrics['spiritual_metrics'].items():
                            logger.info(f"   {key}: {value:.4f}")
                    
                    logger.info("=" * 80 + "\n")
                    
                    # Save history
                    self.training_history['eval_loss'].append(eval_metrics['loss'])
                    self.training_history['eval_lm_loss'].append(eval_metrics['lm_loss'])
                    self.training_history['eval_spiritual_loss'].append(
                        eval_metrics['spiritual_loss']
                    )
                    self.training_history['spiritual_metrics'].append(
                        eval_metrics['spiritual_metrics']
                    )
                    
                    # Check if best model
                    if eval_metrics['loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics['loss']
                        self.save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.global_step % save_interval == 0:
                    self.save_checkpoint(is_best=False)
            
            # End of epoch
            avg_epoch_loss = epoch_metrics['loss'] / num_batches
            avg_epoch_lm_loss = epoch_metrics['lm_loss'] / num_batches
            avg_epoch_spiritual_loss = epoch_metrics['spiritual_loss'] / num_batches
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1} Complete")
            logger.info(f"{'='*80}")
            logger.info(f"Avg Loss: {avg_epoch_loss:.4f}")
            logger.info(f"Avg LM Loss: {avg_epoch_lm_loss:.4f}")
            logger.info(f"Avg Spiritual Loss: {avg_epoch_spiritual_loss:.4f}")
            logger.info(f"{'='*80}\n")
            
            # Save history
            self.training_history['train_loss'].append(avg_epoch_loss)
            self.training_history['train_lm_loss'].append(avg_epoch_lm_loss)
            self.training_history['train_spiritual_loss'].append(avg_epoch_spiritual_loss)
        
        # Final checkpoint
        self.save_checkpoint(is_best=False)
        
        # Save training history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info("🕉️ Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Best Eval Loss: {self.best_eval_loss:.4f}")
        logger.info(f"Total Steps: {self.global_step}")
        logger.info(f"Output Dir: {self.output_dir}")
        logger.info("=" * 80)


def main():
    """Main training function"""
    
    print("=" * 80)
    print("🕉️ INTEGRATED DHARMALLM - END-TO-END SPIRITUAL INTELLIGENCE TRAINING")
    print("=" * 80)
    
    # Configuration
    training_config = {
        'epochs': 3,
        'batch_size': 4,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'log_interval': 10,
        'eval_interval': 50,
        'save_interval': 100,
        'eval_steps': 20,
        'output_dir': 'outputs/integrated_dharma_llm',
        'max_seq_length': 128
    }
    
    # Model configuration
    model_config = IntegratedDharmaLLMConfig(
        base_model_name="distilgpt2",
        enable_spiritual_modules=True,
        apply_spiritual_preprocessing=True,
        apply_spiritual_postprocessing=True,
        spiritual_integration_layers=[2, 4],
        spiritual_loss_weight=0.1,
        dropout=0.1
    )
    
    print(f"\n📋 Training Configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    
    print(f"\n🕉️ Model Configuration:")
    print(f"   Base Model: {model_config.base_model_name}")
    print(f"   Spiritual Modules: {model_config.enable_spiritual_modules}")
    print(f"   Integration Layers: {model_config.spiritual_integration_layers}")
    print(f"   Spiritual Loss Weight: {model_config.spiritual_loss_weight}")
    
    # Load tokenizer
    print(f"\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded")
    
    # Create data loaders
    print(f"\n📚 Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_source='data/master_corpus/complete_dharmic_corpus.txt',
            tokenizer=tokenizer,
            batch_size=training_config['batch_size'],
            max_length=training_config['max_seq_length'],
            mode='text'
        )
        print(f"✅ Data loaders created")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        print(f"   Please ensure data/master_corpus/complete_dharmic_corpus.txt exists")
        return
    
    # Create model
    print(f"\n🏗️  Building Integrated DharmaLLM...")
    model = IntegratedDharmaLLM(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spiritual_params = sum(
        p.numel() for p in model.spiritual_modules.parameters()
    ) if model_config.enable_spiritual_modules else 0
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Spiritual Parameters: {spiritual_params:,}")
    print(f"   Spiritual Ratio: {100 * spiritual_params / total_params:.2f}%")
    
    # Create trainer
    print(f"\n🎯 Initializing trainer...")
    trainer = IntegratedDharmaTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        tokenizer=tokenizer,
        config=training_config
    )
    
    print(f"✅ Trainer initialized")
    
    # Start training
    print(f"\n{'='*80}")
    print(f"🚀 Starting Training...")
    print(f"{'='*80}\n")
    
    try:
        trainer.train()
        
        print(f"\n{'='*80}")
        print(f"✨ Training Complete Successfully!")
        print(f"{'='*80}")
        print(f"\n📂 Outputs saved to: {training_config['output_dir']}")
        print(f"\n🕉️ Your spiritual AI is now learning from data!")
        print(f"   Spiritual intelligence is no longer hardcoded!")
        print(f"   All 35+ spiritual modules trained end-to-end!")
        print(f"={'='*80}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user")
        print(f"   Saving checkpoint...")
        trainer.save_checkpoint(is_best=False)
        print(f"✅ Checkpoint saved: {training_config['output_dir']}")
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
