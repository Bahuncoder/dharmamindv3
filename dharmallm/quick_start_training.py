#!/usr/bin/env python3
"""
Quick Start Training Script for DharmaMind LLM

This script runs a quick test training session to validate the entire pipeline.
Use this for your first run to ensure everything works before full training.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.unified_data_loader import create_dataloaders
from training.embeddings import DharmicEmbeddingModel
from training.training_utils import TrainingConfig
from training.advanced_trainer import AdvancedTrainer
from training.checkpoint_manager import CheckpointManager, RetentionPolicy


def main():
    """Run a quick test training session."""
    
    print("=" * 80)
    print("🕉️  DHARMAMIND LLM - QUICK START TRAINING")
    print("=" * 80)
    print()
    
    # 1. Check system
    print("📊 System Check:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   CUDA Memory: {mem_gb:.1f} GB")
    print()
    
    # 2. Setup paths
    corpus_path = (project_root / "data" / "master_corpus" / 
                   "complete_dharmic_corpus.txt")
    checkpoint_dir = project_root / "training" / "checkpoints" / "quick_test"
    
    print("📁 Paths:")
    print(f"   Corpus: {corpus_path}")
    print(f"   Checkpoints: {checkpoint_dir}")
    print(f"   Corpus exists: {corpus_path.exists()}")
    print()
    
    if not corpus_path.exists():
        print("❌ ERROR: Corpus not found!")
        print(f"   Expected: {corpus_path}")
        print()
        print("💡 Build corpus with:")
        print("   python data/scripts/build_master_corpus.py")
        return 1
    
    # 3. Load tokenizer
    print("🔤 Loading Tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token
        print("   ✅ Tokenizer: distilgpt2")
        print(f"   ✅ Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return 1
    print()
    
    # 4. Load data (UNIFIED - auto-detects text or JSON)
    print("📚 Loading Data (Unified Loader - Auto Mode)...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_source=str(corpus_path),  # Auto-detects .txt format
            tokenizer=tokenizer,
            batch_size=4,
            max_length=256,
            min_length=10,
            num_workers=0,
            mode="auto",  # Auto-detect: text or JSON
        )
        print(f"   ✅ Train batches: {len(train_loader)}")
        print(f"   ✅ Val batches: {len(val_loader)}")
        print(f"   ✅ Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()
    
    # 5. Initialize embeddings
    print("🔮 Loading Embeddings...")
    try:
        embedding_model = DharmicEmbeddingModel()
        print(f"   ✅ Model: {embedding_model.model_name}")
        print(f"   ✅ Dimensions: {embedding_model.dimension}")
        print(f"   ✅ Index: {embedding_model.index is not None}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()
    
    # 6. Initialize model (small GPT-2 for quick test)
    print("🤖 Initializing Model...")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "distilgpt2",
            torch_dtype=torch.float32,
        )
        print("   ✅ Model: distilgpt2 (82M parameters)")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"   ✅ Device: {device}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return 1
    print()
    
    # 7. Setup training config
    print("⚙️  Training Configuration:")
    config = TrainingConfig(
        learning_rate=5e-5,
        warmup_steps=10,
        max_steps=500,  # Short test (about 2 epochs with 238 train batches)
        max_grad_norm=1.0,
        weight_decay=0.01,
        use_fp16=False,
        use_bf16=False,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        logging_steps=5,
        eval_steps=50,
        save_steps=100,
        train_batch_size=4,
        eval_batch_size=4,
    )
    print(f"   Max Steps: {config.max_steps}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Batch Size: {config.train_batch_size}")
    print()
    
    # 8. Setup checkpoint manager
    print("💾 Checkpoint Manager:")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        retention_policy=RetentionPolicy(keep_best_n=2, keep_latest_n=1),
    )
    print(f"   Directory: {checkpoint_dir}")
    print()
    
    # 9. Initialize trainer
    print("🎯 Initializing Trainer...")
    try:
        trainer = AdvancedTrainer(
            model=model,
            config=config,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            optimizer=torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate
            ),
        )
        print("   ✅ Trainer ready")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()
    
    # 10. Run training
    print("=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    print()
    print(f"This will run {config.max_steps} steps (about 2 epochs).")
    print("Logging every 5 steps, eval every 50 steps.")
    print("Press Ctrl+C to stop training gracefully")
    print()
    
    try:
        history = trainer.train()
        print()
        print("=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print()
        print("📊 Final Metrics:")
        if history['train_losses']:
            print(f"   Final Train Loss: {history['train_losses'][-1]:.4f}")
        if history['val_losses']:
            print(f"   Final Val Loss: {history['val_losses'][-1]:.4f}")
        print(f"   Total Steps: {len(history['train_losses'])}")
        print()
        
    except KeyboardInterrupt:
        print()
        print("⚠️  Training interrupted by user")
        print()
    except Exception as e:
        print()
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 11. Show checkpoint info
    print("💾 Saved Checkpoints:")
    checkpoints = checkpoint_manager.list_checkpoints()
    if checkpoints:
        for i, ckpt_info in enumerate(checkpoints[:3], 1):
            print(f"   {i}. Step {ckpt_info['step']}, "
                  f"Loss: {ckpt_info['loss']:.4f}")
    else:
        print("   (No checkpoints saved)")
    print()
    
    # 12. Next steps
    print("=" * 80)
    print("🎉 SUCCESS! Your training pipeline is working!")
    print("=" * 80)
    print()
    print("📚 Next Steps:")
    print()
    print("1. Run Full Training:")
    print("   python run_full_training.py --epochs 10 --fp16")
    print()
    print("2. Check Documentation:")
    print("   - TRAINING_GUIDE.md")
    print("   - API_DOCUMENTATION.md")
    print("   - PERFORMANCE_TUNING.md")
    print()
    print("3. Run Tests:")
    print("   pytest tests/training/ -v")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
