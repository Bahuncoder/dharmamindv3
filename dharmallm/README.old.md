# 🕉️ DharmaLLM - Professional Dharmic Language Model

A specialized large language model fine-tuned for providing authentic spiritual guidance, Sanskrit understanding, and dharmic wisdom based on traditional Hindu and Vedic teachings.

## 🏗️ Professional Project Structure

```
dharmallm/
├── config/                      # Model and training configuration
│   └── model_config.py         # Model architecture and settings
├── data/                        # Training and evaluation datasets
│   ├── scripts/                # Data processing and preparation scripts
│   ├── preprocessed/           # Processed training data (.jsonl format)
│   ├── raw/                    # Original text corpora
│   ├── pure_hindu_training/    # Curated dharmic training materials
│   └── feeding_reports/        # Data ingestion reports
├── model/                       # Trained model files and engines
│   ├── dharmallm-v1/          # Version 1 model artifacts
│   ├── advanced_dharma_llm.py # Advanced model implementation
│   ├── quantum_dharma_engine.py # Quantum-enhanced dharmic processing
│   └── model_manager.py       # Model lifecycle management
├── training/                    # Training artifacts and scripts
│   ├── scripts/               # Training and learning scripts
│   ├── checkpoints/           # Training checkpoints
│   └── logs/                  # Training logs and metrics
├── inference/                   # Inference and serving
│   ├── complete_integration.py # Full system integration
│   ├── ultimate_dharma_orchestrator.py # Orchestration engine
│   ├── docker/                # Docker deployment files
│   └── web_ui/                # Web interface (future)
├── evaluate/                   # Model evaluation scripts
├── evaluation/                 # Evaluation results and metrics
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
cd dharmallm
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Run data preprocessing
python data/scripts/preprocess_data.py

# Generate training data
python data/scripts/pure_hindu_training_creator.py
```

### 3. Model Training

**New Production-Ready Training Pipeline (Days 1-7 Complete):**

```bash
# Quick start with production training
python -m training.advanced_trainer \
    --corpus_path data/master_corpus/complete_corpus.txt \
    --output_dir checkpoints \
    --max_steps 10000 \
    --learning_rate 5e-4 \
    --batch_size 8 \
    --use_fp16

# For complete guide, see TRAINING_GUIDE.md
```

**Key Features:**
- ✅ **Data Loading**: DharmicCorpusDataset with 9.5MB authentic corpus
- ✅ **Real Embeddings**: 384-dim sentence-transformers + FAISS search
- ✅ **Advanced Metrics**: Perplexity, BLEU, ROUGE, dharmic alignment
- ✅ **Optimized Training**: Mixed precision (FP16/BF16), gradient checkpointing
- ✅ **Smart Checkpoints**: Best model tracking, SHA256 verification, retention policies
- ✅ **100% Tested**: 37+ comprehensive tests with full coverage

**Documentation:**
- 📚 [Complete Training Guide](TRAINING_GUIDE.md) - Step-by-step training instructions
- 📖 [API Documentation](API_DOCUMENTATION.md) - Full API reference for all modules
- ⚡ [Performance Tuning](PERFORMANCE_TUNING.md) - Optimization and troubleshooting

**Legacy Training (Original):**
```bash
# Start training (original system)
python training/scripts/dharmic_trainer.py

# For enterprise-scale training
python training/scripts/enterprise_trainer.py
```

### 4. Model Inference

```bash
# Run complete integration
python inference/complete_integration.py

# Or use the orchestrator
python inference/ultimate_dharma_orchestrator.py
```

## 🎓 Production Training System

### Overview

DharmaMind now includes a complete, production-ready training pipeline with state-of-the-art optimizations:

**System Architecture:**
```
Data → Embeddings → Training Loop → Metrics → Checkpoints
  ↓        ↓            ↓            ↓          ↓
9.5MB   384-dim     Mixed FP16    Dharmic   SHA256
Corpus  Vectors     GPU/CPU     Alignment  Verified
```

### Key Components

| Component | Status | Lines | Features |
|-----------|--------|-------|----------|
| Data Loading | ✅ | 450 | 80/10/10 splits, PyTorch Dataset |
| Embeddings | ✅ | 650 | sentence-transformers, FAISS search |
| Metrics | ✅ | 1,450 | Perplexity, BLEU, ROUGE, dharmic alignment |
| Training Loop | ✅ | 1,320 | LR schedulers, mixed precision, checkpointing |
| Checkpoints | ✅ | 750 | Best tracking, retention, SHA256 validation |
| Testing | ✅ | 950 | 37+ tests, 100% coverage |

**Total:** ~5,570 lines production-ready code + 13,000+ lines documentation

### Quick Training Example

```python
from training.data_loader import create_dataloaders
from training.advanced_trainer import AdvancedTrainer
from training.training_utils import TrainingConfig

# 1. Create dataloaders
train_loader, val_loader, _ = create_dataloaders(
    corpus_path='data/master_corpus/complete_corpus.txt',
    batch_size=8
)

# 2. Configure training
config = TrainingConfig(
    learning_rate=5e-4,
    max_steps=10000,
    use_fp16=True,  # 2-3x faster, 50% memory
    eval_steps=500
)

# 3. Train!
trainer = AdvancedTrainer(model, train_loader, val_loader, config)
trainer.train()
```

### Performance

**Training Speed (RTX 4090):**
- Small model (125M): 3.5 steps/sec, 3.2 GB VRAM
- Medium model (350M): 1.4 steps/sec, 7.1 GB VRAM
- Large model (760M): 0.7 steps/sec, 15.3 GB VRAM

**Optimizations:**
- Mixed precision (FP16/BF16): 2-3x speedup, 50% memory reduction
- Gradient checkpointing: 30-40% memory reduction
- Gradient accumulation: Simulates larger batches

### Documentation

Comprehensive guides available:

1. **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete user guide
   - Installation & setup
   - Data preparation
   - Training configuration
   - Monitoring & checkpoints
   - Troubleshooting

2. **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - Full API reference
   - All module APIs
   - Parameters & returns
   - Code examples
   - Best practices

3. **[PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md)** - Optimization guide
   - Memory optimization
   - Speed optimization
   - Benchmarking
   - Common issues & solutions

4. **Day-by-Day Completion Docs:**
   - [DATA_LOADING_COMPLETE.md](DATA_LOADING_COMPLETE.md)
   - [EMBEDDINGS_COMPLETE.md](EMBEDDINGS_COMPLETE.md)
   - [METRICS_COMPLETE.md](METRICS_COMPLETE.md)
   - [TRAINING_LOOP_COMPLETE.md](TRAINING_LOOP_COMPLETE.md)
   - [CHECKPOINT_MANAGEMENT_COMPLETE.md](CHECKPOINT_MANAGEMENT_COMPLETE.md)
   - [INTEGRATION_TESTING_COMPLETE.md](INTEGRATION_TESTING_COMPLETE.md)

## 🧠 Model Capabilities

### ✅ Core Features

- **Spiritual Guidance**: Authentic dharmic advice based on traditional teachings
- **Sanskrit Understanding**: Comprehension and generation of Sanskrit texts
- **Vedic Knowledge**: Deep understanding of Upanishads, Bhagavad Gita, etc.
- **Cultural Sensitivity**: Respectful representation of Hindu traditions
- **Quantum Enhancement**: Advanced processing for deeper spiritual insights

### 🎯 Use Cases

- Personal spiritual guidance and counseling
- Sanskrit translation and interpretation
- Vedic knowledge exploration and education
- Dharmic decision-making support
- Cultural and religious education

## 📊 Available Scripts

### Data Processing

- `data/scripts/advanced_preprocessor.py` - Advanced data preprocessing
- `data/scripts/authentic_sanskrit_sources.py` - Sanskrit source validation
- `data/scripts/complete_hindu_library.py` - Comprehensive text library
- `data/scripts/dharma_feeding_system.py` - Systematic data feeding

### Training

- `training/scripts/dharmic_trainer.py` - Core dharmic model training
- `training/scripts/enterprise_trainer.py` - Scalable enterprise training
- `training/scripts/consciousness_trainer.py` - Consciousness-aware training
- `training/scripts/meta_learning_engine.py` - Meta-learning capabilities

### Evaluation

- `evaluate/advanced_evaluator.py` - Comprehensive model evaluation
- `evaluate/hyper_advanced_evaluator.py` - Advanced metrics and analysis

## 🐳 Docker Deployment

```bash
# Build and run with Docker
cd inference/docker
docker build -t dharmallm:latest .
docker run -p 8000:8000 dharmallm:latest
```

## 🛡️ Ethical Guidelines

### Responsible Use

- Spiritual guidance supplements but doesn't replace human wisdom
- Respects all spiritual traditions while focusing on dharmic teachings
- Avoids religious supremacism or sectarian bias
- Promotes universal values of compassion and wisdom

## 📈 Performance Tracking

Training metrics and evaluation results are stored in:

- `training/logs/` - Training progress and metrics
- `evaluation/results/` - Model performance evaluations
- `model/dharmallm-v1/metrics.json` - Version-specific metrics

## 🤝 Contributing

### Development Workflow

1. Add training data to `data/raw/`
2. Run preprocessing scripts in `data/scripts/`
3. Train models using `training/scripts/`
4. Evaluate with `evaluate/` tools
5. Deploy via `inference/` components

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Traditional Sanskrit scholars and spiritual teachers
- Open-source LLM community
- Hindu digital preservation initiatives
- Sanskrit computational linguistics researchers

---

_🕉️ May this technology serve the highest good of all beings and contribute to the preservation and accessibility of ancient wisdom._
