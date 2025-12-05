# 📚 DharmaMind Training API Documentation

**Complete API Reference for All Training Modules**

Version: 1.0  
Date: October 27, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [training.data_loader](#trainingdata_loader)
3. [training.embeddings](#trainingembeddings)
4. [training.metrics](#trainingmetrics)
5. [training.dharmic_metrics](#trainingdharmic_metrics)
6. [training.training_utils](#trainingtraining_utils)
7. [training.advanced_trainer](#trainingadvanced_trainer)
8. [training.checkpoint_manager](#trainingcheckpoint_manager)
9. [Complete Examples](#complete-examples)

---

## Overview

The DharmaMind training pipeline consists of 7 main modules:

```
training/
├── data_loader.py          # Data loading and preprocessing
├── embeddings.py           # Sentence embeddings and vector search
├── metrics.py              # Training metrics (perplexity, BLEU, ROUGE)
├── dharmic_metrics.py      # Dharmic alignment scoring
├── training_utils.py       # Training utilities (schedulers, optimization)
├── advanced_trainer.py     # Complete training loop
└── checkpoint_manager.py   # Checkpoint management
```

### Installation

```bash
pip install torch transformers sentence-transformers faiss-cpu nltk rouge-score
```

### Quick Import

```python
from training.data_loader import DharmicCorpusDataset, create_dataloaders
from training.embeddings import DharmicEmbeddingModel
from training.metrics import MetricsComputer
from training.dharmic_metrics import DharmicAlignmentScorer
from training.training_utils import TrainingConfig, LearningRateScheduler
from training.advanced_trainer import AdvancedTrainer
from training.checkpoint_manager import CheckpointManager, RetentionPolicy
```

---

## training.data_loader

### `DharmicCorpusDataset`

PyTorch Dataset for loading dharmic corpus.

#### Class Definition

```python
class DharmicCorpusDataset(torch.utils.data.Dataset):
    """
    Dataset for dharmic corpus with tokenization.
    
    Loads corpus file (one document per line) and provides
    tokenized batches for training.
    """
```

#### Constructor

```python
def __init__(
    self,
    corpus_path: str,
    max_length: int = 512,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    cache_dir: Optional[str] = None
)
```

**Parameters:**
- `corpus_path` (str): Path to corpus file (UTF-8 text, one doc per line)
- `max_length` (int, default=512): Maximum sequence length
- `tokenizer` (Optional[PreTrainedTokenizer], default=None): HuggingFace tokenizer (defaults to GPT2)
- `cache_dir` (Optional[str], default=None): Cache directory for preprocessed data

**Example:**
```python
from training.data_loader import DharmicCorpusDataset

dataset = DharmicCorpusDataset(
    corpus_path='data/master_corpus/complete_corpus.txt',
    max_length=512
)

print(f"Dataset size: {len(dataset)} documents")
print(f"Corpus size: {dataset.corpus_size / 1024 / 1024:.2f} MB")
```

#### Methods

##### `__len__()`

```python
def __len__(self) -> int:
    """Return number of documents in corpus."""
```

**Returns:** int - Number of documents

**Example:**
```python
num_docs = len(dataset)
print(f"Total documents: {num_docs}")
```

##### `__getitem__(idx)`

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """
    Get tokenized document at index.
    
    Returns:
        dict with keys:
        - input_ids: Token IDs (shape: [max_length])
        - attention_mask: Attention mask (shape: [max_length])
        - labels: Labels for language modeling (shape: [max_length])
    """
```

**Parameters:**
- `idx` (int): Document index (0 to len(dataset)-1)

**Returns:** Dict[str, torch.Tensor] with keys:
- `input_ids`: Token IDs tensor
- `attention_mask`: Attention mask tensor
- `labels`: Label tensor (same as input_ids for language modeling)

**Example:**
```python
sample = dataset[0]
print(f"Input IDs shape: {sample['input_ids'].shape}")
print(f"First 10 tokens: {sample['input_ids'][:10]}")
```

#### Properties

```python
@property
def corpus_size(self) -> int:
    """Total corpus size in bytes."""

@property
def avg_doc_length(self) -> float:
    """Average document length in characters."""

@property
def vocab_size(self) -> int:
    """Tokenizer vocabulary size."""
```

**Example:**
```python
print(f"Corpus size: {dataset.corpus_size / 1024 / 1024:.2f} MB")
print(f"Average doc length: {dataset.avg_doc_length:.0f} chars")
print(f"Vocabulary size: {dataset.vocab_size}")
```

---

### `create_dataloaders()`

Create train, validation, and test dataloaders with automatic splitting.

#### Function Definition

```python
def create_dataloaders(
    corpus_path: str,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    pin_memory: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
```

**Parameters:**
- `corpus_path` (str): Path to corpus file
- `batch_size` (int, default=8): Batch size
- `max_length` (int, default=512): Maximum sequence length
- `num_workers` (int, default=4): DataLoader workers (CPU cores)
- `train_split` (float, default=0.8): Training set proportion (0-1)
- `val_split` (float, default=0.1): Validation set proportion (0-1)
- `test_split` (float, default=0.1): Test set proportion (0-1)
- `shuffle` (bool, default=True): Shuffle training data
- `pin_memory` (bool, default=True): Pin memory for faster GPU transfer
- `seed` (int, default=42): Random seed for reproducibility

**Returns:** Tuple[DataLoader, DataLoader, DataLoader]
- Training DataLoader
- Validation DataLoader
- Test DataLoader

**Example:**
```python
from training.data_loader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    corpus_path='data/master_corpus/complete_corpus.txt',
    batch_size=8,
    max_length=512,
    num_workers=4
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Iterate over batches
for batch in train_loader:
    print(f"Batch shape: {batch['input_ids'].shape}")
    break  # First batch only
```

---

## training.embeddings

### `DharmicEmbeddingModel`

Sentence embedding model for dharmic texts with vector search.

#### Class Definition

```python
class DharmicEmbeddingModel:
    """
    Sentence transformer for dharmic texts.
    
    Uses 'paraphrase-multilingual-MiniLM-L12-v2' model
    to encode texts into 384-dimensional vectors.
    """
```

#### Constructor

```python
def __init__(
    self,
    model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
    device: Optional[str] = None,
    use_faiss: bool = True,
    cache_dir: Optional[str] = None
)
```

**Parameters:**
- `model_name` (str, default='paraphrase-multilingual-MiniLM-L12-v2'): sentence-transformers model
- `device` (Optional[str], default=None): 'cuda', 'cpu', or None (auto-detect)
- `use_faiss` (bool, default=True): Enable FAISS vector search
- `cache_dir` (Optional[str], default=None): Model cache directory

**Example:**
```python
from training.embeddings import DharmicEmbeddingModel

# Auto-detect device (GPU if available)
model = DharmicEmbeddingModel()

# Force CPU
model = DharmicEmbeddingModel(device='cpu')

# Custom model
model = DharmicEmbeddingModel(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
```

#### Methods

##### `encode()`

```python
def encode(
    self,
    texts: Union[str, List[str]],
    batch_size: int = 32,
    show_progress_bar: bool = False,
    normalize_embeddings: bool = True
) -> np.ndarray:
    """
    Encode texts to embedding vectors.
    
    Returns:
        np.ndarray of shape (num_texts, 384) with embeddings
    """
```

**Parameters:**
- `texts` (Union[str, List[str]]): Single text or list of texts
- `batch_size` (int, default=32): Batch size for encoding
- `show_progress_bar` (bool, default=False): Show progress bar
- `normalize_embeddings` (bool, default=True): L2 normalize embeddings

**Returns:** np.ndarray of shape (num_texts, 384)

**Example:**
```python
# Single text
text = "ॐ नमः शिवाय"
embedding = model.encode(text)
print(f"Embedding shape: {embedding.shape}")  # (384,)

# Multiple texts
texts = [
    "योगश्चित्तवृत्तिनिरोधः",
    "अहिंसा परमो धर्मः",
    "सत्यं वद धर्मं चर"
]
embeddings = model.encode(texts, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 384)
```

##### `encode_batch()`

```python
def encode_batch(
    self,
    texts: List[str],
    batch_size: int = 32
) -> np.ndarray:
    """
    Batch encode texts (optimized for large datasets).
    
    Returns:
        np.ndarray of shape (num_texts, 384)
    """
```

**Parameters:**
- `texts` (List[str]): List of texts to encode
- `batch_size` (int, default=32): Batch size

**Returns:** np.ndarray of shape (num_texts, 384)

**Example:**
```python
# Large dataset
with open('data/corpus.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f]

# Efficient batch encoding
embeddings = model.encode_batch(texts, batch_size=64)
print(f"Encoded {len(texts)} texts -> {embeddings.shape}")
```

##### `similarity()`

```python
def similarity(
    self,
    text1: Union[str, np.ndarray],
    text2: Union[str, np.ndarray]
) -> float:
    """
    Compute cosine similarity between two texts or embeddings.
    
    Returns:
        float in range [-1, 1] (higher = more similar)
    """
```

**Parameters:**
- `text1` (Union[str, np.ndarray]): First text or embedding
- `text2` (Union[str, np.ndarray]): Second text or embedding

**Returns:** float - Cosine similarity score

**Example:**
```python
text1 = "योगश्चित्तवृत्तिनिरोधः"
text2 = "Yoga is the cessation of mental fluctuations"

similarity = model.similarity(text1, text2)
print(f"Similarity: {similarity:.3f}")  # 0.823

# Pre-computed embeddings
emb1 = model.encode(text1)
emb2 = model.encode(text2)
similarity = model.similarity(emb1, emb2)
```

##### `build_index()`

```python
def build_index(
    self,
    texts: List[str],
    batch_size: int = 32
) -> None:
    """
    Build FAISS index for fast similarity search.
    
    Encodes all texts and creates searchable index.
    """
```

**Parameters:**
- `texts` (List[str]): Texts to index
- `batch_size` (int, default=32): Encoding batch size

**Returns:** None (index stored internally)

**Example:**
```python
# Load corpus
with open('data/corpus.txt', 'r', encoding='utf-8') as f:
    corpus = [line.strip() for line in f]

# Build searchable index
model.build_index(corpus, batch_size=64)
print(f"Index built with {len(corpus)} documents")
```

##### `search()`

```python
def search(
    self,
    query: Union[str, np.ndarray],
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for k most similar texts in index.
    
    Returns:
        Tuple of (similarities, indices)
        - similarities: np.ndarray of shape (k,) with similarity scores
        - indices: np.ndarray of shape (k,) with document indices
    """
```

**Parameters:**
- `query` (Union[str, np.ndarray]): Query text or embedding
- `k` (int, default=5): Number of results to return

**Returns:** Tuple[np.ndarray, np.ndarray]
- similarities: Similarity scores (shape: k,)
- indices: Document indices (shape: k,)

**Example:**
```python
# Search for similar texts
query = "What is dharma?"
similarities, indices = model.search(query, k=5)

print("Top 5 similar texts:")
for i, (sim, idx) in enumerate(zip(similarities, indices)):
    print(f"{i+1}. Similarity: {sim:.3f}, Index: {idx}")
    print(f"   Text: {corpus[idx][:100]}...")
```

##### `save_index()`

```python
def save_index(self, path: str) -> None:
    """Save FAISS index to disk."""
```

**Parameters:**
- `path` (str): Path to save index file

**Example:**
```python
model.save_index('data/embeddings/dharmic_index.faiss')
```

##### `load_index()`

```python
def load_index(self, path: str) -> None:
    """Load FAISS index from disk."""
```

**Parameters:**
- `path` (str): Path to index file

**Example:**
```python
model.load_index('data/embeddings/dharmic_index.faiss')
```

#### Properties

```python
@property
def embedding_dim(self) -> int:
    """Embedding dimension (384 for default model)."""

@property
def device(self) -> str:
    """Device model is running on ('cuda' or 'cpu')."""

@property
def index_size(self) -> int:
    """Number of vectors in FAISS index."""
```

**Example:**
```python
print(f"Embedding dimension: {model.embedding_dim}")
print(f"Device: {model.device}")
print(f"Index size: {model.index_size} documents")
```

---

## training.metrics

### `MetricsComputer`

Compute training metrics (perplexity, BLEU, ROUGE, accuracy).

#### Class Definition

```python
class MetricsComputer:
    """
    Compute standard NLP metrics for training.
    """
```

#### Constructor

```python
def __init__(
    self,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    device: Optional[str] = None
)
```

**Parameters:**
- `tokenizer` (Optional[PreTrainedTokenizer], default=None): Tokenizer for decoding
- `device` (Optional[str], default=None): Device for computations

**Example:**
```python
from training.metrics import MetricsComputer
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
metrics = MetricsComputer(tokenizer=tokenizer)
```

#### Methods

##### `compute_perplexity()`

```python
def compute_perplexity(self, loss: float) -> float:
    """
    Compute perplexity from loss.
    
    Perplexity = exp(loss)
    Lower is better (typical range: 10-100)
    """
```

**Parameters:**
- `loss` (float): Cross-entropy loss

**Returns:** float - Perplexity score

**Example:**
```python
loss = 3.45
perplexity = metrics.compute_perplexity(loss)
print(f"Loss: {loss:.2f}, Perplexity: {perplexity:.2f}")
# Output: Loss: 3.45, Perplexity: 31.50
```

##### `compute_accuracy()`

```python
def compute_accuracy(
    self,
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute token-level accuracy.
    
    Accuracy = correct_predictions / total_predictions
    """
```

**Parameters:**
- `logits` (torch.Tensor): Model predictions (shape: [batch, seq_len, vocab_size])
- `labels` (torch.Tensor): Ground truth labels (shape: [batch, seq_len])
- `ignore_index` (int, default=-100): Label index to ignore (padding)

**Returns:** float - Accuracy (0-1)

**Example:**
```python
# During training
outputs = model(input_ids)
accuracy = metrics.compute_accuracy(outputs.logits, labels)
print(f"Accuracy: {accuracy:.2%}")
# Output: Accuracy: 45.67%
```

##### `compute_bleu()`

```python
def compute_bleu(
    self,
    predictions: List[str],
    references: List[str],
    max_order: int = 4
) -> float:
    """
    Compute BLEU score.
    
    BLEU = geometric mean of n-gram precision (n=1 to max_order)
    Range: 0-1 (higher is better)
    """
```

**Parameters:**
- `predictions` (List[str]): Predicted texts
- `references` (List[str]): Reference texts
- `max_order` (int, default=4): Maximum n-gram order (BLEU-4)

**Returns:** float - BLEU score (0-1)

**Example:**
```python
predictions = [
    "योगश्चित्तवृत्तिनिरोधः",
    "अहिंसा परमो धर्मः"
]
references = [
    "योगश्चित्तवृत्तिनिरोधः तदा द्रष्टुः",
    "अहिंसा परमो धर्मः धर्म हिंसा"
]

bleu = metrics.compute_bleu(predictions, references)
print(f"BLEU score: {bleu:.3f}")
# Output: BLEU score: 0.456
```

##### `compute_rouge()`

```python
def compute_rouge(
    self,
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Returns:
        dict with keys: rouge1, rouge2, rougeL
        Each value is F1 score (0-1, higher is better)
    """
```

**Parameters:**
- `predictions` (List[str]): Predicted texts
- `references` (List[str]): Reference texts

**Returns:** Dict[str, float] with keys:
- `rouge1`: ROUGE-1 F1 score
- `rouge2`: ROUGE-2 F1 score
- `rougeL`: ROUGE-L F1 score

**Example:**
```python
rouge_scores = metrics.compute_rouge(predictions, references)
print(f"ROUGE-1: {rouge_scores['rouge1']:.3f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.3f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.3f}")
# Output:
# ROUGE-1: 0.567
# ROUGE-2: 0.423
# ROUGE-L: 0.534
```

##### `compute_all_metrics()`

```python
def compute_all_metrics(
    self,
    predictions: List[str],
    references: List[str],
    loss: Optional[float] = None,
    logits: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute all metrics at once.
    
    Returns:
        dict with all metric scores
    """
```

**Parameters:**
- `predictions` (List[str]): Predicted texts
- `references` (List[str]): Reference texts
- `loss` (Optional[float], default=None): Loss for perplexity
- `logits` (Optional[torch.Tensor], default=None): Logits for accuracy
- `labels` (Optional[torch.Tensor], default=None): Labels for accuracy

**Returns:** Dict[str, float] with all metrics

**Example:**
```python
all_metrics = metrics.compute_all_metrics(
    predictions=predictions,
    references=references,
    loss=3.45,
    logits=logits,
    labels=labels
)

for metric, value in all_metrics.items():
    print(f"{metric}: {value:.3f}")
# Output:
# perplexity: 31.500
# accuracy: 0.456
# bleu: 0.234
# rouge1: 0.567
# rouge2: 0.423
# rougeL: 0.534
```

---

## training.dharmic_metrics

### `DharmicAlignmentScorer`

Score dharmic alignment of generated text.

#### Class Definition

```python
class DharmicAlignmentScorer:
    """
    Evaluate dharmic alignment of text.
    
    Uses embedding similarity to core dharmic concepts.
    """
```

#### Constructor

```python
def __init__(
    self,
    embedding_model: Optional[DharmicEmbeddingModel] = None,
    threshold: float = 0.7,
    dharmic_concepts: Optional[List[str]] = None
)
```

**Parameters:**
- `embedding_model` (Optional[DharmicEmbeddingModel], default=None): Embedding model (creates one if None)
- `threshold` (float, default=0.7): Alignment threshold (0-1)
- `dharmic_concepts` (Optional[List[str]], default=None): Core concepts (uses defaults if None)

**Default Dharmic Concepts:**
```python
[
    "dharma", "karma", "moksha", "ahimsa",
    "satya", "brahman", "atman", "yoga",
    "vedas", "upanishads", "bhagavad gita",
    "righteousness", "duty", "cosmic law",
    "spiritual knowledge", "liberation", "truth"
]
```

**Example:**
```python
from training.dharmic_metrics import DharmicAlignmentScorer

# Use defaults
scorer = DharmicAlignmentScorer()

# Custom threshold
scorer = DharmicAlignmentScorer(threshold=0.8)

# Custom concepts
custom_concepts = [
    "dharma", "karma", "moksha", "ahimsa",
    "satya", "अहिंसा", "सत्य", "धर्म"
]
scorer = DharmicAlignmentScorer(
    threshold=0.75,
    dharmic_concepts=custom_concepts
)
```

#### Methods

##### `score()`

```python
def score(self, text: str) -> float:
    """
    Compute dharmic alignment score for text.
    
    Returns:
        float in range [0, 1]
        0 = no alignment, 1 = perfect alignment
        > threshold = considered aligned
    """
```

**Parameters:**
- `text` (str): Text to score

**Returns:** float - Alignment score (0-1)

**Example:**
```python
# Aligned text
text1 = "अहिंसा परमो धर्मः। सत्यं वद धर्मं चर।"
score1 = scorer.score(text1)
print(f"Score: {score1:.3f}")  # 0.856

# Non-aligned text
text2 = "The weather is nice today."
score2 = scorer.score(text2)
print(f"Score: {score2:.3f}")  # 0.123
```

##### `score_batch()`

```python
def score_batch(self, texts: List[str]) -> List[float]:
    """
    Batch score multiple texts.
    
    Returns:
        List of alignment scores
    """
```

**Parameters:**
- `texts` (List[str]): List of texts to score

**Returns:** List[float] - Alignment scores

**Example:**
```python
texts = [
    "योगश्चित्तवृत्तिनिरोधः",
    "अहिंसा परमो धर्मः",
    "Hello world"
]

scores = scorer.score_batch(texts)
for text, score in zip(texts, scores):
    print(f"Text: {text[:30]}... Score: {score:.3f}")
# Output:
# Text: योगश्चित्तवृत्तिनिरोधः... Score: 0.867
# Text: अहिंसा परमो धर्मः... Score: 0.923
# Text: Hello world... Score: 0.045
```

##### `is_aligned()`

```python
def is_aligned(self, text: str) -> bool:
    """
    Check if text is dharmic aligned (score > threshold).
    
    Returns:
        bool: True if aligned, False otherwise
    """
```

**Parameters:**
- `text` (str): Text to check

**Returns:** bool - True if score > threshold

**Example:**
```python
text = "धर्मो रक्षति रक्षितः"
aligned = scorer.is_aligned(text)
print(f"Aligned: {aligned}")  # True

text = "Random text"
aligned = scorer.is_aligned(text)
print(f"Aligned: {aligned}")  # False
```

##### `get_top_concepts()`

```python
def get_top_concepts(
    self,
    text: str,
    k: int = 5
) -> List[Tuple[str, float]]:
    """
    Get top k most similar dharmic concepts.
    
    Returns:
        List of (concept, similarity) tuples
    """
```

**Parameters:**
- `text` (str): Text to analyze
- `k` (int, default=5): Number of top concepts

**Returns:** List[Tuple[str, float]] - (concept, similarity) pairs

**Example:**
```python
text = "योगश्चित्तवृत्तिनिरोधः तदा द्रष्टुः स्वरूपेऽवस्थानम्"
top_concepts = scorer.get_top_concepts(text, k=5)

print("Top 5 related concepts:")
for i, (concept, sim) in enumerate(top_concepts):
    print(f"{i+1}. {concept}: {sim:.3f}")
# Output:
# Top 5 related concepts:
# 1. yoga: 0.923
# 2. spiritual knowledge: 0.867
# 3. moksha: 0.834
# 4. atman: 0.812
# 5. upanishads: 0.789
```

---

## training.training_utils

### `TrainingConfig`

Training configuration dataclass.

#### Class Definition

```python
@dataclass
class TrainingConfig:
    """Complete training configuration."""
```

#### Fields

```python
# Learning rate
learning_rate: float = 5e-4
min_learning_rate: float = 1e-6
warmup_steps: int = 1000

# Batch sizes
batch_size: int = 8
gradient_accumulation_steps: int = 1

# Training duration
max_steps: int = 50000
max_epochs: int = 10

# Evaluation
eval_steps: int = 500
eval_batch_size: int = 16
save_steps: int = 1000

# Optimization
optimizer: str = 'adamw'
weight_decay: float = 0.01
max_grad_norm: float = 1.0
scheduler_type: str = 'cosine'

# Mixed precision
use_fp16: bool = False
use_bf16: bool = False
use_gradient_checkpointing: bool = False

# Memory optimization
offload_optimizer: bool = False
cpu_offload: bool = False

# Checkpointing
checkpoint_dir: str = 'checkpoints'
resume_from_checkpoint: Optional[str] = None

# Logging
log_steps: int = 10
log_level: str = 'INFO'
wandb_project: Optional[str] = None
wandb_run_name: Optional[str] = None

# Reproducibility
seed: int = 42
```

**Example:**
```python
from training.training_utils import TrainingConfig

# Default config
config = TrainingConfig()

# Custom config
config = TrainingConfig(
    learning_rate=1e-3,
    batch_size=16,
    max_steps=10000,
    use_fp16=True,
    gradient_checkpointing=True,
    seed=42
)

# Update fields
config.eval_steps = 200
config.save_steps = 500
```

---

### `LearningRateScheduler`

Learning rate scheduler with warmup.

#### Class Definition

```python
class LearningRateScheduler:
    """
    Learning rate scheduler with multiple strategies.
    
    Supports: cosine, linear, polynomial, constant with warmup
    """
```

#### Constructor

```python
def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_warmup_steps: int = 1000,
    num_training_steps: int = 50000,
    min_lr: float = 1e-6,
    power: float = 1.0
)
```

**Parameters:**
- `optimizer` (torch.optim.Optimizer): PyTorch optimizer
- `scheduler_type` (str, default='cosine'): 'cosine', 'linear', 'polynomial', 'constant'
- `num_warmup_steps` (int, default=1000): Warmup steps (linear increase)
- `num_training_steps` (int, default=50000): Total training steps
- `min_lr` (float, default=1e-6): Minimum learning rate
- `power` (float, default=1.0): Power for polynomial decay

**Example:**
```python
from training.training_utils import LearningRateScheduler
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# Cosine with warmup (recommended)
scheduler = LearningRateScheduler(
    optimizer=optimizer,
    scheduler_type='cosine',
    num_warmup_steps=1000,
    num_training_steps=10000
)

# Linear decay
scheduler = LearningRateScheduler(
    optimizer=optimizer,
    scheduler_type='linear',
    num_warmup_steps=500,
    num_training_steps=10000
)
```

#### Methods

##### `step()`

```python
def step(self, current_step: Optional[int] = None) -> None:
    """Update learning rate for current step."""
```

**Parameters:**
- `current_step` (Optional[int], default=None): Current step (increments if None)

**Example:**
```python
for step in range(10000):
    # Training code...
    optimizer.step()
    scheduler.step()  # Update LR
    
    if step % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Step {step}, LR: {current_lr:.6f}")
```

##### `get_lr()`

```python
def get_lr(self, step: int) -> float:
    """Get learning rate for specific step."""
```

**Parameters:**
- `step` (int): Step number

**Returns:** float - Learning rate

**Example:**
```python
# Plot LR schedule
import matplotlib.pyplot as plt

steps = range(10000)
lrs = [scheduler.get_lr(step) for step in steps]

plt.plot(steps, lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()
```

---

## training.advanced_trainer

### `AdvancedTrainer`

Complete training loop with all optimizations.

#### Class Definition

```python
class AdvancedTrainer:
    """
    Advanced trainer with mixed precision, gradient checkpointing,
    gradient accumulation, and comprehensive logging.
    """
```

#### Constructor

```python
def __init__(
    self,
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    metrics_computer: Optional[MetricsComputer] = None,
    criterion: Optional[nn.Module] = None
)
```

**Parameters:**
- `model` (nn.Module): PyTorch model to train
- `train_dataloader` (DataLoader): Training DataLoader
- `eval_dataloader` (Optional[DataLoader], default=None): Validation DataLoader
- `config` (Optional[TrainingConfig], default=None): Training configuration
- `checkpoint_manager` (Optional[CheckpointManager], default=None): Checkpoint manager
- `metrics_computer` (Optional[MetricsComputer], default=None): Metrics computer
- `criterion` (Optional[nn.Module], default=None): Loss function (defaults to CrossEntropyLoss)

**Example:**
```python
from training.advanced_trainer import AdvancedTrainer
from training.training_utils import TrainingConfig
from training.checkpoint_manager import CheckpointManager, RetentionPolicy
import torch.nn as nn

# Setup config
config = TrainingConfig(
    learning_rate=5e-4,
    max_steps=10000,
    use_fp16=True
)

# Setup checkpoint manager
retention_policy = RetentionPolicy(best_n=3, latest_n=2)
checkpoint_manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    retention_policy=retention_policy
)

# Create trainer
trainer = AdvancedTrainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    config=config,
    checkpoint_manager=checkpoint_manager
)
```

#### Methods

##### `train()`

```python
def train(self) -> None:
    """
    Run complete training loop.
    
    Trains model for config.max_steps or config.max_epochs,
    evaluates every config.eval_steps,
    saves checkpoints every config.save_steps.
    """
```

**Returns:** None

**Example:**
```python
# Start training
print("🕉️  Starting training...")
trainer.train()
print("✅ Training complete!")
```

##### `evaluate()`

```python
def evaluate(self) -> Dict[str, float]:
    """
    Run evaluation on validation set.
    
    Returns:
        dict with metrics: loss, perplexity, accuracy, etc.
    """
```

**Returns:** Dict[str, float] - Evaluation metrics

**Example:**
```python
# Evaluate current model
metrics = trainer.evaluate()

print("Validation Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}")
```

##### `save_checkpoint()`

```python
def save_checkpoint(
    self,
    step: int,
    metrics: Optional[Dict[str, float]] = None
) -> str:
    """
    Save checkpoint.
    
    Returns:
        str: Path to saved checkpoint
    """
```

**Parameters:**
- `step` (int): Current training step
- `metrics` (Optional[Dict[str, float]], default=None): Metrics to save

**Returns:** str - Checkpoint path

**Example:**
```python
# Manual checkpoint
checkpoint_path = trainer.save_checkpoint(
    step=1000,
    metrics={'loss': 3.45, 'perplexity': 31.5}
)
print(f"Saved: {checkpoint_path}")
```

##### `load_checkpoint()`

```python
def load_checkpoint(self, checkpoint_path: str) -> None:
    """
    Load checkpoint and restore training state.
    """
```

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint file

**Returns:** None

**Example:**
```python
# Resume training
trainer.load_checkpoint('checkpoints/checkpoint_step_5000.pt')
print(f"Resumed from step {trainer.global_step}")
trainer.train()  # Continue training
```

#### Properties

```python
@property
def global_step(self) -> int:
    """Current training step."""

@property
def current_epoch(self) -> int:
    """Current training epoch."""

@property
def best_metric(self) -> float:
    """Best metric value seen."""
```

**Example:**
```python
print(f"Current step: {trainer.global_step}")
print(f"Current epoch: {trainer.current_epoch}")
print(f"Best perplexity: {trainer.best_metric}")
```

---

## training.checkpoint_manager

### `CheckpointManager`

Manage model checkpoints with retention policies.

#### Class Definition

```python
class CheckpointManager:
    """
    Checkpoint manager with automatic cleanup and verification.
    """
```

#### Constructor

```python
def __init__(
    self,
    checkpoint_dir: str = 'checkpoints',
    retention_policy: Optional[RetentionPolicy] = None,
    verify_checksums: bool = True,
    auto_cleanup: bool = True
)
```

**Parameters:**
- `checkpoint_dir` (str, default='checkpoints'): Directory for checkpoints
- `retention_policy` (Optional[RetentionPolicy], default=None): Retention policy (keeps all if None)
- `verify_checksums` (bool, default=True): Enable SHA256 verification
- `auto_cleanup` (bool, default=True): Automatic cleanup on save

**Example:**
```python
from training.checkpoint_manager import CheckpointManager, RetentionPolicy

# Default (keep all checkpoints)
manager = CheckpointManager(checkpoint_dir='checkpoints')

# With retention policy (keep best 3 + latest 2)
retention_policy = RetentionPolicy(
    best_n=3,
    latest_n=2,
    metric_name='perplexity',
    mode='min'
)
manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    retention_policy=retention_policy,
    verify_checksums=True
)
```

#### Methods

##### `save_checkpoint()`

```python
def save_checkpoint(
    self,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> CheckpointMetadata:
    """
    Save checkpoint with metadata.
    
    Returns:
        CheckpointMetadata: Metadata for saved checkpoint
    """
```

**Parameters:**
- `model` (nn.Module): Model to save
- `optimizer` (torch.optim.Optimizer): Optimizer to save
- `scheduler` (Optional[Any], default=None): LR scheduler to save
- `step` (int, default=0): Current training step
- `epoch` (int, default=0): Current epoch
- `metrics` (Optional[Dict[str, float]], default=None): Metrics to save
- `config` (Optional[Dict[str, Any]], default=None): Configuration to save

**Returns:** CheckpointMetadata - Checkpoint metadata

**Example:**
```python
metadata = manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    step=1000,
    epoch=2,
    metrics={'loss': 3.45, 'perplexity': 31.5}
)

print(f"Saved: {metadata.path}")
print(f"Size: {metadata.size_mb:.2f} MB")
print(f"SHA256: {metadata.sha256[:16]}...")
```

##### `load_checkpoint()`

```python
def load_checkpoint(
    self,
    checkpoint_path: str,
    verify_checksum: bool = True
) -> Dict[str, Any]:
    """
    Load checkpoint from file.
    
    Returns:
        dict: Checkpoint data
    """
```

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint file
- `verify_checksum` (bool, default=True): Verify SHA256 checksum

**Returns:** Dict[str, Any] - Checkpoint data

**Example:**
```python
checkpoint = manager.load_checkpoint('checkpoints/checkpoint_step_1000.pt')

# Restore model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Loaded from step {checkpoint['step']}")
print(f"Metrics: {checkpoint['metrics']}")
```

##### `list_checkpoints()`

```python
def list_checkpoints(
    self,
    sort_by: str = 'step'
) -> List[CheckpointMetadata]:
    """
    List all checkpoints.
    
    Returns:
        List of CheckpointMetadata sorted by specified field
    """
```

**Parameters:**
- `sort_by` (str, default='step'): Sort field ('step', 'timestamp', 'size')

**Returns:** List[CheckpointMetadata] - Sorted checkpoints

**Example:**
```python
checkpoints = manager.list_checkpoints(sort_by='step')

print(f"Total checkpoints: {len(checkpoints)}")
for ckpt in checkpoints:
    print(f"Step {ckpt.step}: {ckpt.metrics.get('perplexity', 'N/A')}")
```

##### `get_best_checkpoint()`

```python
def get_best_checkpoint(
    self,
    metric_name: str,
    mode: str = 'min'
) -> Optional[CheckpointMetadata]:
    """
    Get best checkpoint by metric.
    
    Returns:
        CheckpointMetadata or None if no checkpoints
    """
```

**Parameters:**
- `metric_name` (str): Metric name ('loss', 'perplexity', 'accuracy')
- `mode` (str, default='min'): 'min' or 'max'

**Returns:** Optional[CheckpointMetadata] - Best checkpoint or None

**Example:**
```python
best_ckpt = manager.get_best_checkpoint('perplexity', mode='min')

if best_ckpt:
    print(f"Best model at step {best_ckpt.step}")
    print(f"Perplexity: {best_ckpt.metrics['perplexity']:.2f}")
    
    # Load best model
    checkpoint = manager.load_checkpoint(best_ckpt.path)
    model.load_state_dict(checkpoint['model_state_dict'])
```

##### `verify_checkpoint()`

```python
def verify_checkpoint(self, checkpoint_path: str) -> bool:
    """
    Verify checkpoint integrity via SHA256.
    
    Returns:
        bool: True if valid, False if corrupted
    """
```

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint

**Returns:** bool - True if valid

**Example:**
```python
is_valid = manager.verify_checkpoint('checkpoints/checkpoint_step_1000.pt')

if is_valid:
    print("✅ Checkpoint integrity verified")
else:
    print("❌ Checkpoint corrupted!")
```

##### `cleanup_old_checkpoints()`

```python
def cleanup_old_checkpoints(self) -> int:
    """
    Apply retention policy and delete old checkpoints.
    
    Returns:
        int: Number of checkpoints deleted
    """
```

**Returns:** int - Number deleted

**Example:**
```python
deleted = manager.cleanup_old_checkpoints()
print(f"Deleted {deleted} old checkpoints")
```

---

### `RetentionPolicy`

Checkpoint retention policy.

#### Class Definition

```python
@dataclass
class RetentionPolicy:
    """Policy for keeping/deleting checkpoints."""
    
    best_n: Optional[int] = None     # Keep N best checkpoints
    latest_n: Optional[int] = None   # Keep N latest checkpoints
    metric_name: str = 'loss'        # Metric for ranking
    mode: str = 'min'                # 'min' or 'max'
```

**Example:**
```python
from training.checkpoint_manager import RetentionPolicy

# Keep best 3 + latest 2
policy = RetentionPolicy(
    best_n=3,
    latest_n=2,
    metric_name='perplexity',
    mode='min'
)

# Keep only best 1
policy = RetentionPolicy(
    best_n=1,
    latest_n=0,
    metric_name='accuracy',
    mode='max'
)

# Keep all checkpoints
policy = RetentionPolicy(
    best_n=None,
    latest_n=None
)
```

---

## Complete Examples

### Example 1: Basic Training Script

```python
#!/usr/bin/env python3
"""Basic DharmaMind training script."""

import torch
import torch.nn as nn
from training.data_loader import create_dataloaders
from training.advanced_trainer import AdvancedTrainer
from training.training_utils import TrainingConfig
from training.checkpoint_manager import CheckpointManager, RetentionPolicy

# 1. Configuration
config = TrainingConfig(
    learning_rate=5e-4,
    batch_size=8,
    max_steps=10000,
    eval_steps=500,
    save_steps=1000,
    use_fp16=True,
    seed=42
)

# 2. Data
train_loader, val_loader, _ = create_dataloaders(
    corpus_path='data/master_corpus/complete_corpus.txt',
    batch_size=config.batch_size,
    max_length=512,
    num_workers=4
)

# 3. Model (replace with your model)
model = nn.Sequential(
    nn.Embedding(50000, 768),
    nn.Linear(768, 50000)
)

# 4. Checkpoint manager
retention_policy = RetentionPolicy(best_n=3, latest_n=2)
checkpoint_manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    retention_policy=retention_policy
)

# 5. Trainer
trainer = AdvancedTrainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    config=config,
    checkpoint_manager=checkpoint_manager
)

# 6. Train!
print("🕉️  Starting training...")
trainer.train()
print("✅ Training complete!")

# 7. Load best model
best_ckpt = checkpoint_manager.get_best_checkpoint('perplexity', mode='min')
if best_ckpt:
    print(f"💎 Best model: {best_ckpt.path}")
```

### Example 2: Training with Custom Metrics

```python
from training.metrics import MetricsComputer
from training.dharmic_metrics import DharmicAlignmentScorer

# Custom metrics computer
class CustomMetrics(MetricsComputer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignment_scorer = DharmicAlignmentScorer(threshold=0.7)
    
    def compute_all_metrics(self, predictions, references, **kwargs):
        # Base metrics
        metrics = super().compute_all_metrics(predictions, references, **kwargs)
        
        # Add dharmic alignment
        alignment_scores = self.alignment_scorer.score_batch(predictions)
        metrics['dharmic_alignment'] = sum(alignment_scores) / len(alignment_scores)
        
        return metrics

# Use custom metrics
metrics_computer = CustomMetrics()
trainer = AdvancedTrainer(
    model=model,
    train_dataloader=train_loader,
    config=config,
    metrics_computer=metrics_computer
)
```

### Example 3: Resume Training

```python
# Load existing checkpoint
checkpoint_path = 'checkpoints/checkpoint_step_5000.pt'

trainer = AdvancedTrainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    config=config,
    checkpoint_manager=checkpoint_manager
)

# Resume
trainer.load_checkpoint(checkpoint_path)
print(f"▶️  Resuming from step {trainer.global_step}")

# Continue training
trainer.train()
```

---

## Summary

The DharmaMind training API provides:

1. ✅ **Data Loading**: `DharmicCorpusDataset`, `create_dataloaders()`
2. ✅ **Embeddings**: `DharmicEmbeddingModel` with FAISS search
3. ✅ **Metrics**: `MetricsComputer` (perplexity, BLEU, ROUGE)
4. ✅ **Dharmic Metrics**: `DharmicAlignmentScorer` for spiritual alignment
5. ✅ **Training Utilities**: `TrainingConfig`, `LearningRateScheduler`
6. ✅ **Advanced Trainer**: `AdvancedTrainer` with all optimizations
7. ✅ **Checkpoints**: `CheckpointManager` with retention policies

All modules are production-ready with comprehensive documentation and examples!

---

*Last updated: October 27, 2025*
