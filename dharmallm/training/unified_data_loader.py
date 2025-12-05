"""
🕉️ Unified DharmaLLM Data Loader
=================================

Complete data loading pipeline supporting BOTH:
1. Fast plain text loading (instant, for training)
2. Rich JSON loading (with metadata, for corpus building)

Features:
- Dual-mode: text files OR JSON files
- Fast text loading: <1 second for training
- JSON loading: full metadata preservation
- Train/val/test splitting (80/10/10)
- HuggingFace tokenizer integration
- Memory-efficient processing

Author: DharmaMind Team
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DharmicTextSample:
    """Single training sample from dharmic corpus"""
    
    text: str
    source: str = "unknown"
    category: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict] = None


class UnifiedDharmicDataset(Dataset):
    """
    Unified PyTorch Dataset supporting both text and JSON sources.
    
    Mode 1: Fast text loading (one line per sample)
    Mode 2: Rich JSON loading (with metadata)
    """
    
    def __init__(
        self,
        data_source: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        min_length: int = 10,
        mode: str = "auto",  # "auto", "text", "json"
    ):
        """
        Initialize unified dataset.
        
        Args:
            data_source: Path to text file, JSON file, or directory
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            min_length: Minimum text length to include
            mode: Loading mode - "auto", "text", or "json"
        """
        self.data_source = Path(data_source)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        
        # Auto-detect mode if needed
        if mode == "auto":
            self.mode = self._detect_mode()
        else:
            self.mode = mode
        
        # Load samples
        self.samples: List[DharmicTextSample] = []
        self._load_data()
        
        logger.info(
            f"Loaded {len(self.samples)} samples using {self.mode} mode"
        )
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_source}")
    
    def _detect_mode(self) -> str:
        """Auto-detect loading mode based on data source."""
        if not self.data_source.exists():
            raise FileNotFoundError(f"Data source not found: {self.data_source}")
        
        if self.data_source.is_file():
            if self.data_source.suffix == ".txt":
                return "text"
            elif self.data_source.suffix == ".json":
                return "json"
            else:
                # Try text first, fall back to JSON
                return "text"
        elif self.data_source.is_dir():
            # Directory: look for JSON files
            return "json"
        else:
            return "text"
    
    def _load_data(self):
        """Load data based on mode."""
        if self.mode == "text":
            self._load_from_text()
        elif self.mode == "json":
            self._load_from_json()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _load_from_text(self):
        """Fast loading from plain text file (one sample per line)."""
        logger.info(f"Fast loading from text: {self.data_source}")
        
        with open(self.data_source, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                text = line.strip()
                if len(text) >= self.min_length:
                    sample = DharmicTextSample(
                        text=text,
                        source=str(self.data_source.name),
                        metadata={"line_number": line_num}
                    )
                    self.samples.append(sample)
    
    def _load_from_json(self):
        """Load from JSON file(s) with full metadata."""
        if self.data_source.is_file():
            self._load_json_file(self.data_source)
        elif self.data_source.is_dir():
            # Load from directory (old data_loader behavior)
            logger.info(f"Loading JSON files from directory: {self.data_source}")
            
            # Check for main corpus file
            main_corpus = self.data_source / "authentic_sources" / \
                         "COMPLETE_AUTHENTIC_CORPUS.json"
            if main_corpus.exists():
                self._load_json_file(main_corpus)
            
            # Load other JSON files
            patterns = [
                "authentic_sources/**/*.json",
                "complete_sanskrit_library/**/*.json",
                "*.json"
            ]
            
            for pattern in patterns:
                for json_file in self.data_source.glob(pattern):
                    if json_file.name != "COMPLETE_AUTHENTIC_CORPUS.json":
                        try:
                            self._load_json_file(json_file)
                        except Exception as e:
                            logger.warning(f"Failed to load {json_file}: {e}")
    
    def _load_json_file(self, file_path: Path):
        """Load samples from a single JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                for item in data:
                    self._add_sample_from_dict(item, file_path.name)
            elif isinstance(data, dict):
                if 'text' in data:
                    # Single sample
                    self._add_sample_from_dict(data, file_path.name)
                else:
                    # Dictionary of samples
                    for value in data.values():
                        if isinstance(value, dict) and 'text' in value:
                            self._add_sample_from_dict(value, file_path.name)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    self._add_sample_from_dict(
                                        item, file_path.name
                                    )
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    def _add_sample_from_dict(self, item: Dict, source_file: str):
        """Add a sample from dictionary."""
        if not isinstance(item, dict):
            return
        
        text = item.get('text', '').strip()
        if len(text) < self.min_length:
            return
        
        sample = DharmicTextSample(
            text=text,
            source=item.get('source', source_file),
            category=item.get('category'),
            url=item.get('url'),
            metadata=item.get('metadata'),
        )
        self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sample.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),
        }


def create_dataloaders(
    data_source: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    min_length: int = 10,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    mode: str = "auto",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with unified loading.
    
    Supports both:
    - Fast text files: data_source="path/to/corpus.txt"
    - Rich JSON files: data_source="path/to/data_dir"
    
    Args:
        data_source: Path to text file, JSON file, or directory
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        min_length: Minimum text length
        train_split: Fraction for training (default 0.8)
        val_split: Fraction for validation (default 0.1)
        test_split: Fraction for testing (default 0.1)
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        mode: "auto", "text", or "json"
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        f"Splits must sum to 1.0, got {train_split + val_split + test_split}"
    
    # Create full dataset
    full_dataset = UnifiedDharmicDataset(
        data_source=data_source,
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min_length,
        mode=mode,
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    logger.info(
        f"Dataset splits: train={train_size}, val={val_size}, test={test_size}"
    )
    
    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    logger.info(
        f"Created dataloaders: "
        f"train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches, "
        f"test={len(test_loader)} batches"
    )
    
    return train_loader, val_loader, test_loader


# Convenience aliases
create_fast_dataloaders = create_dataloaders  # Backward compatibility
