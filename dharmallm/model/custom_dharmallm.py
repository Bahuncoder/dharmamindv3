"""
🕉️ Custom DharmaLLM - Pure PyTorch Transformer
==============================================

A completely custom transformer language model for dharmic wisdom.
NO GPT-2, NO HuggingFace dependencies - pure PyTorch implementation.

Architecture:
- Custom BPE tokenizer for Sanskrit/English
- Transformer decoder (causal language model)
- Rotary positional embeddings (RoPE)
- Multi-head self-attention with causal masking
- Optimized for dharmic text generation

May this model embody the eternal wisdom of Sanatan Dharma! 🙏
"""

import math
import json
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

class DharmicTokenizer:
    """
    Custom BPE tokenizer optimized for dharmic texts (Sanskrit + English).
    
    Features:
    - Byte-Pair Encoding (BPE) for subword tokenization
    - Special handling for Devanagari script
    - Dharmic special tokens ([DHARMA], [KARMA], etc.)
    - Efficient encoding/decoding
    """
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    
    DHARMIC_TOKENS = [
        "[DHARMA]", "[KARMA]", "[MOKSHA]", "[YOGA]", "[MEDITATION]",
        "[WISDOM]", "[COMPASSION]", "[PEACE]", "[OM]", "[MANTRA]",
        "[VEDAS]", "[GITA]", "[SANSKRIT]", "[RISHI]", "[GURU]"
    ]
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.trained = False
        
        # Initialize special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens."""
        special = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        special.extend(self.DHARMIC_TOKENS)
        
        for idx, token in enumerate(special):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def train(self, texts: List[str], min_frequency: int = 2):
        """Train tokenizer using BPE on given texts."""
        logger.info(f"Training tokenizer on {len(texts)} texts...")
        
        # Get word frequencies
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            for word in words:
                # Add end-of-word marker
                word = ' '.join(list(word)) + ' </w>'
                word_freqs[word] += 1
        
        # Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs:
            for char in word.split():
                vocab.add(char)
        
        # Add characters to token mapping
        current_idx = len(self.token_to_id)
        for char in sorted(vocab):
            if char not in self.token_to_id:
                self.token_to_id[char] = current_idx
                self.id_to_token[current_idx] = char
                current_idx += 1
        
        # BPE merges
        num_merges = self.vocab_size - current_idx
        logger.info(f"Performing {num_merges} BPE merges...")
        
        for i in range(num_merges):
            # Get pair frequencies
            pairs = Counter()
            for word, freq in word_freqs.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j + 1])] += freq
            
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break
            
            # Merge pair
            new_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_token
            
            if new_token not in self.token_to_id:
                self.token_to_id[new_token] = current_idx
                self.id_to_token[current_idx] = new_token
                current_idx += 1
            
            # Update word frequencies
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = word.replace(f"{best_pair[0]} {best_pair[1]}", new_token)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
            
            if (i + 1) % 500 == 0:
                logger.info(f"  Completed {i + 1} merges, vocab size: {len(self.token_to_id)}")
        
        self.trained = True
        logger.info(f"✅ Tokenizer trained! Final vocab size: {len(self.token_to_id)}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if not self.trained:
            raise RuntimeError("Tokenizer not trained!")
        
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.token_to_id[self.BOS_TOKEN])
        
        # Tokenize each word
        for word in text.split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        if add_special_tokens:
            tokens.append(self.token_to_id[self.EOS_TOKEN])
        
        return tokens
    
    def _tokenize_word(self, word: str) -> List[int]:
        """Tokenize a single word using BPE."""
        # Convert to character sequence with end marker
        word = ' '.join(list(word)) + ' </w>'
        
        # Apply merges
        while True:
            symbols = word.split()
            if len(symbols) == 1:
                break
            
            # Find applicable merge
            merged = False
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self.merges:
                    symbols = symbols[:i] + [self.merges[pair]] + symbols[i + 2:]
                    word = ' '.join(symbols)
                    merged = True
                    break
            
            if not merged:
                break
        
        # Convert to IDs
        tokens = []
        for symbol in word.split():
            if symbol in self.token_to_id:
                tokens.append(self.token_to_id[symbol])
            else:
                tokens.append(self.token_to_id[self.UNK_TOKEN])
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        special_ids = {self.token_to_id[t] for t in [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN] + self.DHARMIC_TOKENS}
        
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid in self.id_to_token:
                tokens.append(self.id_to_token[tid])
            else:
                tokens.append(self.UNK_TOKEN)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        text = ' '.join(text.split())  # Clean whitespace
        
        return text
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'merges': {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            'trained': self.trained
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DharmicTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        tokenizer.merges = {tuple(k.split('|||')): v for k, v in data['merges'].items()}
        tokenizer.trained = data['trained']
        
        return tokenizer
    
    def __len__(self):
        return len(self.token_to_id)
    
    @property
    def pad_token_id(self):
        return self.token_to_id[self.PAD_TOKEN]
    
    @property
    def bos_token_id(self):
        return self.token_to_id[self.BOS_TOKEN]
    
    @property
    def eos_token_id(self):
        return self.token_to_id[self.EOS_TOKEN]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DharmaLLMConfig:
    """Configuration for custom DharmaLLM."""
    vocab_size: int = 8000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'max_seq_length': self.max_seq_length,
            'dropout': self.dropout,
            'layer_norm_eps': self.layer_norm_eps,
            'rope_theta': self.rope_theta
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DharmaLLMConfig':
        return cls(**d)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DharmaLLMConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════════
# ROTARY POSITIONAL EMBEDDING (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding."""
    
    def __init__(self, dim: int, max_seq_length: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cos/sin cache
        self._build_cache(max_seq_length)
    
    def _build_cache(self, seq_length: int):
        t = torch.arange(seq_length, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_length > self.cos_cached.shape[0]:
            self._build_cache(seq_length)
        return self.cos_cached[:seq_length], self.sin_cached[:seq_length]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key.
    
    q, k: [batch, num_heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim]
    """
    # Reshape cos/sin to [1, 1, seq_len, head_dim] for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION MECHANISM
# ═══════════════════════════════════════════════════════════════════════════════

class DharmaAttention(nn.Module):
    """Multi-head self-attention with causal masking and RoPE."""
    
    def __init__(self, config: DharmaLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.dropout
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_length=config.max_seq_length,
            theta=config.rope_theta
        )
        self.attn_dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(q, seq_length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=hidden_states.device),
            diagonal=1
        ).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# FEED-FORWARD NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class DharmaFeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, config: DharmaLLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class DharmaTransformerLayer(nn.Module):
    """Single transformer decoder layer."""
    
    def __init__(self, config: DharmaLLMConfig):
        super().__init__()
        self.attention = DharmaAttention(config)
        self.feed_forward = DharmaFeedForward(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm architecture (like LLaMA)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class DharmaLLM(nn.Module):
    """
    🕉️ Custom DharmaLLM - Pure PyTorch Transformer
    
    A decoder-only transformer language model optimized for dharmic wisdom.
    """
    
    def __init__(self, config: DharmaLLMConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DharmaTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language model head (tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Calculate parameters
        self.num_parameters = sum(p.numel() for p in self.parameters())
        logger.info(f"🕉️ DharmaLLM initialized: {self.num_parameters/1e6:.1f}M parameters")
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_length]
            attention_mask: Attention mask [batch, seq_length]
            labels: Labels for language modeling loss [batch, seq_length]
            
        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)
        
        # Create attention mask
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final norm
        hidden_states = self.final_layernorm(hidden_states)
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        output = {'logits': logits}
        
        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs [batch, seq]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            
        Returns:
            Generated token IDs [batch, seq + max_new_tokens]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        for _ in range(max_new_tokens):
            # Truncate if too long
            if input_ids.shape[1] > self.config.max_seq_length:
                input_ids = input_ids[:, -self.config.max_seq_length:]
            
            # Forward pass
            outputs = self(input_ids)
            logits = outputs['logits'][:, -1, :]  # Last token logits
            
            if do_sample:
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def save(self, path: str):
        """Save model to file."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save(str(path / 'config.json'))
        
        # Save weights
        torch.save(self.state_dict(), str(path / 'model.pt'))
        
        logger.info(f"✅ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'DharmaLLM':
        """Load model from file."""
        path = Path(path)
        
        # Load config
        config = DharmaLLMConfig.load(str(path / 'config.json'))
        
        # Create model
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(str(path / 'model.pt'), map_location=device)
        model.load_state_dict(state_dict)
        
        logger.info(f"✅ Model loaded from {path}")
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class DharmicDataset(Dataset):
    """Dataset for dharmic text training."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: DharmicTokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Tokenize all texts
        for text in texts:
            if len(text.strip()) > 10:
                tokens = tokenizer.encode(text)
                # Chunk into max_length segments
                for i in range(0, len(tokens), max_length):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) >= 10:  # Minimum sequence length
                        self.examples.append(chunk)
        
        logger.info(f"Created dataset with {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        
        # Pad if necessary
        if len(tokens) < self.max_length:
            attention_mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
            attention_mask = [1] * self.max_length
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_model_small() -> Tuple[DharmaLLMConfig, DharmaLLM]:
    """Create small model (~25M params)."""
    config = DharmaLLMConfig(
        vocab_size=8000,
        hidden_size=384,
        num_layers=6,
        num_heads=6,
        intermediate_size=1536,
        max_seq_length=512,
        dropout=0.1
    )
    return config, DharmaLLM(config)


def create_model_medium() -> Tuple[DharmaLLMConfig, DharmaLLM]:
    """Create medium model (~85M params)."""
    config = DharmaLLMConfig(
        vocab_size=8000,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        intermediate_size=2048,
        max_seq_length=512,
        dropout=0.1
    )
    return config, DharmaLLM(config)


def create_model_large() -> Tuple[DharmaLLMConfig, DharmaLLM]:
    """Create large model (~200M params)."""
    config = DharmaLLMConfig(
        vocab_size=8000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        max_seq_length=1024,
        dropout=0.1
    )
    return config, DharmaLLM(config)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🕉️ Testing Custom DharmaLLM")
    print("=" * 60)
    
    # Create small model for testing
    config, model = create_model_small()
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Parameters: {model.num_parameters/1e6:.1f}M")
    
    # Test forward pass
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    outputs = model(input_ids, labels=input_ids)
    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    
    # Test generation
    print(f"\nGeneration test:")
    start_tokens = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=0.8)
    print(f"  Generated shape: {generated.shape}")
    
    print("\n✅ Custom DharmaLLM working!")

